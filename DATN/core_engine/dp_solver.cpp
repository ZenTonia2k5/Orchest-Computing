#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdint>

// ---------------------------------------------------------------------------
// Native layer-partitioning solver for Orchest Computing.
//
// Searches the space of (ordered selection of distinct nodes x contiguous layer
// cuts) with a Held-Karp style subset DP:
//
//     D[S][v][i]            min cost to run layers 0..i using exactly node set S,
//                           with v in S running the final stage
//     D[{v}][v][i]        = c(0, i, v)
//     D[S|{u}][u][j]      = min over v in S, i < j of
//                              D[S][v][i] (+ | max) w(i) + c(i+1, j, u),  u not in S
//     answer              = min over S, v in S of D[S][v][N-1]
//
// The `u not in S` guard is a correctness requirement, not an optimization. The
// memory ceiling is a per-node aggregate, so a node holding two disjoint segments
// would need its working sets summed before the ceiling is applied. The previous
// dp[layer][node] formulation here omitted that guard and could therefore return
// node-reusing plans whose memory was only ever checked one segment at a time --
// i.e. plans that OOM on the device while scoring as feasible.
//
// NOTE ON COST-MODEL DIVERGENCE: the helpers below are deliberately cruder than
// the Python model in orchest/partitioner/cost_matrix.py. They check weights only
// (no KV cache, no transient activations, no backend workspace) and assume a
// single uniform link for every node pair. This file fixes the SEARCH SPACE only;
// the two cost models are not unified, so results will not match the Python
// solver numerically. Treat Python's PartitionCostCalculator as authoritative.
// ---------------------------------------------------------------------------

namespace {

// 2^K * K * N floats of DP state, so the fleet size has to be bounded. At K=12 and
// N=36 the tables are ~21 MB; beyond that we degrade to the fixed-order DP instead
// of allocating gigabytes.
constexpr int kMaxSubsetNodes = 12;
constexpr float kAlpha = 0.90f;

constexpr float kInf = std::numeric_limits<float>::infinity();

}  // namespace

struct NodeCap {
    int node_id;
    float compute_tflops;
    float vram_available_mb;
};

struct LayerCost {
    int layer_idx;
    float weight_mb;
    float flops;
    int tensor_bytes;
};

struct SliceResult {
    int node_id;
    int layer_start;
    int layer_end;
    float est_compute_ms;
    float est_comm_ms;
};

namespace {

// c(a, b, v): cost of layers [a..b] on nodes[v], or +inf if the segment's weights
// breach the node's safety ceiling. Flattened as comp[v * N * N + a * N + b].
std::vector<float> precompute_compute_ms(
    const LayerCost* layers, int num_layers,
    const NodeCap* nodes, int num_nodes
) {
    const int N = num_layers;
    std::vector<float> comp(static_cast<size_t>(num_nodes) * N * N, kInf);

    for (int v = 0; v < num_nodes; ++v) {
        const float ceiling = nodes[v].vram_available_mb * kAlpha;
        const float speed = std::max(nodes[v].compute_tflops, 0.05f);
        const size_t base_v = static_cast<size_t>(v) * N * N;

        for (int a = 0; a < N; ++a) {
            const size_t row = base_v + static_cast<size_t>(a) * N;
            float weight = 0.0f;
            float flops = 0.0f;
            for (int b = a; b < N; ++b) {
                weight += layers[b].weight_mb;
                flops += layers[b].flops;
                // Monotone in b: once the segment overflows, so does every longer
                // segment from the same start.
                if (weight > ceiling) break;
                comp[row + b] = ((flops / 1e12f) / speed) * 1000.0f;
            }
        }
    }
    return comp;
}

// w(p): cost of shipping the activation produced by layer p to the next stage.
// Endpoint-independent here because the ABI carries one uniform link rather than
// an RTT/bandwidth matrix; the DP still keeps a last-node dimension so a per-pair
// matrix can be dropped in without reshaping the state.
std::vector<float> precompute_comm_ms(
    const LayerCost* layers, int num_layers,
    float default_rtt_ms, float default_bw_mbps
) {
    std::vector<float> comm(static_cast<size_t>(num_layers), 0.0f);
    const float bw = std::max(default_bw_mbps, 1.0f);
    for (int p = 0; p + 1 < num_layers; ++p) {
        const float transfer_sec = (layers[p].tensor_bytes * 8.0f / 1e6f) / bw;
        comm[p] = (std::max(default_rtt_ms, 0.05f) / 2.0f) + (transfer_sec * 1000.0f);
    }
    return comm;
}

// Fallback for fleets too large for 2^K state: the fixed-order prefix DP, where
// stage s runs on nodes[s]. Each node is used at most once by construction, so the
// per-node memory check stays sound; the plan is only optimal for the given node
// ordering. Returns the number of slices written to out_results.
int solve_fixed_order(
    const LayerCost* layers, int num_layers,
    const NodeCap* nodes, int num_nodes,
    const std::vector<float>& comp,
    const std::vector<float>& comm,
    int objective,
    SliceResult* out_results
) {
    const int N = num_layers;
    const int K = num_nodes;
    const bool use_sum = (objective == 0);

    std::vector<float> dp(static_cast<size_t>(K) * N, kInf);
    std::vector<int> par(static_cast<size_t>(K) * N, -1);

    for (int i = 0; i < N; ++i) {
        const float c = comp[static_cast<size_t>(i)];  // v = 0, a = 0
        if (!std::isinf(c)) dp[static_cast<size_t>(i)] = c;
    }

    for (int s = 1; s < K; ++s) {
        const size_t srow = static_cast<size_t>(s) * N;
        const size_t prow = static_cast<size_t>(s - 1) * N;
        const size_t cbase = static_cast<size_t>(s) * N * N;
        for (int i = 0; i < N; ++i) {
            for (int k = 1; k <= i; ++k) {
                const float prev = dp[prow + (k - 1)];
                if (std::isinf(prev)) continue;
                const float c = comp[cbase + static_cast<size_t>(k) * N + i];
                if (std::isinf(c)) continue;
                const float stage = comm[k - 1] + c;
                const float cand = use_sum ? (prev + stage) : std::max(prev, stage);
                if (cand < dp[srow + i]) {
                    dp[srow + i] = cand;
                    par[srow + i] = k - 1;
                }
            }
        }
    }

    int best_s = -1;
    float best = kInf;
    for (int s = 0; s < K; ++s) {
        const float cost = dp[static_cast<size_t>(s) * N + (N - 1)];
        if (cost < best) { best = cost; best_s = s; }
    }
    if (best_s == -1) return 0;

    std::vector<SliceResult> trace;
    int curr_i = N - 1;
    int curr_s = best_s;
    while (curr_s >= 0 && curr_i >= 0) {
        const int cut = par[static_cast<size_t>(curr_s) * N + curr_i];
        const int start_l = (cut == -1) ? 0 : cut + 1;
        const float c = comp[static_cast<size_t>(curr_s) * N * N +
                             static_cast<size_t>(start_l) * N + curr_i];
        const float w = (cut == -1) ? 0.0f : comm[start_l - 1];
        trace.push_back({nodes[curr_s].node_id, start_l, curr_i, c, w});
        if (cut == -1) break;
        curr_i = cut;
        curr_s -= 1;
    }

    std::reverse(trace.begin(), trace.end());
    for (size_t i = 0; i < trace.size(); ++i) out_results[i] = trace[i];
    return static_cast<int>(trace.size());
}

}  // namespace

extern "C" {

// Exact subset DP over node subsets.
//
//   objective: 0 = minimize the sum of stage times   (single-request prefill TTFT)
//              1 = minimize the slowest stage        (steady-state throughput)
//
// Both objectives decompose over the same per-stage quantity t_s = w_in(s) + c(s),
// i.e. the incoming transfer is charged to the downstream stage, which models a
// blocking recv-then-compute transport.
//
// out_results must have room for num_nodes entries. Returns the number of slices
// written, or 0 when no assignment satisfies the memory ceiling.
int solve_dp_partition_ex(
    const LayerCost* layers, int num_layers,
    const NodeCap* nodes, int num_nodes,
    float default_rtt_ms, float default_bw_mbps,
    int objective,
    SliceResult* out_results
) {
    if (num_layers <= 0 || num_nodes <= 0) return 0;
    if (objective != 0 && objective != 1) return 0;

    const int N = num_layers;
    const int K = num_nodes;
    const bool use_sum = (objective == 0);

    const std::vector<float> comp = precompute_compute_ms(layers, N, nodes, K);
    const std::vector<float> comm = precompute_comm_ms(layers, N, default_rtt_ms, default_bw_mbps);

    if (K > kMaxSubsetNodes) {
        std::cerr << "[orchest_core] fleet of " << K << " nodes exceeds the subset-DP "
                  << "limit of " << kMaxSubsetNodes << "; falling back to the "
                  << "fixed-order DP (optimal only for the given node ordering)."
                  << std::endl;
        return solve_fixed_order(layers, N, nodes, K, comp, comm, objective, out_results);
    }

    const uint32_t num_masks = 1u << K;
    const size_t state_count = static_cast<size_t>(num_masks) * K * N;
    std::vector<float> dp(state_count, kInf);
    std::vector<int> par_v(state_count, -1);   // node index of the preceding stage
    std::vector<int> par_i(state_count, -1);   // last layer of the preceding stage

    auto state = [K, N](uint32_t mask, int v, int i) -> size_t {
        return (static_cast<size_t>(mask) * K + v) * N + i;
    };

    // Base case: node v alone covers layers 0..i
    for (int v = 0; v < K; ++v) {
        const size_t cbase = static_cast<size_t>(v) * N * N;  // a = 0
        for (int i = 0; i < N; ++i) {
            const float c = comp[cbase + i];
            if (!std::isinf(c)) dp[state(1u << v, v, i)] = c;
        }
    }

    // Forward relaxation. mask | (1 << u) > mask, so ascending mask order is a valid
    // topological order over the state graph.
    for (uint32_t mask = 1; mask < num_masks; ++mask) {
        for (int v = 0; v < K; ++v) {
            if (!((mask >> v) & 1u)) continue;
            for (int i = 0; i + 1 < N; ++i) {
                const float cur = dp[state(mask, v, i)];
                if (std::isinf(cur)) continue;
                const float link = comm[i];
                for (int u = 0; u < K; ++u) {
                    if ((mask >> u) & 1u) continue;
                    const uint32_t nmask = mask | (1u << u);
                    const size_t cbase = static_cast<size_t>(u) * N * N +
                                         static_cast<size_t>(i + 1) * N;
                    for (int j = i + 1; j < N; ++j) {
                        const float c = comp[cbase + j];
                        if (std::isinf(c)) break;  // monotone in j
                        const float stage = link + c;
                        const float cand = use_sum ? (cur + stage) : std::max(cur, stage);
                        const size_t idx = state(nmask, u, j);
                        if (cand < dp[idx]) {
                            dp[idx] = cand;
                            par_v[idx] = v;
                            par_i[idx] = i;
                        }
                    }
                }
            }
        }
    }

    uint32_t best_mask = 0;
    int best_v = -1;
    float best = kInf;
    for (uint32_t mask = 1; mask < num_masks; ++mask) {
        for (int v = 0; v < K; ++v) {
            if (!((mask >> v) & 1u)) continue;
            const float cost = dp[state(mask, v, N - 1)];
            if (cost < best) { best = cost; best_mask = mask; best_v = v; }
        }
    }
    if (best_v == -1) return 0;  // no assignment fits the memory ceiling

    std::vector<SliceResult> trace;
    uint32_t mask = best_mask;
    int v = best_v;
    int i = N - 1;
    while (true) {
        const size_t idx = state(mask, v, i);
        const int pv = par_v[idx];
        const int pi = par_i[idx];
        const int start_l = (pv == -1) ? 0 : pi + 1;
        const float c = comp[static_cast<size_t>(v) * N * N +
                             static_cast<size_t>(start_l) * N + i];
        const float w = (pv == -1) ? 0.0f : comm[start_l - 1];
        trace.push_back({nodes[v].node_id, start_l, i, c, w});
        if (pv == -1) break;
        mask ^= (1u << v);
        v = pv;
        i = pi;
    }

    std::reverse(trace.begin(), trace.end());
    for (size_t t = 0; t < trace.size(); ++t) out_results[t] = trace[t];
    return static_cast<int>(trace.size());
}

// Stable entry point kept for existing callers: latency-sum objective.
int solve_dp_partition(
    const LayerCost* layers, int num_layers,
    const NodeCap* nodes, int num_nodes,
    float default_rtt_ms, float default_bw_mbps,
    SliceResult* out_results
) {
    return solve_dp_partition_ex(
        layers, num_layers, nodes, num_nodes,
        default_rtt_ms, default_bw_mbps, /*objective=*/0, out_results
    );
}

}  // extern "C"
