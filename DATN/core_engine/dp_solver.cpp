#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

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

extern "C" {

// Fast C++ O(N * K^2) Dynamic Programming partition solver
int solve_dp_partition(
    const LayerCost* layers, int num_layers,
    const NodeCap* nodes, int num_nodes,
    float default_rtt_ms, float default_bw_mbps,
    SliceResult* out_results
) {
    if (num_layers <= 0 || num_nodes <= 0) return 0;

    std::vector<std::vector<float>> dp(num_layers, std::vector<float>(num_nodes, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<std::pair<int, int>>> parent(num_layers, std::vector<std::pair<int, int>>(num_nodes, {-1, -1}));

    // Helper: Compute time in ms
    auto get_compute_ms = [&](int start_l, int end_l, int n_idx) -> float {
        float total_weight = 0.0f;
        float total_flops = 0.0f;
        for (int l = start_l; l <= end_l; ++l) {
            total_weight += layers[l].weight_mb;
            total_flops += layers[l].flops;
        }
        if (total_weight > nodes[n_idx].vram_available_mb * 0.90f) {
            return std::numeric_limits<float>::infinity();
        }
        float speed = std::max(nodes[n_idx].compute_tflops, 0.05f);
        float sec = (total_flops / 1e12f) / speed;
        return sec * 1000.0f;
    };

    // Helper: Comm time in ms
    auto get_comm_ms = [&](int tensor_bytes, int from_n, int to_n) -> float {
        if (from_n == to_n) return 0.0f;
        float transfer_sec = (tensor_bytes * 8.0f / 1e6f) / default_bw_mbps;
        return (default_rtt_ms / 2.0f) + (transfer_sec * 1000.0f);
    };

    // Base case: layer 0
    for (int j = 0; j < num_nodes; ++j) {
        float c = get_compute_ms(0, 0, j);
        if (!std::isinf(c)) {
            dp[0][j] = c;
        }
    }

    // DP transitions
    for (int i = 1; i < num_layers; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            for (int k = 0; k <= i; ++k) {
                float c_time = get_compute_ms(k, i, j);
                if (std::isinf(c_time)) continue;

                if (k == 0) {
                    if (c_time < dp[i][j]) {
                        dp[i][j] = c_time;
                        parent[i][j] = {-1, -1};
                    }
                } else {
                    for (int p = 0; p < num_nodes; ++p) {
                        if (std::isinf(dp[k - 1][p])) continue;
                        float comm_time = get_comm_ms(layers[k - 1].tensor_bytes, p, j);
                        float total = dp[k - 1][p] + comm_time + c_time;
                        if (total < dp[i][j]) {
                            dp[i][j] = total;
                            parent[i][j] = {k - 1, p};
                        }
                    }
                }
            }
        }
    }

    // Reconstruct solution
    int best_node = -1;
    float min_cost = std::numeric_limits<float>::infinity();
    for (int j = 0; j < num_nodes; ++j) {
        if (dp[num_layers - 1][j] < min_cost) {
            min_cost = dp[num_layers - 1][j];
            best_node = j;
        }
    }

    if (best_node == -1) return 0; // Solution infeasible

    std::vector<SliceResult> trace;
    int curr_l = num_layers - 1;
    int curr_n = best_node;

    while (curr_l >= 0) {
        auto p = parent[curr_l][curr_n];
        int start_l = (p.first == -1) ? 0 : p.first + 1;
        
        float c_time = get_compute_ms(start_l, curr_l, curr_n);
        float comm_time = (p.first != -1) ? get_comm_ms(layers[start_l - 1].tensor_bytes, p.second, curr_n) : 0.0f;

        trace.push_back({nodes[curr_n].node_id, start_l, curr_l, c_time, comm_time});

        if (p.first == -1) break;
        curr_l = p.first;
        curr_n = p.second;
    }

    std::reverse(trace.begin(), trace.end());
    for (size_t i = 0; i < trace.size(); ++i) {
        out_results[i] = trace[i];
    }

    return static_cast<int>(trace.size());
}

} // extern "C"
