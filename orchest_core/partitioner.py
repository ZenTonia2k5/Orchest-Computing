import logging

logger = logging.getLogger("orchest.partitioner")

class ModelSpecification:
    """Defines the computational and memory specifications of a model."""
    def __init__(self, name="Qwen2-1.5B", num_layers=28, total_params=1.54e9, hidden_size=2048):
        self.name = name
        self.num_layers = num_layers
        self.total_params = total_params
        self.hidden_size = hidden_size
        
        # Determine byte precision dynamically based on model size (simulating quantization)
        if total_params > 1.5e10:
            self.bytes_per_param = 0.5  # 4-bit quantization (standard for 30B+ models on consumer GPUs)
        elif total_params > 5.0e9:
            self.bytes_per_param = 1.0  # 8-bit quantization
        else:
            self.bytes_per_param = 2.0  # FP16 precision
            
        self.base_mem_mb = 250.0
        self.layer_mem_mb = (self.total_params * self.bytes_per_param) / (1024 * 1024) / num_layers
        self.layer_flops = (2 * self.total_params) / num_layers
        self.activation_size_mb = (self.hidden_size * 384 * 2.0) / (1024 * 1024) # Activations are kept in FP16

    def get_layer_memory(self, layer_idx):
        return self.layer_mem_mb

    def get_layer_flops(self, layer_idx):
        return self.layer_flops

    def get_activation_size(self, layer_idx):
        return self.activation_size_mb

class OrchestPartitioner:
    def __init__(self, model_spec: ModelSpecification):
        self.model_spec = model_spec

    def partition(self, nodes):
        """
        Partitions model layers across stages: Host -> Worker-1 -> Worker-2 -> ... -> Host
        to minimize total latency, ensuring contiguous slices and respecting memory limits.
        """
        N = self.model_spec.num_layers
        node_ids = list(nodes.keys())
        M = len(node_ids)
        
        # Find host node ID
        host_node_id = None
        for nid, ninfo in nodes.items():
            if ninfo.get("is_host", False):
                host_node_id = nid
                break
        if not host_node_id and M > 0:
            host_node_id = node_ids[0]
            
        # Get workers list
        workers = [nid for nid in node_ids if nid != host_node_id]
        
        # Stages sequence: Host -> Workers in order -> Host
        stage_nodes = [host_node_id] + workers + [host_node_id]
        S = len(stage_nodes)
        
        # Network helper
        def get_network(src_id, dest_id):
            if src_id == dest_id:
                return 10000.0, 0.0
            src_info = nodes[src_id]
            bw = src_info.get("network_bandwidth", {}).get(dest_id, 100.0)
            ping = src_info.get("network_ping", {}).get(dest_id, 5.0)
            return max(bw, 0.01), ping / 1000.0

        INF = float('inf')
        # dp[i][s] = min latency to execute layers i..N-1 starting with stage s
        dp = [[INF] * (S + 1) for _ in range(N + 1)]
        parent = [[None] * (S + 1) for _ in range(N + 1)]
        
        # Base case: completed all layers
        for s in range(S + 1):
            dp[N][s] = 0.0
            
        # Fill DP table bottom-up
        for i in range(N - 1, -1, -1): # layer index from N-1 down to 0
            for s in range(S - 1, -1, -1): # stage index from S-1 down to 0
                node_id = stage_nodes[s]
                node_mem_limit = nodes[node_id].get("available_ram_mb", 1024.0)
                
                # Option 1: Bypass stage s (assign 0 layers to stage s)
                if dp[i][s+1] < dp[i][s]:
                    dp[i][s] = dp[i][s+1]
                    parent[i][s] = ("bypass", s+1)
                    
                # Option 2: Assign contiguous slice layers i..k to stage s
                accumulated_mem = 0.0
                accumulated_flops = 0.0
                
                for k in range(i, N):
                    accumulated_mem += self.model_spec.get_layer_memory(k)
                    accumulated_flops += self.model_spec.get_layer_flops(k)
                    
                    if accumulated_mem > node_mem_limit:
                        # Exceeded memory for this stage slice
                        break
                        
                    # Calculate compute time
                    compute_time = accumulated_flops / (nodes[node_id].get("gflops", 10.0) * 1e9)
                    
                    # Communication to next stage
                    if s == S - 1:
                        # Final stage, no communication needed to next stage
                        comm_time = 0.0
                    else:
                        next_node_id = stage_nodes[s+1]
                        bw, ping_s = get_network(node_id, next_node_id)
                        act_size = self.model_spec.get_activation_size(k)
                        comm_time = 0.0 if node_id == next_node_id else (ping_s + (act_size / bw))
                        
                    cost = compute_time + comm_time + dp[k+1][s+1]
                    if cost < dp[i][s]:
                        dp[i][s] = cost
                        parent[i][s] = (k, s+1)
                        
        min_latency = dp[0][0]
        if min_latency == INF:
            logger.error("No feasible stage partition found! Fallback triggered.")
            # Fallback partition: distribute evenly, ignoring memory
            fallback_assignment = []
            for idx in range(N):
                # Distribute layers across all available nodes
                fallback_assignment.append(node_ids[idx % M])
            return fallback_assignment, 999.0
            
        # Reconstruct path
        assignment = [None] * N
        curr_layer = 0
        curr_stage = 0
        
        while curr_layer < N and curr_stage < S:
            p_val = parent[curr_layer][curr_stage]
            if not p_val:
                break
            if p_val[0] == "bypass":
                curr_stage = p_val[1]
            else:
                k, next_stage = p_val
                node_id = stage_nodes[curr_stage]
                for l in range(curr_layer, k + 1):
                    assignment[l] = node_id
                curr_layer = k + 1
                curr_stage = next_stage
                
        # Fill in any unassigned layers just in case of edge cases
        for l in range(N):
            if assignment[l] is None:
                assignment[l] = host_node_id
                
        # Warn if memory is exceeded on any node across its slices
        actual_mem_usage = {nid: 0.0 for nid in node_ids}
        for l in range(N):
            assigned_node = assignment[l]
            actual_mem_usage[assigned_node] += self.model_spec.get_layer_memory(l)
            
        for nid, mem_used in actual_mem_usage.items():
            limit = nodes[nid].get("available_ram_mb", 1024.0)
            if mem_used > limit:
                logger.warning(f"Node {nid} allocated {mem_used:.1f}MB which exceeds limit {limit:.1f}MB.")
                
        return assignment, min_latency
