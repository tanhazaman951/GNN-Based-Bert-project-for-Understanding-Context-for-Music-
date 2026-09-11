import os
import json
import torch
import torch.nn.functional as F
import numpy as np

def build_segment_graph(signal, num_samples=220500, num_segments=10):
    """
    Constructs dynamic audio segment graph from raw audio signal tensor (1, T).
    
    Node Features:
      - 10 temporal frames.
      - Mean and std extracted per window into a 24-dimensional feature vector.
      
    Edges:
      - Temporal edges: Bi-directional connections between adjacent segments (i <-> i+1).
      - Similarity edges: Cosine similarity > 0.5 for non-adjacent segments.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
        
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    window_size = num_samples // num_segments
    segment_feats = []
    
    for i in range(num_segments):
        seg = signal[:, i * window_size : (i + 1) * window_size]
        mean_val = seg.mean(dim=-1, keepdim=True)
        std_val = seg.std(dim=-1, keepdim=True) + 1e-6
        feat = torch.cat([mean_val, std_val], dim=0).view(-1)
        
        # Expand or pad to fixed 24-dim feature vector per segment
        if feat.shape[0] < 24:
            feat = F.pad(feat, (0, 24 - feat.shape[0]))
        else:
            feat = feat[:24]
        segment_feats.append(feat)

    node_features = torch.stack(segment_feats)  # (10, 24)

    # Build Graph Edges
    edges = []
    # 1. Temporal adjacent edges
    for i in range(num_segments - 1):
        edges.extend([[i, i + 1], [i + 1, i]])

    # 2. Cosine similarity edges (> 0.5)
    norm_feat = F.normalize(node_features, p=2, dim=-1)
    sim_matrix = torch.mm(norm_feat, norm_feat.t())
    for i in range(num_segments):
        for j in range(i + 2, num_segments):
            if sim_matrix[i, j] > 0.5:
                edges.extend([[i, j], [j, i]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_features, edge_index

def save_preprocessed_graph_samples(output_dir="data/processed", num_samples=25):
    """
    Generates and saves at least 25 example .pt and .json graph samples into data/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    torch.manual_seed(42)

    saved_paths = []
    for idx in range(num_samples):
        # Generate synthetic audio signal
        t = np.linspace(0, 10.0, 220500, endpoint=False)
        freq = 220.0 + idx * 15.0
        signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        node_feats, edge_index = build_segment_graph(signal_tensor)

        graph_dict = {
            "sample_id": f"sample_{idx:03d}",
            "num_nodes": int(node_feats.shape[0]),
            "feature_dim": int(node_feats.shape[1]),
            "node_features": node_feats.tolist(),
            "edge_index": edge_index.tolist()
        }

        # Save JSON representation
        json_path = os.path.join(output_dir, f"sample_{idx:03d}_graph.json")
        with open(json_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

        # Save PyTorch .pt representation
        pt_path = os.path.join(output_dir, f"sample_{idx:03d}_graph.pt")
        torch.save({
            "x": node_feats,
            "edge_index": edge_index
        }, pt_path)

        saved_paths.append(pt_path)

    print(f"Saved {len(saved_paths)} preprocessed graph samples (.pt and .json) in '{output_dir}'.")
    return saved_paths

if __name__ == "__main__":
    save_preprocessed_graph_samples()
