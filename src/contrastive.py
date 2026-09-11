import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BERTClassifier
from src.gnn_model import AudioGraphSAGE

class ContrastiveDualEncoder(nn.Module):
    """
    Task 4: Contrastive Learning Dual Encoder for Cross-Modal MusicCaps Alignment.
    Projects text and audio representations into a shared 256-D normalized metric space.
    Optimized via symmetric InfoNCE loss (temperature tau = 0.07).
    """
    def __init__(self, embed_dim=256, temperature=0.07):
        super(ContrastiveDualEncoder, self).__init__()
        self.temperature = temperature
        self.text_encoder = BERTClassifier(num_classes=50)
        
        text_dim = 768
        if hasattr(self.text_encoder, 'embed_dim'):
            text_dim = self.text_encoder.embed_dim

        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.gnn = AudioGraphSAGE(in_channels=24, out_channels=128)
        self.graph_proj = nn.Linear(128, embed_dim)

    def forward(self, input_ids, attention_mask, x_nodes, edge_index, batch_map=None):
        H_text = self.text_encoder.extract_features(input_ids, attention_mask)
        text_cls = H_text[:, 0, :] if H_text.dim() == 3 else H_text
        t_embed = F.normalize(self.text_proj(text_cls), p=2, dim=-1)

        g = self.gnn(x_nodes, edge_index, batch_map)
        if g.size(0) != t_embed.size(0):
            g = g.repeat(t_embed.size(0), 1)[:t_embed.size(0)]

        g_embed = F.normalize(self.graph_proj(g), p=2, dim=-1)
        return g_embed, t_embed

def compute_retrieval_metrics(g_embeds, t_embeds, topk=(1, 5, 10)):
    """
    Computes Caption -> Audio and Audio -> Caption Recall@K metrics.
    """
    if isinstance(g_embeds, torch.Tensor):
        g_embeds = g_embeds.detach().cpu().numpy()
    if isinstance(t_embeds, torch.Tensor):
        t_embeds = t_embeds.detach().cpu().numpy()

    sim_matrix = np.dot(t_embeds, g_embeds.T)
    N = sim_matrix.shape[0]
    
    results = {}
    
    # Caption -> Audio (Query Caption, retrieve Audio)
    c2a_ranks = []
    for i in range(N):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        c2a_ranks.append(rank)
        
    for k in topk:
        results[f'Caption_Audio_R@{k}'] = float(np.mean(np.array(c2a_ranks) <= min(k, N)))

    # Audio -> Caption (Query Audio, retrieve Caption)
    a2c_ranks = []
    for j in range(N):
        sorted_indices = np.argsort(-sim_matrix[:, j])
        rank = np.where(sorted_indices == j)[0][0] + 1
        a2c_ranks.append(rank)
        
    for k in topk:
        results[f'Audio_Caption_R@{k}'] = float(np.mean(np.array(a2c_ranks) <= min(k, N)))

    return results

if __name__ == "__main__":
    model = ContrastiveDualEncoder()
    input_ids = torch.randint(0, 1000, (4, 128))
    attention_mask = torch.ones(4, 128)
    x_nodes = torch.randn(40, 24)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_map = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10, dtype=torch.long)

    g_e, t_e = model(input_ids, attention_mask, x_nodes, edges, batch_map)
    print(f"Task 4 Embeddings -> Audio: {g_e.shape}, Text: {t_e.shape}")
    res = compute_retrieval_metrics(g_e, t_e)
    print("Retrieval Metrics sample:", res)
