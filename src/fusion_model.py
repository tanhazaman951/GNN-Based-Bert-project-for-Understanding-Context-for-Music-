import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BERTClassifier
from src.gnn_model import AudioGraphSAGE

class CrossAttentionFusion(nn.Module):
    """
    Task 3: Cross-Attention Fusion Layer.
    Queries text token sequence states (H_text) using audio graph embedding g.
    """
    def __init__(self, g_dim=128, text_dim=768, hidden_dim=256):
        super(CrossAttentionFusion, self).__init__()
        self.W_q = nn.Linear(g_dim, hidden_dim)
        self.W_k = nn.Linear(text_dim, hidden_dim)
        self.W_v = nn.Linear(text_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, g, H_text, attention_mask=None):
        # g: (B, g_dim)
        # H_text: (B, L, text_dim)
        Q = self.W_q(g).unsqueeze(1)    # (B, 1, hidden_dim)
        K = self.W_k(H_text)            # (B, L, hidden_dim)
        V = self.W_v(H_text)            # (B, L, hidden_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale # (B, 1, L)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # (B, 1, L)
        context = torch.bmm(attn_weights, V).squeeze(1) # (B, hidden_dim)
        
        # Fused vector z = [g || context] shape (B, g_dim + hidden_dim)
        return torch.cat([g, context], dim=-1)


class GNNBERTFusionModel(nn.Module):
    """
    Task 3: GNN-BERT Multitask Fusion Model.
    Predicts 50-D aspect tags (BCE) and 2-D valence/arousal emotion targets (MSE).
    """
    def __init__(self, num_classes=50, num_emotion=2, text_dim=768):
        super(GNNBERTFusionModel, self).__init__()
        self.text_encoder = BERTClassifier(num_classes=num_classes)
        self.gnn = AudioGraphSAGE(in_channels=24, out_channels=128)
        
        # Determine actual text representation dimension (768 or 256)
        if hasattr(self.text_encoder, 'embed_dim'):
            text_dim = self.text_encoder.embed_dim
            
        self.fusion = CrossAttentionFusion(g_dim=128, text_dim=text_dim, hidden_dim=256)
        
        fused_dim = 128 + 256 # 384D
        self.classifier = nn.Linear(fused_dim, num_classes)
        self.emotion_head = nn.Linear(fused_dim, num_emotion)

    def forward(self, input_ids, attention_mask, x_nodes, edge_index, batch_map=None):
        H_text = self.text_encoder.extract_features(input_ids, attention_mask) # (B, L, text_dim)
        g = self.gnn(x_nodes, edge_index, batch_map)                            # (B, 128)
        
        if g.size(0) != H_text.size(0):
            g = g.repeat(H_text.size(0), 1)[:H_text.size(0)]

        z = self.fusion(g, H_text, attention_mask=attention_mask)              # (B, 384)

        tag_logits = self.classifier(z)
        emotion_preds = self.emotion_head(z)
        return tag_logits, emotion_preds

if __name__ == "__main__":
    model = GNNBERTFusionModel(num_classes=50, num_emotion=2)
    input_ids = torch.randint(0, 1000, (4, 128))
    attention_mask = torch.ones(4, 128)
    x_nodes = torch.randn(40, 24)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_map = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10, dtype=torch.long)

    tag_logits, emotion_preds = model(input_ids, attention_mask, x_nodes, edges, batch_map)
    print(f"Task 3 Fusion Tag Logits shape: {tag_logits.shape}, Emotion Preds shape: {emotion_preds.shape}")
