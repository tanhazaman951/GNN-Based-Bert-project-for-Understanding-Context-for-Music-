import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing PyTorch Geometric SAGEConv and global_mean_pool
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

class AudioCNNBaseline(nn.Module):
    """
    Task 2 Baseline: 2D Mel-Spectrogram Convolutional Neural Network.
    """
    def __init__(self, num_classes=50):
        super(AudioCNNBaseline, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FallbackSAGEConv(nn.Module):
    """Fallback GraphSAGE Conv layer when PyG binary wheels are offline."""
    def __init__(self, in_channels, out_channels):
        super(FallbackSAGEConv, self).__init__()
        self.fc_self = nn.Linear(in_channels, out_channels, bias=False)
        self.fc_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        neigh_sum = torch.zeros(N, x.size(1), device=x.device)
        deg = torch.zeros(N, 1, device=x.device)

        neigh_sum.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
        deg = torch.clamp(deg, min=1.0)
        neigh_mean = neigh_sum / deg

        out = self.fc_self(x) + self.fc_neigh(neigh_mean) + self.bias
        return F.relu(out)

class AudioGraphSAGE(nn.Module):
    """
    Task 2 GNN: GraphSAGE message passing over 10-node segment graphs.
    """
    def __init__(self, in_channels=24, hidden_channels=64, out_channels=128):
        super(AudioGraphSAGE, self).__init__()
        self.pyg = PYG_AVAILABLE
        if self.pyg:
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            self.conv1 = FallbackSAGEConv(in_channels, hidden_channels)
            self.conv2 = FallbackSAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        if self.pyg:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            if batch is not None:
                return global_mean_pool(x, batch)
            else:
                return x.mean(dim=0, keepdim=True)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            if batch is None:
                return x.mean(dim=0, keepdim=True)
            else:
                num_graphs = int(batch.max().item()) + 1
                g = torch.zeros(num_graphs, x.size(1), device=x.device)
                counts = torch.zeros(num_graphs, 1, device=x.device)
                g.index_add_(0, batch, x)
                counts.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
                return g / torch.clamp(counts, min=1.0)

class AudioGNNClassifier(nn.Module):
    """
    Task 2 Classifier operating on batched segment graphs.
    """
    def __init__(self, num_classes=50):
        super(AudioGNNClassifier, self).__init__()
        self.gnn = AudioGraphSAGE(in_channels=24, out_channels=128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch=None):
        g = self.gnn(x, edge_index, batch)
        return self.fc(g)

if __name__ == "__main__":
    cnn = AudioCNNBaseline(num_classes=50)
    spec = torch.randn(4, 1, 128, 100)
    cnn_logits = cnn(spec)
    print(f"Task 2 CNN Logits shape: {cnn_logits.shape}")

    gnn = AudioGNNClassifier(num_classes=50)
    x_nodes = torch.randn(40, 24)
    edges = torch.tensor([[0, 1, 10, 11], [1, 0, 11, 10]], dtype=torch.long)
    batch_map = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10, dtype=torch.long)
    gnn_logits = gnn(x_nodes, edges, batch_map)
    print(f"Task 2 GNN Logits shape: {gnn_logits.shape}")
