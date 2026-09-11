import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score

def compute_classification_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes Macro-F1, Micro-F1, and mean AUC-PR matching Kaggle metric evaluation.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > threshold).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))

    auc_pr_list = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            precision, recall, _ = precision_recall_curve(y_true[:, k], y_probs[:, k])
            auc_pr_list.append(auc(recall, precision))

    mean_auc_pr = float(np.mean(auc_pr_list)) if len(auc_pr_list) > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "auc_pr": round(mean_auc_pr, 4)
    }

def plot_f1_curves(history, output_path="results/plots/f1_curves.png"):
    """Plots Macro-F1 and Micro-F1 curves vs training epochs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_f1"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], 'o-', label="Train Macro-F1", color="#1f77b4")
    plt.plot(epochs, history["val_f1"], 's-', label="Val Macro-F1", color="#ff7f0e")
    if "val_micro_f1" in history:
        plt.plot(epochs, history["val_micro_f1"], '^--', label="Val Micro-F1", color="#2ca02c")

    plt.title("GNN-BERT Model Training & Validation F1 Curves", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved F1 curves plot to '{output_path}'")

def plot_tsne_clusters(embeddings, labels, output_path="results/plots/tsne_clusters.png"):
    """Generates 2D t-SNE / PCA plot of fused z representations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    embeddings = np.array(embeddings)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    unique_labels = list(set(labels))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(9, 7))
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(
            coords[idx, 0], coords[idx, 1],
            label=label, alpha=0.8, edgecolors='w', s=70, color=cmap(i % 10)
        )

    plt.title("t-SNE / PCA Visualization of Fused GNN-BERT Representations (z)", fontsize=12, fontweight='bold')
    plt.xlabel("Component 1", fontsize=11)
    plt.ylabel("Component 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Primary Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to '{output_path}'")
