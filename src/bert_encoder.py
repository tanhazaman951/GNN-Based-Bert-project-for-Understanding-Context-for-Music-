import torch
import torch.nn as nn

class BERTClassifier(nn.Module):
    """
    Task 1: BERT Multi-Label Text Baseline Classifier.
    Fine-tunes bert-base-uncased transformer backbone or self-contained PyTorch Transformer layer.
    """
    def __init__(self, num_classes=50, pretrained_name="bert-base-uncased", force_fast=True):
        super(BERTClassifier, self).__init__()
        self.num_classes = num_classes
        self.use_hf = False

        if not force_fast:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, local_files_only=True)
                self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
                self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
                self.use_hf = True
            except Exception:
                self.use_hf = False

        if not self.use_hf:
            embed_dim = 256
            self.token_embed = nn.Embedding(5000, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, 128, embed_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(embed_dim, num_classes)
            self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None):
        if self.use_hf:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
            logits = self.fc(pooled_output)
            return logits
        else:
            B, L = input_ids.shape
            x = self.token_embed(input_ids) + self.pos_embed[:, :L, :]
            h = self.transformer(x)
            pooled_output = h.mean(dim=1)
            logits = self.fc(pooled_output)
            return logits

    def extract_features(self, input_ids, attention_mask=None):
        """Returns sequence representation H_text (B, L, 768 or 256)."""
        if self.use_hf:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
        else:
            B, L = input_ids.shape
            x = self.token_embed(input_ids) + self.pos_embed[:, :L, :]
            return self.transformer(x)

if __name__ == "__main__":
    model = BERTClassifier(num_classes=50)
    input_ids = torch.randint(0, 1000, (4, 128))
    logits = model(input_ids)
    print(f"Task 1 BERT Classifier logits shape: {logits.shape}")
