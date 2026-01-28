import json
import argparse
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
import copy
import os


def plot_training_history(history: dict, output_path: str):
    """Plots and saves the training history curves for loss and accuracy."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWarning: matplotlib not installed. Skipping plot generation.")
        print("Please run 'pip install matplotlib' to enable plotting.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MLP Probe Training History', fontsize=16)

    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    plt.close(fig)
    print(f"Training plot saved to {output_path}")


class RepeatDataset(Dataset):
    def __init__(self, jsonl_path: str, prefix_tokens: int = 32):
        self.samples: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.prefix_tokens = prefix_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        q = item.get("q") or item.get("Q")
        t = item.get("t") or item.get("T") or item.get("answer")
        label = int(item.get("repeat", 0))
        # 若标注阶段已经为每条样本记录了 "prefix_len"（表示在 <think> 中被判定为重复的前缀 token 数），
        # 则优先使用该动态长度；否则使用初始化时给定的 self.prefix_tokens。
        dynamic_len = item.get("prefix_len", None)
        if isinstance(dynamic_len, int) and dynamic_len > 0:
            ptoks = dynamic_len
        else:
            ptoks = self.prefix_tokens
        prefix = " ".join(t.split()[: ptoks])
        return q, prefix, label


def collate_fn(batch, embedder):
    qs, ps, labels = zip(*batch)
    embs_q = embedder.encode(list(qs), show_progress_bar=False)
    embs_p = embedder.encode(list(ps), show_progress_bar=False)
    feats = np.concatenate([embs_q, embs_p], axis=1)
    feats = torch.tensor(feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return feats, labels


class RepeatDetector(nn.Module):
    """A two-layer MLP for detecting repeats, with an option for a linear probe."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Linear probe if hidden_dim is 0
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a probe MLP for repeat detection, inspired by best practices."
    )
    parser.add_argument("data", help="Labeled JSONL file for training.")
    parser.add_argument("output", help="Path to save the best model state_dict.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer.")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden layer dimension. Set to 0 for a linear probe.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--val_split_ratio", type=float, default=0.2, help="Ratio of data to use for validation.")
    parser.add_argument("--loss_alpha", type=float, default=1.0, help="Scaling factor for weighted loss.")
    parser.add_argument("--embedding_model", default="/path/to/your/embedding_model/", help="SentenceTransformer model name.")
    parser.add_argument("--prefix_tokens", type=int, default=32, help="Number of tokens from the start of the answer to consider.")
    parser.add_argument("--log_file", help="Optional path to save training history (e.g., losses, accuracies) as a JSON file.")
    parser.add_argument("--plot_file", help="Optional path to save a plot of training history.")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedder = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    dataset = RepeatDataset(args.data, prefix_tokens=args.prefix_tokens)

    # Split dataset
    val_size = int(args.val_split_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Calculate pos_weight for weighted loss from the training set
    train_labels = [train_dataset.dataset.samples[i]["repeat"] for i in train_dataset.indices]
    num_positives = sum(train_labels)
    num_negatives = len(train_labels) - num_positives
    
    if num_positives == 0 or num_negatives == 0:
        print("Warning: Training data contains only one class. Weighted loss is disabled.")
        pos_weight = torch.tensor(1.0, device=device)
    else:
        w = num_negatives / num_positives
        pos_weight = torch.tensor(args.loss_alpha * w, device=device)
        print(f"Calculated pos_weight for loss: {pos_weight.item():.2f} (alpha={args.loss_alpha}, w={w:.2f})")


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, embedder),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, embedder),
    )

    input_dim = embedder.get_sentence_embedding_dimension() * 2
    model = RepeatDetector(input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }

    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            preds = model(feats)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted_labels = torch.sigmoid(preds) > 0.5
            train_total += labels.size(0)
            train_correct += (predicted_labels == (labels > 0.5)).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                preds = model(feats)
                loss = loss_fn(preds, labels)
                total_val_loss += loss.item()

                predicted_labels = torch.sigmoid(preds) > 0.5
                val_total += labels.size(0)
                val_correct += (predicted_labels == (labels > 0.5)).sum().item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model found with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs with no improvement.")
            break
    
    print("\nTraining finished.")
    if best_model_state:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(best_model_state, args.output)
        print(f"Best model saved to {args.output} (validation loss: {best_val_loss:.4f})")
    else:
        print("Warning: No best model state was saved. This might happen if validation did not improve.")

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_file, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {args.log_file}")

    if args.plot_file:
        plot_training_history(history, args.plot_file)


if __name__ == "__main__":
    main()


