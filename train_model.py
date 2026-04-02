import os
import json
import copy
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Settings
# =========================
DATA_PATH = "data.csv"
MODEL_PATH = "sentiment_gnn_model.pth"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LABELMAP_PATH = "label_map.json"

HIDDEN_DIM = 64
EPOCHS = 200
PATIENCE = 20
K_NEIGHBORS = 10
MAX_FEATURES = 5000


# =========================
# Load and clean data
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH} in the current folder.")

df = pd.read_csv(DATA_PATH, encoding="latin-1", on_bad_lines="skip")
df.columns = [c.strip().lower() for c in df.columns]

if "phrase" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("CSV must contain exactly these columns: phrase, sentiment")

df["phrase"] = (
    df["phrase"]
    .astype(str)
    .str.replace('"', "", regex=False)
    .str.strip()
)

df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.replace('"', "", regex=False)
    .str.lower()
    .str.strip()
)

df = df[(df["phrase"] != "") & (df["sentiment"] != "")].copy()

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(label_map)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["phrase", "label"]).reset_index(drop=True)

print("Rows after cleaning:", len(df))
print("Label counts:")
print(df["label"].value_counts().sort_index())
print()

if len(df) < 10:
    raise ValueError("Dataset is too small. Add more rows before training.")

if df["label"].nunique() < 2:
    raise ValueError("Need at least 2 classes to train a classifier.")

id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

with open(LABELMAP_PATH, "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, indent=2)


# =========================
# TF-IDF features
# =========================
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["phrase"])
y_np = df["label"].to_numpy()

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("TF-IDF shape:", X.shape)


# =========================
# Build graph with k-NN
# =========================
n_samples = X.shape[0]
if n_samples < 2:
    raise ValueError("Need at least 2 samples to build a graph.")

k = min(K_NEIGHBORS, n_samples - 1)

nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)

edges = []
for i in range(n_samples):
    for j in indices[i, 1:]:
        edges.append([i, int(j)])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index, num_nodes=n_samples)

x = torch.tensor(X.toarray(), dtype=torch.float)
y = torch.tensor(y_np, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print("Graph nodes:", data.num_nodes)
print("Graph edges:", data.num_edges)
print()


# =========================
# Train / Val / Test masks
# =========================
all_idx = np.arange(n_samples)

try:
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=SEED,
        stratify=y_np
    )

    temp_labels = y_np[temp_idx]
    if len(np.unique(temp_labels)) > 1 and min(np.bincount(temp_labels)) >= 2:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=SEED,
            stratify=temp_labels
        )
    else:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=SEED
        )
except ValueError:
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED
    )

train_mask = torch.zeros(n_samples, dtype=torch.bool)
val_mask = torch.zeros(n_samples, dtype=torch.bool)
test_mask = torch.zeros(n_samples, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print("Train/Val/Test sizes:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
print()


# =========================
# Model
# =========================
class SentimentGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=0.3)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)

        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

model = SentimentGAT(
    input_dim=data.num_node_features,
    hidden_dim=HIDDEN_DIM,
    num_classes=3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# =========================
# Helpers
# =========================
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        acc = accuracy(out[mask], data.y[mask])
    return loss, acc


# =========================
# Training
# =========================
best_val_acc = -1.0
best_state = None
patience_count = 0

print("Training started...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    train_loss, train_acc = evaluate(data.train_mask)
    val_loss, val_acc = evaluate(data.val_mask)
    test_loss, test_acc = evaluate(data.test_mask)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"Test Acc {test_acc:.4f}"
        )

    if patience_count >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =========================
# Save best model
# =========================
if best_state is None:
    best_state = model.state_dict()

torch.save(
    {
        "model_state_dict": best_state,
        "input_dim": data.num_node_features,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": 3
    },
    MODEL_PATH
)

print("\nSaved:")
print(f"Model      -> {MODEL_PATH}")
print(f"Vectorizer -> {VECTORIZER_PATH}")
print(f"Labels     -> {LABELMAP_PATH}")

model.load_state_dict(best_state)
final_test_loss, final_test_acc = evaluate(data.test_mask)
print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")