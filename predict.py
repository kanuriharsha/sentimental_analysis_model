import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# =========================
# File paths
# =========================
MODEL_PATH = "sentiment_gnn_model.pth"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LABELMAP_PATH = "label_map.json"


# =========================
# Model class must match training exactly
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


# =========================
# Load vectorizer
# =========================
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# =========================
# Load label map
# =========================
with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
    raw_label_map = json.load(f)

# Make it robust whether JSON keys are strings or ints
id_to_label = {}
for k, v in raw_label_map.items():
    try:
        id_to_label[int(k)] = v
    except Exception:
        id_to_label[k] = v


# =========================
# Load model checkpoint
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

input_dim = checkpoint["input_dim"]
hidden_dim = checkpoint["hidden_dim"]
num_classes = checkpoint["num_classes"]

model = SentimentGAT(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =========================
# Predict function
# =========================
def predict(text: str):
    text = str(text).strip()
    if not text:
        return {
            "prediction": "neutral",
            "confidence": {
                "negative": 0.0,
                "neutral": 1.0,
                "positive": 0.0
            }
        }

    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float)

    # Single-node graph for inference
    data = Data(
        x=vec,
        edge_index=torch.empty((2, 0), dtype=torch.long)
    )

    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = id_to_label.get(pred_id, id_to_label.get(str(pred_id), "neutral"))

    return {
        "prediction": pred_label,
        "confidence": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        }
    }


# =========================
# Demo
# =========================
if __name__ == "__main__":
    samples = [
        "today was good",
        "i don't like this experience",
        "the experience is neither good nor bad"
    ]

    for s in samples:
        result = predict(s)
        print("\nText:", s)
        print("Prediction:", result["prediction"])
        print("Confidence:", result["confidence"])
