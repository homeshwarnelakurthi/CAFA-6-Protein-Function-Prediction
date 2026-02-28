# 🧬 CAFA-6 Protein Function Prediction

> **Kaggle Competition | Biomedical AI | Multi-Label Classification**  
> Predicting biological functions of proteins using Gene Ontology (GO) terms from amino acid sequences.

[![Kaggle](https://img.shields.io/badge/Kaggle-CAFA--6-blue)](https://kaggle.com/competitions/cafa-6-protein-function-prediction)

---

## 📌 Overview

Proteins drive nearly every biological process in the human body — from oxygen transport to immune response. Yet the functions of thousands of proteins remain unknown. The **CAFA-6 (Critical Assessment of Functional Annotation)** competition challenges participants to predict **Gene Ontology (GO) terms** for proteins solely from their amino acid sequences.

This repository documents a full ML engineering pipeline — from data loading and embedding, through multi-architecture neural network design, ensemble strategies, and ontology-aware post-processing — achieving a **best public leaderboard score of 0.209** across **30+ systematic experiments**.

---

## 🏆 Results Summary

| Metric | Value |
|--------|-------|
| **Best Public Leaderboard Score** | **0.209** |
| Best Approach | 3-Model Ensemble (1500 terms each) + GO Graph Propagation |
| Total Submissions | 30+ |
| Training Proteins | 82,404 |
| Test Proteins | 224,309 |
| Evaluation Metric | Information-Accretion Weighted F1 |

---

## 🧠 Problem Formulation

Given a protein's amino acid sequence, predict which **Gene Ontology (GO) terms** apply across three subontologies:

| Subontology | Code | What It Describes |
|---|---|---|
| **Molecular Function** | `F` | What the protein does molecularly (e.g., binding, catalysis) |
| **Biological Process** | `P` | Which biological processes the protein participates in |
| **Cellular Component** | `C` | Where in the cell the protein is located |

**Core challenges solved in this project:**
- **Multi-label classification** — each protein maps to dozens of GO terms simultaneously
- **Hierarchical label structure** — GO terms form a Directed Acyclic Graph (DAG); child term predictions must propagate to parents
- **Label noise & incompleteness** — open-world annotations (absence ≠ non-function)
- **Class imbalance** — top GO terms appear thousands of times; rare terms appear once
- **Scale** — 224,309 test proteins, 1500–5000 GO terms per subontology

---

## 📁 Repository Structure

```
CAFA-6-Protein-Function-Prediction/
│
├── bottleneck-compression-network-prop.ipynb   # SE-ResNet + Bottleneck architectures
├── cafa6-knn-ensemble-v1.ipynb                 # KNN retrieval + weighted ensemble
├── cafa6-optimization-v4-2000terms.ipynb       # Optimization across 2000 GO terms + pseudo-labeling
├── new-model2.ipynb                            # DeepTax model + KNN blend (taxonomy features)
├── new-models-goo.ipynb                        # Diverse architecture ensemble (Simple/Deep/Wide)
├── newmodel.ipynb                              # 3-model ensemble baseline (best score: 0.209)
├── README.md
└── LICENSE
```

---

## 🔬 Technical Pipeline

### Step 1 — Data Loading & Preprocessing

All experiments used **ESM (Evolutionary Scale Modeling)** protein embeddings of shape `(N, 1280)` — 1280-dimensional feature vectors per protein, pre-computed from amino acid sequences using Meta's ESM protein language model.

```python
# Load pre-computed ESM embeddings
train_emb = np.load("train_embeds.npy").astype(np.float32)  # shape: (82404, 1280)
test_emb  = np.load("test_embeds.npy").astype(np.float32)   # shape: (224309, 1280)

# Z-score normalization (critical for stable training)
mean = train_emb.mean(axis=0)
std  = train_emb.std(axis=0) + 1e-6
train_emb = (train_emb - mean) / std
test_emb  = (test_emb  - mean) / std
```

**Target matrix construction** — for each aspect (F/P/C), build a sparse binary label matrix:

```python
# Select top-N most frequent GO terms
top_terms  = aspect_terms['term'].value_counts().index[:1500].tolist()
term_map   = {t: i for i, t in enumerate(top_terms)}

# Build binary label matrix
label_matrix = np.zeros((len(train_ids), len(top_terms)), dtype=np.float32)
for _, row in relevant.iterrows():
    if row['id'] in id_to_idx:
        label_matrix[id_to_idx[row['id']], term_map[row['term']]] = 1.0
```

---

### Step 2 — Model Architectures

#### Architecture A: Simple MLP (Baseline — Score: 0.195)

The starting point — a clean 2-layer MLP trained independently per GO aspect:

```python
class SimpleModel(nn.Module):
    def __init__(self, n_feat, n_class):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_class)
        )
    def forward(self, x): return self.net(x)
```

- **Loss:** `BCEWithLogitsLoss` (numerically stable binary cross-entropy)
- **Optimizer:** Adam (lr=1e-3)
- **Epochs:** 10
- **Batch size:** 256

---

#### Architecture B: SE-ResNet (Squeeze-and-Excitation ResNet)

Recalibrates the importance of each embedding dimension dynamically using channel attention:

```python
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(x.view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)

class SEResNet1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 1280)
        self.se1 = SEBlock(1280)
        self.res1 = nn.Sequential(nn.Linear(1280, 1280), nn.ReLU(), nn.Dropout(0.2))
        self.se2 = SEBlock(1280)
        self.res2 = nn.Sequential(nn.Linear(1280, 1280), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        residual = x
        x_se = self.se1(x.view(x.size(0), x.size(1), 1)).view(x.size(0), x.size(1))
        x = self.res1(x_se) + residual
        residual = x
        x_se = self.se2(x.view(x.size(0), x.size(1), 1)).view(x.size(0), x.size(1))
        x = self.res2(x_se) + residual
        return self.classifier(x)
```

**Key insight:** Treating the 1280 ESM embedding dimensions as "channels" allows SE attention to learn which embedding features matter most for GO term prediction.

---

#### Architecture C: Bottleneck Compression Network (Score: 0.157)

Forces the model to learn a compressed biological representation before predicting GO terms:

```python
class BottleneckNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),   # ← The bottleneck (compression)
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.decoder_classifier = nn.Sequential(
            nn.Linear(256, 1024),   # ← Expansion
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.decoder_classifier(self.encoder(x))
```

**Key insight:** The 256-dimensional bottleneck forces the network to capture the most biologically relevant features, acting as an information bottleneck.

---

#### Architecture D: DeepTax Model + KNN Blend (Score: 0.145)

Incorporates **taxonomic (species) information** as a learnable embedding alongside protein sequence features:

```python
class DeepTaxModel(nn.Module):
    def __init__(self, n_feat, n_taxons, n_class):
        super().__init__()
        self.tax_emb = nn.Embedding(n_taxons, 128)  # 9835 species → 128-dim embedding
        self.net = nn.Sequential(
            nn.Linear(n_feat + 128, 2048),  # concat ESM + taxonomy
            nn.BatchNorm1d(2048),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, n_class)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, self.tax_emb(t)], dim=1))
```

**Training details:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=15)
- Gradient clipping: max_norm=1.0
- Blended with KNN predictions: `final = 0.6 × neural + 0.4 × knn`

---

#### Architecture E: Diverse Architecture Ensemble (Score: 0.165)

Three architecturally distinct models trained and averaged:

```python
class Model_A(nn.Module):  # Shallow (2-layer, 512 hidden)
    ...
class Model_B(nn.Module):  # Deep (3-layer: 1024 → 512 → output)
    ...
class Model_C(nn.Module):  # Wide + BatchNorm (1024 hidden, dropout=0.4)
    ...

# Average predictions from all three
final_preds = np.mean([preds_A, preds_B, preds_C], axis=0)
```

---

### Step 3 — GO Hierarchy Propagation (Post-Processing)

**Every top-performing submission** used ontology-aware propagation. Since GO terms form a Directed Acyclic Graph, a prediction for a child term implies the parent term must also be predicted.

```python
import obonet

# Parse GO ontology graph
graph = obonet.read_obo("go-basic.obo")
parent_map = {n: list(graph.successors(n)) for n in graph.nodes()}

# BFS propagation: push scores up the GO DAG
for pid, preds in submission_dict.items():
    final_scores = preds.copy()
    queue = list(preds.keys())
    visited = set(queue)

    while queue:
        term = queue.pop(0)
        score = final_scores.get(term, 0.0)
        for parent in parent_map.get(term, []):
            if final_scores.get(parent, 0.0) < score:
                final_scores[parent] = score       # propagate max score upward
                if parent not in visited:
                    queue.append(parent)
                    visited.add(parent)
```

**Why this matters:** Without propagation, scores drop significantly because the weighted F1 metric penalizes missing parent terms that are implied by predicted child terms.

---

### Step 4 — Pseudo-Labeling (Advanced Technique)

To expand the training set, high-confidence test predictions were used as additional training labels:

```python
CONFIDENCE_THRESHOLD = 0.75  # Only use very confident predictions

# Stage 1: Generate high-confidence pseudo-labels from base ensemble
for char in ['F', 'P', 'C']:
    final_preds = np.mean([model_preds_1, model_preds_2, model_preds_3], axis=0)
    for i, pid in enumerate(test_ids):
        high_conf = np.where(final_preds[i] > CONFIDENCE_THRESHOLD)[0]
        if len(high_conf) > 0:
            pseudo_labels[char][pid] = {terms[idx]: final_preds[i, idx] for idx in high_conf}

# Result: 20,333 total pseudo-labels generated
# F: 5,854 | P: 1,406 | C: 13,073

# Stage 2: Augment training set
augmented_emb    = np.vstack([train_emb, pseudo_emb])       # 82,404 + 5,000 = 87,404
augmented_labels = np.vstack([original_labels, pseudo_label_matrix])

# Stage 3: Retrain on augmented data (lr=8e-4, epochs=12)
```

---

### Step 5 — Best Submission: 3-Model Ensemble (Score: **0.209**)

```python
CONFIG = {
    "n_terms": 1500,
    "n_models": 3,
}

# Train 3 models with different random seeds per aspect
for model_idx in range(3):
    torch.manual_seed(42 + model_idx)
    # Train SimpleModel on aspect F/P/C

# Ensemble: average predictions across all 3 models
final_preds = np.mean([preds_model_0, preds_model_1, preds_model_2], axis=0)

# Post-process: threshold → top-k selection → GO graph propagation
```

**Training output (Aspect F, 3 models):**
```
Model 1/3: Epoch 3 Loss=0.0040 | Epoch 6 Loss=0.0033 | Epoch 9 Loss=0.0030
Model 2/3: Epoch 3 Loss=0.0040 | Epoch 6 Loss=0.0033 | Epoch 9 Loss=0.0030
Model 3/3: Epoch 3 Loss=0.0040 | Epoch 6 Loss=0.0033 | Epoch 9 Loss=0.0030
```

**Scale of predictions generated:**
- 82,404 training proteins across 9,835 species
- 224,309 test proteins predicted
- Up to 1,500 GO terms per protein per aspect
- Final submission: ~10M+ prediction rows before pruning

---

## 📊 Full Submission Log

| Score | Approach |
|-------|----------|
| **0.209** | 🥇 3-Model Ensemble (1500 terms each) + Graph Prop |
| 0.195 | Simple Baseline (1500 terms, 2-layer MLP) |
| 0.187 | kSIM-2 + MLP + Ontology Propagation (Graph Consistency) |
| 0.178 | WeakMLP (1500 terms) + GO Hierarchy (Child-Parent) |
| 0.178 | kSIM-2 + Simple 2-Layer MLP + Graph Prop |
| 0.175 | Ensemble (WeakMLP + Weighted KNN) + Hierarchy Prop + Top-75 |
| 0.170 | kSIM-MLP + Hierarchy Prop + Balanced Threshold |
| 0.165 | Diamond Ensemble (8 Models) + Graph Propagation |
| 0.165 | Diamond Ensemble (Standard) + MLP + Dropout + Graph Prop |
| 0.162 | WeakMLP (1500) + Hierarchy Propagation (Child-Parent) |
| 0.160 | Diverse k-NN Ensemble (Single+Deep+Wide) 1500 terms |
| 0.157 | SE-ResNet + Bottleneck Ensemble + Propagation |
| 0.157 | kSIM-MLP + Hierarchy Prop + Dynamic Threshold |
| 0.145 | DeepTax (kSIM-2 + taxonomy info) + Graph Propagation |
| 0.140 | Threshold-0.2 |
| 0.133 | Deep Tax + KNN Blend (k=51) + Graph Prop |
| 0.132 | Taxonomy Model (kSIM-2 + tax info2) + Graph Prop |
| 0.111 | kSIM-2 SUM + Multi-Head Neural Network |
| 0.089 | KNN Baseline (k=51) + Cosine Sim + Top 1500 Terms |
| 0.047 | kSIM-2 Embeddings + MLP (M=2 Only) + 1500 Clusters |

---

## 🧰 Tech Stack

| Tool | Role |
|------|------|
| **Python 3.10** | Primary language |
| **PyTorch** | All neural network architectures |
| **TensorFlow** | Alternative model experiments |
| **scikit-learn** | KNN (NearestNeighbors, cosine metric), preprocessing |
| **LightGBM / XGBoost** | Gradient boosting ensemble experiments |
| **Hugging Face Transformers** | ESM, ProtBERT protein language models |
| **obonet** | GO ontology graph parsing |
| **networkx** | Graph traversal for propagation |
| **NumPy / Pandas** | Data processing and label matrix construction |
| **CUDA (GPU)** | Training acceleration on Kaggle T4/P100 |

---

## 📈 Key Learnings

**1. Ensembles beat complex single models**
A 3-model ensemble of simple MLPs (0.209) outperformed a deep SE-ResNet (0.157) and Bottleneck architecture (0.157) — showing that variance reduction through ensembling is more valuable than architectural complexity for this task.

**2. GO hierarchy propagation is non-negotiable**
Every submission without propagation scored 0.03–0.05 lower. The GO DAG structure means unpropagated predictions violate the hierarchical annotation contract and are penalized heavily by the weighted F1 metric.

**3. Simpler is often better with sparse labels**
The label matrix sparsity was ~0.001 (0.1%) — meaning most proteins have very few positive labels. In this regime, overly complex models (DeepTax, 5000-term models) overfit or underfit more easily than a well-regularized 2-layer MLP.

**4. Threshold calibration has outsized impact**
Testing thresholds from 0.005 to 0.02 showed meaningful score differences. The best threshold was 0.01 — below that, too many false positives; above that, recall dropped sharply for rare GO terms.

**5. Taxonomy features didn't generalize well**
Despite incorporating 9,835 species embeddings into the DeepTaxModel, scores (0.132–0.145) were lower than the MLP baseline. Hypothesis: species information added noise rather than signal because many test proteins come from under-represented species in training.

**6. Pseudo-labeling showed theoretical promise**
Generating 20,333 high-confidence pseudo-labels (threshold=0.75) and retraining on augmented data reduced training loss further but wasn't directly validated on leaderboard due to submission limits.

---

## 🔗 References

- [CAFA-6 Kaggle Competition](https://kaggle.com/competitions/cafa-6-protein-function-prediction)
- [ESM: Evolutionary Scale Modeling — Meta AI](https://github.com/facebookresearch/esm)
- [Gene Ontology Consortium](http://geneontology.org/)
- [obonet — GO graph parser](https://github.com/dhimmel/obonet)
- [Jiang et al. (2016) — Weighted F1 Evaluation Metric](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6)
- [ProtBERT (Hugging Face)](https://huggingface.co/Rostlab/prot_bert)

---

## 👤 Author

**Homeshwar Nelakurthi**  
Master's in Data Science | ML Engineer | Healthcare AI  
[GitHub](https://github.com/homeshwarnelakurthi)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
