# Semantic Matching Solver for Linguistic Olympiad

&gt; A prototype system that tackles **semantic-matching problems** in linguistics olympiads via **graph neural networks**, **compositional word semantics**, and **combinatorial search**.

---

## 1. Problem Statement

In a typical problem instance you are given

| Side | Content |
|------|---------|
| **Left** | A set of **unknown-language word strings** (possibly sharing sub-strings) |
| **Right** | A list of **native-language words/phrases** whose **meanings correspond 1-to-1** to the left words |

Both lists are **scrambled**.  
Goal: recover the **correct bijection** so that each unknown word is matched to its native translation.

We treat this as a **graph-based matching problem**:

1. Build a **bipartite graph** whose left nodes are unknown words and right nodes are native phrases.
2. Left edges encode **morphological overlap** (currently “share a token ⇔ edge exists”).
3. Right similarities are **FastText cosine scores**.
4. Search for the permutation that **best aligns structural and semantic neighbourhoods**.

---

## 2. System Overview
| Composer (frozen) |----->| Data Generator |----->| Graph Model |

| Module | Role |
|--------|------|
| `datagen.py` | Produce **synthetic matching instances** |
| `wmdl.py` | **Composer**: merges several token vectors into one phrase vector |
| `g_model.py` | **Heterogeneous GNN** that infers token vectors from structure alone |
| `matchit.py` | **Matching solver** (exact enumeration for n≤10) |
| `exft.py` | FastText wrapper for semantic similarities |
| `sa_.py` | Generic **Simulated Annealing** framework (unused yet) |

---

## 3. Data Generation Pipeline

1. **Random graph**  
   - Skeleton nodes + leaves → degree sequence → `nx.configuration_model`
   - Remove self-loops, ensure simplicity

2. **Token sampling**  
   - Randomly pick `n` word vectors from a FastText slice (row 200‥10 000)

3. **Phrase composition**  
   - For every hyper-edge `(i,j,…)` run **frozen Composer** → phrase vector  
   - Hyper-edges = graph edges + random triples + singletons

4. **Graph construction**  
   - Nodes: `token` (zero-feature) and `phrase` (embedding)  
   - Edges: `token ─occurs in→ phrase` and reverse  
   - Save tuple `(gProblem_T, gold_token_vectors)`

---

## 4. Training the Graph Model

| Item | Setting |
|------|---------|
| Framework | PyTorch Geometric |
| Model | `TokenSemantics` (2-layer Hetero-TransformerConv) |
| Input | `token.x = zeros (n, 1)`; `phrase.x = composer_emb`; edge_index |
| Target | **Pre-generated token vectors** (`gold_token_vectors`) |
| Loss | `MSELoss(pred_token_vec, gold_token_vec)` |
| Optimiser | AdamW |
| Scheduler | Cosine annealing |
| Batch | Many small graphs packed into one large disconnected graph |

> **Note**: phrase vectors are **only used as node features**; they **never appear in the loss**.

---

## 5. Inference & Solving

After training we freeze the GNN.

Given a **real problem instance**:

4. Search for the **permutation** that maximises structural-semantic agreement  
   - Currently **full enumeration** (feasible for n≤9~10)
   - SA backend ready but **not integrated**
1. Build the same bipartite graph (left words → right phrases)
2. Run GNN → obtain **inferred token vectors** for unknown words
3. Compute **cosine similarity** between these vectors and FastText vectors of native phrases


---

## 6. Quick Start
### 6.1 training & data generating...
### 6.2 solving the puzzle

## 7. Current Limitations
| Limitation                                                 | Plan                                          |
| ---------------------------------------------------------- | --------------------------------------------- |
| Only **synthetic data** used; no fine-tune on real puzzles | Collect & annotate real problems              |
| Composer is **frozen**; may generate biased phrases        | Train Composer on genuine token→phrase pairs  |
| Exact enumeration → **O(n!)**                              | Plug in **Simulated Annealing** (`sa_.py`)    |
| Left graph uses **binary edges** only                      | Introduce **edge types** (morph, syllable, …) |
| Single-language FastText                                   | Switch to **multilingual embeddings**         |