# scfm-repro

Make single-cell foundation model (scFM) embeddings reproducible, and check it.

UCE, scPRINT and scGPT contain random sampling steps without a seed, so each run gives a
different cell embedding. This package (1) seeds everything, (2) replaces those steps with
fixed rules, and (3) checks two runs with the kNN Jaccard index.

```python
import scfm_repro as R
R.seed_everything(0)
R.fix("scgpt")            # "uce", "scprint" or "scgpt"; call before running the official code
# ... run the model twice -> Z1, Z2 ...
print(R.knn_jaccard(Z1, Z2))   # 1.0 = same neighbours
```

| model   | random step replaced |
|---------|----------------------|
| UCE     | expression-weighted gene sampling → largest-remainder allocation; chromosome shuffle → fixed order |
| scPRINT | random padding genes → fixed order |
| scGPT   | random bin for tied values → fixed bin; random gene subset (>1,200 genes) → top-expressed genes |

Paper: *Same Cosine, Different Neighbours: Unseeded Sampling Makes Single-Cell Foundation Model Embeddings Irreproducible* (ISBRA 2027 submission).
