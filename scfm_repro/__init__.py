"""scfm-repro: make single-cell foundation model (scFM) embeddings reproducible, and check it.

Usage (call before running the model's official inference code):
    import scfm_repro as R
    R.seed_everything(0)      # seed Python, NumPy and PyTorch
    R.fix("uce")              # or "scprint" / "scgpt": replace random sampling with fixed rules
    ...run the model...
    R.knn_jaccard(Z1, Z2)     # check two runs: mean kNN Jaccard (1 = same neighbours)
"""
import os, random, types
import numpy as np

__all__ = ["seed_everything", "fix", "knn_jaccard", "centred_cosine"]


def seed_everything(seed=0):
    """Seed Python, NumPy and PyTorch (if installed)."""
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _np_with_random(**overrides):
    """A copy of the numpy module whose np.random has some functions replaced."""
    rnd = types.SimpleNamespace(**{k: getattr(np.random, k) for k in dir(np.random) if not k.startswith("__")})
    for k, v in overrides.items():
        setattr(rnd, k, v)
    mod = types.ModuleType("numpy_fixed"); mod.__dict__.update(np.__dict__); mod.random = rnd
    return mod


def _fix_uce():
    """UCE: genes are allocated in proportion to expression by largest-remainder rounding
    (instead of np.random.choice), and chromosomes keep ascending order (instead of shuffle)."""
    import eval_data  # module of the official UCE code

    def quota(a, size=None, p=None, replace=True):
        q = np.floor(p * size).astype(int); r = size - q.sum()
        if r > 0:
            frac = p * size - np.floor(p * size)
            q[np.lexsort((np.arange(len(p)), -frac))[:r]] += 1
        return np.repeat(np.asarray(a), q)

    eval_data.np = _np_with_random(choice=quota, shuffle=lambda x: x.sort())


def _fix_scprint():
    """scPRINT: padding genes are taken in a fixed order (lowest indices) instead of at random."""
    import scdataloader.collator as C
    C.np = _np_with_random(choice=lambda n, size, replace=False, p=None: np.arange(size))


def _fix_scgpt():
    """scGPT: values on a bin edge go to a fixed bin; cells with more genes than max_length
    keep their most highly expressed genes instead of a random subset."""
    import torch, scgpt.preprocess as P, scgpt.data_collator as D
    dig = P._digitize
    P._digitize = lambda x, bins, side="both": dig(x, bins, side="one")

    def top(self, genes, expressions, max_length):
        n = self.keep_first_n_tokens
        idx = torch.argsort(expressions[n:], descending=True, stable=True)[: max_length - n] + n
        keep = torch.cat([torch.arange(n), idx])
        return genes[keep], expressions[keep]

    D.DataCollator._sample = top


def fix(model):
    """Replace the unseeded random steps of `model` ('uce', 'scprint' or 'scgpt') with fixed rules."""
    {"uce": _fix_uce, "scprint": _fix_scprint, "scgpt": _fix_scgpt}[model.lower()]()


def knn_jaccard(Z1, Z2, k=15):
    """Mean over cells of |N1 ∩ N2| / |N1 ∪ N2|, where N are each cell's k nearest neighbours
    (Euclidean) in the two embeddings. 1 = identical neighbours."""
    from sklearn.neighbors import NearestNeighbors
    nn = lambda Z: NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    A, B = nn(np.asarray(Z1)), nn(np.asarray(Z2))
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]))


def centred_cosine(Z1, Z2):
    """Median per-cell cosine after removing the mean embedding of Z1 (raw cosine is
    inflated by the direction shared by all cells)."""
    Z1, Z2 = np.asarray(Z1, float), np.asarray(Z2, float); mu = Z1.mean(0)
    A, B = Z1 - mu, Z2 - mu
    return float(np.median((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))
