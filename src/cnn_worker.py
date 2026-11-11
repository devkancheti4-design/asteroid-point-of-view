"""CNN worker for sheet_comparison.py, run as its own process.

PyTorch ships its own OpenMP runtime, while LightGBM, XGBoost and GPBoost load Homebrew's. With both runtimes in
one process PyTorch's backward pass segfaults on macOS, so the CNN trains here, in a process that imports only
numpy and torch. Inputs arrive already imputed and standardized (fitted on the training fold by the caller).

Usage: python cnn_worker.py <jobs.pkl> <results.pkl>
  jobs.pkl     list of dicts: A_fit, y_fit, A_val, y_val (None for full-batch jobs), B (test rows), seed, threads,
               full_batch, epochs, patience, batch
  results.pkl  list of (test scores, info) in the same order
"""
from __future__ import annotations

import copy
import math
import pickle
import sys
import time

import numpy as np
import torch


def build(n_features: int) -> torch.nn.Module:
    """1-D CNN over the feature vector: two conv layers, dense head, one logit."""
    nn = torch.nn
    return nn.Sequential(nn.Unflatten(1, (1, n_features)),
                         nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                         nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(64 * n_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


def train(job: dict) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    net = build(job["A_fit"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xf, yf = torch.from_numpy(job["A_fit"]), torch.from_numpy(job["y_fit"])
    info: dict = {}
    if job["full_batch"]:                       # tiny tables: full batch, fixed number of epochs
        for _ in range(job["epochs"]):
            net.train()
            opt.zero_grad()
            loss_fn(net(xf).squeeze(1), yf).backward()
            opt.step()
        info["epochs"] = job["epochs"]
    else:                                       # mini-batches, early stopping on the validation rows
        xv, yv = torch.from_numpy(job["A_val"]), torch.from_numpy(job["y_val"])
        best, best_state, wait, epoch = math.inf, copy.deepcopy(net.state_dict()), 0, 0
        for epoch in range(job["epochs"]):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(yf)))
            for s in range(0, len(perm), job["batch"]):
                b = perm[s:s + job["batch"]]
                opt.zero_grad()
                loss_fn(net(xf[b]).squeeze(1), yf[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(xv).squeeze(1), yv).item()
            if v < best - 1e-6:
                best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= job["patience"]:
                    break
        net.load_state_dict(best_state)
        info.update(epochs=epoch + 1, val_loss=best)
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(torch.from_numpy(job["B"])).squeeze(1)).numpy().astype(float)
    info["fit_seconds"] = time.time() - t0
    return scores, info


def main(jobs_path: str, out_path: str) -> None:
    with open(jobs_path, "rb") as f:
        jobs = pickle.load(f)                   # written by sheet_comparison.py in the same run
    torch.set_num_threads(jobs[0]["threads"] if jobs else 1)
    torch.use_deterministic_algorithms(True)
    results = [train(job) for job in jobs]
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=5)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
