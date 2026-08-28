"""Proof-step 2: with weights forced to 1.0, LambdaRank must reproduce BPR EXACTLY.

If this fails, my pair enumeration differs from epoch_bpr's, and any score
difference later would be attributable to that rather than to the position
discount -- which would invalidate the whole comparison.

Runs both losses for a few epochs from an identical seed and compares the
resulting model parameters bitwise-ish.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/khantminn/Desktop/Tiktok-TechJam-2026/runtime")
os.environ.setdefault("KUAIRAND_CACHE",
                      "/Users/khantminn/Desktop/Tiktok-TechJam-2026/runtime/cache")
import train_lib


def run_loss(loss_name, epochs=3):
    splits, meta = train_lib.load_cache()
    enc, dim, offsets, dims = train_lib.encode_features(splits, meta, "none")
    cfg = {"dim": dim, "k": 16, "lr": 1e-3, "bs": 8192, "epochs": epochs,
           "patience": 99, "seed": 0, "loss": loss_name, "history": "none",
           "multitask": "none", "model": "fm_numpy", "training": "default",
           "aux_tasks": []}
    res = train_lib.train_numpy_fm(cfg, enc, splits, meta, log=lambda *_: None)
    m = res["model"]
    return m.V.copy(), m.W.copy(), float(m.b), res["scores_valid"].copy()


print("running bpr_pairwise ...")
V1, W1, b1, s1 = run_loss("bpr_pairwise")
print("running _lambdarank_uniform (weights forced to 1.0) ...")
V2, W2, b2, s2 = run_loss("_lambdarank_uniform")

dV = np.abs(V1 - V2).max()
dW = np.abs(W1 - W2).max()
db = abs(b1 - b2)
ds = np.abs(s1 - s2).max()
print()
print(f"max |dV| = {dV:.3e}")
print(f"max |dW| = {dW:.3e}")
print(f"    |db| = {db:.3e}")
print(f"max |d valid scores| = {ds:.3e}")
ok = dV < 1e-9 and dW < 1e-9 and db < 1e-9 and ds < 1e-6
print()
print("RESULT:", "PASS -- uniform-weight LambdaRank == BPR exactly; pair "
      "enumeration is identical, so any later difference is the discount alone"
      if ok else "FAIL -- enumeration differs from BPR; comparison would be invalid")
sys.exit(0 if ok else 1)
