"""
FMAP Experiment 3 — Phase 0 MVP.

Question this run answers:
    Does a functional-map alignment residual between an EARLY and a LATE layer
    (calibrated on benign text only) separate adversarial/harmful inputs from
    benign — and does it beat (a) Mahalanobis on activations and (b) a plain
    activation-space early->late ridge residual (the "no spectral basis" control,
    i.e. the Mumcu & Yilmaz mechanic without the spectral step)?

Method (Phase 0 instantiation):
    - Extract last-token residual-stream activations at several candidate layers.
    - Per layer: standardize, then a kernel-PCA (RBF) spectral basis Phi.
      KernelPCA gives a clean out-of-sample `.transform`, standing in for the
      graph-Laplacian eigenfunctions + Nystrom extension we'd use in later phases.
    - Functional map: ridge-regress Phi_late from Phi_early on benign points
      (identical tokens => free identity correspondence). Residual = ||Phi_late - C(Phi_early)||.
    - Score benign vs. attack by residual magnitude; report AUROC, sweeping a small
      grid of (early, late) layer pairs.

Runs on Modal (GPU). Locally you only need the `modal` client (managed by uv):

    uv sync
    uv run modal token new                      # one-time CLI login
    uv run modal run mvp/phase0.py              # default: Qwen2.5-1.5B-Instruct
    uv run modal run mvp/phase0.py --model-name Qwen/Qwen2.5-7B-Instruct --n-cal 3000 --k 96
"""

import modal

app = modal.App("fmap-exp3-phase0")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.44",
        "accelerate>=0.33",
        "datasets>=2.20",
        "scikit-learn>=1.4",
        "scipy>=1.11",
        "numpy>=1.26,<2",
    )
)

# Persist the HF model cache between runs so weights download only once.
hf_cache = modal.Volume.from_name("fmap-hf-cache", create_if_missing=True)


# --------------------------------------------------------------------------------------
# Helpers. Imports are deferred into each function so the local entrypoint (which has
# only the `modal` client installed) never tries to import torch/sklearn.
# --------------------------------------------------------------------------------------

def _pick_text_col(columns):
    for c in ["Goal", "goal", "prompt", "Prompt", "Behavior", "behavior", "instruction", "text"]:
        if c in columns:
            return c
    return columns[0]


def get_activations(prompts, model, tok, layers, batch_size, max_length):
    """Last-token residual-stream activations at each hidden_states index in `layers`."""
    import numpy as np
    import torch

    feats = {l: [] for l in layers}
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = []
        for p in batch:
            try:
                texts.append(tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True))
            except Exception:
                texts.append(p)
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last_idx = enc["attention_mask"].sum(1) - 1            # last real token per sequence
        rows = torch.arange(enc["input_ids"].size(0))
        for l in layers:
            h = out.hidden_states[l]                            # (B, T, d)
            feats[l].append(h[rows, last_idx].float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


def fit_kpca(X, k, gamma):
    from sklearn.decomposition import KernelPCA
    kp = KernelPCA(n_components=k, kernel="rbf", gamma=gamma)
    Z = kp.fit_transform(X)
    return kp, Z


def ridge_residual(Xe_tr, Xl_tr, Xe_te, Xl_te, alpha):
    """Fit late ~ C(early) on calibration; return per-row residual norm on test."""
    import numpy as np
    from sklearn.linear_model import Ridge
    C = Ridge(alpha=alpha, fit_intercept=True).fit(Xe_tr, Xl_tr)
    return np.linalg.norm(Xl_te - C.predict(Xe_te), axis=1)


def maha_scores(X_tr, X_te):
    from sklearn.covariance import EmpiricalCovariance
    cov = EmpiricalCovariance().fit(X_tr)
    return cov.mahalanobis(X_te)                                # squared distance; higher = more OOD


# --------------------------------------------------------------------------------------
# Remote job
# --------------------------------------------------------------------------------------

@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=3600)
def run_phase0(model_name, n_cal, k, batch_size, max_length, ridge_alpha):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import numpy as np
    import torch
    from datasets import load_dataset, get_dataset_config_names
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # ----- data: benign calibration corpus (Alpaca) + JBB harmful/benign test -----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign_cal = [r["instruction"] for r in alpaca if r["input"].strip() == ""][:n_cal]

    try:
        harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        ben = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    except Exception as ex:  # surface the real schema so it's a one-line fix on first run
        cfgs = get_dataset_config_names("JailbreakBench/JBB-Behaviors")
        raise RuntimeError(f"JBB load failed: {ex}. Available configs: {cfgs}")

    col = _pick_text_col(harm.column_names)
    attack = [r[col] for r in harm]
    test_benign = [r[col] for r in ben]
    print(f"[data] cal_benign={len(benign_cal)} test_benign={len(test_benign)} "
          f"attack={len(attack)} text_col={col!r}")

    # ----- model -----
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to("cuda").eval()

    L = model.config.num_hidden_layers
    cand = sorted({max(1, min(L, round(f * L))) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})
    print(f"[model] {model_name} num_layers={L} candidate hidden_state indices={cand}")

    # ----- activations -----
    A_cal = get_activations(benign_cal, model, tok, cand, batch_size, max_length)
    A_te = get_activations(test_benign + attack, model, tok, cand, batch_size, max_length)
    y = np.array([0] * len(test_benign) + [1] * len(attack))    # 1 = attack

    # ----- per-layer standardize + spectral basis -----
    Xs_cal, Xs_te, Phi_cal, Phi_te = {}, {}, {}, {}
    for l in cand:
        sc = StandardScaler().fit(A_cal[l])
        Xs_cal[l], Xs_te[l] = sc.transform(A_cal[l]), sc.transform(A_te[l])
        kp, Z = fit_kpca(Xs_cal[l], k, gamma=None)
        Phi_cal[l], Phi_te[l] = Z, kp.transform(Xs_te[l])

    early = [l for l in cand if l <= round(0.4 * L)]
    late = [l for l in cand if l >= round(0.6 * L)]

    # ----- score: fmap residual + baselines -----
    results = []
    for e in early:
        for la in late:
            if la <= e:
                continue
            r_fm = ridge_residual(Phi_cal[e], Phi_cal[la], Phi_te[e], Phi_te[la], ridge_alpha)
            r_act = ridge_residual(Xs_cal[e], Xs_cal[la], Xs_te[e], Xs_te[la], ridge_alpha)
            results.append({"method": "fmap_residual", "early": e, "late": la,
                            "auroc": float(roc_auc_score(y, r_fm))})
            results.append({"method": "act_residual", "early": e, "late": la,
                            "auroc": float(roc_auc_score(y, r_act))})
    for la in late:
        results.append({"method": "mahalanobis", "early": None, "late": la,
                        "auroc": float(roc_auc_score(y, maha_scores(Xs_cal[la], Xs_te[la])))})

    # Log to the remote stdout (captured by `modal app logs`) and persist to the volume,
    # so results are reviewable via CLI without re-running.
    import json
    results.sort(key=lambda r: -r["auroc"])
    print("\n=== Phase 0 — AUROC (attack vs. benign) ===")
    print(f"{'method':16}{'early':>7}{'late':>7}{'AUROC':>9}")
    for r in results:
        print(f"{r['method']:16}{str(r['early']):>7}{str(r['late']):>7}{r['auroc']:>9.3f}")

    os.makedirs("/cache/results", exist_ok=True)
    safe = model_name.replace("/", "__")
    with open(f"/cache/results/phase0_{safe}.json", "w") as fh:
        json.dump({"model": model_name, "n_cal": n_cal, "k": k, "results": results}, fh, indent=2)
    hf_cache.commit()
    return results


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", n_cal: int = 2000,
         k: int = 64, batch_size: int = 16, max_length: int = 256, ridge_alpha: float = 1.0):
    res = run_phase0.remote(model_name=model_name, n_cal=n_cal, k=k,
                            batch_size=batch_size, max_length=max_length, ridge_alpha=ridge_alpha)
    res.sort(key=lambda r: -r["auroc"])

    print("\n=== Phase 0 — AUROC (attack vs. benign), higher = better separation ===")
    print(f"{'method':16}{'early':>7}{'late':>7}{'AUROC':>9}")
    for r in res:
        print(f"{r['method']:16}{str(r['early']):>7}{str(r['late']):>7}{r['auroc']:>9.3f}")

    fm = [r for r in res if r["method"] == "fmap_residual"]
    if fm:
        best = max(fm, key=lambda r: r["auroc"])
        print(f"\nBest fmap_residual AUROC = {best['auroc']:.3f} "
              f"(early={best['early']}, late={best['late']})")
    print("Gate: fmap AUROC >> 0.5 => signal exists. Compare vs. act_residual / mahalanobis for the edge.")
