"""
FMAP Experiment 3 — Phase 0 DIAGNOSTIC.

Phase 0 showed the functional-map residual (0.48-0.65 AUROC) badly trailing
Mahalanobis (0.93) and a plain early->late activation residual (0.92). This run
decides whether that's an instantiation artifact or fundamental, by answering:

  1. k-sweep: does a larger spectral basis (k = 64 -> 256 -> 512) recover the
     fmap residual?
  2. coverage: does a linear probe on the *spectral embedding* match a linear
     probe on the *raw activations*?
       - spectral_probe << raw_probe  => the basis is discarding the signal
         (artifact; larger k / different basis may help).
       - spectral_probe ~ raw_probe but fmap residual stays low => the basis is
         fine and the residual-of-map construction is the weak part (fundamental
         to the detection framing).

    uv run modal run mvp/phase0_diag.py
    uv run modal run mvp/phase0_diag.py --model-name Qwen/Qwen2.5-7B-Instruct --n-cal 3000
"""

import modal

app = modal.App("fmap-exp3-phase0-diag")

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
hf_cache = modal.Volume.from_name("fmap-hf-cache", create_if_missing=True)


def _pick_text_col(columns):
    for c in ["Goal", "goal", "prompt", "Prompt", "Behavior", "behavior", "instruction", "text"]:
        if c in columns:
            return c
    return columns[0]


def get_activations(prompts, model, tok, layers, batch_size, max_length):
    import numpy as np
    import torch
    feats = {l: [] for l in layers}
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = []
        for p in batch:
            try:
                texts.append(tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
            except Exception:
                texts.append(p)
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last_idx = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(enc["input_ids"].size(0))
        for l in layers:
            feats[l].append(out.hidden_states[l][rows, last_idx].float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


def fit_kpca(X, k, gamma):
    from sklearn.decomposition import KernelPCA
    kp = KernelPCA(n_components=k, kernel="rbf", gamma=gamma)
    return kp, kp.fit_transform(X)


def ridge_residual(Xe_tr, Xl_tr, Xe_te, Xl_te, alpha):
    import numpy as np
    from sklearn.linear_model import Ridge
    C = Ridge(alpha=alpha, fit_intercept=True).fit(Xe_tr, Xl_tr)
    return np.linalg.norm(Xl_te - C.predict(Xe_te), axis=1)


def maha_scores(X_tr, X_te):
    from sklearn.covariance import EmpiricalCovariance
    return EmpiricalCovariance().fit(X_tr).mahalanobis(X_te)


def cv_probe_auroc(X, y):
    """5-fold CV AUROC of a linear (logreg) probe — measures whether the class
    signal is linearly present in this representation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=3600)
def run_diag(model_name, n_cal, batch_size, max_length, ridge_alpha):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json
    import numpy as np
    import torch
    from datasets import load_dataset, get_dataset_config_names
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    ks = [64, 256, 512]

    # ----- data -----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign_cal = [r["instruction"] for r in alpaca if r["input"].strip() == ""][:n_cal]
    try:
        harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        ben = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    except Exception as ex:
        raise RuntimeError(f"JBB load failed: {ex}. configs={get_dataset_config_names('JailbreakBench/JBB-Behaviors')}")
    col = _pick_text_col(harm.column_names)
    attack = [r[col] for r in harm]
    test_benign = [r[col] for r in ben]
    print(f"[data] cal_benign={len(benign_cal)} test_benign={len(test_benign)} attack={len(attack)} col={col!r}")

    # ----- model + activations -----
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda").eval()
    L = model.config.num_hidden_layers
    cand = sorted({max(1, min(L, round(f * L))) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})
    print(f"[model] {model_name} num_layers={L} indices={cand}")

    A_cal = get_activations(benign_cal, model, tok, cand, batch_size, max_length)
    A_te = get_activations(test_benign + attack, model, tok, cand, batch_size, max_length)
    y = np.array([0] * len(test_benign) + [1] * len(attack))

    Xs_cal, Xs_te = {}, {}
    for l in cand:
        sc = StandardScaler().fit(A_cal[l])
        Xs_cal[l], Xs_te[l] = sc.transform(A_cal[l]), sc.transform(A_te[l])
    early = [l for l in cand if l <= round(0.4 * L)]
    late = [l for l in cand if l >= round(0.6 * L)]

    # ----- references (k-independent) -----
    raw_probe = {l: cv_probe_auroc(Xs_te[l], y) for l in late}
    maha = {l: float(roc_auc_score(y, maha_scores(Xs_cal[l], Xs_te[l]))) for l in late}
    act_res = {f"{e}->{l}": float(roc_auc_score(y, ridge_residual(Xs_cal[e], Xs_cal[l], Xs_te[e], Xs_te[l], ridge_alpha)))
               for e in early for l in late if l > e}

    # ----- k-sweep (spectral) -----
    sweep = []
    for k in ks:
        Phi_cal, Phi_te = {}, {}
        for l in cand:
            kp, Z = fit_kpca(Xs_cal[l], k, None)
            Phi_cal[l], Phi_te[l] = Z, kp.transform(Xs_te[l])
        fmap = {f"{e}->{l}": float(roc_auc_score(y, ridge_residual(Phi_cal[e], Phi_cal[l], Phi_te[e], Phi_te[l], ridge_alpha)))
                for e in early for l in late if l > e}
        sprobe = {l: cv_probe_auroc(Phi_te[l], y) for l in late}
        sweep.append({"k": k, "best_fmap": max(fmap.values()), "best_spectral_probe": max(sprobe.values()),
                      "fmap_by_pair": fmap, "spectral_probe_by_layer": {str(l): v for l, v in sprobe.items()}})

    out = {"model": model_name, "n_cal": n_cal,
           "best_raw_probe": max(raw_probe.values()), "raw_probe": {str(l): v for l, v in raw_probe.items()},
           "best_maha": max(maha.values()), "best_act_residual": max(act_res.values()),
           "act_residual": act_res, "sweep": sweep}

    # ----- log + persist -----
    print("\n=== Phase 0 diagnostic ===")
    print(f"reference (k-independent):  raw_probe={out['best_raw_probe']:.3f}  "
          f"mahalanobis={out['best_maha']:.3f}  act_residual={out['best_act_residual']:.3f}")
    print(f"{'k':>5}{'fmap_residual':>16}{'spectral_probe':>17}")
    for s in sweep:
        print(f"{s['k']:>5}{s['best_fmap']:>16.3f}{s['best_spectral_probe']:>17.3f}")

    os.makedirs("/cache/results", exist_ok=True)
    with open(f"/cache/results/phase0_diag_{model_name.replace('/', '__')}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", n_cal: int = 2000,
         batch_size: int = 16, max_length: int = 256, ridge_alpha: float = 1.0):
    out = run_diag.remote(model_name=model_name, n_cal=n_cal,
                          batch_size=batch_size, max_length=max_length, ridge_alpha=ridge_alpha)
    print("\n=== Phase 0 diagnostic — is the spectral basis discarding the signal? ===")
    print(f"Reference (raw activations, k-independent):")
    print(f"  raw linear probe (logreg, 5-fold CV)   AUROC = {out['best_raw_probe']:.3f}")
    print(f"  mahalanobis                            AUROC = {out['best_maha']:.3f}")
    print(f"  act_residual (early->late ridge)       AUROC = {out['best_act_residual']:.3f}")
    print(f"\nk-sweep (spectral basis):")
    print(f"  {'k':>5}{'fmap_residual':>16}{'spectral_probe':>17}")
    for s in out["sweep"]:
        print(f"  {s['k']:>5}{s['best_fmap']:>16.3f}{s['best_spectral_probe']:>17.3f}")
    print("\nRead:")
    print("  spectral_probe << raw_probe        => basis discards signal (artifact; try larger k / other basis)")
    print("  spectral_probe ~ raw_probe, fmap low => residual-of-map construction is the weak part (fundamental)")
