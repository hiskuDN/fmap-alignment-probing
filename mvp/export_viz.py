"""
Export data for the FMAP activation-atlas web viz.

Recomputes last-token residual-stream activations for benign + anomaly classes,
then for each candidate layer projects them to 2D (PCA fit on benign; UMAP fit on
benign, if available) and attaches per-point detector scores (fmap_residual,
act_residual, mahalanobis) + truncated prompt text. Writes `viz/data.js`
(`window.FMAP_DATA = {...}`) locally so the static page works from file://.

    uv run modal run mvp/export_viz.py
"""

import modal

app = modal.App("fmap-exp3-export-viz")

# adds umap-learn to the Phase-0 image (one-time rebuild); rest identical.
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
        "umap-learn>=0.5.6",
    )
)
hf_cache = modal.Volume.from_name("fmap-hf-cache", create_if_missing=True)


def _pick_text_col(columns):
    for c in ["Goal", "goal", "prompt", "Prompt", "Behavior", "behavior", "instruction", "text"]:
        if c in columns:
            return c
    return columns[0]


def fetch_artifacts(method, access, max_n):
    import requests
    urls = []
    try:
        api = f"https://api.github.com/repos/JailbreakBench/artifacts/contents/attack-artifacts/{method}/{access}"
        r = requests.get(api, timeout=30); r.raise_for_status()
        urls = [it["download_url"] for it in r.json() if it.get("name", "").endswith(".json")]
    except Exception as ex:
        print(f"[artifacts] API listing failed for {method}/{access} ({ex}); raw fallback")
        raw = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
        urls = [f"{raw}/{method}/{access}/{m}.json" for m in ("vicuna-13b-v1.5", "llama-2-7b-chat-hf")]
    prompts = []
    for url in urls:
        try:
            data = requests.get(url, timeout=30).json()
        except Exception:
            continue
        for jb in data.get("jailbreaks", []):
            p = jb.get("prompt")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
        if len(prompts) >= max_n:
            break
    return prompts[:max_n]


def make_ood_random(tok, n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    vocab = getattr(tok, "vocab_size", None) or len(tok)
    return [tok.decode(rng.integers(0, vocab, size=int(rng.integers(20, 60))).tolist()) for _ in range(n)]


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
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last_idx = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(enc["input_ids"].size(0))
        for l in layers:
            feats[l].append(out.hidden_states[l][rows, last_idx].float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=3600)
def export(model_name, n_cal, n_test, n_plot_benign, early, late, batch_size, max_length, ridge_alpha):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.metrics import roc_auc_score

    # ----- data -----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign_all = [r["instruction"] for r in alpaca if r["input"].strip() == ""]
    benign_cal = benign_all[:n_cal]
    benign_test = benign_all[n_cal:n_cal + n_test][:n_plot_benign]

    harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    harmful = [r[_pick_text_col(harm.column_names)] for r in harm]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    gcg = fetch_artifacts("GCG", "white_box", 150)
    pair = fetch_artifacts("PAIR", "black_box", 150)
    ood = make_ood_random(tok, 150, seed=0)

    plot_groups = {"benign": benign_test, "harmful": harmful, "gcg": gcg, "pair": pair, "ood_random": ood}
    plot_groups = {g: p for g, p in plot_groups.items() if p}
    print("[data] cal=%d | %s" % (len(benign_cal), " ".join(f"{g}={len(p)}" for g, p in plot_groups.items())))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda").eval()
    L = model.config.num_hidden_layers
    cand = sorted({max(1, min(L, round(f * L))) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})
    print(f"[model] {model_name} L={L} layers={cand} score_pair=({early},{late})")

    A_cal = get_activations(benign_cal, model, tok, cand, batch_size, max_length)
    G = {g: get_activations(p, model, tok, cand, batch_size, max_length) for g, p in plot_groups.items()}

    # ----- standardize (fit on cal) -----
    sc = {l: StandardScaler().fit(A_cal[l]) for l in cand}
    Xc = {l: sc[l].transform(A_cal[l]) for l in cand}
    Xg = {g: {l: sc[l].transform(G[g][l]) for l in cand} for g in plot_groups}

    # ----- 2D projections per layer: PCA (fit on cal) + UMAP (fit on cal subsample) -----
    pca = {l: PCA(n_components=3).fit(Xc[l]) for l in cand}
    coords = {g: {"pca": {}, "umap": {}} for g in plot_groups}
    for l in cand:
        for g in plot_groups:
            coords[g]["pca"][str(l)] = pca[l].transform(Xg[g][l])
    has_umap = False
    try:
        import umap
        sub = Xc[cand[0]].shape[0]
        idx = np.random.default_rng(0).choice(sub, size=min(800, sub), replace=False)
        for l in cand:
            reducer = umap.UMAP(n_components=3, random_state=42).fit(Xc[l][idx])
            for g in plot_groups:
                coords[g]["umap"][str(l)] = reducer.transform(Xg[g][l])
        has_umap = True
    except Exception as ex:
        print(f"[umap] skipped ({ex}); PCA only")

    # ----- detector scores (fit on cal) at fixed (early,late) / late -----
    fmap_dummy = None  # fmap residual here = activation-space spectral was the failing one; for the panel
    # we expose the three detectors actually compared: act_residual (early->late), mahalanobis (late),
    # and the spectral fmap_residual (KernelPCA early->late) for completeness.
    from sklearn.decomposition import KernelPCA
    actC = Ridge(alpha=ridge_alpha).fit(Xc[early], Xc[late])
    maha = EmpiricalCovariance().fit(Xc[late])
    kpe, kpl = KernelPCA(n_components=128, kernel="rbf").fit(Xc[early]), KernelPCA(n_components=128, kernel="rbf").fit(Xc[late])
    Pe_c, Pl_c = kpe.transform(Xc[early]), kpl.transform(Xc[late])
    fmapC = Ridge(alpha=ridge_alpha).fit(Pe_c, Pl_c)

    def scores_for(Xe, Xl):
        act = np.linalg.norm(Xl - actC.predict(Xe), axis=1)
        mh = maha.mahalanobis(Xl)
        fm = np.linalg.norm(kpl.transform(Xl) - fmapC.predict(kpe.transform(Xe)), axis=1)
        return fm, act, mh

    scores = {}
    for g in plot_groups:
        fm, act, mh = scores_for(Xg[g][early], Xg[g][late])
        scores[g] = {"fmap": fm, "act": act, "maha": mh}

    # ----- AUROC per class vs benign at this pair/layer (for panel labels) -----
    auroc = {}
    yb = scores["benign"]
    for g in plot_groups:
        if g == "benign":
            continue
        auroc[g] = {}
        for d in ("fmap", "act", "maha"):
            s = np.concatenate([yb[d], scores[g][d]])
            y = np.array([0] * len(yb[d]) + [1] * len(scores[g][d]))
            auroc[g][d] = float(roc_auc_score(y, s))

    # ----- assemble compact point records -----
    def trunc(s, n=160):
        s = " ".join(s.split())
        return s[:n] + ("…" if len(s) > n else "")

    points = []
    for g, prompts in plot_groups.items():
        for i, p in enumerate(prompts):
            rec = {"cls": g, "prompt": trunc(p),
                   "pca": {l: [round(float(coords[g]["pca"][l][i][0]), 3), round(float(coords[g]["pca"][l][i][1]), 3), round(float(coords[g]["pca"][l][i][2]), 3)] for l in coords[g]["pca"]},
                   "scores": {"fmap": round(float(scores[g]["fmap"][i]), 4),
                              "act": round(float(scores[g]["act"][i]), 4),
                              "maha": round(float(scores[g]["maha"][i]), 4)}}
            if has_umap:
                rec["umap"] = {l: [round(float(coords[g]["umap"][l][i][0]), 3), round(float(coords[g]["umap"][l][i][1]), 3), round(float(coords[g]["umap"][l][i][2]), 3)] for l in coords[g]["umap"]}
            points.append(rec)

    out = {
        "model": model_name, "layers": cand, "score_pair": {"early": early, "late": late},
        "classes": list(plot_groups.keys()), "has_umap": has_umap,
        "colors": {"benign": "#9aa0a6", "harmful": "#f9ab00", "gcg": "#ea4335", "pair": "#a142f4", "ood_random": "#202124"},
        "auroc": auroc, "points": points,
    }
    print(f"[export] points={len(points)} has_umap={has_umap} auroc={ {k: {d: round(v[d],3) for d in v} for k,v in auroc.items()} }")
    return out


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", n_cal: int = 2000, n_test: int = 400,
         n_plot_benign: int = 200, early: int = 6, late: int = 18,
         batch_size: int = 16, max_length: int = 512, ridge_alpha: float = 1.0):
    import json, os
    out = export.remote(model_name=model_name, n_cal=n_cal, n_test=n_test, n_plot_benign=n_plot_benign,
                        early=early, late=late, batch_size=batch_size, max_length=max_length, ridge_alpha=ridge_alpha)
    os.makedirs("viz", exist_ok=True)
    with open("viz/data.js", "w") as f:
        f.write("window.FMAP_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote viz/data.js  ({len(out['points'])} points, has_umap={out['has_umap']})")
