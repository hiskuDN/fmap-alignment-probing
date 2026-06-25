"""
FMAP Experiment 3 — Phase 0, the RIGHT task: computational anomaly, not content.

Task 1 (phase0.py) varied *content* (harmful vs. benign requests) — both flow
through the model normally, so the early->late map doesn't break and the residual
stayed flat (~0.65) while a probe/Mahalanobis won (~0.93).

This run varies *whether the computation is anomalous*. Benign = normal requests;
anomaly classes:
  - gcg          : GCG adversarial-suffix jailbreaks (gibberish tokens that hijack
                   attention) — the prime "flow-breakage" case.
  - pair         : PAIR jailbreaks (semantic rephrasings) — more content-like.
  - ood_random   : random-token gibberish — pure far-OOD.
  - harmful      : JBB harmful behaviors — kept as the Task-1 contrast (expect ~0.65).
Adversarial prompts are pulled from the JailbreakBench/artifacts repo.

Decisive read: if fmap_residual spikes on gcg/ood_random while staying flat on
harmful, the residual is a *computational-anomaly* detector (a real, differentiated
capability). If it stays flat on gcg/ood too, the residual idea is dead -> pivot.

    uv run modal run mvp/phase0_adv.py
"""

import modal

app = modal.App("fmap-exp3-phase0-adv")

# NOTE: image definition kept byte-identical to phase0.py so Modal reuses the cached
# build. `requests` is already present transitively (huggingface_hub/datasets).
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


def fetch_artifacts(method, access, max_n):
    """Pull adversarial prompts from JailbreakBench/artifacts (GitHub API + raw fallback)."""
    import requests
    urls = []
    try:
        api = f"https://api.github.com/repos/JailbreakBench/artifacts/contents/attack-artifacts/{method}/{access}"
        r = requests.get(api, timeout=30)
        r.raise_for_status()
        urls = [it["download_url"] for it in r.json() if it.get("name", "").endswith(".json")]
    except Exception as ex:
        print(f"[artifacts] API listing failed for {method}/{access} ({ex}); trying raw fallback")
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


def make_ood_random(tok, n, seed, min_tok=20, max_tok=60):
    import numpy as np
    rng = np.random.default_rng(seed)
    vocab = getattr(tok, "vocab_size", None) or len(tok)
    out = []
    for _ in range(n):
        L = int(rng.integers(min_tok, max_tok))
        ids = rng.integers(0, vocab, size=L).tolist()
        out.append(tok.decode(ids))
    return out


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


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=3600)
def run_adv(model_name, n_cal, n_test, k, batch_size, max_length, ridge_alpha):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.metrics import roc_auc_score

    # ----- benign corpus (cal + disjoint test) -----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign_all = [r["instruction"] for r in alpaca if r["input"].strip() == ""]
    benign_cal, benign_test = benign_all[:n_cal], benign_all[n_cal:n_cal + n_test]

    # ----- anomaly classes -----
    harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    harmful = [r[_pick_text_col(harm.column_names)] for r in harm]
    gcg = fetch_artifacts("GCG", "white_box", max_n=150)
    pair = fetch_artifacts("PAIR", "black_box", max_n=150)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ood_random = make_ood_random(tok, n=200, seed=0)

    groups = {"benign_test": benign_test, "harmful": harmful,
              "gcg": gcg, "pair": pair, "ood_random": ood_random}
    groups = {g: p for g, p in groups.items() if p}
    print("[data] cal_benign=%d | %s" % (len(benign_cal),
          " ".join(f"{g}={len(p)}" for g, p in groups.items())))

    # ----- model + activations -----
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda").eval()
    L = model.config.num_hidden_layers
    cand = sorted({max(1, min(L, round(f * L))) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})
    print(f"[model] {model_name} num_layers={L} indices={cand}")

    A_cal = get_activations(benign_cal, model, tok, cand, batch_size, max_length)
    G_act = {g: get_activations(p, model, tok, cand, batch_size, max_length) for g, p in groups.items()}

    # ----- standardize + spectral basis, all fit on benign cal -----
    scalers = {l: StandardScaler().fit(A_cal[l]) for l in cand}
    Xs_cal = {l: scalers[l].transform(A_cal[l]) for l in cand}
    Xs_g = {g: {l: scalers[l].transform(G_act[g][l]) for l in cand} for g in groups}

    kp = {}
    Phi_cal, Phi_g = {}, {g: {} for g in groups}
    for l in cand:
        kp[l], Phi_cal[l] = fit_kpca(Xs_cal[l], k, None)
        for g in groups:
            Phi_g[g][l] = kp[l].transform(Xs_g[g][l])

    early = [l for l in cand if l <= round(0.4 * L)]
    late = [l for l in cand if l >= round(0.6 * L)]
    pairs = [(e, la) for e in early for la in late if la > e]

    # ----- fit maps / detectors on benign cal -----
    fmap_C = {(e, la): Ridge(alpha=ridge_alpha).fit(Phi_cal[e], Phi_cal[la]) for e, la in pairs}
    act_C = {(e, la): Ridge(alpha=ridge_alpha).fit(Xs_cal[e], Xs_cal[la]) for e, la in pairs}
    maha = {la: EmpiricalCovariance().fit(Xs_cal[la]) for la in late}

    def fmap_score(g, e, la):
        return np.linalg.norm(Phi_g[g][la] - fmap_C[(e, la)].predict(Phi_g[g][e]), axis=1)

    def act_score(g, e, la):
        return np.linalg.norm(Xs_g[g][la] - act_C[(e, la)].predict(Xs_g[g][e]), axis=1)

    # ----- AUROC per anomaly class vs benign_test, best over layer pair/layer -----
    table = {}
    for c in [g for g in groups if g != "benign_test"]:
        row = {}
        fm = [(roc_auc_score([0]*len(benign_test) + [1]*len(groups[c]),
               np.concatenate([fmap_score("benign_test", e, la), fmap_score(c, e, la)])), (e, la))
               for e, la in pairs]
        ac = [(roc_auc_score([0]*len(benign_test) + [1]*len(groups[c]),
               np.concatenate([act_score("benign_test", e, la), act_score(c, e, la)])), (e, la))
               for e, la in pairs]
        mh = [(roc_auc_score([0]*len(benign_test) + [1]*len(groups[c]),
               np.concatenate([maha[la].mahalanobis(Xs_g["benign_test"][la]),
                               maha[la].mahalanobis(Xs_g[c][la])])), la)
               for la in late]
        bf, bfp = max(fm); ba, bap = max(ac); bm, bml = max(mh)
        row = {"n": len(groups[c]),
               "fmap_residual": float(bf), "fmap_pair": bfp,
               "act_residual": float(ba), "act_pair": bap,
               "mahalanobis": float(bm), "maha_layer": bml}
        table[c] = row

    out = {"model": model_name, "n_cal": n_cal, "n_test_benign": len(benign_test), "k": k, "table": table}

    print("\n=== Phase 0 adversarial/OOD — AUROC vs benign_test (best over layers) ===")
    print(f"{'class':14}{'n':>5}{'fmap_residual':>15}{'act_residual':>14}{'mahalanobis':>13}")
    for c, r in table.items():
        print(f"{c:14}{r['n']:>5}{r['fmap_residual']:>15.3f}{r['act_residual']:>14.3f}{r['mahalanobis']:>13.3f}")

    os.makedirs("/cache/results", exist_ok=True)
    with open(f"/cache/results/phase0_adv_{model_name.replace('/', '__')}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", n_cal: int = 2000, n_test: int = 256,
         k: int = 128, batch_size: int = 16, max_length: int = 512, ridge_alpha: float = 1.0):
    out = run_adv.remote(model_name=model_name, n_cal=n_cal, n_test=n_test, k=k,
                         batch_size=batch_size, max_length=max_length, ridge_alpha=ridge_alpha)
    t = out["table"]
    print("\n=== Phase 0 adversarial/OOD — AUROC vs benign_test (higher = more separable) ===")
    print(f"{'class':14}{'n':>5}{'fmap_residual':>15}{'act_residual':>14}{'mahalanobis':>13}")
    for c, r in t.items():
        print(f"{c:14}{r['n']:>5}{r['fmap_residual']:>15.3f}{r['act_residual']:>14.3f}{r['mahalanobis']:>13.3f}")
    print("\nRead: residual high on gcg/ood_random but ~0.65 on harmful => catches computation, not content.")
    print("      residual flat everywhere => pivot.")
