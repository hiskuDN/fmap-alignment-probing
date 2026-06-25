# Experiment 3: Minimal Viable Experiment (scoping)

*Scopes the cheapest run that can falsify the [Experiment 3](proposal.md) bet, and the open-source stack to build it on. Companion to [`proposal.md`](proposal.md) and [`RELATED_WORK.md`](RELATED_WORK.md).*

## The question this MVP answers

1. **Signal:** does the functional-map alignment residual separate adversarial/OOD inputs from benign at all?
2. **Edge:** does it beat (a) Mahalanobis-on-activations (Lee et al. 2018) and (b) the no-spectral-basis early→deep regressor (Mumcu & Yilmaz 2025), the latter being "our idea minus the basis"?
3. **Localization:** is the residual *structured* enough to point at a layer transition + subspace that patching confirms?

If (1) fails, the idea is dead cheaply. If (1) passes but (2) fails, the spectral basis adds nothing and the honest result is "use the regressor." Only if (1)+(2)+(3) hold is there a paper.

## Build-from stack (all open source; licenses not individually verified, confirm before redistribution)

- **Activations + patching:** `TransformerLensOrg/TransformerLens` (cache residual stream; activation patching for the localization step).
- **Functional-map primitives:** `RobinMagnet/pyFM` (Laplacian eigenfunctions, C-estimation, refinement).
- **Detection baselines:** `kkirchheim/pytorch-ood` (Mahalanobis); port the early→deep residual idea from `furkanmumcu/Layer-Regression` (image-domain, reimplement for LLM layers).
- **Lens baseline:** `AlignmentResearch/tuned-lens`.
- **Data:** `JailbreakBench/jailbreakbench` + `artifacts` (benign + attack families); `centerforaisafety/HarmBench` to widen attack diversity.

**Custom code (the only real build):** the LFM-on-representations adapter, kNN graph over activations → graph-Laplacian eigenfunctions → estimate `C` against the free identity anchors → per-input residual, with **Nyström out-of-sample extension** so new inputs can be scored against the benign-calibrated basis without recomputing the eigendecomposition.

## Protocol

**Model.** A small open white-box instruct model for fast iteration. Phase 0 defaults to **Qwen2.5-1.5B-Instruct** (ungated/Apache-2.0, no HF token friction), configurable up to **Qwen2.5-7B-Instruct** via a flag.

**Map setup.** Intra-model, inter-layer: pick an early layer `ℓ_e` and a late layer `ℓ_l` (start ~25% and ~75% depth). Over a benign corpus, take last-token (or mean) residual-stream activations at both layers, identical tokens give a free identity correspondence. Build the graph, take the top `k` Laplacian eigenfunctions (start `k ≈ 64–128`), estimate `C` by regularized least squares. Residual `r(x) = ‖Φ_l(x) − Φ_e(x)·C‖`.

**Data / splits.** Benign = JBB benign behaviors (+ a generic instruction set). Attacks = JBB artifacts grouped by family (PAIR, GCG, JailbreakChat, …). Calibrate `C` on benign only.

**Detection metric.** AUROC separating attacks from benign by residual magnitude. Baselines on the same activations: Mahalanobis (`pytorch-ood`), linear probe, ported early→deep regressor, tuned-lens trajectory.

**Cross-attack OOD (the differentiator).** Calibrate/threshold on attack family A only, evaluate on held-out family B. Three arms: unsupervised residual (ours), linear-probe-on-activations (expected to collapse), probe-on-fmap-features (ablation). Detector stays unsupervised; the probe is only the ablation.

**Localization + legibility.** From the residual, extract the top-rank subspace; patch it (TransformerLens) and measure suppression of the attack effect vs. a random equal-rank subspace; adopt the Makelov et al. (2023) subspace-patching-illusion controls. Falsification: high AUROC but unstable/illegible/random-patching subspace ⇒ detector works, localization claim fails (report as such).

## Go / kill gates (run in order)

- **Phase 0 (signal, ½–1 day once data+model load):** residual AUROC vs benign on one attack family. Gate: AUROC ≫ 0.5. Kill if not.
- **Phase 1 (edge):** beat Mahalanobis and the ported early→deep regressor on AUROC, and survive the cross-attack split where the linear probe collapses. Gate: ≥ baselines on held-out family.
- **Phase 2 (localization):** patching the residual subspace suppresses the attack more than a random subspace, under the illusion controls. Gate: significant, stable effect.

## Setup (decided)

- **Compute:** Modal (serverless GPU). Dependencies via **uv**, the heavy ML stack (torch, transformers, scikit-learn, datasets) lives in the Modal image; locally only the `modal` client is installed.
- **Model:** Qwen2.5-1.5B-Instruct default (ungated, fast), bump to 7B via flag.
- **Scope:** Phase 0 first (does the residual separate attack from benign at all?).

## How to run (Phase 0)

Code: [`../mvp/phase0.py`](../mvp/phase0.py) (Modal app), [`../pyproject.toml`](../pyproject.toml) (uv project).

```bash
uv sync                                   # creates the local env (just the modal client)
uv run modal token new                    # one-time Modal CLI login
uv run modal run mvp/phase0.py            # default: Qwen2.5-1.5B-Instruct
# larger model / more calibration data:
uv run modal run mvp/phase0.py --model-name Qwen/Qwen2.5-7B-Instruct --n-cal 3000 --k 96
```

The job pulls a benign calibration corpus (Alpaca) + JailbreakBench harmful/benign behaviors, extracts early/late residual-stream activations on a Modal GPU, fits the functional-map residual and the two baselines (activation-space ridge residual, Mahalanobis), and prints **AUROC per (early, late) layer pair**. Gate: best `fmap_residual` AUROC ≫ 0.5 means signal exists; the side-by-side `act_residual` / `mahalanobis` columns are an early read on the edge.

**Notes / known wrinkles to confirm on first run:** the JBB config/split names (`"behaviors"` → `harmful`/`benign`) and text column are auto-detected but echo them in the log; KernelPCA(RBF) stands in for the graph-Laplacian eigenbasis + Nyström extension (a Phase-1 refinement); switching to a *gated* model (Gemma/Llama) needs a Modal HF-token secret.

## Results & verdict (2026-06-25)

Phase 0 ran on **Qwen2.5-1.5B-Instruct** via Modal. **Outcome: the functional-map residual is dominated by trivial baselines and is the wrong readout for these signals. Detection via the fmap residual is falsified.**

### Run 1: content task (harmful vs. benign behaviors), best AUROC

| method | AUROC |
|---|---|
| mahalanobis | 0.925 |
| act_residual (early→late ridge, *no spectral basis*) | 0.917 |
| **fmap_residual** | **0.649** |

### Diagnostic: is the spectral basis at fault? (No.)

A linear probe on the spectral embedding scored **0.928** (≈ probe on raw activations, 0.959), while sweeping `k ∈ {64, 256, 512}` left the residual flat (0.649 → 0.657). So the basis *retains* the signal; the residual-of-map construction simply can't read it. Not an instantiation artifact.

### Run 2: computational-anomaly task (the "right" task), AUROC vs. benign

| class | n | fmap_residual | act_residual | mahalanobis |
|---|---|---|---|---|
| harmful | 100 | 0.818 | 0.956 | 0.957 |
| gcg | 150 | 0.764 | **1.000** | 0.985 |
| pair | 150 | 0.763 | 0.999 | 0.970 |
| ood_random | 200 | **0.245** | **1.000** | 0.997 |

### Verdict

- The fmap residual **never beats** the baselines, on any class.
- It is **inverted on far-OOD (0.245)**: KernelPCA `.transform` collapses far points to ≈0 (all RBF affinities vanish), so the residual *shrinks* as inputs get more OOD, a structural flaw of any benign-fit spectral-embedding residual, not specific to KernelPCA.
- **`act_residual` = 1.000** on GCG/OOD: the plain activation-space early→late residual is already perfect, leaving **no headroom** for the spectral version. We did not pursue a graph-Laplacian + Nyström basis because a perfect baseline cannot be beaten.

### Positive byproduct

The **plain early→late activation residual** (≈ Mumcu & Yilmaz's image-domain mechanic, ported to an LLM) is a near-perfect, near-free adversarial/OOD monitor (AUROC ≈ 1.0 on GCG/OOD). Deployable independent of fmap.

### Reassessment

The residual-as-monitor falls to the point we derived analytically earlier: in the **single-model** setting the coordinate frame is shared, so a plain linear/activation diff is the right tool and the functional map's coordinate-free machinery is dead weight (here, worse, it inverts on OOD). fmap can only earn its place where incumbents structurally don't apply: **cross-model alignment** (no shared frame, no trivial distance baseline), i.e. Experiment 2. Experiment 1 (intra-architecture diffing) shares the same shared-frame vulnerability. **Recommendation:** treat Experiment 2 (cross-model probe transfer) as the make-or-break test of whether fmap earns a place in safety at all; a clean negative there is a legitimate stopping point.

### Reproduction

- Scripts: `mvp/phase0.py` (Run 1), `mvp/phase0_diag.py` (diagnostic), `mvp/phase0_adv.py` (Run 2).
- Results JSON persisted to the Modal volume `fmap-hf-cache` under `/cache/results/`.
- Modal runs (org `stackonehq`): phase0 `ap-WYLsIGAY72dWtL164Amw97`, diag `ap-vNYYSKCb4FGECG4PdPQ6Ag`, adv `ap-fqFAXKcbB6dxN8MhmslXnx`.
