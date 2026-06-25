# FMAP: Functional Maps for Alignment Probing

Can a **functional-map alignment residual** act as a runtime monitor for adversarial /
out-of-distribution inputs to a language model, flagging *and* localizing where the model's
computation goes anomalous?

**The idea:** calibrate a functional map between an *early* and a *late* layer on benign text. At
inference, the residual (what the map cannot explain) should spike on inputs that break the normal
early→late flow. The bet is that this readable, unsupervised operator beats, or at least matches,
more cheaply, entrenched detectors like Mahalanobis and linear probes, while also pointing at
*where* the breakdown happens.

## The 3D activation atlas

<img width="1473" height="759" alt="image" src="https://github.com/user-attachments/assets/a13cfb57-7e0d-4968-8119-865499db5a4e" />

*Each glowing point is one input's last-token residual-stream activation, projected to 3D. Benign
inputs (grey) stay centered while adversarial (GCG/PAIR) and far-OOD inputs fling outward as depth
increases.*

[`viz/index.html`](viz/index.html) is an interactive 3D map of the model's internals, Three.js +
Plotly, no build step. Open it directly:

```bash
open viz/index.html                      # macOS; or serve viz/ and browse
uv run modal run mvp/export_viz.py       # regenerate viz/data.js from a model
```

- **Orbit** (drag), **zoom** (scroll), **morph across layers** (slider / Play), watch the manifold
  reorganize with depth.
- Toggle **PCA / UMAP**, toggle classes, **hover** a point for its prompt + detector scores.
- The right panel histograms each detector's benign-vs-anomaly scores, with the AUROC.

**Why it matters:** the atlas makes the Phase-0 result *visible*. Adversarial/OOD inputs separate
from benign by the late layers, which is exactly why a trivial distance/residual baseline nails
them (AUROC ≈ 1.0), and why the spectral functional-map residual (which even *inverts* on far-OOD)
does not earn its place.

## Status, Phase 0 (2026-06-25): falsified

First test executed, a Phase-0 MVP on Modal (Qwen2.5-1.5B). The bar was never "fill an empty space"
but **beating entrenched methods on tasks where they already work**, and the residual monitor
didn't clear it.

**The functional-map residual was beaten outright** by a plain activation-space early→late residual
and by Mahalanobis (already perfect, AUROC ≈ 1.0, on adversarial/OOD inputs), and was *inverted* on
far-OOD (more-OOD → smaller residual). In the single-model setting the coordinate frame is shared,
so a plain diff is the right tool and the spectral machinery is dead weight. The residual-as-detector
idea is dead; if fmap earns a place anywhere it's in cross-model settings, where these single-model
baselines structurally don't apply.

Full results, baselines, and reproduction: [`docs/exp3_mvp.md`](docs/exp3_mvp.md).

## Repo layout

**Code** (uv project; heavy deps run inside the Modal image)
- `mvp/phase0.py`: content task (harmful vs. benign behaviors).
- `mvp/phase0_diag.py`: k-sweep + spectral-vs-raw probe coverage diagnostic.
- `mvp/phase0_adv.py`: adversarial/OOD task (GCG + PAIR + random-token OOD).
- `mvp/export_viz.py`: exports 3D PCA/UMAP coords + scores for the atlas.
- `viz/`: the interactive 3D activation atlas (`index.html` + generated `data.js`).

**Docs**
- [`docs/exp3_mvp.md`](docs/exp3_mvp.md): MVP scope, run instructions, and the Phase-0 results.
- [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md): prior-art survey + baselines (all arXiv IDs verified 2026-06-25).
- [`docs/proposal.md`](docs/proposal.md): the full research proposal and cross-cutting caveats (broader context).
