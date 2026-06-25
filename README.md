# FMAP: Functional Maps for Alignment Probing

A research proposal testing whether spectral representation-alignment operators — the
[functional-maps](https://arxiv.org/abs/2406.14183) framework from geometry processing — are
useful for **detecting and localizing hidden misalignment** in language models.

The core bet: many interpretability-for-safety problems are really "compare two representation
spaces and find where they diverge" problems, which is exactly what a functional map computes — a
small, readable operator `C` between two spaces plus an alignment residual (what `C` cannot
explain). This proposal asks whether that operator earns a place next to SAE crosscoders and linear
probes as a screener and localizer, tested on safety testbeds where ground truth already exists.

## Contents

- [`docs/proposal.md`](docs/proposal.md) — the full proposal: motivation, scope, four experiments, and
  cross-cutting caveats.
- [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) — prior-art survey and baseline list behind the
  Experiment 3 framing (all arXiv IDs verified 2026-06-25).
- [`docs/exp3_mvp.md`](docs/exp3_mvp.md) — Experiment 3 MVP: scope, run instructions, and the
  Phase-0 results (the residual monitor was **falsified** — see verdict).

## The experiments at a glance

| # | Experiment | Role | What it tests |
|---|---|---|---|
| 2 | Cross-model transfer of a safety probe | **next candidate (live bet)** | Does a refusal/lie-detection direction carry across models through the map, beating no-transfer and a naive linear map? |
| 1 | Functional-map model diffing for hidden misalignment | secondary | Can `C` and its residual flag and localize a backdoor / emergent-misalignment fine-tune? |
| 3 | Runtime alignment-breakdown monitor (detect + localize) | **falsified (Phase 0)** | MVP: fmap residual beaten outright by a plain activation-space residual + Mahalanobis (perfect on adversarial/OOD), inverted on far-OOD. See `docs/exp3_mvp.md`. |
| 4 | Checkpoint drift | exploratory | When in training does a safety-relevant structure form? |

## Status

First experiment executed (Phase 0 MVP on Modal). The bar was never "fill an empty space" but
**beating entrenched methods (SAE crosscoders, linear probes, Mahalanobis) on tasks where they
already work** — and the first test held to it.

**Phase-0 finding (2026-06-25):** Experiment 3's residual-as-detector core is **falsified** — the
functional-map residual was beaten outright by a plain activation-space early→late residual and by
Mahalanobis (already perfect, AUROC ≈ 1.0, on adversarial/OOD inputs) and was *inverted* on far-OOD.
The single-model setting has a shared coordinate frame, so a plain diff is the right tool and the
spectral machinery is dead weight. Full results in [`docs/exp3_mvp.md`](docs/exp3_mvp.md).

**Next:** Experiment 2 (cross-model transfer) — the one setting where the single-model baselines
that just won structurally don't apply, and where LFM's published results live. The make-or-break
test of whether fmap earns a place in safety.

All references in the proposal were verified against arXiv on 2026-06-25.
