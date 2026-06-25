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

## The experiments at a glance

| # | Experiment | Role | What it tests |
|---|---|---|---|
| 3 | Runtime alignment-breakdown monitor (detect + localize) | **primary — current focus** | Does the residual flag adversarial/OOD inputs *and* localize where/how the breakdown enters, as a readable operator rather than a bare score? |
| 1 | Functional-map model diffing for hidden misalignment | secondary | Can `C` and its residual flag and localize a backdoor / emergent-misalignment fine-tune? |
| 2 | Cross-model transfer of a safety probe | universality test | Does a refusal/lie-detection direction carry across models through the map? |
| 4 | Checkpoint drift | exploratory | When in training does a safety-relevant structure form? |

## Status

Proposal stage — no code yet. The bar is not "fill an empty space" but **beating entrenched
methods (SAE crosscoders, linear probes) on tasks where they already work.** See the proposal's
*Cross-cutting caveats* for the honest limitations (correlational-only signal, anchor dependence,
adversarial evasion).

**Current focus — Experiment 3:** a runtime monitor that both *detects* adversarial/OOD inputs (residual magnitude) and *localizes* where the breakdown enters (residual structure, rendered as an operator-flow visualization), validated by patching.

All references in the proposal were verified against arXiv on 2026-06-25.
