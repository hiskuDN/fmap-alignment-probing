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

- [`proposal.md`](proposal.md) — the full proposal: motivation, scope, four experiments, and
  cross-cutting caveats.

## The experiments at a glance

| # | Experiment | Role | What it tests |
|---|---|---|---|
| 1 | Functional-map model diffing for hidden misalignment | primary | Can `C` and its residual flag and localize a backdoor / emergent-misalignment fine-tune? |
| 2 | Cross-model transfer of a safety probe | universality test | Does a refusal/lie-detection direction carry across models through the map? |
| 3 | Alignment breakdown as a runtime monitor | secondary | Does the residual separate adversarial/OOD inputs from benign? |
| 4 | Checkpoint drift | exploratory | When in training does a safety-relevant structure form? |

## Status

Proposal stage — no code yet. The bar is not "fill an empty space" but **beating entrenched
methods (SAE crosscoders, linear probes) on tasks where they already work.** See the proposal's
*Cross-cutting caveats* for the honest limitations (correlational-only signal, anchor dependence,
adversarial evasion).

All references in the proposal were verified against arXiv on 2026-06-25.
