# FMAP: Functional Maps for Alignment Probing

A research proposal for testing whether spectral representation-alignment operators are useful for detecting and localizing hidden misalignment in language models.

## Motivation

Most interpretability-for-safety work that reads model internals uses one of a few tools: linear probes, activation or path patching, circuit discovery, and sparse-autoencoder (SAE) crosscoders. A search of the safety literature turns up no work pointing the functional-maps framework at a safety problem, even though its two core operations map cleanly onto safety questions: comparing two representation spaces with a readable operator, and transferring structure between them.

Many internals-based safety problems are, underneath, a "compare two spaces and find where they diverge" problem. That is exactly what functional maps are built for. This proposal tests whether a functional-maps alignment operator earns a place next to SAE crosscoders and probes as a screener and localizer, using safety testbeds where ground truth already exists.

The bar is not "fill an empty space." The bar is beating entrenched methods (SAE crosscoders, linear probes) on a task where they already work. The experiments below are framed to test that directly.

## Background and scope

There are two distinct ways to "use something like this," and they have opposite prospects.

**Architecture path (out of scope).** Building a model out of Functional Attention (FUNCATTN) so it is interpretable by construction. This does not help here: FUNCATTN is a layer you train from scratch, so it tells you nothing about an already-trained transformer, and you cannot retrofit it onto a model whose attention is standard softmax. The bases come out legible in the original paper because the input is a discretized continuous field with low intrinsic dimension; language tokens are not samples of a continuum, the learned bases have no pressure to be meaningful, and the authors themselves list NLP as untested. A low-rank operator is not automatically interpretable. We do not pursue this path.

**Post-hoc alignment path (this proposal).** Repurposing the functional-maps machinery as a probe applied to existing models. This already exists as a general tool in Latent Functional Maps (LFM, Fumero et al. 2024), which FUNCATTN cites. LFM is not a safety paper: its motivation and experiments are stitching, retrieval, and cross-modal alignment. The primitive it provides is the one we want.

What the primitive gives us: an operator `C` between two representation spaces, estimated by regularized least squares against a set of anchor correspondences. The structure of `C` (its near-diagonality, its spectrum) plus the alignment residual (what `C` cannot explain) is a graded, readable signal of how and where two spaces correspond. It is unsupervised or weakly supervised, and cheap relative to training SAEs or running large patching sweeps.

Its edge over the incumbents is being a structured, global, unsupervised, architecture-flexible operator. It pays for that with a correlational-only signal and a dependence on anchor quality. The experiments are designed to test whether the edge is real.

## Experiments

### Experiment 1: Functional-map model diffing for hidden misalignment (secondary)

Detect and localize a hidden behavioral change between a base model and a compromised counterpart.

**Setup.** Take a base model and a known-compromised version from the model-organisms-of-misalignment line: a sleeper agent or backdoored model trained to act unsafely only when a trigger is present (Hubinger et al. 2024), or an emergent-misalignment fine-tune (Betley et al. 2025). Because the two models share architecture and tokenizer, feed identical inputs and use them as anchors. This removes the anchor-matching problem that usually limits functional maps. Estimate the operator `C` between the two representation spaces, per layer.

**Hypothesis.** On benign inputs, `C` is close to identity. A compromised model produces a structured, low-rank deviation concentrated on the directions the backdoor or misalignment recruits, and the alignment residual spikes on or near trigger-relevant structure.

**What we measure.**
1. Detection without the trigger: can the residual flag that a model is compromised when the auditor has black-box access, suspects poisoning, but does not know the trigger? This is the realistic and hard case named in the Sleeper Agents work.
2. Localization: does the divergent subspace identify the mechanism, validated by patching that subspace and checking it controls the triggered behavior.

**Baseline to beat.** SAE crosscoder diffing (e.g. Delta-Crosscoder, 2026), which returns a set of changed features. The functional-map pitch is a single global operator whose spectrum and diagonality give one readable, unsupervised signal of how much and where two models diverge, without training crosscoders.

**Failure modes.** The signal is correlational, so a divergent subspace is a hypothesis, not proof; patching validates it. A backdoor adversarially trained to minimize representation drift could evade the residual, the same situational-awareness problem every internals detector faces.

### Experiment 2: Cross-model transfer of a safety probe (universality test)

Test whether a safety-relevant direction learned on one model can be carried to another through the functional map, and what that says about shared structure.

**Setup.** Train a safety-relevant probe on model A: a refusal direction, a harmful-intent classifier, or a lie-detection direction in the style of Azaria & Mitchell (2023). Build a functional map from A to B using a modest anchor set of shared prompts. Apply the map to carry the probe into B's space. Compare the transferred probe against a probe trained natively on B.

**What we learn.** If transfer accuracy is close to native, safety-relevant structure is shared geometry across models. That buys two things: cheap auditing of a new model from a trusted reference, and evidence on whether safety representations are universal. If transfer fails, that is equally informative: safety structure is model-specific and reference-based auditing will not generalize.

**Why it fits.** This reuses the capability LFM actually validated (transfer between spaces), pointed at safety. The metric is clean and falsifiable either way.

**Failure modes.** Anchors are the whole game. Cross-tokenizer and cross-family alignment is hard. Start with same-family-different-size or same-architecture-different-seed before attempting cross-family transfer.

### Experiment 3: Runtime alignment-breakdown monitor — detect *and* localize (falsified in Phase 0)

> **Status (2026-06-25).** A Phase-0 MVP falsified the residual-as-detector core: the functional-map residual was dominated by a plain activation-space early→late residual and by Mahalanobis on both content and adversarial/OOD inputs (and *inverted* on far-OOD), with the trivial baseline already perfect (AUROC ≈ 1.0) — leaving no headroom for the spectral version. See [`exp3_mvp.md`](exp3_mvp.md). The live fmap bet moves to Experiment 2 (cross-model transfer), where single-model baselines don't apply. The description below is retained for the record.

Use the functional-map residual as a runtime signal for adversarial or out-of-distribution inputs — but read both its *magnitude* (detection) and its *structure* (localization: where in the network the breakdown enters, and along which subspace). The contribution is the second half. A bare residual-magnitude detector is not novel and is not the pitch.

**Setup.** Calibrate the map offline on a benign reference: either (a) between an early and a late layer of the same model (intra-model, inter-layer), or (b) between a benign reference manifold and the live representation. The map needs a *corpus* of benign inputs to define the manifold — a single forward pass cannot — so this is offline-calibrated and applied at runtime, not built live. At inference, score each input by its residual against the calibrated map.

**Hypothesis.** Benign inputs stay close to the calibrated correspondence: small, diffuse residual. Adversarial or far-OOD inputs produce a residual that is both *large* (the detection signal) and *structured* — concentrated at a particular layer transition and in a low-rank subspace (the localization signal). "Structured" is the load-bearing word, and it is tested, not assumed.

**What we measure.**
1. **Detection (scalar).** AUROC separating attack/OOD from benign by residual magnitude, against a linear-probe-on-activations baseline and a Mahalanobis / kNN distance-to-benign-manifold baseline.
2. **Localization (structured).** From the residual, read which layer transition and which activation subspace the anomaly recruits. Validate causally: patching that subspace should suppress the attack's effect; patching a random subspace of equal rank should not. Adopt the controls from the subspace-patching-illusion result (Makelov et al. 2023) so the validated subspace is genuinely causal rather than an artifact of the patch.
3. **OOD / unseen-attack generalization.** The differentiator over trained probes is that the residual is unsupervised, so it should generalize to attacks it was never calibrated against — whereas probes latch onto surface cues and collapse out of distribution (Wang et al. 2025). Test this on a *cross-attack split*: calibrate on attack family A, evaluate on held-out family B. Three arms — the unsupervised residual (ours), a linear-probe-on-activations (incumbent, expected to collapse), and a probe trained on the fmap features (an ablation isolating whether any robustness comes from the geometry of the fmap space or merely from being unsupervised). The probe-on-fmap is an ablation, never the detector: making the detector supervised would walk straight into the critique it is meant to sidestep.
4. **Legibility (falsification).** Directly test whether the "structured spike" is real. If detection AUROC is high but the recovered subspace is unstable across inputs, illegible, or no better than random under patching, then the detector works but the localization claim fails — the residual is a black box, not an explanation. That negative is informative and is reported as such.

**Visualization.** The operator-flow rendering — the sequence of inter-layer functional maps and their residual structure, shown as compact operators rather than a point cloud — is the human-facing readout of measurement 2: a view of *where* the computation diverges from its benign correspondence. It draws the map, not the dots, which is the one thing point-embedding tools (PCA, t-SNE, UMAP) structurally cannot. Its evaluation is the patching task above, optionally backed by a human-in-the-loop study: can an auditor localize the attack mechanism faster with the operator-flow view than with per-layer UMAP or the logit/tuned lens?

**Baseline to beat.** Detection: Mahalanobis-on-activations (Lee et al. 2018) and a linear-probe-on-activations, plus the nearest mechanistic competitor — an early→deep-layer feature regressor scored on its prediction residual (Mumcu & Yilmaz 2025), which is our idea minus the spectral basis. Localization: per-layer attribution and activation-patching sweeps, which answer "where" but expensively. The pitch is not "beat the probe on AUROC" (likely a wash) but "match it on detection while additionally localizing the breakdown to a patch-validated subspace from one calibrated operator."

**Prior art and positioning.** Detection from internal activations is crowded and mature, which is why this experiment is *not* pitched on detection. The defensible novelty is narrow and specific: a functional-map alignment residual used as a *per-input runtime* signal (functional maps have otherwise been offline-only — alignment, stitching, retrieval in Fumero et al. 2024, or a static cross-modal diagnostic in Sarkar 2026), feeding a *joint* detect-and-structurally-localize output validated by patching. Note what is **not** new and must not be claimed: operator-valued cross-layer comparison already exists (CAST, Fu et al. 2025; the Jacobian-spectral analysis of Fernando & Guitchounts 2026), and the latter already shows perturbations funnel into a low-rank residual-stream subspace. The pieces exist separately; the unified runtime object does not. Full survey, ranked closest-prior-work, and the complete baseline list are in [`RELATED_WORK.md`](RELATED_WORK.md).

**Failure modes.** Correlational, as everywhere — a localized subspace is a hypothesis patching must confirm. An attack tuned against the monitor can suppress the residual (the standard situational-awareness problem). And the legibility test may simply fail, collapsing this experiment back to a scalar detector with no edge over the probe — which is exactly why the test is run.

### Experiment 4: Checkpoint drift (exploratory)

Localize when, during training, a safety-relevant structure forms.

**Setup.** Track the functional map between consecutive training checkpoints, restricted to a safety probe direction. The per-checkpoint operator gives a "when did this change, and in what subspace" readout, which current tools answer awkwardly.

**Status.** More open-ended; depends on access to a checkpoint sequence with a known behavioral change.

## Cross-cutting caveats

These apply to every experiment above and should be stated up front in any writeup.

- **Correlational, not causal.** Functional maps give geometric correspondence, never causation. A divergent subspace or a residual spike is a localizer and a screener that hands a hypothesis to causal methods (patching). It does not establish what drives a behavior, and it does not replace patching or circuit analysis.
- **Anchor quality decides everything.** The map is only as good as its anchor correspondences and the basis built over the representation similarity graph. Same-model diffing sidesteps this by using identical inputs; cross-model and cross-modal settings inherit the full difficulty, and choosing the basis well is the real work.
- **Adversarial evasion.** A model trained to be situationally aware, or a backdoor trained against a detector, can suppress the representation drift the method keys on. No internals detector is safe from this, and functional maps are no exception.
- **Conceptual, not propositional.** Like SAEs, this aligns representation geometry, not propositions. It can tell you whether two geometries correspond; it will not natively give you "the model represents claim X as false." Capturing propositional content needs a binding or propositional layer on top (cf. Chalmers 2025).
- **"Is the basis meaningful" persists.** A compact, low-rank operator is easier to read than a dense affinity matrix, but readability is not interpretability. The same caution that sinks naive intrinsic-interpretability hopes applies to the learned or spectral basis here.
- **The comparison is against strong incumbents.** SAEs have a growing critical literature (faithfulness failures, feature absorption and hedging, sensitivity to hyperparameters, questions about beating random baselines), so the absolute bar is not high. But functional maps do not solve those SAE problems; they answer a different question. Frame each experiment as testing whether the method catches something the incumbents miss, or catches it cheaper, on testbeds with ground truth.
- **Empirically tested once (2026-06-25), and it lost.** A Phase-0 MVP of the Experiment 3 residual monitor was beaten outright by trivial baselines — a plain activation-space early→late residual (AUROC 1.0 on adversarial/OOD) and Mahalanobis — confirming the analytic worry: in the single-model setting the coordinate frame is shared, so a plain diff is the right tool and the spectral machinery does not earn its place. fmap's remaining case rests on cross-model settings where those baselines structurally don't apply. See [`exp3_mvp.md`](exp3_mvp.md).

## References

### Method and architecture

- Xiao, J., Gao, M., Weber, S., Yang, G., Cremers, D. (2026). *Functional Attention: From Pairwise Affinities to Functional Correspondences.* ICML 2026 (PMLR 306). [arXiv:2605.31559](https://arxiv.org/abs/2605.31559).
- Fumero, M., Pegoraro, M., Maiorca, V., Locatello, F., Rodolà, E. (2024). *Latent Functional Maps: a spectral framework for representation alignment.* NeurIPS 2024. [arXiv:2406.14183](https://arxiv.org/abs/2406.14183).
- Ovsjanikov, M., Ben-Chen, M., Solomon, J., Butscher, A., Guibas, L. (2012). *Functional maps: a flexible representation of maps between shapes.* ACM Transactions on Graphics 31(4). [doi:10.1145/2185520.2185526](https://doi.org/10.1145/2185520.2185526).
- Wu, H., Luo, H., Wang, H., Wang, J., Long, M. (2024). *Transolver: A fast transformer solver for PDEs on general geometries.* [arXiv:2402.02366](https://arxiv.org/abs/2402.02366).
- Garnelo, M., Czarnecki, W. M. (2023). *Exploring the space of key-value-query models with Intention.* [arXiv:2305.10203](https://arxiv.org/abs/2305.10203).

### Interpretability methods (baselines and comparisons)

- *A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models* (2025). [arXiv:2503.05613](https://arxiv.org/abs/2503.05613).
- *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* (2026). [arXiv:2602.14111](https://arxiv.org/abs/2602.14111).
- *Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes* (2026). [arXiv:2603.04426](https://arxiv.org/abs/2603.04426).
- Nasiri-Sarvi, A., Rivaz, H., Hosseini, M. S. (2025). *SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability.* TMLR 2026. [arXiv:2507.06265](https://arxiv.org/abs/2507.06265).
- Chalmers, D. (2025). *Propositional Interpretability in Artificial Intelligence.* [arXiv:2501.15740](https://arxiv.org/abs/2501.15740).
- Azaria, A., Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* EMNLP Findings 2023. [arXiv:2304.13734](https://arxiv.org/abs/2304.13734).
- Ravindran, S. K. (2025). *Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers.* [arXiv:2507.09406](https://arxiv.org/abs/2507.09406).
- *Mechanistic Interpretability for Large Language Model Alignment: Progress, Challenges, and Future Directions* (2026). [arXiv:2602.11180](https://arxiv.org/abs/2602.11180).

### Safety testbeds and threat models

- Hubinger, E., et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* [arXiv:2401.05566](https://arxiv.org/abs/2401.05566).
- Betley, J., et al. (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* [arXiv:2502.17424](https://arxiv.org/abs/2502.17424).
- Rando, J., Croce, F., Mitka, K., Shabalin, S., Andriushchenko, M., Flammarion, N., Tramèr, F. (2024). *Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs* (RLHF Trojan Competition). IEEE SaTML 2024. [arXiv:2404.14461](https://arxiv.org/abs/2404.14461).
- *Stealthy and Persistent Unalignment on Large Language Models via Backdoor Injections* (2023). [arXiv:2312.00027](https://arxiv.org/abs/2312.00027).

---

*Note on reference IDs:* all arXiv IDs above were verified against arXiv on 2026-06-25, including the Azaria & Mitchell (2304.13734) and Betley et al. (2502.17424) entries that were previously unverified.
