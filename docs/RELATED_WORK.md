# Related Work & Prior-Art Survey

*Background research behind the **Experiment 3** framing in [`proposal.md`](proposal.md): an offline-calibrated functional map (early↔late layer, or benign-reference↔live representation) whose **alignment residual** drives (1) detection, residual magnitude → AUROC, and (2) localization, residual structure → which layer transition and which low-rank subspace, validated by patching.*

*All arXiv IDs below were verified against arXiv on 2026-06-25. One-line summaries are drawn from abstracts and a literature-search pass; confirm characterizations against full text before relying on them in a formal writeup.*

## TL;DR, where the gap is

- **Detection from internal activations is crowded and mature.** We will not win on detection alone, and should not pitch it as the contribution.
- **Detect *and* structurally localize (layer transition + low-rank subspace, patch-validated) as a single runtime object is largely open.** The pieces exist separately; nobody ships them together. This is where our novelty concentrates.
- **Functional maps applied to neural-network internals is nearly empty**, one static, cross-modal diagnostic exists; nobody uses the functional-map *residual* as a per-input runtime signal.
- **"Operator-valued cross-layer comparison" is NOT new** (CAST; Fernando & Guitchounts). Frame our operator as a *functional map used per-input at runtime*, not as the first operator-valued layer comparison.

## 1. Detection from internal activations (our detection baselines)

- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888), Lee et al., 2018. arXiv:1807.03888. Canonical baseline: class-conditional Gaussians on hidden features, score = **Mahalanobis distance** to nearest mean, fused across layers; detects both OOD and adversarial.
- [Mahalanobis++: Improving OOD Detection via Feature Normalization](https://arxiv.org/abs/2505.18032), Mueller et al., 2025. arXiv:2505.18032. Feature-normalized refresh of Mahalanobis OOD.
- [Enhancing Out-of-Distribution Detection with Multitesting-based Layer-wise Feature Fusion](https://arxiv.org/abs/2403.10803), Li et al., 2024. arXiv:2403.10803. Multiple-testing across feature levels; representative **multi-layer aggregation** detector.
- [Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers](https://arxiv.org/abs/2506.20816), Mumcu & Yilmaz, 2025. arXiv:2506.20816. Trains a regressor to **predict deep-layer features from early-layer features**, scores per-input on the prediction residual weighted by per-layer attack impact. **Structurally the nearest detection competitor to our early→late residual** (see §5).
- [Adversarial Sample Detection Through Neural Network Transport Dynamics](https://arxiv.org/abs/2306.04252), Karkar et al., 2023. arXiv:2306.04252. Treats the network as a discrete dynamical system; detects by comparing the **layer-to-layer vector field** of clean vs. abnormal inputs, conceptually adjacent to our "flow of inter-layer operators."
- [HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](https://arxiv.org/abs/2502.14744), Jiang et al., 2025. arXiv:2502.14744. Training-free; monitors hidden-state patterns / refusal-direction similarity at inference.
- [Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics](https://arxiv.org/abs/2504.00446), Zhou et al., 2025. arXiv:2504.00446. Detects hallucinations and jailbreaks via **layer-specific activation patterns** at runtime.
- [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805), McKenzie et al., 2025. arXiv:2506.10805. Lightweight activation probes reusing the model's own activations; the cheap-monitor archetype.
- [False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize](https://arxiv.org/abs/2509.03888), Wang et al., 2025. arXiv:2509.03888. Trained probes latch onto trigger words and **collapse OOD**. Motivates an unsupervised geometry-based detector and mandates an OOD-generalization test.

**Takeaway:** detection-from-activations is mature. Mahalanobis (1807.03888) and the early→deep layer-impact regressor (2506.20816) are the two to beat on AUROC; HiddenDetect and activation probes are the cheap-monitor baselines.

## 2. Localizing where an anomaly enters

- [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112), Belrose et al., 2023. arXiv:2303.08112. Per-layer affine decoders to the vocabulary; the abstract explicitly notes the **trajectory of latent predictions can detect malicious inputs**. Strongest existing "per-layer readout used as a detector."
- [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030), Makelov et al., 2023. arXiv:2311.17030. The standard tool for "which layer/subspace is causally responsible," **plus a cautionary result**: a patched subspace can look causal while being an artifact. Our patching-validation step must adopt its controls.

**Verdict:** the constituent pieces (lens trajectories, layer-adaptive OOD, subspace patching, per-layer adversarial impact) exist separately, but **no runtime monitor delivers detection *and* structured layer-transition-plus-low-rank-subspace localization as one object with patching validation.** That joint framing is where our novelty concentrates.

## 3. Functional maps / spectral methods on NN internals

- [Functional Maps: A Flexible Representation of Maps Between Shapes](https://doi.org/10.1145/2185520.2185526), Ovsjanikov et al., 2012. ACM ToG 31(4) (no arXiv). Correspondence as a compact linear operator between Laplace–Beltrami eigenbases, the origin of the framework.
- [Latent Functional Maps: a spectral framework for representation alignment](https://arxiv.org/abs/2406.14183), Fumero et al., 2024 (NeurIPS). arXiv:2406.14183. Our base framework. Used for similarity / stitching / retrieval, **never for runtime monitoring, anomaly detection, or safety.**
- [On the Spectral Geometry of Cross-Modal Representations: A Functional Map Diagnostic for Multimodal Alignment](https://arxiv.org/abs/2604.08579), Sarkar, 2026. arXiv:2604.08579. **The single closest functional-map-on-NN-internals paper:** vision↔language correspondence as a compact operator between graph-Laplacian eigenbases. But it is a **static, offline cross-modal diagnostic**, no per-input runtime score, no layer localization, no patching. (Single-author 2026 preprint; treat as nearest/concurrent prior art.)
- [ResiDual Transformer Alignment with Spectral Decomposition](https://arxiv.org/abs/2411.00246), Basile et al., 2024 (TMLR). arXiv:2411.00246. Same lab lineage as LFM; **spectral decomposition of the residual stream** to amplify task-relevant components, an alignment/adaptation method, not a monitor.

**Takeaway:** functional maps on NN internals is genuinely sparse, and nobody uses the **functional-map residual as a per-input runtime signal.**

## 4. Representation similarity across layers (the "operator-valued" framing)

- [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](https://arxiv.org/abs/1706.05806), Raghu et al., 2017. arXiv:1706.05806. Affine-invariant cross-layer comparison.
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414), Kornblith et al., 2019 (ICML). arXiv:1905.00414. Introduces **CKA**, the scalar cross-layer similarity heatmap we are generalizing.
- [Do Wide and Deep Networks Learn the Same Things?](https://arxiv.org/abs/2010.15327), Nguyen et al., 2021 (ICLR). arXiv:2010.15327. The **"block structure"** finding, canonical cross-layer CKA structure.
- [CAST: Compositional Analysis via Spectral Tracking for Understanding Transformer Layer Functions](https://arxiv.org/abs/2510.14262), Fu et al., 2025. arXiv:2510.14262. **Most threatening to the "operator-valued generalization" claim:** estimates the realized per-layer transformation matrix (pseudoinverse) and reads its spectrum, i.e., already replaces scalar cross-layer similarity with a per-layer **operator**. Offline/analytic, not a per-input runtime detector or localizer.
- [Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology](https://arxiv.org/abs/2605.14258), Fernando & Guitchounts, 2026. arXiv:2605.14258. Per-layer **Jacobian eigendecomposition**; finds perturbations funnel into a **low-rank residual-stream bottleneck**, relevant prior art for our "the anomaly recruits a low-rank subspace" claim. Static/analytic, not a monitor.

**Verdict:** operator-valued / spectral per-layer characterization **already exists** (CAST; Fernando & Guitchounts). Our defensible narrower novelty: an operator that is (a) a *functional map in a Laplacian eigenbasis* (not a pseudoinverse transformation matrix or a Jacobian) and (b) used to produce a *per-input residual* for runtime detection + localization, not an offline characterization of the trained network.

## 5. Closest prior work, ranked

1. **Mumcu & Yilmaz, 2025 (2506.20816)**, closest on the **detection mechanic** (offline early→deep residual, per-input scoring, nonuniform layer impact). Differs: unstructured regressor (not a spectral functional map), magnitude-only (no structured subspace localization), no patching validation. *The detection baseline we most need to beat and differentiate from.*
2. **Sarkar, 2026 (2604.08579)**, closest on the **method** (functional map between graph-Laplacian eigenbases of NN reps). Differs: cross-modal, static offline diagnostic, no per-input runtime score, no localization, no patching.
3. **CAST, 2025 (2510.14262)**, closest on the **operator-valued cross-layer** framing. Differs: offline network characterization, not per-input, not a detector/localizer.
4. **Fernando & Guitchounts, 2026 (2605.14258)**, closest on **per-layer spectral operators + low-rank perturbation subspaces.** Differs: Jacobian eigendecomposition, analytic study, no runtime detection/localization/patching.
5. **Tuned Lens, 2023 (2303.08112)**, closest on **a per-layer readout already used to detect malicious inputs.** Differs: vocab-space affine probe per layer (not an inter-set functional-map operator), trajectory-based, no subspace localization, no patching.

## Novelty verdict

**Defensible:**
- A **functional-map alignment residual** used as a **per-input runtime monitor**, functional maps have only been used offline for alignment/stitching/retrieval (2406.14183) or static cross-modal diagnosis (2604.08579), never as an inference-time anomaly signal.
- The **joint detect-and-structurally-localize** output (magnitude → detection; structure → layer transition + low-rank subspace; validated by patching). The pieces exist separately; the unified runtime object does not.
- The **"flow of compact inter-layer operators"** framing, a map between representation sets rather than point-cloud distances or scalar similarities, instantiated specifically as a functional map.

**Must concede:**
- "First to make cross-layer comparison operator-valued", **false** (CAST; Fernando & Guitchounts). Frame ours as a *functional-map operator used per-input at runtime*.
- "The low-rank subspace an anomaly recruits is a new observation", temper it; 2605.14258 already shows perturbations funnel into a low-rank residual-stream bottleneck.
- Detection-from-activations broadly, and early→late residual scoring specifically (2506.20816), are established.
- OOD signal being depth-dependent / layer-localizable is established (2403.10803).

## Risks & controls to adopt

- **Subspace-patching illusion (2311.17030):** a patched subspace can look causal while being an artifact. The localization claim needs that paper's controls.
- **Probes collapse OOD (2509.03888):** trained probes latch onto surface cues. Since "geometry generalizes where probes don't" is our differentiator, explicitly test unseen-attack / OOD generalization (cross-attack split), not just in-distribution AUROC.

## Baselines we must beat

**Detection (AUROC vs. benign):**
1. Mahalanobis, Lee et al. 2018 (1807.03888), mandatory canonical baseline (optionally Mahalanobis++, 2505.18032).
2. Early→deep layer-regression residual w/ nonuniform layer weighting, Mumcu & Yilmaz 2025 (2506.20816), nearest mechanistic competitor.
3. Multi-layer / layer-adaptive OOD fusion, Li et al. 2024 (2403.10803).
4. Transport/dynamics vector-field detector, Karkar et al. 2023 (2306.04252).
5. Activation-probe / HiddenDetect cheap monitors, McKenzie et al. 2025 (2506.10805); Jiang et al. 2025 (2502.14744), cost-matched.
6. Tuned-lens trajectory malicious-input detector, Belrose et al. 2023 (2303.08112).

**Localization (show residual structure beats these at pinpointing layer + subspace):**
7. Per-layer Mahalanobis-at-depth profile (1807.03888 / 2403.10803).
8. Tuned/logit-lens per-layer trajectory (2303.08112).
9. Activation/subspace patching as ground-truth-ish localizer, with the illusion controls (2311.17030).
