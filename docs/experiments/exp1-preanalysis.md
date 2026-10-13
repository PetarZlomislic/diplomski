# Experiment 1 — pre-analysis plan

**Ch. 4: comparative study of SAR/optical fusion mechanisms under sensor degradation.**

Written before any run. Hypotheses, decision rules and analysis are committed here so that results are *interpreted*, not rationalised. Deviations from this document get recorded as deviations, with reasons.

---

## 1. The question

> **Does the choice of fusion mechanism determine how gracefully a SAR/optical model degrades when one sensor fails — and does the right mechanism depend on the *structure* of the failure?**

Two parts. The first is the ordinary comparative question. The second is the thesis's actual claim, and it is the one worth the compute.

**What this experiment is not.** It is not a bid for state-of-the-art clean accuracy on reBEN. 54 pretrained reBEN models already exist at `BIFOLD-BigEarthNetv2-0` in S1-only, S2-only and S1+S2 variants, so published clean comparisons on this exact data already exist. **The contribution is the degradation behaviour**, and the write-up must say so plainly rather than competing on a leaderboard it cannot win.

---

## 2. Design

### 2.1 Independent variable A — fusion mechanism (6 trained configurations)

Note this is **not** a full factorial: gate granularity only applies to the gating arm.

| ID | Configuration | What it tests |
|---|---|---|
| `opt` | Optical-only (S2) | Unimodal floor / ceiling |
| `sar` | SAR-only (S1) | Unimodal floor / ceiling |
| `early` | Channel concatenation before the encoder | Fusion with no capacity to isolate a corrupted stream |
| `late` | Separate encoders, combined at the logit level | Fusion that can ensemble but not reweight by content |
| `gate_global` | Attention gating, **one weight per modality per sample** | Content-adaptive, spatially uniform |
| `gate_spatial` | Attention gating, **per-token weights** | Content-adaptive, spatially varying |

### 2.2 Independent variable B — degradation (applied at evaluation only, no retraining)

| Degradation | Target | Structure | Severity grid |
|---|---|---|---|
| `none` | — | — | {0} |
| `cloud` | optical | Spatially localised, total loss inside mask | {0, .25, .5, .75, 1.0} = fraction of patch occluded |
| `speckle` | SAR | Global, multiplicative | {0, .25, .5, .75, 1.0} → multiplicative gamma noise, mapped to a stated look-number range |
| `dropout` | either | Binary, global | {drop-optical, drop-SAR} |
| `temporal` | optical↔SAR pairing | Semantically valid, temporally offset | {0, 1, 2, 4, 8} acquisition steps Δ |

`temporal` is included because it is the one degradation **no gating mechanism can fix in principle** — misaligned features don't correspond, so any reweighting of them weights the wrong pixels. It is the control that makes the other results interpretable.

### 2.3 Replication

**3 seeds per configuration.** 6 configs × 3 seeds = **18 training runs**.

Evaluations: 18 checkpoints × (4 cloud + 4 speckle + 2 dropout + 4 temporal severities, excluding the shared clean point) + 18 clean = **270 evaluation runs**.

### 2.4 Dependent variables

- **Primary:** macro-F1 on the **test** split (macro, not micro — reBEN's label distribution is heavily skewed and micro-F1 would hide rare-class collapse)
- Secondary: micro-F1, per-class F1, accuracy
- Cost: parameter count, FLOPs, inference latency

---

## 3. Derived quantities and the measurement decision that matters

For each (config, degradation) pair define:

- **Absolute curve** `F1(s)` — raw performance at severity *s*
- **Relative retention** `R(s) = F1(s) / F1(0)` — performance as a fraction of that model's own clean performance
- **AUDC** — normalised area under the curve, the single number used for ranking: `AUDC = mean_s F1(s)`, with `AUDC_rel = mean_s R(s)`

**These answer different questions and can disagree.** Absolute AUDC answers *"which model would I deploy under cloud?"* Relative retention answers *"which fusion mechanism is intrinsically most robust?"* — it divides out differences in clean accuracy and backbone capacity, isolating the mechanism.

**Pre-committed:** the headline scientific claim is made on **relative retention**, because it isolates the mechanism from capacity. Absolute curves are reported alongside, always. **If the two orderings disagree, that disagreement is itself a reported finding**, not something to resolve by picking the flattering one.

---

## 4. Hypotheses

Directional, falsifiable, committed in advance. Each cites why it is predicted.

**H1 — Optical dominance.** On clean data, `opt` ≈ all fusion configs; SAR adds little when optical is intact. *Why:* land cover is largely spectrally determined, and modality imbalance during training (the premise of OGM-GE) predicts the SAR branch is undertrained.
*Rejected if:* any fusion config exceeds `opt` on clean macro-F1 by more than the seed spread.

**H2 — Asymmetric degradation.** Degrading optical costs the fusion configs more than degrading SAR does. *Why:* follows directly from H1.
*Rejected if:* AUDC drop under `speckle` ≥ AUDC drop under `cloud` for the fusion configs.

**H3 — Graceful-degradation ordering.** Under optical degradation, relative retention orders `gate_* > late > early`. *Why:* early fusion mixes modalities at the input and cannot isolate a corrupted stream; late fusion can ensemble; gating can explicitly reallocate.
*Rejected if:* `early` ≥ `gate_*` — which would mean the gate is not learning what we claim it learns.

**H4 — Granularity × structure interaction. ⭐ The headline.** `gate_spatial` beats `gate_global` under `cloud`, but **not** under `speckle` or `dropout`. *Why:* cloud is spatially localised, so a global gate must discard the clean 70% of a 30%-clouded patch; speckle and dropout are global, so spatial resolution in the gate buys nothing.
*Rejected if:* granularity helps uniformly across degradation types, or not at all.

**H5 — Architecture over capacity.** AUDC ranking is not predicted by parameter count. *Why:* MuteBench found architecture family predicts robustness better than parameter count, with a ~256M-param model among the most vulnerable.
*Rejected if:* AUDC correlates with params across the six configs.

**H6 — Temporal misalignment is unresponsive.** All configs degrade similarly under `temporal`; gating confers no advantage. *Why:* misaligned features don't correspond, so reweighting cannot help.
*Rejected if:* any gating config retains meaningfully more than `early` under `temporal` — in which case **suspect the simulation before believing the result** (§7).

---

## 5. Decision rules

Written in advance so the result cannot be re-interpreted after the fact.

| Outcome | Interpretation | Action |
|---|---|---|
| **H1 holds** (fusion ≈ optical on clean) | Expected, **not a failure.** It makes the experiment *more* informative: if fusion has value, it must appear under degradation or nowhere | Record and proceed |
| **H1 holds AND H3 fails** | Fusion adds nothing anywhere — the SAR branch is undertrained, the predicted most-likely failure | **Apply OGM-GE** (gradient modulation by confidence discrepancy ratio + noise injection) and re-run. Report both arms |
| **H3 holds, H4 holds** | Headline result: mechanism matters, and the right mechanism depends on degradation structure | Carry `gate_spatial` to Ch. 5/6. Write up the interaction as the chapter's contribution |
| **H3 holds, H4 fails** (granularity irrelevant) | Gating helps, but the architecture↔degradation matching claim is wrong | **Report as a negative result.** Revisit Ch. 6's framing, which leans on reliability being spatially meaningful |
| **H4 holds only partially** (e.g. spatial wins everywhere) | Spatial gate may be winning on capacity, not structure | Add a parameter-matched global-gate control before claiming the interaction |
| **H5 fails** (params predict AUDC) | The comparison is confounded by capacity | Parameter-match the configs and re-run, or restrict claims to relative retention only |
| **H6 fails** | Either a real and surprising finding, or a broken simulation | **Assume broken first.** Verify the misalignment actually produces non-corresponding features before writing anything |
| **All six configs within seed spread everywhere** | The degradation protocol is too weak to separate anything | Hard stop. Increase severity range and re-pilot before spending Stage 2 budget |

---

## 6. Controls

- **Identical backbone, schedule, augmentation, and optimiser** across all six configs. Differences must be attributable to the fusion mechanism.
- **Identical initialisation.** All configs start from the same checkpoint with randomly-initialised fusion layers. **Do not** use the per-variant BIFOLD S1/S2/S1+S2 weights for the compared arms — those three were trained separately, and using them would confound mechanism with pretraining. BIFOLD numbers are used *only* as an external sanity check on clean performance.
- **`degradation_seed` is independent of the training seed.** Same severity ⇒ byte-identical corruption across all configs. Without this the comparison is void.
- **`severity = 0` must be an exact identity.** Asserted in tests; verified again as a pre-flight check (§8).
- **Split discipline.** `val` for checkpoint selection, `test` for the degradation sweep. No checkpoint is ever selected using a degraded metric.
- **Spatial autocorrelation** is handled by reBEN's geographically-disjoint splits — the reason for using reBEN over BigEarthNet v1.

---

## 7. Threats to validity

Stated now so they appear in the limitations section rather than being discovered by an examiner.

1. **Cloud simulation realism.** Synthetic masks are not real cloud. The claim is about *response to localised occlusion*, not about operational cloud performance. Document the mask-generation method precisely and keep the claim scoped.
2. **Speckle model.** Multiplicative gamma noise is the standard model but real speckle is correlated and sensor-dependent. State the look-number range simulated.
3. **Three seeds is thin.** Report mean and range; treat an effect as real only if it exceeds the seed spread. **Do not run significance tests that three samples cannot support** — a descriptive uncertainty band is the honest presentation.
4. **Shared pretrained initialisation may already encode robustness**, compressing differences between mechanisms. Report that all configs share it; note it as a possible ceiling on effect size.
5. **Severity scales are not comparable across degradation types.** `cloud=0.5` and `speckle=0.5` are not equivalent damage. Never compare AUDC *across* degradation types — only within.
6. **`temporal` depends on reBEN offering multiple acquisitions per tile.** Verify this before committing to the arm; if unavailable, simulate by pairing geographically adjacent tiles and say so.

---

## 8. Execution: two stages

**Stage 1 — pilot (cheap, throwaway).** 6 configs × **1 seed**, small backbone, `reben-sweep` (15%) subset, reduced severity grid {0, 0.5, 1.0}.
*Purpose:* verify the pipeline end to end and get a rough ordering. ~6 GPU-hours, roughly €6 on A10G.
*Gate:* do not proceed to Stage 2 until every pre-flight check below passes.

**Stage 2 — the reported experiment.** 6 configs × 3 seeds, full severity grids, full splits.
Budget guided by what Stage 1 reveals; if the pilot shows two configs indistinguishable everywhere, consider dropping one rather than paying for it three times.

### Pre-flight checks — all must pass before Stage 2

- [ ] `severity=0` evaluation is **bit-identical** to a clean evaluation, for every degradation
- [ ] Same `degradation_seed` produces byte-identical corrupted tensors across two different trained models
- [ ] `opt` is unaffected by `speckle`, and `sar` is unaffected by `cloud` — **if either moves, the degradation is leaking into the wrong modality**
- [ ] Clean macro-F1 for `opt` and `sar` is within a plausible band of the published BIFOLD S2-only / S1-only numbers. Far below ⇒ a training bug, not a finding
- [ ] `data_wait_frac < 0.3` on the pilot runs — otherwise you are paying GPU rates to wait on I/O
- [ ] Per-class F1 shows no class at exactly zero across all configs, which would indicate a broken label or a sampling failure

That third check is the highest-value one in the list: it catches a wiring error that would otherwise produce a plausible-looking but entirely meaningless set of curves.

---

## 9. Deliverables

1. **Figure 1** — absolute degradation curves, one panel per degradation type, six lines each, with seed-spread bands
2. **Figure 2** — relative retention `R(s)`, same layout. The headline figure
3. **Figure 3** — the H4 interaction: `gate_global` vs `gate_spatial` across degradation types, showing the effect is cloud-specific or isn't
4. **Table 1** — clean performance, AUDC and AUDC_rel per (config, degradation), plus params and latency
5. **Table 2** — hypothesis outcomes: held / rejected / inconclusive, with the number that decided each
6. **A stated selection** — which mechanism carries forward to Ch. 5/6, and the specific result that justifies it
7. **A deviations log** — anything done differently from this document, and why
