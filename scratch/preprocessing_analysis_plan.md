# Preprocessing Analysis Plan

## Data Overview

```
Raw signal:     3 US channels × 6172 pulses × 1996 ADC samples (int16, 50 MSPS)
Participants:   4 (P0–P3)
Sessions:       6 per participant (24 total)
Experiments:    5 per session (120 total)
Tokens (post):  371,735 (window=20, stride=2)
Labels:         5 classes [Noise 24%, Up 21%, Down 19%, Left 19%, Right 18%]
```

---

## 1. Preprocessing Classification: Physics vs Analysis vs Ablation

### Tier A: Physics-Mandated (justify from instrumentation, never ablate)

| Step | Setting | Justification | How to Justify |
|------|---------|---------------|----------------|
| **Clip** | remove first 107, last 589 of 1996 samples (retain 1300 = 26 µs) | Removes transducer ring-down and attenuated tail. | Depth-wise SNR profile, show energy collapse in removed regions. |
| **Bandpass** | 8–12 MHz, order 10, fs=50 MSPS | ~10 MHz probe center freq. Rejects baseline drift, electronics noise, harmonics. | Show averaged PSD with passband overlay. State probe spec. Note: order-10 Butterworth is steep — confirm zero-phase (filtfilt) or argue envelope discards phase. |
| **Envelope** | Hilbert analytic signal, padding=30 (constant) | Removes carrier phase, retains backscatter morphology. Task is amplitude/morphology-based, not phase-coherent. | State that downstream task depends on amplitude morphology, not phase-coherent scattering. Padding suppresses Hilbert edge artifacts. |
| **Anti-alias + Decimation** | Lowpass ~2 MHz, decimate by 10 (50→5 MSPS) | Post-decimation Nyquist = 2.5 MHz > 2 MHz cutoff. Removes oversampling, reduces compute. | Show post-envelope PSD, confirm negligible energy above 2.5 MHz. One plot is sufficient. |

### Tier B: Physically Analyzable (justify with signal-domain diagnostics)

| Step | Setting | Justification | How to Analyze |
|------|---------|---------------|----------------|
| **TGC** | α = 0.3 dB/(MHz·cm) | Compensates depth-dependent attenuation. 0.3 plausible for soft tissue (literature: 0.3–0.7 for muscle/tendon). | Fit slope to mean log-envelope vs depth, show slope reduction. Compare α = 0.2, 0.3, 0.5. **Risk**: α too high flattens real depth structure. |
| **Log compression** | 50 dB | Reduces heavy-tailed amplitude distribution, stabilizes dynamic range. | Histogram skewness/kurtosis before vs after. Compare 40, 50, 60 dB on subset — distribution diagnostics only. **Risk**: too aggressive → noise floor dominates. |
| **Normalization** | Peak | Ensures consistent input scale across acquisitions. | Compare scaling factor stability: peak vs RMS vs percentile(99) vs z-score. Input statistics only, no ML needed. **Risk**: single spike sets entire sample's scale. |

### Tier C: Data-Facing Ablation (must test empirically, task-dependent)

| Step | Setting | Justification | How to Ablate |
|------|---------|---------------|---------------|
| **Differentiation** | 1st-order temporal gradient along pulse axis | Emphasizes motion/transitions. Also amplifies noise. | On/off is the single most important ablation. Also test: order 1 vs 2, gradient vs diff. **Risk**: amplifies jitter, may destroy static features. Hardest step to justify universally. |
| **Percentile clip** | Currently disabled (apply: false) | Good — shows restraint. | Only consider if outlier analysis shows need. Justify omission: avoids suppressing rare high-amplitude events that may be physically meaningful. |

---

## 2. Experiment Design: Data Selection

### Three-Scope Approach

| Scope | Data | Size | Purpose |
|-------|------|------|---------|
| **Full training set** | All 371K tokens (or raw equivalents) | 4P × 6S × 5E = 120 experiments | Cheap aggregate diagnostics that run in seconds |
| **Stratified subset** | 1 experiment per session per participant | 4P × 6S × 1E = 24 experiments, ~74K tokens (~20%) | Parameter screening & visual comparisons |
| **Pilot set** | 1 experiment from 2 sessions per participant, include 1 "difficult" case | 4P × 2S × 1E = 8 experiments | Waveform inspection, debugging, artifact discovery |

### Critical Rules

- Use **training data only** — never validation/test for preprocessing selection
- Sample at **experiment level**, not token level (respects hierarchy)
- Include at least **1 stress case** per participant (low SNR, boundary artifact)

### Stratified Subset Selection (Concrete)

Pick experiment 0 from every session:

```
P0: S0/E0, S1/E0, S2/E0, S3/E0, S4/E0, S5/E0    (6 experiments)
P1: S0/E0, S1/E0, S2/E0, S3/E0, S4/E0, S5/E0    (6 experiments)
P2: S0/E0, S1/E0, S2/E0, S3/E0, S4/E0, S5/E0    (6 experiments)
P3: S0/E0, S1/E0, S2/E0, S3/E0, S4/E0, S5/E0    (6 experiments)
────────────────────────────────────────────────────
Total: 24 experiments, ~74K tokens, all P×S combinations covered
```

Pilot set: pick 2 sessions per participant, include 1 with known low-quality or unusual characteristics (inspect first). Total: 8 experiments for manual waveform review.

---

## 3. Concrete Analysis Plan

### Plot Set 1: Depth Diagnostics — supports Clip, TGC

**Data**: full training set

| Plot | What | Why |
|------|------|-----|
| (a) | Mean \|amplitude\| vs depth (raw RF, all 3 channels) | Shows ring-down region, attenuation slope, noise floor |
| (b) | Variance vs depth | Confirms where SNR collapses |
| (c) | Mean log-envelope vs depth: before TGC, after TGC | Show slope reduction (quantify: dB/sample) |
| (d) | Compare TGC α = {0.2, 0.3, 0.5}: residual slope per setting | Justifies α=0.3 as partial compensation |

### Plot Set 2: Spectral Diagnostics — supports Bandpass, Envelope lowpass, Decimation

**Data**: full training set

| Plot | What | Why |
|------|------|-----|
| (a) | Averaged PSD of raw RF (pre-bandpass) | Overlay 8–12 MHz passband, mark probe center freq |
| (b) | PSD after bandpass | Show out-of-band suppression |
| (c) | PSD after envelope + lowpass | Show effective modulation bandwidth |
| (d) | Vertical line at 2.5 MHz (post-decimation Nyquist) | Confirm negligible energy above, quantify alias risk (fraction of energy > 2.5 MHz) |

### Plot Set 3: Distribution Diagnostics — supports Log compression, Normalization

**Data**: stratified subset

| Plot | What | Why |
|------|------|-----|
| (a) | Amplitude histograms: before vs after log compression | Report: skewness, kurtosis, dynamic range (99th/1st percentile) |
| (b) | Log compression comparison: 40 vs 50 vs 60 dB | Histogram overlay, just 3 settings on subset |
| (c) | Per-sample scaling factor distribution for: peak, RMS, percentile(99), z-score | Coefficient of variation of scaling factor. If peak has high CV → unstable |
| (d) | Cross-participant amplitude alignment check | Per-participant histograms after normalization should overlap |

### Plot Set 4: Differentiation Analysis — supports Differentiation (Tier C)

**Data**: stratified subset

| Plot | What | Why |
|------|------|-----|
| (a) | Example tokens: raw envelope vs differentiated | Visual: does it emphasize class-relevant transitions? |
| (b) | SNR before vs after differentiation | Quantify noise amplification |
| (c) | Fisher discriminant ratio: between-class / within-class variance, with and without differentiation | If Fisher ratio improves → differentiation helps |
| (d) | Per-class temporal profiles with/without differentiation | Visual class separability |

### Tiny ML Screen — only for Tier C, after signal diagnostics

**Data**: 1 participant (within-participant CV), 50 epochs, early stop

| # | Condition | Tests |
|---|-----------|-------|
| 1 | Full pipeline (baseline) | — |
| 2 | Full pipeline − differentiation | Differentiation value |
| 3 | Full pipeline, TGC α=0.2 | TGC sensitivity |
| 4 | Full pipeline, TGC α=0.5 | TGC sensitivity |
| 5 | Full pipeline, RMS norm instead of peak | Normalization robustness |

Metric: val_balanced_accuracy. This is a screening experiment, not a publication result.

---

## 4. Priority Order

| Priority | Analysis | Effort | Blocks |
|----------|----------|--------|--------|
| 1 (first) | Plot set 1 (depth) | low | nothing |
| 2 | Plot set 2 (spectral) | low | nothing |
| 3 | Plot set 3 (distributions) | medium | needs stratified subset selection |
| 4 | Plot set 4 (differentiation) | medium | needs stratified subset |
| 5 (last) | Tiny ML screen | high | needs plots 1–4 to narrow conditions |

Plot sets 1 and 2 run on full data and are cheap (aggregate statistics over numpy arrays). Start there — they defend 6 of 8 active preprocessing steps without touching the ML pipeline.

---

## 5. Paper-Ready Framing

### Narrative Structure

> Preprocessing was selected based on ultrasound acquisition physics and statistical conditioning for learning, rather than by exhaustive end-to-end ablation. Steps that correct known nuisances in the measurement process were treated as mandatory, whereas representation-shaping transforms were tuned using signal-domain diagnostics and limited screening experiments.

### Three Conceptual Stages

1. **Stage 1 — Acquisition-consistent RF conditioning**: clip, TGC, bandpass
2. **Stage 2 — Amplitude-domain representation**: envelope, lowpass, decimation, log compression
3. **Stage 3 — ML scale conditioning**: optional differentiation, normalization

### Invariance Argument

Each preprocessing step suppresses a specific nuisance variability while preserving task-relevant variability:

| Nuisance source | Suppressed by |
|-----------------|---------------|
| Depth-dependent attenuation | TGC |
| Front-end transients | Clip |
| Out-of-band noise | Bandpass |
| Raw dynamic range | Log compression |
| Acquisition gain differences | Normalization |
| Oversampling redundancy | Decimation |
| Carrier phase oscillation | Envelope extraction |

---

## 6. Reviewer Vulnerability Assessment

| Likely reviewer target | Your defense |
|------------------------|-------------|
| Differentiation — why? | Fisher ratio analysis + on/off ablation |
| Peak normalization — outlier sensitive? | Scaling factor CV comparison across methods |
| TGC α=0.3 — why not tissue-specific? | "Nominal regularizing correction, not precise tissue inversion" + slope reduction plot |
| Constant padding — edge artifacts? | Compare first/last 50 samples under constant vs reflect padding |
| No full ablation? | "Signal-domain criteria per step + targeted screening of uncertain steps" |