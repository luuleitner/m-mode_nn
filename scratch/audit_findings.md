# ML Pipeline Audit Findings

Full audit of preprocessing, data loading, model architecture, training, and config system.
Conducted 2026-02-12.

---

## TIER 1: Actively Hurting Training Right Now

### 1. CNN classifier has NO learning rate scheduler (cosine_warmup unhandled)

- **File:** `src/training/train_cnn_cls.py:205-221`
- **Impact:** Every CNN training run uses constant LR instead of warmup + cosine decay
- `_create_scheduler()` handles `cosine`, `plateau`, `onecycle` only. `"cosine_warmup"` from config falls through to `self.scheduler = None`.
- BaseTrainer (UNet) handles `cosine_warmup` correctly.
- **Fix:** Add `cosine_warmup` case to `_create_scheduler()` using `SequentialLR(LinearLR + CosineAnnealingLR)`, matching BaseTrainer pattern at `base_trainer.py:121-132`.

### 2. UNet gets wrong dropout values -- `get_regularization_config()` drops `dropout` sub-dict

- **File:** `config/configurator.py:198-210` -> `src/training/train_unet_ae.py:163-167`
- **Config says:** `dropout.spatial = 0.3`, `dropout.fc = 0.5`
- **UNet gets:** `spatial_dropout = 0.0`, `classifier_dropout = 0.3` (fallback defaults)
- `get_regularization_config()` returns only `{grad_clip_norm, batch_norm}`, never includes the nested `dropout` dict.
- CNN trainer bypasses this by reading OmegaConf directly -- only UNet affected.
- **Fix:** Include `dropout` sub-dict in `get_regularization_config()` return value.

### 3. Envelope padding config silently ignored -- wrong `analytic_signal` imported

- **File:** `preprocessing/processor.py:13,696` vs `preprocessing/signal_utils.py:96`
- Config: `envelope.padding.apply: true, mode: "constant", amount: 30`
- Processor imports `analytic_signal` from `dasIT` (no padding parameter). The local `signal_utils.analytic_signal` supports padding but is never used.
- Padding config values loaded into instance vars but never passed anywhere.
- **Impact:** Hilbert transform boundary artifacts (Gibbs ringing) uncorrected.
- **Fix:** Import `analytic_signal` from `signal_utils` instead of `dasIT`, or pass padding params.

### 4. OneCycleLR never stepped per-batch in BaseTrainer (UNet)

- **File:** `src/training/base_trainer.py:263-323` (train_epoch), line 527 (epoch step)
- `OneCycleLR` requires `.step()` after every batch. BaseTrainer skips epoch-level step but never calls per-batch step in `train_epoch()`.
- CNN trainer does this correctly at `train_cnn_cls.py:357-358`.
- **Currently masked:** config uses `cosine_warmup`, not `onecycle`. Latent but critical.
- **Fix:** Add per-batch scheduler step for OneCycleLR in `BaseTrainer.train_epoch()`.

### 5. Differentiation applied AFTER log-compression -- computes log-derivative, not velocity

- **File:** `preprocessing/processor.py:695-728`
- Pipeline: Envelope -> Lowpass -> Decimation -> **Log-compress** -> **Differentiate**
- Result: `d/dt[log(env)] = env'(t) / env(t)` (relative rate of change, not velocity)
- If intent is tissue velocity: differentiation must come BEFORE log-compression.
- **Decision needed:** Is log-derivative intentional? If so, document. If not, reorder.

---

## TIER 2: Correctness Issues

### 6. `train_test_split` return names swapped

- **File:** `src/data/datasets.py:543-548,558-562`
- `test_data, val_data = train_test_split(test_val_data, test_size=test_ratio, ...)`
- sklearn returns `(first=1-test_size, second=test_size)`, so test_data gets `1-ratio` and val_data gets `ratio`.
- **Masked** with current `test_val_split_ratio=0.5` (symmetric). Any non-0.5 ratio inverts proportions.
- **Fix:** Swap variable names or use `test_size=1-test_ratio`.

### 7. Per-experiment normalization leaks across train/test boundary

- **File:** `preprocessing/signal_utils.py:116-130`
- `peak_normalization` computes min/max across all pulses of an experiment.
- **Currently safe** with `split_level: "experiment"`. Fragile if split level changes to `"sequence"`.

### 8. Division by zero in normalization for constant-value channels

- **File:** `preprocessing/signal_utils.py:116-122`
- `data /= (maximum - minimum)` -- NaN if max == min. No guard.
- Same in `Z_normalization` (divides by sigma=0).
- **Fix:** Add epsilon or constant-signal check.

### 9. CheckpointCallback always tracks best val_loss, ignoring balanced accuracy monitor

- **File:** `src/training/callbacks/checkpoint.py:48-54`
- Early stopping monitors `val_balanced_accuracy` but CheckpointCallback saves "best" on `val_loss`.
- CNN trainer handles this with own `is_best` logic. UNet via BaseTrainer saves wrong "best".
- **Fix:** Make CheckpointCallback accept a `monitor` parameter matching early stopping.

### 10. Joystick/ultrasound sample count mismatch silently zero-padded

- **File:** `preprocessing/processor.py:887-897`
- Mismatch silently truncated or padded with zeros (= noise class). No max threshold check.
- **Fix:** Raise error if mismatch exceeds configurable threshold (e.g., 1% of samples).

### 11. `test_mae` metric is actually MSE

- **File:** `src/training/base_trainer.py:595-596`
- `'test_mae': val_metrics['mse']` -- wrong metric name, misleading downstream consumers.
- **Fix:** Compute actual MAE or remove the key.

---

## TIER 3: Architectural / Design Issues

### 12. UNet final upsample discards ~44% of width output via hard crop

- **File:** `src/models/unet_ae.py:193-198,286-289`
- Default dims (130x18 input): ConvTranspose2d produces 144x32, cropped to 130x18.
- 14 columns (44% of width) discarded. Final stage has NO skip connection from input.
- **Fix:** Use `F.interpolate` for final upsample, or add input-level skip connection.

### 13. Global RNG state corruption in dataset `__getitem__`

- **File:** `src/data/datasets.py:895`, `src/data/augmentations.py:62-63`
- `np.random.seed(augment_seed)` inside `__getitem__` mutates global state.
- With `num_workers=8`: cross-worker interference, correlated augmentations.
- Balance-augmented samples have frozen RNG state from pickle serialization.
- **Fix:** Use `np.random.RandomState(seed)` or `np.random.default_rng(seed)` locally. Don't serialize augmenter into pickle.

### 14. Three independent split computations -- fragile determinism

- **File:** `src/data/datasets.py:1128-1145`
- Train/val/test each independently recompute the full split. Works only because global RNG is seeded identically.
- **Fix:** Compute split once, pass split DataFrames to each instance.

### 15. No `worker_init_fn` on DataLoaders -- determinism broken

- Config: `num_workers: 8, behaviour: deterministic`
- No DataLoader passes `worker_init_fn` or `generator=torch.Generator()`.
- Workers fork with identical RNG state causing correlated augmentations.
- **Fix:** Add `worker_init_fn` that seeds each worker uniquely.

### 16. `np.roll` temporal shift wraps data around edges

- **File:** `src/data/augmentations.py:124-127`
- With `window=25, max_shift=3`: 12% of temporal signal wrapped.
- **Fix:** Use zero-pad or edge-pad after shift instead of `np.roll`.

### 17. Dropout2d after BatchNorm -- train/eval distribution mismatch

- **File:** `src/models/unet_ae.py:46-61`
- BatchNorm running stats from pre-dropout training, but eval has no dropout. Known issue (Li et al., 2019).
- **Fix:** Place Dropout2d BEFORE BatchNorm.

### 18. Label threshold from absolute max -- outlier vulnerable

- **File:** `preprocessing/label_logic/label_logic.py:84-88`
- `vel_range = np.max(np.abs(velocity))` -- single spike inflates threshold, disabling detection.
- **Fix:** Use robust percentile (e.g., 99th).

---

## TIER 4: Config / Code Hygiene

| #  | Issue | Location |
|----|-------|----------|
| 19 | WandB API key in plaintext, committed to git | `config.yaml:320` |
| 20 | `get_validation_config()` looks for early_stopping in wrong YAML path + `enable` vs `enabled` typo | `configurator.py:212-231` |
| 21 | Loss weight defaults (mse=0.8, cls=0.0) drastically differ from config (mse=0.2, cls=0.8) | `configurator.py:125-139` |
| 22 | `validate_configuration()` only checks mse+l1, ignores classification/contrastive | `configurator.py:411-416` |
| 23 | `include_noise: false` will crash preprocessing (no remap implemented) | `processor.py:144`, `soft_labels.py:47` |
| 24 | Bare `except` in processor swallows all errors including KeyboardInterrupt | `processor.py:609-623` |
| 25 | Hardcoded 3 classes in statistics functions (pipeline uses 5) | `precompute_datasets.py:81-83` |
| 26 | UNet docstring says `[B,C,Depth,Pulses]` but receives `[B,C,Pulses,Depth]` after adapter transpose | `unet_ae.py:7` |
| 27 | Incompatible checkpoint formats between BaseTrainer and DirectClassifierTrainer | `checkpoint.py` vs `train_cnn_cls.py` |
| 28 | Majority vote ties in soft labels systematically favor noise (lowest index) | `soft_labels.py:121-124` |
| 29 | `logcompression` internally calls `envelope()` again + `log10(0) = -inf` risk | `dasIT/features/signal.py:80-89` |
| 30 | Sequence grouping inflates class weight calculation (overlapping tokens counted multiple times) | `datasets.py:610-640` |
| 31 | Early stopping patience resets to 0 on training restart | `early_stopping.py:56-63` |
| 32 | Contrastive loss mask uses `-1e9` which overflows in float16/AMP | `losses.py:131-135` |
| 33 | CNN MaxPool2d(2,2) in block3 fails if input_pulses < 2; even kernels break padding with kernel_scale=2 | `direct_cnn_classifier.py:90-127` |
| 34 | Unbounded UNet embedding (no normalization) interacts poorly with contrastive loss | `unet_ae.py:178` |
| 35 | CNN OneCycleLR uses `max_lr = lr * 10` vs BaseTrainer uses `max_lr = lr` (inconsistent) | `train_cnn_cls.py:213-219` vs `base_trainer.py:111-116` |
| 36 | `squeeze()` without dim arg can collapse batch dimension for batch_size=1 | `base_trainer.py:166`, `train_cnn_cls.py:317` |
| 37 | `best_val_loss` stores loss at best-monitored-metric epoch, not actual best val_loss | `train_cnn_cls.py:809-815` |
| 38 | Redundant CSV separator conditional: `',' if .csv else ','` | `datasets.py:201-204` |
| 39 | Two `analytic_signal` functions exist with different signatures (dasIT vs signal_utils) | `signal_utils.py:96` vs `dasIT` |
| 40 | `*kwargs` instead of `**kwargs` in signal_utils `analytic_signal` | `signal_utils.py:100` |
| 41 | Reversal detection can look before `start_idx` (examines pre-movement samples) | `label_logic.py:146-158` |
| 42 | Dead code in TwoStageClassifier.predict (for loop overwritten by direct indexing) | `two_stage_classifier.py:293-302` |
| 43 | TwoStageClassifier.predict_proba runs Stage 2 on out-of-distribution noise samples | `two_stage_classifier.py:257` |
| 44 | Missing balanced_accuracy in CNN test results (only computed for validation) | `train_cnn_cls.py:486-495` |

---

## Recommended Fix Priority

```
IMMEDIATE (actively degrading results)
────────────────────────────────────────
#1  Add cosine_warmup to CNN _create_scheduler()
#2  Include dropout sub-dict in get_regularization_config()
#3  Import analytic_signal from signal_utils (with padding)
#5  Decide: log-derivative vs velocity. Document or reorder.
#19 Rotate WandB API key, move to env variable

NEXT SPRINT (correctness)
────────────────────────────────────────
#6  Swap test_data/val_data names in train_test_split
#8  Add max==min guard in peak_normalization
#9  Make CheckpointCallback monitor-aware
#10 Add max-mismatch threshold for joystick/US alignment
#11 Fix test_mae metric or remove it

ARCHITECTURE (design decisions needed)
────────────────────────────────────────
#12 UNet final stage: interpolate or add skip connection
#13 Replace global np.random.seed with local RandomState
#14 Compute split once, share across dataset instances
#15 Add worker_init_fn to all DataLoaders
#17 Move Dropout2d before BatchNorm in encoder blocks
```
