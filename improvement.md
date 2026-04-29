# Moonshine Fine-Tuning Improvement Plan

This file captures the current investigation, the most likely causes of the
fine-tuning regression, and the plan to align the English workflow with the
original authors' French-oriented setup.

## Current Observation

Fine-tuning the English Moonshine model is reducing performance relative to the
base model. The clearest comparison found so far:

- Base model on the 4-10s test subset: 13.01% WER, 4.61% CER.
- Phase 1 fine-tuned model on the same 500 samples: 31.62% WER, 12.70% CER.
- Punctuation-stripped WER narrows the gap but does not remove it: base around
  8.85%, phase 1 around 13.90%.

This suggests a real model-quality regression, with some extra WER inflation
from punctuation/case/text-format mismatch.

## What Looks Different From The Authors' Original Setup

The original repository setup and docs were aimed at French fine-tuning. The
French no-curriculum config uses:

- a local MLS French split
- `UsefulSensors/moonshine-tiny`
- 4-20s audio
- config-driven no-curriculum training
- schedule-free AdamW
- effective batch size 64
- evaluation and model selection from the same training script/config surface

The current English run differs in several important ways:

1. `configs/english_accent_local_no_curriculum.yaml` says
   `curriculum.enabled: false`, but `train.py` now always runs hardcoded
   phase-based curriculum training.
2. `train_job.sh` always passes `--phase` and appends `-phaseN`, so the active
   run is not the no-curriculum path described by the config name.
3. `train.py` ignores many config values, including `optim`,
   `lr_scheduler_type`, `training.learning_rate`, `training.max_steps`,
   configured batch sizes, audio duration limits, and generation penalties.
4. The current script bypasses `MoonshineDataLoader.prepare_dataset`, so
   `preprocessing.normalize_audio: true` is not applied during training.
   Inference/evaluation does normalize audio, creating a train/eval mismatch.
5. Phase training evaluates against the full eval set or the first N eval
   examples, not a duration-matched phase validation subset.
6. The trainer WER metric is raw/case/punctuation sensitive. The phase model
   often drops punctuation, so checkpoint selection can be distorted.
7. The English dataset is synthetic and highly repeated by transcript across
   voices. The train/test split is mostly transcript-disjoint, which is useful
   for generalization but different from the authors' MLS French assumptions.

## Investigation Plan

The goal is to separate three possible failure modes:

- checkpoint selection is choosing bad checkpoints
- training itself degrades the model quickly
- evaluation/text normalization is overstating the degradation

Steps:

1. Evaluate base, early checkpoints, later checkpoints, and final on the exact
   same duration-filtered subset.
2. Report both raw lowercased WER and punctuation-stripped WER for every model.
3. Compare phase 1 checkpoints such as `checkpoint-200`, `checkpoint-1000`,
   `checkpoint-4000`, and `final`.
4. Repeat on phase 2 after phase 2 completes or at available checkpoints.
5. Inspect the worst samples to determine whether errors are mostly punctuation,
   genuine word errors, truncations, deletions, or accent/domain substitutions.
6. Run a small "author-aligned" smoke train once fixes are ready:
   no curriculum, config-controlled optimizer, config-controlled duration
   filtering, training audio normalization, and duration-matched eval.
7. Only after the smoke run behaves sanely, scale to the full English dataset.

## Fix Plan

The fixes should prioritize restoring the original authors' training semantics
before adding new experiments.

1. Restore a real no-curriculum path in `train.py`.
   - If `curriculum.enabled: false`, train on the globally duration-filtered
     dataset instead of `PHASE_CONFIGS`.
   - Use `training.output_dir` directly, without forcing `-phaseN`.

2. Make `train.py` config-driven again.
   - Respect `training.optim`, `learning_rate`, `lr_scheduler_type`,
     `warmup_steps`, `max_steps`, batch sizes, eval/save/log steps, and
     generation settings.
   - Keep phase constants only for explicit curriculum mode.

3. Restore preprocessing parity.
   - Apply configured audio normalization during training preprocessing, or
     remove normalization from both training and inference if that is the chosen
     policy.
   - Include preprocessing settings in the prepared-dataset cache signature so
     stale caches cannot silently reuse differently processed features.

4. Fix evaluation/model selection.
   - For phase training, evaluate against a phase-matched validation subset.
   - For no-curriculum training, evaluate against the same global duration range
     used for training.
   - Consider a punctuation-stripped secondary metric for diagnostics while
     retaining raw WER if the deployment target expects punctuation.

5. Align the SLURM job with the selected mode.
   - For no-curriculum author-aligned runs, do not pass `--phase`.
   - Keep separate scripts or explicit environment flags for curriculum
     experiments.

6. Add a small regression/debug harness.
   - Compare base vs checkpoints/final on identical sample sets.
   - Persist summary CSV, sample CSV, and JSON outputs for every run.

## Applied Alignment Fixes

The training entrypoint has now been changed back to an author-aligned
no-curriculum default:

- `curriculum.enabled: false` runs one full training job instead of forcing
  `PHASE_CONFIGS`.
- No-curriculum output uses `training.output_dir` directly; `-phaseN` suffixes
  are only used when curriculum is explicitly enabled.
- Training/eval splits are both filtered to the same active duration range
  before preprocessing.
- The English no-curriculum config now uses the French setup's 4-20s duration
  range, effective batch size 64, schedule-free AdamW, constant-with-warmup,
  8000 max steps, and the original anti-repetition generation defaults.
- Training preprocessing now applies the configured RMS normalization, matching
  the inference/evaluation path.
- Prepared dataset cache signatures now include duration and audio
  normalization settings, so stale caches cannot silently reuse incompatible
  features.
- `train_job.sh` now submits the no-curriculum config by default and only
  resumes when `RESUME=1` is provided.

## New Debug Evaluation Tooling

Use `scripts/debug_checkpoint_eval.py` for the checkpoint investigation. It
compares the base model and selected checkpoints on the same duration-filtered
subset and reports raw WER plus punctuation-stripped WER.

Example local run:

```bash
python scripts/debug_checkpoint_eval.py \
  --config configs/english_accent_local_no_curriculum.yaml \
  --run-id 12216527 \
  --phase 1 \
  --checkpoints checkpoint-200 checkpoint-1000 checkpoint-4000 final \
  --max-samples 500
```

Example SLURM run:

```bash
sbatch debug_checkpoint_eval_job.sh
```

Useful overrides:

```bash
sbatch --export=ALL,RUN_ID=12216527,PHASE=1,MAX_SAMPLES=1000 debug_checkpoint_eval_job.sh
sbatch --export=ALL,CHECKPOINTS="checkpoint-200 checkpoint-1000 final" debug_checkpoint_eval_job.sh
```

Outputs are written under `results/` as:

- `debug_checkpoint_eval_summary_<timestamp>.csv`
- `debug_checkpoint_eval_samples_<timestamp>.csv`
- `debug_checkpoint_eval_suite_<timestamp>.json`
