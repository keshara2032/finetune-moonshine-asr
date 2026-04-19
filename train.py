"""
Main training script for Moonshine fine-tuning with curriculum learning.

Based on Moonshine paper (arXiv:2410.15608v2) recommendations:
- Training instances in [4, 30] seconds
- Bimodal distribution (naturally emerges)
- <0.5% of data should be <1s (avoids repetitions, >100% WER)

Examples:
    # Phase 1 with Common Voice French
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 1

    # Phase 2 (resume from Phase 1)
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 2 \\
                   --resume ./results-moonshine-fr-phase1

    # Phase 3 (resume from Phase 2)
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 3 \\
                   --resume ./results-moonshine-fr-phase2

    # Train without curriculum learning
    python train.py --config moonshine_ft/configs/common_voice_fr.yaml --no-curriculum
"""

import argparse
import yaml
import os
from pathlib import Path

import torch
import numpy as np
import evaluate
from transformers import (
    AutoProcessor,
    MoonshineForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from moonshine_ft.data_loader import MoonshineDataLoader
from moonshine_ft.curriculum import CurriculumScheduler
from moonshine_ft.storage import configure_storage, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune Moonshine ASR with Curriculum Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 1
  python train.py --config moonshine_ft/configs/common_voice_fr.yaml --phase 2 --resume ./results-moonshine-fr-phase1
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Curriculum learning phase (1-3)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Checkpoint directory to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning (train on full data)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override max training steps'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push model to HuggingFace Hub after training'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: use only 100 samples for quick validation'
    )

    return parser.parse_args()


# ============================================
# Data Collator for Moonshine
# ============================================
class DataCollatorMoonshineSeq2SeqWithPadding:
    """
    Data collator for Moonshine that handles:
    - Padding audio inputs
    - Padding labels with -100 (ignored in loss)
    - Creating decoder_input_ids (shifted right with BOS token)

    Based on original implementation but optimized for readability.
    """

    def __init__(self, processor, decoder_start_token_id, pad_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # Extract input values and labels
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # Pad audio inputs
        batch = self.processor.feature_extractor.pad(
            input_values,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Pad labels using PyTorch's efficient pad_sequence
        label_tensors = [torch.tensor(labels, dtype=torch.long) for labels in label_features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100  # Ignored in loss calculation
        )

        # Create decoder_input_ids: [BOS, t1, t2, ..., tN]
        # Labels are: [t1, t2, ..., tN, EOS]
        # So decoder_input_ids = [BOS] + labels[:-1]
        decoder_input_tensors = [
            torch.tensor(
                [self.decoder_start_token_id] + labels[:-1],
                dtype=torch.long
            )
            for labels in label_features
        ]

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels_padded

        return batch


# ============================================
# Custom Trainer for Moonshine
# ============================================
class MoonshineSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom trainer that properly handles Moonshine's generate() method.

    Moonshine uses variable-length sequences, so we need to:
    1. Calculate max_new_tokens based on audio duration
    2. Use phase-specific generation parameters
    """

    def __init__(self, *args, generation_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config or {}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for Moonshine."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Calculate generation parameters based on audio length
        # Paper: Training instances in [4, 30] seconds
        if 'input_values' in inputs:
            audio_length = inputs['input_values'].shape[-1]
            audio_duration = audio_length / 16000  # 16kHz sampling rate
            # Roughly 6 tokens per second (conservative estimate)
            max_new_tokens = max(5, min(int(audio_duration * 6), 50))
        else:
            max_new_tokens = 50

        labels = inputs.get("labels", None)

        # If only computing loss, don't generate
        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, None, None)

        # Generate predictions
        with torch.no_grad():
            # Calculate loss first (if labels available)
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                loss = None

            # Generate transcriptions with phase-specific parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                **self.generation_config
            }

            # Moonshine generate expects input_values (not input_features)
            generated_tokens = model.generate(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask", None),
                **generation_kwargs
            )

        if labels is not None:
            labels = labels.detach()

        return (loss, generated_tokens, labels)


def get_duration_column(dataset):
    """Return the available duration column name for a dataset split."""
    for column_name in ("duration", "audio_duration"):
        if column_name in dataset.column_names:
            return column_name
    return None


def summarize_duration_profile(dataset):
    """Compute duration statistics used to size batches sanely."""
    duration_column = get_duration_column(dataset)
    if duration_column is None:
        return None

    durations = np.asarray(dataset[duration_column], dtype=np.float32)
    if durations.size == 0:
        return None

    return {
        "count": int(durations.size),
        "min": float(np.min(durations)),
        "mean": float(np.mean(durations)),
        "median": float(np.median(durations)),
        "p90": float(np.percentile(durations, 90)),
        "p95": float(np.percentile(durations, 95)),
        "max": float(np.max(durations)),
        "under_30": int(np.sum(durations <= 30.0)),
        "under_60": int(np.sum(durations <= 60.0)),
        "under_90": int(np.sum(durations <= 90.0)),
    }


def tune_training_config_for_duration(train_config, duration_profile):
    """
    Downshift batch settings automatically when the dataset contains long-form audio.

    Moonshine's processor keeps raw-waveform length in `input_values`, so extremely long
    clips combined with large per-device batches make the first training step look hung.
    """
    if duration_profile is None:
        return dict(train_config), []

    adjusted = dict(train_config)
    notes = []

    long_audio_detected = (
        duration_profile["p90"] > 45.0 or
        duration_profile["mean"] > 30.0
    )

    if not long_audio_detected:
        return adjusted, notes

    target_audio_seconds = float(
        adjusted.get("max_audio_seconds_per_device_batch", 180.0)
    )
    safe_train_batch = max(1, int(target_audio_seconds // max(duration_profile["p90"], 1.0)))
    safe_eval_batch = max(1, int(target_audio_seconds // max(duration_profile["p95"], 1.0)))

    if adjusted["per_device_train_batch_size"] > safe_train_batch:
        notes.append(
            "Long-form audio detected; reducing per-device train batch size "
            f"from {adjusted['per_device_train_batch_size']} to {safe_train_batch}."
        )
        adjusted["per_device_train_batch_size"] = safe_train_batch

    if adjusted["per_device_eval_batch_size"] > safe_eval_batch:
        notes.append(
            "Long-form audio detected; reducing per-device eval batch size "
            f"from {adjusted['per_device_eval_batch_size']} to {safe_eval_batch}."
        )
        adjusted["per_device_eval_batch_size"] = safe_eval_batch

    max_long_audio_grad_accum = int(
        adjusted.get("max_long_audio_gradient_accumulation_steps", 4)
    )
    if adjusted["gradient_accumulation_steps"] > max_long_audio_grad_accum:
        notes.append(
            "Long-form audio detected; reducing gradient accumulation steps "
            f"from {adjusted['gradient_accumulation_steps']} to {max_long_audio_grad_accum} "
            "so the first optimizer step does not take forever."
        )
        adjusted["gradient_accumulation_steps"] = max_long_audio_grad_accum

    if adjusted.get("logging_steps", 1) > 1:
        notes.append(
            f"Long-form audio detected; reducing logging_steps from {adjusted['logging_steps']} to 1."
        )
        adjusted["logging_steps"] = 1

    adjusted["logging_first_step"] = True

    if duration_profile["under_30"] < 100:
        notes.append(
            "This dataset is almost entirely longer than Moonshine's usual 4-30s regime. "
            "Training can still run with conservative batches, but segmenting the audio would be better."
        )

    return adjusted, notes


def main():
    args = parse_args()

    # ============================================
    # Load Configuration
    # ============================================
    print("\n" + "="*80)
    print("MOONSHINE FINE-TUNING WITH CURRICULUM LEARNING")
    print("="*80)
    print(f"Based on Moonshine paper (arXiv:2410.15608v2)")
    print(f"  - Training instances: [4, 30] seconds")
    print(f"  - Bimodal distribution (Phase 3)")
    print(f"  - <0.5% data <1s (avoids repetitions)")
    print("="*80)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration loaded from: {args.config}")
    train_config = dict(config['training'])

    storage = configure_storage(config)
    storage_root = Path(storage["storage_root"]) if storage["storage_root"] else None

    if storage_root is not None:
        print(f"\nStorage root: {storage['storage_root']}")
        print(f"  Model cache:   {storage['model_cache_dir']}")
        print(f"  Dataset cache: {storage['dataset_cache_dir']}")
        print(f"  Metrics cache: {storage['evaluate_cache_dir']}")
        print(f"  Logging dir:   {storage['logging_dir']}")

        if args.output_dir:
            args.output_dir = resolve_path(args.output_dir, storage_root)

        if args.resume:
            args.resume = resolve_path(args.resume, storage_root)

    # Override config with command-line arguments
    if args.no_curriculum:
        config['curriculum']['enabled'] = False
        print("[WARNING] Curriculum learning DISABLED")

    # ============================================
    # Setup Curriculum Learning
    # ============================================
    curriculum = CurriculumScheduler.from_config(config)

    # Print curriculum summary
    curriculum.print_summary()

    if config['curriculum']['enabled']:
        phase_idx = args.phase - 1  # Convert to 0-indexed
        phase = curriculum.get_phase(phase_idx)
        print(f"\n{'='*80}")
        print(f"PHASE {args.phase}: {phase.name}")
        print(f"{'='*80}")
        print(f"Description: {phase.description}")
        print(f"Duration range: [{phase.min_duration}s, {phase.max_duration}s]")
        print(f"Max words: {phase.max_words or 'unlimited'}")
        print(f"Max steps: {phase.max_steps:,}")
        print(f"Learning rate: {phase.learning_rate}")
        print(f"Target WER: <{phase.target_wer}%")
    else:
        phase = curriculum.get_phase(0)  # Use first phase config
        print(f"\nCurriculum learning DISABLED - training on full dataset")
        print(f"Duration range: [{phase.min_duration}s, {phase.max_duration}s]")

    # ============================================
    # Model Configuration
    # ============================================
    model_name = config['model']['name']
    model_path = model_name if args.resume is None else args.resume

    if args.resume:
        print(f"\n[OK] Resuming from checkpoint: {args.resume}")
    else:
        print(f"\n[OK] Starting from pre-trained model: {model_name}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif config['curriculum']['enabled']:
        output_dir = f"{config['training']['output_dir']}-phase{args.phase}"
    else:
        output_dir = config['training']['output_dir']

    print(f"[OK] Output directory: {output_dir}")

    # ============================================
    # Load Processor
    # ============================================
    print("\n" + "="*60)
    print("LOADING PROCESSOR")
    print("="*60)
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=config['model'].get('cache_dir')
    )

    # Register special tokens
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '</s>',
    }
    processor.tokenizer.add_special_tokens(special_tokens)

    print(f"[OK] Tokenizer loaded")
    print(f"  BOS token ID: {processor.tokenizer.bos_token_id}")
    print(f"  EOS token ID: {processor.tokenizer.eos_token_id}")
    print(f"  PAD token ID: {processor.tokenizer.pad_token_id}")

    # ============================================
    # Load Dataset
    # ============================================
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    data_loader = MoonshineDataLoader.from_config(config)
    dataset_dict = data_loader.load_dataset()

    # Test mode: use only 500 samples (enough to get samples in various duration ranges)
    if args.test_mode:
        print("\n[WARNING] TEST MODE: Using only 500 samples per split")
        dataset_dict['train'] = dataset_dict['train'].select(range(min(500, len(dataset_dict['train']))))
        dataset_dict['test'] = dataset_dict['test'].select(range(min(500, len(dataset_dict['test']))))

    # Filter by global duration constraints (remove very short/long)
    print(f"\n{'='*60}")
    print("APPLYING GLOBAL DURATION FILTERS")
    print(f"{'='*60}")

    min_dur = config['audio'].get('min_duration', 1.0)
    max_dur = config['audio'].get('max_duration', 30.0)

    print(f"Global constraints: [{min_dur}s, {max_dur}s]")

    dataset_dict['train'] = data_loader.filter_by_duration(
        dataset_dict['train'],
        max_duration=max_dur,
        min_duration=min_dur
    )
    dataset_dict['test'] = data_loader.filter_by_duration(
        dataset_dict['test'],
        max_duration=max_dur,
        min_duration=min_dur
    )

    # Apply curriculum filtering if enabled
    if config['curriculum']['enabled']:
        print(f"\n{'='*60}")
        print(f"APPLYING CURRICULUM PHASE {args.phase} FILTER")
        print(f"{'='*60}")

        # Detect which duration column exists (different datasets use different names)
        duration_col = 'duration' if 'duration' in dataset_dict['train'].column_names else 'audio_duration'

        dataset_dict['train'] = curriculum.filter_dataset(
            dataset_dict['train'],
            phase,
            duration_column=duration_col,
            text_column='sentence'
        )
        dataset_dict['test'] = curriculum.filter_dataset(
            dataset_dict['test'],
            phase,
            duration_column=duration_col,
            text_column='sentence'
        )

    duration_profile = summarize_duration_profile(dataset_dict['train'])
    if duration_profile is not None:
        print(f"\n{'='*60}")
        print("TRAINING DURATION PROFILE")
        print(f"{'='*60}")
        print(
            "Train durations: "
            f"min={duration_profile['min']:.1f}s, "
            f"mean={duration_profile['mean']:.1f}s, "
            f"median={duration_profile['median']:.1f}s, "
            f"p90={duration_profile['p90']:.1f}s, "
            f"p95={duration_profile['p95']:.1f}s, "
            f"max={duration_profile['max']:.1f}s"
        )
        print(
            "Clips within common Moonshine ranges: "
            f"<=30s: {duration_profile['under_30']:,}, "
            f"<=60s: {duration_profile['under_60']:,}, "
            f"<=90s: {duration_profile['under_90']:,}"
        )

    train_config, duration_tuning_notes = tune_training_config_for_duration(
        train_config,
        duration_profile
    )
    if duration_tuning_notes:
        print(f"\n{'='*60}")
        print("LONG-AUDIO ADJUSTMENTS")
        print(f"{'='*60}")
        for note in duration_tuning_notes:
            print(f"[WARNING] {note}")

    eval_max_samples = train_config.get('eval_max_samples')
    if eval_max_samples:
        original_eval_count = len(dataset_dict['test'])
        capped_eval_count = min(int(eval_max_samples), original_eval_count)
        dataset_dict['test'] = dataset_dict['test'].shuffle(seed=42).select(range(capped_eval_count))
        print(
            f"\n[INFO] Limiting training-time evaluation to "
            f"{capped_eval_count:,}/{original_eval_count:,} samples"
        )

    # Check if we have enough data
    if len(dataset_dict['train']) < 10:
        print(f"\n[WARNING] WARNING: Only {len(dataset_dict['train'])} training samples!")
        print("This is too few for meaningful training. Check your duration filters.")
        if not args.test_mode:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            print("[WARNING] Test mode: Skipping confirmation prompt. Using larger sample in test mode.")
            # In test mode with insufficient data, use more samples
            if len(dataset_dict['train']) == 0:
                print("[WARNING] No training samples found. Exiting.")
                return

    # Prepare datasets (feature extraction + tokenization)
    print(f"\n{'='*60}")
    print("PREPROCESSING DATASETS")
    print(f"{'='*60}")

    dataset_dict['train'] = data_loader.prepare_dataset(
        dataset_dict['train'],
        processor
    )
    dataset_dict['test'] = data_loader.prepare_dataset(
        dataset_dict['test'],
        processor
    )

    # Save encoded datasets
    encoded_path = f'{output_dir}_encoded'
    os.makedirs(encoded_path, exist_ok=True)
    dataset_dict.save_to_disk(encoded_path)
    print(f"\n[OK] Saved encoded datasets to: {encoded_path}")

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(dataset_dict['train']):,} samples")
    print(f"  Test:  {len(dataset_dict['test']):,} samples")

    # ============================================
    # Load Model
    # ============================================
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"Loading from: {model_path}")

    model = MoonshineForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=config['model'].get('cache_dir')
    )

    # Configure model tokens
    model.config.pad_token_id = 2
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.decoder_start_token_id = 1
    model.config.use_cache = False  # Required for gradient checkpointing

    # Configure generation (phase-specific)
    if hasattr(model, 'generation_config'):
        gen_config = curriculum.get_generation_config(phase)
        for key, value in gen_config.items():
            setattr(model.generation_config, key, value)

        print(f"\n[OK] Generation config:")
        print(f"  Repetition penalty: {model.generation_config.repetition_penalty}")
        print(f"  Num beams: {model.generation_config.num_beams}")
        print(f"  No repeat ngram size: {model.generation_config.no_repeat_ngram_size}")

    # Freeze encoder if specified
    if config['model'].get('freeze_encoder', False):
        print("\n[WARNING] Freezing encoder weights (decoder-only training)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = model.num_parameters()
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    else:
        print(f"\n[OK] Model loaded: {model.num_parameters():,} parameters")

    # ============================================
    # Data Collator
    # ============================================
    data_collator = DataCollatorMoonshineSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    # ============================================
    # Evaluation Metrics
    # ============================================
    wer_metric = evaluate.load(
        'wer',
        cache_dir=storage.get('evaluate_cache_dir')
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id for decoding
        label_ids = np.where(label_ids == -100, model.config.pad_token_id, label_ids)
        pred_ids = np.where(pred_ids < 0, model.config.pad_token_id, pred_ids)

        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER (handle empty strings)
        pred_empty = np.array([p.strip() == "" for p in pred_str])
        label_empty = np.array([l.strip() == "" for l in label_str])

        wer_scores = np.ones(len(pred_str))
        wer_scores[pred_empty & label_empty] = 0  # Both empty = perfect

        non_silence = ~label_empty
        if np.any(non_silence):
            non_silence_wer = wer_metric.compute(
                predictions=np.array(pred_str)[non_silence].tolist(),
                references=np.array(label_str)[non_silence].tolist()
            )
            wer_scores[non_silence] = non_silence_wer

        avg_wer = np.mean(wer_scores)

        # Log examples
        print("\n" + "="*80)
        print(f"EVALUATION EXAMPLES (Phase {args.phase if config['curriculum']['enabled'] else 'Full'}):")
        print("="*80)
        for i in range(min(5, len(pred_str))):
            print(f"\nExample {i+1}:")
            print(f"  Prediction: '{pred_str[i]}'")
            print(f"  Reference:  '{label_str[i]}'")
        print("="*80)

        return {"wer": 100 * avg_wer}

    # ============================================
    # Training Arguments
    # ============================================
    # Override with phase-specific and CLI args
    max_steps = args.max_steps or train_config.get('max_steps', phase.max_steps)
    learning_rate = (
        phase.learning_rate
        if config['curriculum']['enabled']
        else train_config.get('learning_rate', phase.learning_rate)
    )

    training_args_kwargs = dict(
        output_dir=output_dir,

        # Batch sizes
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],

        # Optimization
        optim=train_config.get('optim', 'adamw_torch'),  # Default to standard AdamW if not specified
        learning_rate=learning_rate,
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),  # Default to linear
        warmup_steps=train_config.get('warmup_steps', phase.warmup_steps),  # Allow override from config
        max_grad_norm=train_config['max_grad_norm'],
        max_steps=max_steps,
        label_smoothing_factor=phase.label_smoothing,

        # Memory optimization
        gradient_checkpointing=train_config['gradient_checkpointing'],
        fp16=train_config['fp16'],
        fp16_full_eval=train_config.get('fp16_full_eval', True),

        # Evaluation
        eval_strategy=train_config['eval_strategy'],
        eval_steps=train_config['eval_steps'],
        save_steps=train_config['save_steps'],
        logging_steps=train_config['logging_steps'],
        logging_first_step=train_config.get('logging_first_step', True),
        predict_with_generate=train_config['predict_with_generate'],

        # Model selection
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],
        greater_is_better=train_config['greater_is_better'],

        # Logging
        report_to=train_config['report_to'],
        logging_dir=train_config['logging_dir'],

        # Hub
        push_to_hub=args.push_to_hub or train_config.get('push_to_hub', False),
        hub_model_id=train_config.get('hub_model_id'),

        # Run name
        run_name=f"moonshine_phase{args.phase}" if config['curriculum']['enabled'] else "moonshine_full",
    )

    # Transformers renamed this API from `group_by_length` to
    # `train_sampling_strategy="group_by_length"` in newer versions.
    if "group_by_length" in Seq2SeqTrainingArguments.__dataclass_fields__:
        training_args_kwargs["group_by_length"] = train_config['group_by_length']
    else:
        training_args_kwargs["train_sampling_strategy"] = (
            "group_by_length" if train_config['group_by_length'] else "random"
        )

    training_args_kwargs["length_column_name"] = train_config['length_column_name']

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # ============================================
    # Initialize Trainer
    # ============================================
    generation_config = curriculum.get_generation_config(phase)

    trainer = MoonshineSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        generation_config=generation_config,
    )

    # Save processor
    processor.save_pretrained(output_dir)

    # ============================================
    # Start Training
    # ============================================
    print("\n" + "="*80)
    print(f"STARTING TRAINING")
    print("="*80)
    print(f"Training samples: {len(dataset_dict['train']):,}")
    print(f"Evaluation samples: {len(dataset_dict['test']):,}")
    print(f"Max steps: {max_steps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    if config['curriculum']['enabled']:
        print(f"Target WER: <{phase.target_wer}%")
    print("="*80 + "\n")

    trainer.train()

    # ============================================
    # Save Final Model
    # ============================================
    final_path = os.path.join(output_dir, 'final')
    trainer.save_model(final_path)
    print(f"\n[OK] Saved final model to: {final_path}")

    # ============================================
    # Final Evaluation
    # ============================================
    print("\nRunning final evaluation...")

    # Convert model to FP32 before final evaluation to avoid dtype mismatch
    if training_args.fp16:
        print("Converting model to FP32 for final evaluation...")
        model = model.float()
        trainer.model = model

    try:
        results = trainer.evaluate()
    except RuntimeError as e:
        if "should be the same" in str(e):
            print(f"\n⚠️  Final evaluation skipped due to dtype mismatch (this is a known issue with FP16 training)")
            print(f"Your model was saved successfully to: {training_args.output_dir}/final")
            print(f"\nYou can evaluate it separately with:")
            print(f"  python scripts/evaluate.py --model {training_args.output_dir}/final --dataset {config['dataset']['name']} --split test")
            results = None
        else:
            raise

    print("\n" + "="*60)
    if config['curriculum']['enabled']:
        print(f"PHASE {args.phase} TRAINING COMPLETE")
    else:
        print("TRAINING COMPLETE")
    print("="*60)

    if results is not None:
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        if config['curriculum']['enabled']:
            actual_wer = results.get('eval_wer', 100)
            if actual_wer < phase.target_wer:
                print(f"\n[OK] SUCCESS! WER ({actual_wer:.1f}%) < Target ({phase.target_wer}%)")
                if args.phase < len(curriculum.phases):
                    print(f"\n➜ Ready for Phase {args.phase + 1}!")
                    print(f"   Run: python train.py --config {args.config} "
                          f"--phase {args.phase + 1} --resume {output_dir}")
            else:
                print(f"\n[WARNING] WER ({actual_wer:.1f}%) > Target ({phase.target_wer}%)")
                print("   Consider training longer or adjusting parameters.")
    else:
        print("\nNote: Final evaluation was skipped. Run evaluate.py separately for metrics.")

    print("="*60)


if __name__ == '__main__':
    main()
