"""
Config-driven Moonshine fine-tuning for local Hugging Face datasets.

The default path follows the repository author's no-curriculum French setup:
load a local DatasetDict, filter by the global audio duration range in the
config, apply the configured preprocessing, then train one model in
training.output_dir. The notebook-style curriculum phases are still available
when curriculum.enabled is true, but they no longer override a no-curriculum
config by accident.
"""

import argparse
import hashlib
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
import yaml
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoProcessor,
    MoonshineForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from moonshine_ft.storage import configure_storage, resolve_path
from moonshine_ft.utils.preprocessing import normalize_audio


PHASE_CONFIGS = {
    1: {
        "name": "Short Utterances",
        "duration_range": (4.0, 10.0),
        "max_steps": 4000,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 1,
        "generation_max_length": 50,
        "generation_num_beams": 3,
        "expected_duration": "~3-4 hours",
    },
    2: {
        "name": "Medium Utterances",
        "duration_range": (10.0, 20.0),
        "max_steps": 6000,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "warmup_steps": 800,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 25,
        "generation_max_length": 50,
        "generation_num_beams": 5,
        "expected_duration": "~5-6 hours",
    },
    3: {
        "name": "Full Range",
        "duration_range": (4.0, 30.0),
        "max_steps": 5000,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 32,
        "learning_rate": 5e-6,
        "warmup_steps": 400,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 25,
        "generation_max_length": 50,
        "generation_num_beams": 5,
        "expected_duration": "~4-5 hours",
    },
}

TEXT_COLUMN_CANDIDATES = ("text", "transcript", "transcription", "sentence")
PREPARED_DATASET_CACHE_VERSION = 2


def restore_missing_special_tokens(processor, *, model_name: str, cache_dir: Optional[str]) -> None:
    tokenizer = processor.tokenizer
    missing_tokens = {
        "bos_token": tokenizer.bos_token_id is None,
        "eos_token": tokenizer.eos_token_id is None,
        "pad_token": tokenizer.pad_token_id is None,
    }
    if not any(missing_tokens.values()):
        return

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    token_id_fields = {
        "bos_token": "bos_token_id",
        "eos_token": "eos_token_id",
        "pad_token": "pad_token_id",
    }
    restored = {}

    for token_field, token_id_field in token_id_fields.items():
        if not missing_tokens[token_field]:
            continue

        token_id = getattr(config, token_id_field, None)
        if token_id is None:
            continue

        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            raise ValueError(
                f"Could not restore tokenizer {token_field} from model config id {token_id}."
            )

        setattr(tokenizer, token_field, token)
        restored[token_id_field] = getattr(tokenizer, token_id_field)

    if restored:
        print(f"Restored tokenizer special tokens from model config: {restored}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven Moonshine fine-tuning for local datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help=(
            "Curriculum phase to train when curriculum.enabled=true. "
            "Ignored for no-curriculum configs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override training.output_dir. A -phaseN suffix is added only when "
            "curriculum.enabled=true."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest complete checkpoint in the active output directory.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume from a specific Trainer checkpoint directory. "
            "Overrides --resume auto-detection."
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick smoke run with 50 steps and small filtered train/eval subsets.",
    )
    parser.add_argument(
        "--rebuild-dataset-cache",
        action="store_true",
        help="Rebuild the prepared train/eval dataset cache instead of reusing it.",
    )
    return parser.parse_args()


def resolve_text_column(column_names: List[str], preferred: Optional[str]) -> str:
    if preferred and preferred in column_names:
        return preferred

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in column_names:
            return candidate

    raise ValueError(
        "Could not find a transcript column. "
        f"Tried preferred={preferred!r} and fallbacks={TEXT_COLUMN_CANDIDATES}. "
        f"Available columns: {column_names}"
    )


def resolve_eval_split(dataset_dict: DatasetDict, preferred: Optional[str]) -> str:
    if preferred and preferred in dataset_dict:
        return preferred

    for candidate in ("test", "validation", "eval"):
        if candidate in dataset_dict:
            return candidate

    raise ValueError(
        "Could not find an evaluation split. "
        f"Available splits: {list(dataset_dict.keys())}"
    )


def curriculum_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("curriculum", {}).get("enabled", False))


def global_duration_range(config: Dict[str, Any]) -> tuple[float, float]:
    audio_config = config.get("audio", {})
    return (
        float(audio_config.get("min_duration", 4.0)),
        float(audio_config.get("max_duration", 30.0)),
    )


def duration_range_for_run(
    config: Dict[str, Any],
    *,
    phase_number: Optional[int],
) -> tuple[float, float]:
    if curriculum_enabled(config):
        if phase_number is None:
            raise ValueError("A curriculum phase is required when curriculum.enabled=true.")
        return PHASE_CONFIGS[phase_number]["duration_range"]

    return global_duration_range(config)


def run_cache_name(config: Dict[str, Any], *, phase_number: Optional[int]) -> str:
    return f"phase{phase_number}" if curriculum_enabled(config) else "full"


def load_local_dataset(config: Dict[str, Any]) -> DatasetDict:
    dataset_config = config.get("dataset", {})
    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("Config must define dataset.path for local training.")

    print(f"\nLoading local dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict saved with save_to_disk(), got {type(dataset)}"
        )

    if "train" not in dataset:
        raise ValueError(
            f"Dataset must contain a 'train' split. Found: {list(dataset.keys())}"
        )
    if "audio" not in dataset["train"].column_names:
        raise ValueError("Dataset must contain an 'audio' column in the train split.")

    eval_split = resolve_eval_split(dataset, dataset_config.get("eval_split"))
    text_column = resolve_text_column(
        dataset["train"].column_names,
        dataset_config.get("text_column"),
    )

    keep_columns = ["audio", text_column]
    for duration_column in ("duration", "audio_duration"):
        if duration_column in dataset["train"].column_names:
            keep_columns.append(duration_column)
            break

    train_dataset = dataset["train"].select_columns(keep_columns)
    eval_dataset = dataset[eval_split].select_columns(
        [column for column in keep_columns if column in dataset[eval_split].column_names]
    )

    if text_column != "transcription":
        train_dataset = train_dataset.rename_column(text_column, "transcription")
        eval_dataset = eval_dataset.rename_column(text_column, "transcription")

    sampling_rate = config.get("audio", {}).get("sampling_rate", 16000)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    prepared = DatasetDict({"train": train_dataset, "test": eval_dataset})

    print(f"Train samples: {len(prepared['train']):,}")
    print(f"Eval samples:  {len(prepared['test']):,}")
    print(f"Transcript column: transcription")

    return prepared


def prepare_dataset_dict(
    dataset_dict: DatasetDict,
    processor,
    *,
    num_proc: int,
    preprocessing_config: Dict[str, Any],
) -> DatasetDict:
    should_normalize = bool(preprocessing_config.get("normalize_audio", False))
    target_rms = float(preprocessing_config.get("target_rms", 0.075))

    def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        if should_normalize:
            audio_array = normalize_audio(audio_array, target_rms=target_rms)

        inputs = processor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )

        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])

        labels = processor.tokenizer(
            batch["transcription"],
            add_special_tokens=False,
        ).input_ids
        batch["labels"] = labels + [processor.tokenizer.eos_token_id]
        batch["duration"] = len(audio_array) / audio["sampling_rate"]

        return batch

    print("\nPreprocessing dataset...")
    print(f"Audio normalization: {'enabled' if should_normalize else 'disabled'}")
    map_num_proc = num_proc if num_proc and num_proc > 1 else None
    remove_columns = [
        column for column in dataset_dict["train"].column_names
        if column != "duration"
    ]
    return dataset_dict.map(
        prepare_dataset,
        remove_columns=remove_columns,
        num_proc=map_num_proc,
        desc="Preprocessing",
    )


def sanitize_cache_name(value: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in value
    ).strip(".-_")
    return sanitized or "dataset"


def prepared_dataset_cache_root(
    config: Dict[str, Any],
    storage_root: Optional[Path],
) -> Path:
    preprocessing_config = config.get("preprocessing", {})
    configured_cache_dir = preprocessing_config.get("prepared_cache_dir")
    if configured_cache_dir:
        return Path(resolve_path(configured_cache_dir, storage_root))

    if storage_root is not None:
        return storage_root / "cache" / "prepared_datasets"

    return Path(".cache/prepared_datasets").resolve(strict=False)


def prepared_dataset_cache_signature(
    config: Dict[str, Any],
    dataset_dict: DatasetDict,
    *,
    base_model_name: str,
) -> Dict[str, Any]:
    dataset_config = config.get("dataset", {})
    audio_config = config.get("audio", {})
    preprocessing_config = config.get("preprocessing", {})
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    return {
        "cache_version": PREPARED_DATASET_CACHE_VERSION,
        "preprocess_code": "moonshine-input-values-labels-duration-v2",
        "dataset_path": dataset_config.get("path"),
        "train_fingerprint": getattr(train_dataset, "_fingerprint", None),
        "eval_fingerprint": getattr(eval_dataset, "_fingerprint", None),
        "train_columns": list(train_dataset.column_names),
        "eval_columns": list(eval_dataset.column_names),
        "model_name": base_model_name,
        "sampling_rate": audio_config.get("sampling_rate", 16000),
        "global_duration_range": list(global_duration_range(config)),
        "normalize_audio": bool(preprocessing_config.get("normalize_audio", False)),
        "target_rms": float(preprocessing_config.get("target_rms", 0.075)),
        "phase_duration_ranges": {
            str(phase_number): list(phase_config["duration_range"])
            for phase_number, phase_config in sorted(PHASE_CONFIGS.items())
        },
    }


def prepared_dataset_cache_dir(
    config: Dict[str, Any],
    dataset_dict: DatasetDict,
    *,
    base_model_name: str,
    storage_root: Optional[Path],
) -> tuple[Path, Dict[str, Any]]:
    signature = prepared_dataset_cache_signature(
        config,
        dataset_dict,
        base_model_name=base_model_name,
    )
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    dataset_name = sanitize_cache_name(Path(signature.get("dataset_path") or "dataset").name)
    model_name = sanitize_cache_name(base_model_name.split("/")[-1])
    cache_dir = prepared_dataset_cache_root(config, storage_root) / (
        f"{dataset_name}-{model_name}-{digest}"
    )
    return cache_dir, signature


def dataset_cache_ready(dataset_dir: Path) -> bool:
    return (
        dataset_dir.exists()
        and (dataset_dir / "dataset_info.json").exists()
        and (dataset_dir / "state.json").exists()
    )


def read_cache_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def cache_manifest_matches(cache_dir: Path, signature: Dict[str, Any]) -> bool:
    manifest = read_cache_manifest(cache_dir)
    return bool(manifest and manifest.get("signature") == signature)


def write_cache_manifest(cache_dir: Path, signature: Dict[str, Any]) -> None:
    manifest_path = cache_dir / "manifest.json"
    temp_path = cache_dir / f".manifest.tmp-{os.getpid()}"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump({"signature": signature}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, manifest_path)


def save_dataset_to_cache(dataset: Dataset, target_dir: Path, label: str) -> None:
    if dataset_cache_ready(target_dir):
        print(f"Cached {label} already exists: {target_dir}")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir.with_name(f".{target_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"Saving cached {label} dataset to: {target_dir}")
    dataset.save_to_disk(str(temp_dir))

    if target_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Cached {label} appeared while saving; keeping existing copy.")
        return

    temp_dir.rename(target_dir)


def load_prepared_dataset_cache(
    cache_dir: Path,
    signature: Dict[str, Any],
    *,
    train_cache_name: str,
    rebuild: bool,
) -> Optional[tuple[Dataset, Dataset]]:
    train_dir = cache_dir / train_cache_name
    eval_dir = cache_dir / f"{train_cache_name}-eval"

    if rebuild:
        print(f"\nRebuilding prepared dataset cache: {cache_dir}")
        return None

    if not cache_manifest_matches(cache_dir, signature):
        print(f"\nPrepared dataset cache miss: {cache_dir}")
        return None

    if not dataset_cache_ready(train_dir) or not dataset_cache_ready(eval_dir):
        print(f"\nPrepared dataset cache is incomplete: {cache_dir}")
        return None

    print(f"\nLoading prepared datasets from cache: {cache_dir}")
    train_dataset = load_from_disk(str(train_dir))
    eval_dataset = load_from_disk(str(eval_dir))
    summarize_duration_dataset(train_dataset, f"{train_cache_name} training", "duration")
    summarize_duration_dataset(eval_dataset, f"{train_cache_name} evaluation", "duration")
    return train_dataset, eval_dataset


def save_prepared_dataset_cache(
    cache_dir: Path,
    signature: Dict[str, Any],
    *,
    train_cache_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_to_cache(train_dataset, cache_dir / train_cache_name, train_cache_name)
    save_dataset_to_cache(eval_dataset, cache_dir / f"{train_cache_name}-eval", f"{train_cache_name}-eval")
    write_cache_manifest(cache_dir, signature)
    print(f"Prepared dataset cache ready: {cache_dir}")


def duration_column_name(dataset: Dataset) -> Optional[str]:
    for column_name in ("duration", "audio_duration"):
        if column_name in dataset.column_names:
            return column_name
    return None


def audio_duration_seconds(audio: Any, fallback_sampling_rate: int) -> float:
    if isinstance(audio, dict):
        if "array" in audio and "sampling_rate" in audio:
            return float(len(audio["array"])) / float(audio["sampling_rate"])
        if "bytes" in audio and "sampling_rate" in audio and "array" in audio:
            return float(len(audio["array"])) / float(audio["sampling_rate"])

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        return float(samples.data.shape[-1]) / float(samples.sample_rate)

    if hasattr(audio, "__len__"):
        return float(len(audio)) / float(fallback_sampling_rate)

    raise TypeError(f"Unsupported audio value for duration: {type(audio)}")


def ensure_duration_column(
    dataset: Dataset,
    *,
    sampling_rate: int,
    num_proc: int,
    label: str,
) -> tuple[Dataset, str]:
    existing = duration_column_name(dataset)
    if existing is not None:
        if existing != "duration":
            dataset = dataset.rename_column(existing, "duration")
            return dataset, "duration"
        return dataset, existing

    print(f"\nNo duration column found for {label}. Computing durations from audio...")

    def add_duration(example: Dict[str, Any]) -> Dict[str, Any]:
        example["duration"] = audio_duration_seconds(example["audio"], sampling_rate)
        return example

    map_num_proc = num_proc if num_proc and num_proc > 1 else None
    dataset = dataset.map(
        add_duration,
        num_proc=map_num_proc,
        desc=f"Computing {label} durations",
    )
    return dataset, "duration"


def summarize_duration_dataset(dataset: Dataset, label: str, duration_column: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{label} dataset is empty after duration filtering.")

    durations = np.asarray(dataset[duration_column], dtype=np.float32)
    print(f"{label} samples: {len(dataset):,}")
    print(
        f"{label} duration: min={float(np.min(durations)):.1f}s, "
        f"mean={float(np.mean(durations)):.1f}s, "
        f"median={float(np.median(durations)):.1f}s, "
        f"max={float(np.max(durations)):.1f}s"
    )


def filter_duration_dataset(
    dataset: Dataset,
    *,
    duration_column: str,
    min_duration: float,
    max_duration: float,
    label: str,
) -> Dataset:
    print(f"\nFiltering {label} to {min_duration:.1f}-{max_duration:.1f}s...")

    filtered = dataset.filter(
        lambda duration: min_duration <= duration <= max_duration,
        input_columns=[duration_column],
        desc=f"{label} duration filtering",
    )

    print(f"{label} original samples: {len(dataset):,}")
    print(f"{label} filtered samples: {len(filtered):,}")
    print(f"{label} removed samples: {len(dataset) - len(filtered):,}")
    summarize_duration_dataset(filtered, label, duration_column)
    return filtered


def resolve_eval_max_samples(training_config: Dict[str, Any]) -> Optional[int]:
    raw_value = training_config.get(
        "eval_max_samples",
        training_config.get("max_eval_samples"),
    )
    if raw_value in (None, "", False):
        return None

    try:
        max_samples = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.eval_max_samples must be a positive integer, null, or omitted."
        ) from exc

    if max_samples <= 0:
        return None

    return max_samples


def limit_eval_dataset(eval_dataset: Dataset, training_config: Dict[str, Any]) -> Dataset:
    max_samples = resolve_eval_max_samples(training_config)
    total_samples = len(eval_dataset)

    if max_samples is None:
        print(f"\nEvaluation samples: {total_samples:,} (full eval set)")
        return eval_dataset

    limited_samples = min(max_samples, total_samples)
    if limited_samples < total_samples:
        print(
            f"\nLimiting evaluation samples: {limited_samples:,} / {total_samples:,} "
            f"(training.eval_max_samples={max_samples:,})"
        )
    else:
        print(
            f"\nEvaluation samples: {total_samples:,} "
            f"(training.eval_max_samples={max_samples:,}; using all available samples)"
        )

    return eval_dataset.shuffle(seed=42).select(range(limited_samples))


def prepare_or_load_training_datasets(
    dataset_dict: DatasetDict,
    processor,
    *,
    config: Dict[str, Any],
    base_model_name: str,
    storage_root: Optional[Path],
    phase_number: Optional[int],
    num_proc: int,
    test_mode: bool,
    rebuild_cache: bool,
) -> tuple[Dataset, Dataset]:
    min_duration, max_duration = duration_range_for_run(
        config,
        phase_number=phase_number,
    )
    train_cache_name = run_cache_name(config, phase_number=phase_number)

    cache_dir = None
    signature = None
    if not test_mode:
        cache_dir, signature = prepared_dataset_cache_dir(
            config,
            dataset_dict,
            base_model_name=base_model_name,
            storage_root=storage_root,
        )
        cached = load_prepared_dataset_cache(
            cache_dir,
            signature,
            train_cache_name=train_cache_name,
            rebuild=rebuild_cache,
        )
        if cached is not None:
            return cached

    train_raw, train_duration_column = ensure_duration_column(
        dataset_dict["train"],
        sampling_rate=int(config.get("audio", {}).get("sampling_rate", 16000)),
        num_proc=num_proc,
        label="training",
    )
    eval_raw, eval_duration_column = ensure_duration_column(
        dataset_dict["test"],
        sampling_rate=int(config.get("audio", {}).get("sampling_rate", 16000)),
        num_proc=num_proc,
        label="evaluation",
    )

    train_filtered = filter_duration_dataset(
        train_raw,
        duration_column=train_duration_column,
        min_duration=min_duration,
        max_duration=max_duration,
        label=f"{train_cache_name} training",
    )
    eval_filtered = filter_duration_dataset(
        eval_raw,
        duration_column=eval_duration_column,
        min_duration=min_duration,
        max_duration=max_duration,
        label=f"{train_cache_name} evaluation",
    )

    if test_mode:
        print("\nTEST MODE ENABLED - Using small filtered train/eval subsets")
        train_filtered = train_filtered.select(range(min(100, len(train_filtered))))
        eval_filtered = eval_filtered.select(range(min(100, len(eval_filtered))))

    prepared = prepare_dataset_dict(
        DatasetDict({"train": train_filtered, "test": eval_filtered}),
        processor,
        num_proc=num_proc,
        preprocessing_config=config.get("preprocessing", {}),
    )

    if cache_dir is not None and signature is not None:
        save_prepared_dataset_cache(
            cache_dir,
            signature,
            train_cache_name=train_cache_name,
            train_dataset=prepared["train"],
            eval_dataset=prepared["test"],
        )

    return prepared["train"], prepared["test"]


@dataclass
class DataCollatorForMoonshine:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [feature["input_values"] for feature in features]
        batch = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
        )

        labels = [feature["labels"] for feature in features]
        max_label_length = max(len(label) for label in labels)

        padded_labels = []
        decoder_input_ids = []
        bos_token_id = self.processor.tokenizer.bos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id

        for label in labels:
            padded_labels.append(label + [-100] * (max_label_length - len(label)))

            decoder_inputs = [bos_token_id] + label[:-1]
            decoder_inputs += [pad_token_id] * (max_label_length - len(decoder_inputs))
            decoder_input_ids.append(decoder_inputs)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["decoder_input_ids"] = torch.tensor(decoder_input_ids, dtype=torch.long)
        return batch


class MoonshineSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        generation_config: Optional[Dict[str, Any]] = None,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config or {}
        self.sampling_rate = sampling_rate

    # Keep the precomputed length column available long enough for grouped sampling.
    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()

        length_column_name = getattr(self.args, "length_column_name", None)
        if (
            length_column_name
            and self._signature_columns is not None
            and length_column_name not in self._signature_columns
        ):
            self._signature_columns.append(length_column_name)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, None, None)

        generation_kwargs = dict(self.generation_config)
        configured_max_new_tokens = generation_kwargs.pop("max_new_tokens", None)
        if configured_max_new_tokens is None and "input_values" in inputs:
            audio_length = inputs["input_values"].shape[-1]
            audio_duration = float(audio_length) / float(self.sampling_rate)
            configured_max_new_tokens = max(5, min(int(audio_duration * 6), 50))
        elif configured_max_new_tokens is None:
            configured_max_new_tokens = 50
        generation_kwargs["max_new_tokens"] = int(configured_max_new_tokens)

        with torch.no_grad():
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                loss = None

            generated_tokens = model.generate(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
                **generation_kwargs,
            )

        if labels is not None:
            labels = labels.detach()

        return (loss, generated_tokens, labels)


def build_training_arguments(
    *,
    output_dir: str,
    logging_dir: str,
    run_name: str,
    test_mode: bool,
    training_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    phase_config: Optional[Dict[str, Any]] = None,
) -> Seq2SeqTrainingArguments:
    def configured(name: str, default: Any) -> Any:
        if name in training_config:
            return training_config[name]
        if phase_config and name in phase_config:
            return phase_config[name]
        return default

    generation_max_length = generation_config.get(
        "max_length",
        generation_config.get(
            "max_new_tokens",
            configured("generation_max_length", 50),
        ),
    )
    generation_num_beams = generation_config.get(
        "num_beams",
        configured("generation_num_beams", 5),
    )

    kwargs = dict(
        output_dir=output_dir,
        max_steps=50 if test_mode else int(configured("max_steps", 15000)),
        per_device_train_batch_size=int(configured("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(configured("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(configured("gradient_accumulation_steps", 16)),
        learning_rate=float(configured("learning_rate", 1e-5)),
        warmup_steps=10 if test_mode else int(configured("warmup_steps", 500)),
        max_grad_norm=float(configured("max_grad_norm", 1.0)),
        fp16=training_config.get("fp16", True),
        fp16_full_eval=training_config.get("fp16_full_eval", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        eval_steps=25 if test_mode else int(configured("eval_steps", 500)),
        save_steps=25 if test_mode else int(configured("save_steps", configured("eval_steps", 500))),
        logging_steps=1 if test_mode else int(configured("logging_steps", 50)),
        predict_with_generate=training_config.get("predict_with_generate", True),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "wer"),
        greater_is_better=training_config.get("greater_is_better", False),
        report_to=training_config.get("report_to", ["tensorboard"]),
        logging_dir=logging_dir,
        generation_max_length=int(generation_max_length),
        generation_num_beams=int(generation_num_beams),
        run_name=run_name,
    )

    fields = Seq2SeqTrainingArguments.__dataclass_fields__

    if "optim" in fields:
        kwargs["optim"] = training_config.get("optim", "adamw_torch")
    if "lr_scheduler_type" in fields:
        kwargs["lr_scheduler_type"] = training_config.get("lr_scheduler_type", "linear")
    if "label_smoothing_factor" in fields:
        kwargs["label_smoothing_factor"] = float(
            training_config.get("label_smoothing_factor", 0.0)
        )
    if "logging_first_step" in fields:
        kwargs["logging_first_step"] = training_config.get("logging_first_step", True)
    if "save_total_limit" in fields and "save_total_limit" in training_config:
        kwargs["save_total_limit"] = training_config["save_total_limit"]
    if "push_to_hub" in fields:
        kwargs["push_to_hub"] = training_config.get("push_to_hub", False)
    if "hub_model_id" in fields and training_config.get("hub_model_id"):
        kwargs["hub_model_id"] = training_config.get("hub_model_id")
    if "hub_strategy" in fields and training_config.get("hub_strategy"):
        kwargs["hub_strategy"] = training_config.get("hub_strategy")
    if "hub_token" in fields and training_config.get("hub_token"):
        kwargs["hub_token"] = training_config.get("hub_token")

    strategy_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    kwargs[strategy_key] = training_config.get("eval_strategy", training_config.get("evaluation_strategy", "steps"))

    if "group_by_length" in fields:
        kwargs["group_by_length"] = training_config.get("group_by_length", True)
    else:
        kwargs["train_sampling_strategy"] = (
            "group_by_length" if training_config.get("group_by_length", True) else "random"
        )

    if "length_column_name" in fields:
        kwargs["length_column_name"] = training_config.get("length_column_name", "input_length")

    return Seq2SeqTrainingArguments(**kwargs)


def phase_output_dir(base_output_dir: str, phase_number: int) -> str:
    return f"{base_output_dir}-phase{phase_number}"


def phase_logging_dir(base_logging_dir: str, phase_number: int) -> str:
    return f"{base_logging_dir}-phase{phase_number}"


def checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    prefix = "checkpoint-"
    if not checkpoint_dir.name.startswith(prefix):
        return None

    step_text = checkpoint_dir.name[len(prefix):]
    if not step_text.isdigit():
        return None

    return int(step_text)


def is_complete_trainer_checkpoint(checkpoint_dir: Path) -> bool:
    if not checkpoint_dir.is_dir():
        return False

    required_files = (
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
    )
    if not all((checkpoint_dir / filename).is_file() for filename in required_files):
        return False

    return any(
        (checkpoint_dir / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def find_latest_complete_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None

    checkpoint_dirs = []
    for candidate in output_dir.iterdir():
        step = checkpoint_step(candidate)
        if step is not None:
            checkpoint_dirs.append((step, candidate))

    for _, checkpoint_dir in sorted(checkpoint_dirs, reverse=True):
        if is_complete_trainer_checkpoint(checkpoint_dir):
            return checkpoint_dir

        print(f"Skipping incomplete checkpoint: {checkpoint_dir}")

    return None


def resolve_resume_checkpoint(
    *,
    args: argparse.Namespace,
    output_dir: str,
    storage_root: Optional[Path],
) -> Optional[str]:
    if args.resume_from_checkpoint:
        checkpoint = Path(resolve_path(args.resume_from_checkpoint, storage_root))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        if not is_complete_trainer_checkpoint(checkpoint):
            raise ValueError(f"Resume checkpoint is incomplete: {checkpoint}")
        return str(checkpoint)

    if not args.resume:
        return None

    checkpoint = find_latest_complete_checkpoint(Path(output_dir))
    if checkpoint is None:
        raise FileNotFoundError(
            "No complete checkpoint found to resume from in "
            f"{output_dir}. Pass --resume-from-checkpoint to choose one explicitly."
        )

    return str(checkpoint)


def model_source_for_phase(
    *,
    phase_number: int,
    base_model_name: str,
    base_output_dir: str,
) -> str:
    if phase_number == 1:
        return base_model_name

    previous_final = Path(phase_output_dir(base_output_dir, phase_number - 1)) / "final"
    if not previous_final.exists():
        raise FileNotFoundError(
            f"Phase {phase_number} must start from Phase {phase_number - 1}, "
            f"but the checkpoint was not found at {previous_final}"
        )

    return str(previous_final)


def generation_config_for_run(
    config: Dict[str, Any],
    *,
    phase_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {}
    if phase_config is not None:
        generation_config.update({
            "max_new_tokens": phase_config.get("generation_max_length", 50),
            "num_beams": phase_config.get("generation_num_beams", 5),
        })

    generation_config.update(config.get("generation", {}))
    return {
        key: value
        for key, value in generation_config.items()
        if value is not None
    }


def apply_generation_config(model, generation_config: Dict[str, Any]) -> None:
    if not hasattr(model, "generation_config"):
        return

    for key, value in generation_config.items():
        if key == "max_new_tokens":
            continue
        setattr(model.generation_config, key, value)

    print("\nGeneration config:")
    for key in sorted(generation_config):
        print(f"  {key}: {generation_config[key]}")


def make_compute_metrics(processor, wer_metric):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(
            label_ids == -100,
            processor.tokenizer.pad_token_id,
            label_ids,
        )
        pred_ids = np.where(
            pred_ids < 0,
            processor.tokenizer.pad_token_id,
            pred_ids,
        )

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def build_trainer(
    *,
    model,
    args,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
    processor,
    generation_config,
    sampling_rate: int,
):
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    return MoonshineSeq2SeqTrainer(
        **trainer_kwargs,
        generation_config=generation_config,
        sampling_rate=sampling_rate,
    )


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    storage = configure_storage(config)
    storage_root = Path(storage["storage_root"]) if storage["storage_root"] else None

    dataset_config = config.setdefault("dataset", {})
    training_config = config.setdefault("training", {})
    model_config = config.setdefault("model", {})
    preprocessing_config = config.setdefault("preprocessing", {})

    if dataset_config.get("path"):
        dataset_config["path"] = resolve_path(dataset_config["path"], storage_root)

    base_output_dir = args.output_dir or training_config.get("output_dir", "./results/moonshine-en")
    base_logging_dir = training_config.get("logging_dir", "./logs/moonshine-en")

    if args.output_dir:
        base_output_dir = resolve_path(args.output_dir, storage_root)

    use_curriculum = curriculum_enabled(config)
    phase_number = args.phase if args.phase is not None else (1 if use_curriculum else None)
    phase = PHASE_CONFIGS[phase_number] if use_curriculum else None
    if not use_curriculum and args.phase is not None:
        print(
            f"\nIgnoring --phase {args.phase} because curriculum.enabled=false in the config."
        )

    output_dir = (
        phase_output_dir(base_output_dir, phase_number)
        if use_curriculum
        else base_output_dir
    )
    logging_dir = (
        phase_logging_dir(base_logging_dir, phase_number)
        if use_curriculum
        else base_logging_dir
    )
    resume_checkpoint = resolve_resume_checkpoint(
        args=args,
        output_dir=output_dir,
        storage_root=storage_root,
    )

    print("\n" + "=" * 80)
    print("MOONSHINE CONFIG-DRIVEN FINE-TUNING")
    print("=" * 80)
    if use_curriculum:
        print(f"Mode: curriculum phase {phase_number}: {phase['name']}")
    else:
        print("Mode: no curriculum")
    min_duration, max_duration = duration_range_for_run(
        config,
        phase_number=phase_number,
    )
    print(f"Duration range: {min_duration:.1f}s - {max_duration:.1f}s")
    print(f"Output dir: {output_dir}")
    print(f"Logging dir: {logging_dir}")
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")
    print("=" * 80)

    base_model_name = model_config.get("name", "UsefulSensors/moonshine-base")
    model_cache_dir = model_config.get("cache_dir")

    print(f"\nLoading processor from: {base_model_name}")
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        cache_dir=model_cache_dir,
    )
    restore_missing_special_tokens(
        processor,
        model_name=base_model_name,
        cache_dir=model_cache_dir,
    )

    dataset_dict = load_local_dataset(config)
    train_dataset, eval_dataset = prepare_or_load_training_datasets(
        dataset_dict,
        processor,
        config=config,
        base_model_name=base_model_name,
        storage_root=storage_root,
        phase_number=phase_number,
        num_proc=preprocessing_config.get("num_proc", 4),
        test_mode=args.test_mode,
        rebuild_cache=args.rebuild_dataset_cache,
    )
    eval_dataset = limit_eval_dataset(eval_dataset, training_config)

    if resume_checkpoint:
        model_source = resume_checkpoint
    else:
        if use_curriculum:
            model_source = model_source_for_phase(
                phase_number=phase_number,
                base_model_name=base_model_name,
                base_output_dir=base_output_dir,
            )
        else:
            model_source = base_model_name
    print(f"Loading model from: {model_source}")

    model = MoonshineForConditionalGeneration.from_pretrained(
        model_source,
        cache_dir=model_cache_dir,
    )

    if training_config.get("gradient_checkpointing", True):
        model.config.use_cache = False

    if model_config.get("freeze_encoder", False):
        print("\nFreezing encoder weights.")
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = model.num_parameters()
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
    else:
        print(f"\nModel loaded: {model.num_parameters():,} parameters")

    data_collator = DataCollatorForMoonshine(processor=processor)
    wer_metric = evaluate.load("wer", cache_dir=storage.get("evaluate_cache_dir"))
    compute_metrics = make_compute_metrics(processor, wer_metric)
    generation_config = generation_config_for_run(config, phase_config=phase)
    apply_generation_config(model, generation_config)
    training_args = build_training_arguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        run_name=f"moonshine_phase{phase_number}" if use_curriculum else "moonshine_full",
        test_mode=args.test_mode,
        training_config=training_config,
        generation_config=generation_config,
        phase_config=phase,
    )

    trainer = build_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processor=processor,
        generation_config=generation_config,
        sampling_rate=int(config.get("audio", {}).get("sampling_rate", 16000)),
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir)

    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Evaluation samples: {len(eval_dataset):,}")
    print(f"Max steps: {training_args.max_steps:,}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(
        "Effective batch size: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    if args.test_mode:
        print("Expected duration: a few minutes")
    elif phase is not None:
        print(f"Expected duration: {phase['expected_duration']}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print("\nTraining complete!")
    print(f"Saved final model to: {final_dir}")

    if use_curriculum and phase_number < 3:
        print(f"\nNext step: run Phase {phase_number + 1}")
        print(f"  python train.py --config {args.config} --phase {phase_number + 1}")
    elif use_curriculum:
        print("\nFinal curriculum phase complete. Model is ready for evaluation/inference.")


if __name__ == "__main__":
    main()
