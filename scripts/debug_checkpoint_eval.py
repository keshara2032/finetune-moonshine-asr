#!/usr/bin/env python3
"""
Compare base Moonshine against phase checkpoints on one fixed eval subset.

This is a focused debugging tool for the current English fine-tuning regression.
It evaluates base/checkpoint/final models on the same duration-filtered samples
and reports both raw lowercased WER and punctuation-stripped WER.
"""

import argparse
import csv
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in the real env
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import MoonshineInference


DEFAULT_CONFIG = REPO_ROOT / "configs" / "english_accent_local_no_curriculum.yaml"
RESULTS_DIR = REPO_ROOT / "results"
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9']+")
PHASE_DURATION_FILTERS = {
    1: (4.0, 10.0),
    2: (10.0, 20.0),
    3: (4.0, 30.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug Moonshine checkpoint performance against the base model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Training config YAML.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help=(
            "Optional run id used by train_job.sh. If set, output base resolves to "
            "results/<RUN_ID>/moonshine-en-accents under storage.root_dir."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=str,
        help="Base output directory before the -phaseN suffix.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Curriculum phase whose checkpoints should be evaluated.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["checkpoint-200", "checkpoint-1000", "checkpoint-4000", "final"],
        help=(
            "Checkpoint names, numeric steps, 'final', or explicit directories. "
            "Relative checkpoint names resolve under <output-base>-phaseN."
        ),
    )
    parser.add_argument(
        "--strict-checkpoints",
        action="store_true",
        help="Fail if any requested checkpoint is missing instead of skipping it.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Do not evaluate the base model.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Override dataset.path from the config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum duration filter. Defaults to the phase range.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum duration filter. Defaults to the phase range.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples after duration filtering. Use 0 for all matching samples.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Inference device. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 on CUDA.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override generation max_new_tokens. Defaults to duration-based sizing.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="debug_checkpoint_eval",
        help="Prefix for result files under results/.",
    )
    return parser.parse_args()


def resolve_storage_root(config: Dict[str, Any]) -> Optional[Path]:
    root_dir = config.get("storage", {}).get("root_dir")
    if not root_dir:
        return None
    return Path(root_dir).expanduser().resolve(strict=False)


def resolve_path(
    path_value: Optional[str],
    *,
    storage_root: Optional[Path],
    prefer_storage_root: bool = False,
) -> Optional[Path]:
    if path_value is None:
        return None

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)

    candidates: List[Path] = []
    if prefer_storage_root and storage_root is not None:
        candidates.append((storage_root / path).resolve(strict=False))
    candidates.append((REPO_ROOT / path).resolve(strict=False))
    if not prefer_storage_root and storage_root is not None:
        candidates.append((storage_root / path).resolve(strict=False))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_base(
    args: argparse.Namespace,
    config: Dict[str, Any],
    storage_root: Optional[Path],
) -> Path:
    if args.output_base:
        resolved = resolve_path(args.output_base, storage_root=storage_root, prefer_storage_root=True)
        if resolved is None:
            raise ValueError("Could not resolve --output-base.")
        return resolved

    if args.run_id:
        resolved = resolve_path(
            f"results/{args.run_id}/moonshine-en-accents",
            storage_root=storage_root,
            prefer_storage_root=True,
        )
        if resolved is None:
            raise ValueError("Could not resolve --run-id output base.")
        return resolved

    output_dir = config.get("training", {}).get("output_dir")
    resolved = resolve_path(output_dir, storage_root=storage_root, prefer_storage_root=True)
    if resolved is None:
        raise ValueError("Config must define training.output_dir or pass --output-base.")
    return resolved


def resolve_column_name(
    columns: Sequence[str],
    preferred: Optional[str],
    fallbacks: Sequence[str],
    column_kind: str,
) -> str:
    if preferred and preferred in columns:
        return preferred
    for fallback in fallbacks:
        if fallback in columns:
            return fallback
    raise ValueError(
        f"Could not find {column_kind} column. Preferred={preferred!r}, "
        f"fallbacks={list(fallbacks)}, columns={list(columns)}"
    )


def load_eval_dataset(
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
    storage_root: Optional[Path],
) -> Tuple[Dataset, Path, str, str]:
    dataset_path_text = args.dataset_path or config.get("dataset", {}).get("path")
    dataset_path = resolve_path(dataset_path_text, storage_root=storage_root)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            raise KeyError(f"Split {args.split!r} not found. Available: {list(loaded.keys())}")
        dataset = loaded[args.split]
    else:
        dataset = loaded

    text_column = resolve_column_name(
        dataset.column_names,
        config.get("dataset", {}).get("text_column", "text"),
        ("text", "transcript", "transcription", "sentence"),
        "text",
    )
    audio_column = resolve_column_name(
        dataset.column_names,
        "audio",
        ("audio",),
        "audio",
    )
    return dataset, dataset_path, text_column, audio_column


def resolve_audio_path(audio_value: Any, dataset_root: Path) -> Optional[Path]:
    if isinstance(audio_value, dict) and "path" in audio_value and audio_value["path"]:
        audio_value = audio_value["path"]

    if isinstance(audio_value, (str, Path)):
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (dataset_root / audio_path).resolve(strict=False)
        return audio_path

    return None


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling for dataset-decoded audio values."""
    if orig_sr == target_sr:
        return audio_data.astype(np.float32, copy=False)

    duration = len(audio_data) / float(orig_sr)
    target_length = max(1, int(round(duration * target_sr)))
    indices = np.linspace(0, len(audio_data) - 1, target_length)
    return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)


def to_mono_float32(audio_array: Any) -> np.ndarray:
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.ndim == 2 and audio_array.shape[0] <= 2:
        audio_array = audio_array[0] if audio_array.shape[0] == 1 else audio_array.mean(axis=0)
    elif audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    return audio_array.astype(np.float32, copy=False)


def load_audio_value(
    audio_value: Any,
    *,
    dataset_root: Path,
    target_sampling_rate: int,
) -> Tuple[np.ndarray, int]:
    """Load audio from paths, HF Audio dicts, or torchcodec AudioDecoder objects."""
    if hasattr(audio_value, "get_all_samples"):
        audio_samples = audio_value.get_all_samples()
        audio_array = audio_samples.data
        if hasattr(audio_array, "cpu"):
            audio_array = audio_array.cpu().numpy()

        audio_array = to_mono_float32(audio_array)
        sampling_rate = int(audio_samples.sample_rate)
        if sampling_rate != target_sampling_rate:
            audio_array = resample_audio(audio_array, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return audio_array, sampling_rate

    if isinstance(audio_value, dict):
        if "array" in audio_value and "sampling_rate" in audio_value:
            audio_array = to_mono_float32(audio_value["array"])
            sampling_rate = int(audio_value["sampling_rate"])
            if sampling_rate != target_sampling_rate:
                audio_array = resample_audio(audio_array, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate
            return audio_array, sampling_rate
        if "path" in audio_value:
            audio_value = audio_value["path"]

    if isinstance(audio_value, (str, Path)):
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (dataset_root / audio_path).resolve(strict=False)

        audio_array, sampling_rate = sf.read(str(audio_path), always_2d=False)
        audio_array = to_mono_float32(audio_array)
        if sampling_rate != target_sampling_rate:
            audio_array = resample_audio(audio_array, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return audio_array, sampling_rate

    if hasattr(audio_value, "__len__"):
        return to_mono_float32(audio_value), target_sampling_rate

    raise TypeError(f"Unsupported audio value type: {type(audio_value)}")


def audio_duration_seconds(audio_value: Any, dataset_root: Path, sampling_rate: int) -> float:
    audio_path = resolve_audio_path(audio_value, dataset_root)
    if audio_path is not None and audio_path.exists():
        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)

    audio_array, actual_sampling_rate = load_audio_value(
        audio_value,
        dataset_root=dataset_root,
        target_sampling_rate=sampling_rate,
    )
    return float(len(audio_array)) / float(actual_sampling_rate)


def inference_audio_input(audio_value: Any, dataset_root: Path, sampling_rate: int) -> Any:
    audio_path = resolve_audio_path(audio_value, dataset_root)
    if audio_path is not None and audio_path.exists():
        return audio_path

    audio_array, _ = load_audio_value(
        audio_value,
        dataset_root=dataset_root,
        target_sampling_rate=sampling_rate,
    )
    return audio_array


def select_eval_samples(
    dataset: Dataset,
    *,
    dataset_root: Path,
    audio_column: str,
    text_column: str,
    sampling_rate: int,
    min_duration: float,
    max_duration: float,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    desc = f"Selecting {min_duration:g}-{max_duration:g}s samples"

    for dataset_index in tqdm(range(len(dataset)), desc=desc):
        sample = dataset[dataset_index]
        audio_value = sample[audio_column]
        duration = audio_duration_seconds(audio_value, dataset_root, sampling_rate)
        if duration < min_duration or duration > max_duration:
            continue

        selected.append({
            "dataset_index": dataset_index,
            "audio": audio_value,
            "reference": str(sample[text_column]),
            "duration": duration,
        })

        if max_samples is not None and len(selected) >= max_samples:
            break

    if not selected:
        raise ValueError(
            f"No samples matched duration filter {min_duration}-{max_duration}s."
        )
    return selected


def normalize_raw(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text).strip()).lower()


def normalize_punctuation_stripped(text: str) -> str:
    lowered = str(text).strip().lower()
    stripped = PUNCT_RE.sub(" ", lowered)
    return WHITESPACE_RE.sub(" ", stripped).strip()


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def word_errors(reference: str, prediction: str) -> Tuple[int, int]:
    ref_words = reference.split()
    pred_words = prediction.split()
    return edit_distance(ref_words, pred_words), len(ref_words)


def suggest_max_new_tokens(duration_seconds: float) -> int:
    return max(150, min(int(duration_seconds * 5), 1024))


def checkpoint_label(path: Path, phase: int) -> str:
    if path.name == "final":
        return f"phase{phase}_final"
    return f"phase{phase}_{path.name}"


def resolve_checkpoint_specs(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    storage_root: Optional[Path],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    if not args.skip_base:
        specs.append({
            "label": "base",
            "path": config.get("model", {}).get("name", "UsefulSensors/moonshine-base"),
            "source": "base",
        })

    output_base = resolve_output_base(args, config, storage_root)
    phase_dir = Path(f"{output_base}-phase{args.phase}")

    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.is_absolute() or checkpoint_path.exists():
            resolved = checkpoint_path.resolve(strict=False)
        elif checkpoint == "final":
            resolved = phase_dir / "final"
        elif checkpoint.isdigit():
            resolved = phase_dir / f"checkpoint-{checkpoint}"
        else:
            resolved = phase_dir / checkpoint

        if not resolved.exists():
            message = f"Checkpoint not found, skipping: {resolved}"
            if args.strict_checkpoints:
                raise FileNotFoundError(message)
            print(f"WARNING: {message}", file=sys.stderr)
            continue

        specs.append({
            "label": checkpoint_label(resolved, args.phase),
            "path": str(resolved),
            "source": checkpoint,
        })

    if not specs:
        raise ValueError("No models selected for evaluation.")
    return specs


def release_runner(runner: MoonshineInference) -> None:
    del runner
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    *,
    model_spec: Dict[str, Any],
    samples: List[Dict[str, Any]],
    dataset_root: Path,
    sampling_rate: int,
    generation_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    runner = MoonshineInference(
        model_path=model_spec["path"],
        device=args.device,
        fp16=args.fp16,
    )

    raw_errors = 0
    raw_words = 0
    stripped_errors = 0
    stripped_words = 0
    exact_raw = 0
    total_audio_duration = 0.0
    total_inference_time = 0.0
    sample_rows: List[Dict[str, Any]] = []

    try:
        for sample in tqdm(samples, desc=f"Evaluating {model_spec['label']}"):
            max_new_tokens = args.max_new_tokens or suggest_max_new_tokens(sample["duration"])
            start_time = time.time()
            result = runner.transcribe(
                inference_audio_input(sample["audio"], dataset_root, sampling_rate),
                sampling_rate=sampling_rate,
                num_beams=int(generation_config.get("num_beams", 5)),
                repetition_penalty=float(generation_config.get("repetition_penalty", 1.3)),
                no_repeat_ngram_size=int(generation_config.get("no_repeat_ngram_size", 2)),
                max_new_tokens=max_new_tokens,
            )
            elapsed = float(result.get("inference_time", time.time() - start_time))
            prediction = result["text"]

            raw_reference = normalize_raw(sample["reference"])
            raw_prediction = normalize_raw(prediction)
            stripped_reference = normalize_punctuation_stripped(sample["reference"])
            stripped_prediction = normalize_punctuation_stripped(prediction)

            sample_raw_errors, sample_raw_words = word_errors(raw_reference, raw_prediction)
            sample_stripped_errors, sample_stripped_words = word_errors(
                stripped_reference,
                stripped_prediction,
            )

            raw_errors += sample_raw_errors
            raw_words += sample_raw_words
            stripped_errors += sample_stripped_errors
            stripped_words += sample_stripped_words
            exact_raw += int(sample_raw_errors == 0)
            total_audio_duration += sample["duration"]
            total_inference_time += elapsed

            sample_rows.append({
                "model_label": model_spec["label"],
                "dataset_index": sample["dataset_index"],
                "duration": f"{sample['duration']:.4f}",
                "raw_wer": f"{100.0 * sample_raw_errors / sample_raw_words:.4f}" if sample_raw_words else "0.0000",
                "punct_stripped_wer": (
                    f"{100.0 * sample_stripped_errors / sample_stripped_words:.4f}"
                    if sample_stripped_words else "0.0000"
                ),
                "raw_word_errors": sample_raw_errors,
                "punct_stripped_word_errors": sample_stripped_errors,
                "max_new_tokens": max_new_tokens,
                "reference": raw_reference,
                "prediction": raw_prediction,
                "punct_stripped_reference": stripped_reference,
                "punct_stripped_prediction": stripped_prediction,
            })
    finally:
        release_runner(runner)

    summary = {
        "model_label": model_spec["label"],
        "model_source": model_spec["source"],
        "model_path": model_spec["path"],
        "num_samples": len(samples),
        "raw_wer": 100.0 * raw_errors / raw_words if raw_words else 0.0,
        "punct_stripped_wer": (
            100.0 * stripped_errors / stripped_words if stripped_words else 0.0
        ),
        "exact_raw_rate": 100.0 * exact_raw / len(samples) if samples else 0.0,
        "raw_word_errors": raw_errors,
        "raw_reference_words": raw_words,
        "punct_stripped_word_errors": stripped_errors,
        "punct_stripped_reference_words": stripped_words,
        "total_audio_duration": total_audio_duration,
        "total_inference_time": total_inference_time,
        "rtf": total_inference_time / total_audio_duration if total_audio_duration else 0.0,
    }
    return summary, sample_rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve(strict=False)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    storage_root = resolve_storage_root(config)
    dataset, dataset_root, text_column, audio_column = load_eval_dataset(
        config=config,
        args=args,
        storage_root=storage_root,
    )
    sampling_rate = int(config.get("audio", {}).get("sampling_rate", 16000))
    default_min_duration, default_max_duration = PHASE_DURATION_FILTERS[args.phase]
    min_duration = args.min_duration if args.min_duration is not None else default_min_duration
    max_duration = args.max_duration if args.max_duration is not None else default_max_duration
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    model_specs = resolve_checkpoint_specs(args=args, config=config, storage_root=storage_root)
    samples = select_eval_samples(
        dataset,
        dataset_root=dataset_root,
        audio_column=audio_column,
        text_column=text_column,
        sampling_rate=sampling_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        max_samples=max_samples,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary_{timestamp}.csv"
    samples_path = RESULTS_DIR / f"{args.output_prefix}_samples_{timestamp}.csv"
    suite_path = RESULTS_DIR / f"{args.output_prefix}_suite_{timestamp}.json"

    print()
    print("=" * 80)
    print("DEBUG CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"Config:        {config_path}")
    print(f"Dataset:       {dataset_root}")
    print(f"Split:         {args.split}")
    print(f"Text column:   {text_column}")
    print(f"Duration:      {min_duration:g}-{max_duration:g}s")
    print(f"Samples:       {len(samples)}")
    print("Models:")
    for spec in model_specs:
        print(f"  - {spec['label']}: {spec['path']}")
    print("=" * 80)

    summaries: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []
    for model_spec in model_specs:
        summary, sample_rows = evaluate_model(
            model_spec=model_spec,
            samples=samples,
            dataset_root=dataset_root,
            sampling_rate=sampling_rate,
            generation_config=config.get("generation", {}),
            args=args,
        )
        summaries.append(summary)
        all_sample_rows.extend(sample_rows)
        print(
            f"{summary['model_label']}: raw WER {summary['raw_wer']:.2f}% | "
            f"punct-stripped WER {summary['punct_stripped_wer']:.2f}% | "
            f"RTF {summary['rtf']:.3f}x"
        )

    write_csv(
        summary_path,
        summaries,
        fieldnames=[
            "model_label",
            "model_source",
            "model_path",
            "num_samples",
            "raw_wer",
            "punct_stripped_wer",
            "exact_raw_rate",
            "raw_word_errors",
            "raw_reference_words",
            "punct_stripped_word_errors",
            "punct_stripped_reference_words",
            "total_audio_duration",
            "total_inference_time",
            "rtf",
        ],
    )
    write_csv(
        samples_path,
        all_sample_rows,
        fieldnames=[
            "model_label",
            "dataset_index",
            "duration",
            "raw_wer",
            "punct_stripped_wer",
            "raw_word_errors",
            "punct_stripped_word_errors",
            "max_new_tokens",
            "reference",
            "prediction",
            "punct_stripped_reference",
            "punct_stripped_prediction",
        ],
    )
    suite_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "dataset_path": str(dataset_root),
                "split": args.split,
                "phase": args.phase,
                "duration_filter": {
                    "min_seconds": min_duration,
                    "max_seconds": max_duration,
                },
                "num_samples": len(samples),
                "summary_csv": str(summary_path),
                "samples_csv": str(samples_path),
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Wrote:")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Samples CSV: {samples_path}")
    print(f"  Suite JSON:  {suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
