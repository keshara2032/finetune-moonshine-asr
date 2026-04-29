#!/usr/bin/env python3
"""
Moonshine evaluation helper for single runs and simple comparison suites.

Supports:
- single-model evaluation on one split
- base-vs-phase checkpoint comparisons
- phase-duration subset evaluation without sliding windows
- full-dataset evaluation with the repo's sliding-window decoder
- per-run JSON/Markdown reports
- suite-level summary and sample CSVs
"""

import argparse
import csv
import gc
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import MoonshineInference, transcribe_with_sliding_windows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "english_accent_local_no_curriculum.yaml"
RESULTS_DIR = REPO_ROOT / "results"
WHITESPACE_RE = re.compile(r"\s+")
PHASE_DURATION_FILTERS = {
    1: (4.0, 10.0),
    2: (10.0, 20.0),
    3: (4.0, 30.0),
}


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling."""
    if orig_sr == target_sr:
        return audio_data

    duration = len(audio_data) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio_data) - 1, target_length)
    return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)


def load_audio_value(
    audio_value,
    *,
    target_sampling_rate: int,
    dataset_root: Path
) -> Tuple[np.ndarray, int]:
    """Load audio whether the dataset stores decoded audio or plain file paths."""
    if hasattr(audio_value, "get_all_samples"):
        audio_samples = audio_value.get_all_samples()
        audio_array = audio_samples.data

        if hasattr(audio_array, "cpu"):
            audio_array = audio_array.cpu().numpy()

        audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim == 2 and audio_array.shape[0] == 1:
            audio_array = audio_array[0]
        elif audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)

        sampling_rate = int(audio_samples.sample_rate)
        if sampling_rate != target_sampling_rate:
            audio_array = resample_audio(audio_array, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate

        return audio_array, sampling_rate

    if isinstance(audio_value, dict):
        if "array" in audio_value and "sampling_rate" in audio_value:
            audio_array = np.asarray(audio_value["array"], dtype=np.float32)
            return audio_array, int(audio_value["sampling_rate"])
        if "path" in audio_value:
            audio_value = audio_value["path"]

    if isinstance(audio_value, (str, Path)):
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (dataset_root / audio_path).resolve(strict=False)

        audio_array, sampling_rate = sf.read(str(audio_path), always_2d=False)
        audio_array = np.asarray(audio_array, dtype=np.float32)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        if sampling_rate != target_sampling_rate:
            audio_array = resample_audio(audio_array, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate

        return audio_array, sampling_rate

    raise TypeError(f"Unsupported audio value type: {type(audio_value)}")


def resolve_audio_input(audio_value, dataset_root: Path):
    """Prefer raw file-path audio so we can reuse inference.py's exact loading logic."""
    if isinstance(audio_value, dict) and "path" in audio_value:
        audio_value = audio_value["path"]

    if isinstance(audio_value, (str, Path)):
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (dataset_root / audio_path).resolve(strict=False)
        return audio_path

    return audio_value


def resolve_audio_input_for_inference(
    audio_value,
    *,
    target_sampling_rate: int,
    dataset_root: Path
):
    """
    Resolve audio into something inference.py can actually consume.

    Prefer a file path when available so we reuse inference.py's loader.
    Otherwise, decode the dataset-specific audio object to a float32 waveform.
    """
    audio_input = resolve_audio_input(audio_value, dataset_root)
    if isinstance(audio_input, Path):
        return audio_input

    audio_array, _ = load_audio_value(
        audio_input,
        target_sampling_rate=target_sampling_rate,
        dataset_root=dataset_root,
    )
    return audio_array


def estimate_audio_duration_seconds(audio_input, *, target_sampling_rate: int, dataset_root: Path) -> float:
    """Estimate audio duration cheaply for generation-length sizing."""
    audio_input = resolve_audio_input(audio_input, dataset_root)

    if isinstance(audio_input, Path):
        info = sf.info(str(audio_input))
        return float(info.frames) / float(info.samplerate)

    audio_array, sampling_rate = load_audio_value(
        audio_input,
        target_sampling_rate=target_sampling_rate,
        dataset_root=dataset_root
    )
    return len(audio_array) / sampling_rate


def suggest_max_new_tokens(duration_seconds: float) -> int:
    """
    Use a much higher cap than the repo inference default for long dialogue clips.
    The old 150-token ceiling is too small for multi-minute samples.
    """
    return max(150, min(int(duration_seconds * 5), 1024))


def normalize_text(text: str, keep_case: bool) -> str:
    """Normalize whitespace, and lowercase by default for cleaner WER stats."""
    normalized = WHITESPACE_RE.sub(" ", str(text).strip())
    return normalized if keep_case else normalized.lower()


def sanitize_slug(value: str) -> str:
    """Create a filesystem-friendly identifier."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "run"


def resolve_path(
    path_value: Optional[str],
    *,
    storage_root: Optional[Path] = None,
    prefer_storage_root: bool = False
) -> Optional[Path]:
    """Resolve a config path against likely roots."""
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


def resolve_column_name(
    columns: Sequence[str],
    preferred: str,
    fallbacks: Sequence[str],
    column_kind: str
) -> str:
    """Resolve a dataset column with a preferred name plus fallbacks."""
    if preferred in columns:
        return preferred

    for fallback in fallbacks:
        if fallback in columns:
            print(
                f"[INFO] Using {column_kind} column '{fallback}' "
                f"(requested '{preferred}' was not found)"
            )
            return fallback

    raise ValueError(
        f"Could not find {column_kind} column. "
        f"Preferred '{preferred}', fallbacks {list(fallbacks)}, available {list(columns)}"
    )


def edit_distance_length(reference: Sequence[str], prediction: Sequence[str]) -> int:
    """Return Levenshtein edit distance length between two sequences."""
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            substitution_cost = 0 if ref_item == pred_item else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution_cost,
            ))
        previous = current

    return previous[-1]


def align_words(reference_words: List[str], prediction_words: List[str]) -> Tuple[List[Tuple[str, Optional[str], Optional[str]]], Dict[str, int]]:
    """
    Align reference and prediction words with a simple edit-distance backtrace.

    Returns:
        ops: list of (operation, reference_word, prediction_word)
        counts: substitutions/deletions/insertions/hits
    """
    rows = len(reference_words)
    cols = len(prediction_words)

    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    back = [[None] * (cols + 1) for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    operation_rank = {
        "equal": 0,
        "substitute": 1,
        "delete": 2,
        "insert": 3,
    }

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if reference_words[i - 1] == prediction_words[j - 1]:
                diag_cost = dp[i - 1][j - 1]
                diag_op = "equal"
            else:
                diag_cost = dp[i - 1][j - 1] + 1
                diag_op = "substitute"

            candidates = [
                (diag_cost, operation_rank[diag_op], diag_op),
                (dp[i - 1][j] + 1, operation_rank["delete"], "delete"),
                (dp[i][j - 1] + 1, operation_rank["insert"], "insert"),
            ]
            best_cost, _, best_op = min(candidates)
            dp[i][j] = best_cost
            back[i][j] = best_op

    operations: List[Tuple[str, Optional[str], Optional[str]]] = []
    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "hits": 0,
    }

    i = rows
    j = cols
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("equal", "substitute"):
            ref_word = reference_words[i - 1]
            pred_word = prediction_words[j - 1]
            operations.append((op, ref_word, pred_word))
            if op == "equal":
                counts["hits"] += 1
            else:
                counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif op == "delete":
            ref_word = reference_words[i - 1]
            operations.append((op, ref_word, None))
            counts["deletions"] += 1
            i -= 1
        elif op == "insert":
            pred_word = prediction_words[j - 1]
            operations.append((op, None, pred_word))
            counts["insertions"] += 1
            j -= 1
        else:
            break

    operations.reverse()
    return operations, counts


def load_dataset_split(dataset_path: Path, split: str) -> Dataset:
    """Load the requested split from a local Hugging Face dataset directory."""
    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Split '{split}' not found. Available: {list(loaded.keys())}")
        return loaded[split]
    return loaded


def counter_to_records(counter: Counter, key_name: str) -> List[Dict[str, object]]:
    """Format a simple counter for JSON output."""
    return [
        {key_name: item, "count": count}
        for item, count in counter.most_common(15)
    ]


def pair_counter_to_records(counter: Counter) -> List[Dict[str, object]]:
    """Format substitution pair counts for JSON output."""
    return [
        {"reference_word": reference_word, "predicted_word": predicted_word, "count": count}
        for (reference_word, predicted_word), count in counter.most_common(15)
    ]


def resolve_storage_root(config: Dict[str, Any]) -> Optional[Path]:
    """Resolve the optional storage root from the config."""
    root_dir = config.get("storage", {}).get("root_dir")
    if not root_dir:
        return None
    return Path(root_dir).expanduser().resolve(strict=False)


def resolve_default_dataset_path(config: Dict[str, Any], storage_root: Optional[Path]) -> Path:
    """Resolve the default dataset path from config."""
    dataset_path = resolve_path(config["dataset"]["path"], storage_root=storage_root)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    return dataset_path


def resolve_training_output_dir(config: Dict[str, Any], storage_root: Optional[Path]) -> Path:
    """Resolve the base training output directory from config."""
    output_dir = resolve_path(
        config["training"]["output_dir"],
        storage_root=storage_root,
        prefer_storage_root=True
    )
    if output_dir is None:
        raise ValueError("Could not resolve training.output_dir from config.")
    return output_dir


def resolve_phase_model_path(
    config: Dict[str, Any],
    storage_root: Optional[Path],
    phase: int
) -> Path:
    """Resolve a curriculum phase checkpoint directory."""
    base_output_dir = resolve_training_output_dir(config, storage_root)
    model_dir = Path(f"{base_output_dir}-phase{phase}") / "final"
    if not model_dir.exists():
        raise FileNotFoundError(f"Phase {phase} model directory not found: {model_dir}")
    return model_dir


def resolve_single_model_spec(
    args: argparse.Namespace,
    config: Dict[str, Any],
    storage_root: Optional[Path]
) -> Dict[str, Any]:
    """Resolve the original single-model CLI options."""
    if args.model_source == "base":
        resolved_model = config["model"]["name"]
        model_label = "base"
    elif args.model_source == "finetuned":
        output_dir = resolve_training_output_dir(config, storage_root)
        model_dir = output_dir / "final"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        resolved_model = str(model_dir)
        model_label = "finetuned"
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --model-source=path")
        model_path_obj = Path(args.model_path).expanduser().resolve(strict=False)
        resolved_model = str(model_path_obj) if model_path_obj.exists() else args.model_path
        model_label = args.model_label or "custom"

    return {
        "label": model_label,
        "phase": None,
        "source": args.model_source,
        "path": str(resolved_model),
    }


def build_model_specs(
    args: argparse.Namespace,
    config: Dict[str, Any],
    storage_root: Optional[Path]
) -> List[Dict[str, Any]]:
    """Build the list of models that will be evaluated."""
    phases = sorted(set(args.phases or []))
    comparison_mode = bool(args.include_base or phases)

    if not comparison_mode:
        return [resolve_single_model_spec(args, config, storage_root)]

    model_specs: List[Dict[str, Any]] = []

    if args.include_base:
        model_specs.append({
            "label": "base",
            "phase": None,
            "source": "base",
            "path": config["model"]["name"],
        })

    for phase in phases:
        model_specs.append({
            "label": f"phase{phase}",
            "phase": phase,
            "source": f"phase{phase}",
            "path": str(resolve_phase_model_path(config, storage_root, phase)),
        })

    if not model_specs:
        raise ValueError("No models selected for evaluation.")

    return model_specs


def resolve_dataset_arg_path(
    explicit_path: Optional[str],
    *,
    default_path: Path,
    storage_root: Optional[Path]
) -> Path:
    """Resolve an optional dataset-path argument or fall back to the config dataset."""
    if explicit_path is None:
        return default_path

    resolved = resolve_path(explicit_path, storage_root=storage_root)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Dataset directory not found: {resolved}")
    return resolved


def build_dataset_specs(
    args: argparse.Namespace,
    config: Dict[str, Any],
    storage_root: Optional[Path]
) -> List[Dict[str, Any]]:
    """Build the list of dataset/decode configurations that will be evaluated."""
    default_dataset_path = resolve_default_dataset_path(config, storage_root)
    phases = sorted(set(args.phases or []))
    comparison_mode = bool(args.include_base or phases)

    if not comparison_mode:
        return [{
            "label": args.dataset_label or "single_run",
            "path": default_dataset_path,
            "split": args.split,
            "decode_mode": "sliding_window" if args.window_seconds else "full_context",
            "window_seconds": args.window_seconds,
            "stride_seconds": args.window_stride_seconds,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
        }]

    dataset_specs: List[Dict[str, Any]] = []
    phase_filter_phase = args.phase_filter_phase or (phases[0] if phases else 1)
    default_min_duration, default_max_duration = PHASE_DURATION_FILTERS[phase_filter_phase]
    phase_min_duration = args.phase_min_duration
    if phase_min_duration is None:
        phase_min_duration = default_min_duration
    phase_max_duration = args.phase_max_duration
    if phase_max_duration is None:
        phase_max_duration = default_max_duration

    if not args.skip_phase_subset:
        phase_dataset_path = resolve_dataset_arg_path(
            args.phase_dataset_path,
            default_path=default_dataset_path,
            storage_root=storage_root
        )
        dataset_specs.append({
            "label": f"phase{phase_filter_phase}_subset",
            "path": phase_dataset_path,
            "split": args.phase_split or args.split,
            "decode_mode": "full_context",
            "window_seconds": None,
            "stride_seconds": None,
            "min_duration": phase_min_duration,
            "max_duration": phase_max_duration,
        })

    if not args.skip_full_dataset:
        full_dataset_path = resolve_dataset_arg_path(
            args.full_dataset_path,
            default_path=default_dataset_path,
            storage_root=storage_root
        )
        dataset_specs.append({
            "label": "full_dataset_sliding",
            "path": full_dataset_path,
            "split": args.full_split or args.split,
            "decode_mode": "sliding_window",
            "window_seconds": args.full_window_seconds,
            "stride_seconds": args.full_window_stride_seconds,
            "min_duration": None,
            "max_duration": None,
        })

    if not dataset_specs:
        raise ValueError("At least one dataset evaluation must be enabled.")

    return dataset_specs


def filter_dataset_by_duration(
    dataset: Dataset,
    *,
    dataset_root: Path,
    audio_column: str,
    target_sampling_rate: int,
    min_duration: Optional[float],
    max_duration: Optional[float],
    dataset_label: str,
) -> Dataset:
    """Keep only samples that fall inside the requested duration range."""
    if min_duration is None and max_duration is None:
        return dataset

    keep_indices: List[int] = []
    desc = f"Filtering {dataset_label}"

    for index, sample in enumerate(tqdm(dataset, desc=desc)):
        audio_input = resolve_audio_input(sample[audio_column], dataset_root)
        duration_seconds = estimate_audio_duration_seconds(
            audio_input,
            target_sampling_rate=target_sampling_rate,
            dataset_root=dataset_root,
        )

        if min_duration is not None and duration_seconds < min_duration:
            continue
        if max_duration is not None and duration_seconds > max_duration:
            continue
        keep_indices.append(index)

    if not keep_indices:
        raise ValueError(
            f"No samples remained after duration filtering for '{dataset_label}' "
            f"({min_duration}, {max_duration})."
        )

    filtered = dataset.select(keep_indices)
    print(
        f"[INFO] {dataset_label}: kept {len(filtered)} / {len(dataset)} samples "
        f"for duration range {min_duration} - {max_duration} seconds"
    )
    return filtered


def prepare_loaded_dataset_spec(
    dataset_spec: Dict[str, Any],
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Load dataset split, resolve columns, and apply optional duration filters."""
    dataset_path = dataset_spec["path"]
    split = dataset_spec["split"]
    dataset = load_dataset_split(dataset_path, split)

    text_column = resolve_column_name(
        dataset.column_names,
        preferred=config["dataset"].get("text_column", "text"),
        fallbacks=["sentence", "text", "transcript", "transcription"],
        column_kind="text"
    )
    audio_column = resolve_column_name(
        dataset.column_names,
        preferred="audio",
        fallbacks=["audio"],
        column_kind="audio"
    )

    sampling_rate = int(config.get("audio", {}).get("sampling_rate", 16000))
    dataset = filter_dataset_by_duration(
        dataset,
        dataset_root=dataset_path,
        audio_column=audio_column,
        target_sampling_rate=sampling_rate,
        min_duration=dataset_spec["min_duration"],
        max_duration=dataset_spec["max_duration"],
        dataset_label=dataset_spec["label"],
    )

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    loaded_spec = dict(dataset_spec)
    loaded_spec["dataset"] = dataset
    loaded_spec["text_column"] = text_column
    loaded_spec["audio_column"] = audio_column
    return loaded_spec


def write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    """Write a human-readable Markdown summary."""
    metrics = report["metrics"]
    duration_filter = report.get("duration_filter", {})
    sliding_window = report.get("sliding_window", {})

    lines = [
        "# Moonshine Test Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Model label: `{report['model_label']}`",
        f"- Model source: `{report['model_source']}`",
        f"- Model: `{report['model_path']}`",
        f"- Dataset label: `{report['dataset_label']}`",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Split: `{report['split']}`",
        f"- Decode mode: `{report['decode_mode']}`",
        f"- Samples: `{metrics['num_samples']}`",
    ]

    if duration_filter.get("min_seconds") is not None or duration_filter.get("max_seconds") is not None:
        lines.append(
            f"- Duration filter: `{duration_filter.get('min_seconds')}` to "
            f"`{duration_filter.get('max_seconds')}` seconds"
        )

    if report["decode_mode"] == "sliding_window":
        lines.append(f"- Window seconds: `{sliding_window.get('window_seconds')}`")
        lines.append(f"- Window stride seconds: `{sliding_window.get('stride_seconds')}`")

    lines.extend([
        "",
        "## Metrics",
        "",
        f"- WER: `{metrics['wer']:.2f}%`",
        f"- CER: `{metrics['cer']:.2f}%`",
        f"- Exact match rate: `{metrics['exact_match_rate']:.2f}%`",
        f"- Word accuracy: `{metrics['word_accuracy']:.2f}%`",
        f"- Substitutions: `{metrics['substitutions']}`",
        f"- Deletions: `{metrics['deletions']}`",
        f"- Insertions: `{metrics['insertions']}`",
        f"- Total audio duration: `{metrics['total_audio_duration']:.1f}s`",
        f"- Total inference time: `{metrics['total_inference_time']:.1f}s`",
        f"- Real-time factor: `{metrics['rtf']:.2f}x`",
        "",
        "## Most Incorrect Reference Words",
        "",
    ])

    for item in report["top_errors"]["incorrect_reference_words"]:
        lines.append(f"- `{item['reference_word']}`: {item['count']}")

    lines.extend([
        "",
        "## Most Common Substitutions",
        "",
    ])
    for item in report["top_errors"]["substitutions"]:
        lines.append(
            f"- `{item['reference_word']}` -> `{item['predicted_word']}`: {item['count']}"
        )

    lines.extend([
        "",
        "## Most Common Deletions",
        "",
    ])
    for item in report["top_errors"]["deleted_words"]:
        lines.append(f"- `{item['reference_word']}`: {item['count']}")

    lines.extend([
        "",
        "## Most Common Insertions",
        "",
    ])
    for item in report["top_errors"]["inserted_words"]:
        lines.append(f"- `{item['predicted_word']}`: {item['count']}")

    lines.extend([
        "",
        "## Worst Samples",
        "",
    ])
    for sample in report["worst_samples"]:
        lines.extend([
            f"### Sample {sample['index']}",
            f"- WER: `{sample['wer']:.2f}%`",
            f"- Duration: `{sample['duration']:.2f}s`",
            f"- Reference: `{sample['reference']}`",
            f"- Prediction: `{sample['prediction']}`",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_run(
    *,
    runner: MoonshineInference,
    model_spec: Dict[str, Any],
    dataset_spec: Dict[str, Any],
    config: Dict[str, Any],
    config_path: Path,
    keep_case: bool,
    max_new_tokens_override: Optional[int],
    suite_timestamp: str,
) -> Dict[str, Any]:
    """Evaluate one model on one dataset/decode configuration."""
    dataset = dataset_spec["dataset"]
    dataset_path = dataset_spec["path"]
    text_column = dataset_spec["text_column"]
    audio_column = dataset_spec["audio_column"]
    sampling_rate = int(config.get("audio", {}).get("sampling_rate", 16000))
    generation_config = config.get("generation", {})

    total_ref_words = 0
    total_ref_chars = 0
    total_char_edits = 0
    total_hits = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    exact_matches = 0
    total_audio_duration = 0.0
    total_inference_time = 0.0

    substitution_counter: Counter = Counter()
    deleted_word_counter: Counter = Counter()
    inserted_word_counter: Counter = Counter()
    incorrect_reference_counter: Counter = Counter()

    sample_results: List[Dict[str, Any]] = []
    decode_mode = dataset_spec["decode_mode"]

    print()
    print("=" * 80)
    print(f"Model:   {model_spec['label']} ({model_spec['path']})")
    print(f"Dataset: {dataset_spec['label']} ({dataset_path})")
    print(f"Split:   {dataset_spec['split']}")
    print(f"Decode:  {decode_mode}")
    if dataset_spec["min_duration"] is not None or dataset_spec["max_duration"] is not None:
        print(
            f"Filter:  {dataset_spec['min_duration']} - "
            f"{dataset_spec['max_duration']} seconds"
        )
    if decode_mode == "sliding_window":
        print(
            f"Window:  {dataset_spec['window_seconds']}s / "
            f"{dataset_spec['stride_seconds'] or dataset_spec['window_seconds']}s"
        )
    print(f"Samples: {len(dataset)}")
    print("=" * 80)

    progress_desc = f"{model_spec['label']} on {dataset_spec['label']}"
    for index, sample in enumerate(tqdm(dataset, desc=progress_desc)):
        audio = sample[audio_column]
        reference_raw = str(sample[text_column])
        reference_text = normalize_text(reference_raw, keep_case=keep_case)
        reference_words = reference_text.split()

        duration_audio_input = resolve_audio_input(audio, dataset_path)
        duration_seconds = estimate_audio_duration_seconds(
            duration_audio_input,
            target_sampling_rate=sampling_rate,
            dataset_root=dataset_path
        )
        inference_audio_input = resolve_audio_input_for_inference(
            audio,
            target_sampling_rate=sampling_rate,
            dataset_root=dataset_path,
        )

        if decode_mode == "sliding_window":
            sliding_kwargs = {
                "sampling_rate": sampling_rate,
                "window_seconds": dataset_spec["window_seconds"],
                "stride_seconds": dataset_spec["stride_seconds"],
                "num_beams": int(generation_config.get("num_beams", 5)),
                "repetition_penalty": float(generation_config.get("repetition_penalty", 1.3)),
                "no_repeat_ngram_size": int(generation_config.get("no_repeat_ngram_size", 2)),
            }
            if max_new_tokens_override is not None:
                sliding_kwargs["max_new_tokens"] = max_new_tokens_override
            transcription_result = transcribe_with_sliding_windows(
                runner,
                inference_audio_input,
                **sliding_kwargs,
            )
            max_new_tokens_used = max_new_tokens_override
        else:
            max_new_tokens_used = max_new_tokens_override or suggest_max_new_tokens(duration_seconds)
            transcription_result = runner.transcribe(
                inference_audio_input,
                sampling_rate=sampling_rate,
                num_beams=int(generation_config.get("num_beams", 5)),
                repetition_penalty=float(generation_config.get("repetition_penalty", 1.3)),
                no_repeat_ngram_size=int(generation_config.get("no_repeat_ngram_size", 2)),
                max_new_tokens=max_new_tokens_used,
            )

        inference_time = float(transcription_result.get("inference_time", 0.0))
        prediction_raw = transcription_result["text"]

        prediction_text = normalize_text(prediction_raw, keep_case=keep_case)
        prediction_words = prediction_text.split()

        operations, counts = align_words(reference_words, prediction_words)
        char_edits = edit_distance_length(list(reference_text), list(prediction_text))

        sample_ref_words = len(reference_words)
        sample_word_errors = (
            counts["substitutions"] + counts["deletions"] + counts["insertions"]
        )
        if sample_ref_words == 0:
            sample_wer = 0.0 if sample_word_errors == 0 else 100.0
        else:
            sample_wer = 100.0 * sample_word_errors / sample_ref_words

        if sample_word_errors == 0:
            exact_matches += 1

        total_ref_words += sample_ref_words
        total_ref_chars += len(reference_text)
        total_char_edits += char_edits
        total_hits += counts["hits"]
        total_substitutions += counts["substitutions"]
        total_deletions += counts["deletions"]
        total_insertions += counts["insertions"]
        total_audio_duration += duration_seconds
        total_inference_time += inference_time

        for operation, ref_word, pred_word in operations:
            if operation == "substitute":
                substitution_counter[(ref_word, pred_word)] += 1
                incorrect_reference_counter[ref_word] += 1
            elif operation == "delete":
                deleted_word_counter[ref_word] += 1
                incorrect_reference_counter[ref_word] += 1
            elif operation == "insert":
                inserted_word_counter[pred_word] += 1

        sample_results.append({
            "index": index,
            "reference": reference_text,
            "prediction": prediction_text,
            "wer": sample_wer,
            "duration": duration_seconds,
            "inference_time": inference_time,
            "max_new_tokens": max_new_tokens_used,
            "decode_mode": decode_mode,
            "substitutions": counts["substitutions"],
            "deletions": counts["deletions"],
            "insertions": counts["insertions"],
            "word_errors": sample_word_errors,
            "segment_count": len(transcription_result.get("segments", [])),
        })

    if total_ref_words == 0:
        corpus_wer = 0.0 if (total_substitutions + total_deletions + total_insertions) == 0 else 100.0
        word_accuracy = 0.0
    else:
        corpus_wer = 100.0 * (total_substitutions + total_deletions + total_insertions) / total_ref_words
        word_accuracy = 100.0 * total_hits / total_ref_words

    if total_ref_chars == 0:
        corpus_cer = 0.0 if total_char_edits == 0 else 100.0
    else:
        corpus_cer = 100.0 * total_char_edits / total_ref_chars

    worst_samples = sorted(
        sample_results,
        key=lambda item: (item["wer"], item["word_errors"], item["duration"]),
        reverse=True
    )[:10]

    report_stem = (
        f"moonshine_eval_{sanitize_slug(model_spec['label'])}_"
        f"{sanitize_slug(dataset_spec['label'])}_{suite_timestamp}"
    )
    json_path = RESULTS_DIR / f"{report_stem}.json"
    markdown_path = RESULTS_DIR / f"{report_stem}.md"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "model_label": model_spec["label"],
        "model_source": model_spec["source"],
        "model_path": model_spec["path"],
        "phase": model_spec["phase"],
        "dataset_label": dataset_spec["label"],
        "dataset_path": str(dataset_path),
        "split": dataset_spec["split"],
        "decode_mode": decode_mode,
        "duration_filter": {
            "min_seconds": dataset_spec["min_duration"],
            "max_seconds": dataset_spec["max_duration"],
        },
        "sliding_window": {
            "window_seconds": dataset_spec["window_seconds"],
            "stride_seconds": dataset_spec["stride_seconds"] or dataset_spec["window_seconds"],
        } if decode_mode == "sliding_window" else None,
        "metrics": {
            "num_samples": len(sample_results),
            "wer": corpus_wer,
            "cer": corpus_cer,
            "exact_match_rate": 100.0 * exact_matches / len(sample_results) if sample_results else 0.0,
            "word_accuracy": word_accuracy,
            "substitutions": total_substitutions,
            "deletions": total_deletions,
            "insertions": total_insertions,
            "hits": total_hits,
            "reference_word_count": total_ref_words,
            "reference_char_count": total_ref_chars,
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_inference_time,
            "rtf": total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0,
        },
        "top_errors": {
            "incorrect_reference_words": counter_to_records(incorrect_reference_counter, "reference_word"),
            "substitutions": pair_counter_to_records(substitution_counter),
            "deleted_words": counter_to_records(deleted_word_counter, "reference_word"),
            "inserted_words": counter_to_records(inserted_word_counter, "predicted_word"),
        },
        "worst_samples": worst_samples,
        "sample_results": sample_results,
        "report_paths": {
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, markdown_path)

    print()
    print(f"Completed {model_spec['label']} on {dataset_spec['label']}")
    print(f"  WER: {corpus_wer:.2f}%")
    print(f"  CER: {corpus_cer:.2f}%")
    print(f"  Exact match: {report['metrics']['exact_match_rate']:.2f}%")
    print(f"  Word accuracy: {word_accuracy:.2f}%")
    print(f"  RTF: {report['metrics']['rtf']:.2f}x")
    print(f"  JSON report: {json_path}")
    print(f"  Markdown report: {markdown_path}")

    return report


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write CSV rows with consistent columns."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_suite_outputs(reports: List[Dict[str, Any]], suite_timestamp: str) -> Dict[str, Path]:
    """Write suite-level CSV and JSON outputs."""
    summary_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for report in reports:
        metrics = report["metrics"]
        duration_filter = report.get("duration_filter", {})
        sliding_window = report.get("sliding_window") or {}

        summary_rows.append({
            "model_label": report["model_label"],
            "phase": report["phase"] if report["phase"] is not None else "",
            "model_source": report["model_source"],
            "model_path": report["model_path"],
            "dataset_label": report["dataset_label"],
            "dataset_path": report["dataset_path"],
            "split": report["split"],
            "decode_mode": report["decode_mode"],
            "min_duration": duration_filter.get("min_seconds", ""),
            "max_duration": duration_filter.get("max_seconds", ""),
            "window_seconds": sliding_window.get("window_seconds", ""),
            "stride_seconds": sliding_window.get("stride_seconds", ""),
            "num_samples": metrics["num_samples"],
            "wer": f"{metrics['wer']:.4f}",
            "cer": f"{metrics['cer']:.4f}",
            "exact_match_rate": f"{metrics['exact_match_rate']:.4f}",
            "word_accuracy": f"{metrics['word_accuracy']:.4f}",
            "rtf": f"{metrics['rtf']:.4f}",
            "json_report": report["report_paths"]["json"],
            "markdown_report": report["report_paths"]["markdown"],
        })

        for sample in report["sample_results"]:
            sample_rows.append({
                "model_label": report["model_label"],
                "phase": report["phase"] if report["phase"] is not None else "",
                "dataset_label": report["dataset_label"],
                "split": report["split"],
                "decode_mode": report["decode_mode"],
                "sample_index": sample["index"],
                "duration": f"{sample['duration']:.4f}",
                "wer": f"{sample['wer']:.4f}",
                "word_errors": sample["word_errors"],
                "substitutions": sample["substitutions"],
                "deletions": sample["deletions"],
                "insertions": sample["insertions"],
                "inference_time": f"{sample['inference_time']:.4f}",
                "max_new_tokens": sample["max_new_tokens"] if sample["max_new_tokens"] is not None else "",
                "segment_count": sample["segment_count"],
                "reference": sample["reference"],
                "prediction": sample["prediction"],
            })

    summary_csv_path = RESULTS_DIR / f"moonshine_eval_summary_{suite_timestamp}.csv"
    samples_csv_path = RESULTS_DIR / f"moonshine_eval_samples_{suite_timestamp}.csv"
    suite_json_path = RESULTS_DIR / f"moonshine_eval_suite_{suite_timestamp}.json"

    write_csv(
        summary_csv_path,
        summary_rows,
        fieldnames=[
            "model_label",
            "phase",
            "model_source",
            "model_path",
            "dataset_label",
            "dataset_path",
            "split",
            "decode_mode",
            "min_duration",
            "max_duration",
            "window_seconds",
            "stride_seconds",
            "num_samples",
            "wer",
            "cer",
            "exact_match_rate",
            "word_accuracy",
            "rtf",
            "json_report",
            "markdown_report",
        ],
    )
    write_csv(
        samples_csv_path,
        sample_rows,
        fieldnames=[
            "model_label",
            "phase",
            "dataset_label",
            "split",
            "decode_mode",
            "sample_index",
            "duration",
            "wer",
            "word_errors",
            "substitutions",
            "deletions",
            "insertions",
            "inference_time",
            "max_new_tokens",
            "segment_count",
            "reference",
            "prediction",
        ],
    )

    suite_json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "num_runs": len(reports),
                "summary_csv": str(summary_csv_path),
                "samples_csv": str(samples_csv_path),
                "runs": [
                    {
                        "model_label": report["model_label"],
                        "dataset_label": report["dataset_label"],
                        "wer": report["metrics"]["wer"],
                        "cer": report["metrics"]["cer"],
                        "num_samples": report["metrics"]["num_samples"],
                        "json_report": report["report_paths"]["json"],
                        "markdown_report": report["report_paths"]["markdown"],
                    }
                    for report in reports
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": summary_csv_path,
        "samples_csv": samples_csv_path,
        "suite_json": suite_json_path,
    }


def release_runner(runner: MoonshineInference) -> None:
    """Release model resources between comparison runs."""
    del runner
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_suite_summary(reports: List[Dict[str, Any]], suite_outputs: Dict[str, Path]) -> None:
    """Print a compact suite summary to the terminal."""
    print()
    print("=" * 80)
    print("SUITE SUMMARY")
    print("=" * 80)
    for report in reports:
        metrics = report["metrics"]
        print(
            f"{report['model_label']:>10} | "
            f"{report['dataset_label']:<20} | "
            f"WER {metrics['wer']:>7.2f}% | "
            f"CER {metrics['cer']:>7.2f}% | "
            f"samples {metrics['num_samples']:>5}"
        )
    print("=" * 80)
    print(f"Summary CSV: {suite_outputs['summary_csv']}")
    print(f"Samples CSV: {suite_outputs['samples_csv']}")
    print(f"Suite JSON:   {suite_outputs['suite_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Moonshine models on one split or a small comparison suite."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config YAML (default: English-accent config)."
    )
    parser.add_argument(
        "--model-source",
        choices=["base", "finetuned", "path"],
        default="finetuned",
        help="Single-run model source when comparison mode is not used."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom model path or model id when --model-source=path."
    )
    parser.add_argument(
        "--model-label",
        type=str,
        help="Optional label for --model-source=path."
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include the base model in comparison mode."
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        help="Phase checkpoints to evaluate, resolved from training.output_dir-phaseN/final."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Default dataset split to evaluate (default: test)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally limit the number of evaluated samples after any filtering."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 on CUDA."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override generation length cap."
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Preserve original text casing when computing metrics."
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        help="Optional label for single-run mode."
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Optional minimum sample duration for single-run mode."
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Optional maximum sample duration for single-run mode."
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        help="Optional sliding-window size for single-run mode."
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        help="Optional sliding-window stride for single-run mode."
    )
    parser.add_argument(
        "--phase-dataset-path",
        type=str,
        help="Optional dataset path for the phase-subset evaluation in comparison mode."
    )
    parser.add_argument(
        "--phase-split",
        type=str,
        help="Optional split for the phase-subset evaluation (defaults to --split)."
    )
    parser.add_argument(
        "--phase-filter-phase",
        type=int,
        choices=[1, 2, 3],
        help="Use the standard duration range for this phase when filtering the subset dataset."
    )
    parser.add_argument(
        "--phase-min-duration",
        type=float,
        help="Override the phase-subset minimum duration."
    )
    parser.add_argument(
        "--phase-max-duration",
        type=float,
        help="Override the phase-subset maximum duration."
    )
    parser.add_argument(
        "--full-dataset-path",
        type=str,
        help="Optional dataset path for full-dataset sliding-window evaluation."
    )
    parser.add_argument(
        "--full-split",
        type=str,
        help="Optional split for the full-dataset evaluation (defaults to --split)."
    )
    parser.add_argument(
        "--full-window-seconds",
        type=float,
        default=5.0,
        help="Sliding-window size for full-dataset comparison runs (default: 5.0)."
    )
    parser.add_argument(
        "--full-window-stride-seconds",
        type=float,
        help="Sliding-window stride for full-dataset comparison runs (default: same as window size)."
    )
    parser.add_argument(
        "--skip-phase-subset",
        action="store_true",
        help="Skip the duration-filtered phase-subset evaluation in comparison mode."
    )
    parser.add_argument(
        "--skip-full-dataset",
        action="store_true",
        help="Skip the full-dataset sliding-window evaluation in comparison mode."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve(strict=False)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.window_seconds is not None and args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.window_stride_seconds is not None and args.window_stride_seconds <= 0:
        raise ValueError("--window-stride-seconds must be > 0")
    if args.full_window_seconds <= 0:
        raise ValueError("--full-window-seconds must be > 0")
    if args.full_window_stride_seconds is not None and args.full_window_stride_seconds <= 0:
        raise ValueError("--full-window-stride-seconds must be > 0")

    storage_root = resolve_storage_root(config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_specs = build_model_specs(args, config, storage_root)
    dataset_specs = build_dataset_specs(args, config, storage_root)
    loaded_dataset_specs = [
        prepare_loaded_dataset_spec(dataset_spec, config=config, args=args)
        for dataset_spec in dataset_specs
    ]

    suite_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports: List[Dict[str, Any]] = []

    for model_spec in model_specs:
        runner = MoonshineInference(
            model_path=model_spec["path"],
            device=args.device,
            fp16=args.fp16,
        )

        try:
            for dataset_spec in loaded_dataset_specs:
                report = evaluate_run(
                    runner=runner,
                    model_spec=model_spec,
                    dataset_spec=dataset_spec,
                    config=config,
                    config_path=config_path,
                    keep_case=args.keep_case,
                    max_new_tokens_override=args.max_new_tokens,
                    suite_timestamp=suite_timestamp,
                )
                reports.append(report)
        finally:
            release_runner(runner)

    suite_outputs = write_suite_outputs(reports, suite_timestamp)
    print_suite_summary(reports, suite_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
