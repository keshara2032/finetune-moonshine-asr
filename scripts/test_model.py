#!/usr/bin/env python3
"""
Simple Moonshine test-set evaluation script.

This script is intentionally lightweight:
- defaults to the current English-accent config
- can compare the base Moonshine model or your fine-tuned model
- evaluates the configured test split
- prints a compact summary to the terminal
- saves timestamped JSON and Markdown reports to ./results
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
import soundfile as sf
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import MoonshineInference


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "english_accent_local_no_curriculum.yaml"
RESULTS_DIR = REPO_ROOT / "results"
WHITESPACE_RE = re.compile(r"\s+")


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
    The old 150-token ceiling is too small for 2-3 minute samples.
    """
    return max(150, min(int(duration_seconds * 5), 1024))


def normalize_text(text: str, keep_case: bool) -> str:
    """Normalize whitespace, and lowercase by default for cleaner WER stats."""
    normalized = WHITESPACE_RE.sub(" ", str(text).strip())
    return normalized if keep_case else normalized.lower()


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


def write_markdown_report(report: Dict[str, object], output_path: Path) -> None:
    """Write a human-readable Markdown summary."""
    metrics = report["metrics"]

    lines = [
        f"# Moonshine Test Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Model source: `{report['model_source']}`",
        f"- Model: `{report['model_path']}`",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Split: `{report['split']}`",
        f"- Samples: `{metrics['num_samples']}`",
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
    ]

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Moonshine on the configured test dataset.")
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
        help="Which model to evaluate."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom model path or model id when --model-source=path."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally limit the number of evaluated samples."
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
        help="Override generation length cap. By default, the script picks a larger cap for long clips."
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Preserve original text casing when computing metrics."
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve(strict=False)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    storage_root = None
    if config.get("storage", {}).get("root_dir"):
        storage_root = Path(config["storage"]["root_dir"]).expanduser().resolve(strict=False)

    dataset_path = resolve_path(config["dataset"]["path"], storage_root=storage_root)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if args.model_source == "base":
        resolved_model = config["model"]["name"]
    elif args.model_source == "finetuned":
        output_dir = resolve_path(
            config["training"]["output_dir"],
            storage_root=storage_root,
            prefer_storage_root=True
        )
        if output_dir is None:
            raise ValueError("Could not resolve fine-tuned output directory from config.")
        resolved_model = str(output_dir / "final")
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --model-source=path")
        resolved_model = args.model_path

    if args.model_source == "finetuned":
        model_path_obj = Path(resolved_model).expanduser().resolve(strict=False)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path_obj}")
        resolved_model = str(model_path_obj)
    elif args.model_source == "path":
        model_path_obj = Path(resolved_model).expanduser().resolve(strict=False)
        if model_path_obj.exists():
            resolved_model = str(model_path_obj)

    dataset = load_dataset_split(dataset_path, args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

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
    runner = MoonshineInference(
        model_path=resolved_model,
        device=args.device,
        fp16=args.fp16,
    )

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

    sample_results: List[Dict[str, object]] = []

    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")

    for index, sample in enumerate(tqdm(dataset, desc="Testing")):
        audio = sample[audio_column]
        reference_raw = str(sample[text_column])
        reference_text = normalize_text(reference_raw, keep_case=args.keep_case)
        reference_words = reference_text.split()

        audio_input = resolve_audio_input(audio, dataset_path)
        duration_seconds = estimate_audio_duration_seconds(
            audio_input,
            target_sampling_rate=int(config.get("audio", {}).get("sampling_rate", 16000)),
            dataset_root=dataset_path
        )
        max_new_tokens = args.max_new_tokens or suggest_max_new_tokens(duration_seconds)

        start_time = time.time()
        transcription_result = runner.transcribe(
            audio_input,
            sampling_rate=int(config.get("audio", {}).get("sampling_rate", 16000)),
            num_beams=int(config.get("generation", {}).get("num_beams", 5)),
            repetition_penalty=float(config.get("generation", {}).get("repetition_penalty", 1.3)),
            no_repeat_ngram_size=int(config.get("generation", {}).get("no_repeat_ngram_size", 2)),
            max_new_tokens=max_new_tokens
        )
        inference_time = time.time() - start_time
        prediction_raw = transcription_result["text"]

        prediction_text = normalize_text(prediction_raw, keep_case=args.keep_case)
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
            "max_new_tokens": max_new_tokens,
            "substitutions": counts["substitutions"],
            "deletions": counts["deletions"],
            "insertions": counts["insertions"],
            "word_errors": sample_word_errors,
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"moonshine_test_{args.model_source}_{timestamp}"
    json_path = RESULTS_DIR / f"{report_stem}.json"
    markdown_path = RESULTS_DIR / f"{report_stem}.md"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "model_source": args.model_source,
        "model_path": resolved_model,
        "dataset_path": str(dataset_path),
        "split": args.split,
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
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, markdown_path)

    print("\nEvaluation complete")
    print(f"  WER: {corpus_wer:.2f}%")
    print(f"  CER: {corpus_cer:.2f}%")
    print(f"  Exact match: {report['metrics']['exact_match_rate']:.2f}%")
    print(f"  Word accuracy: {word_accuracy:.2f}%")
    print(f"  RTF: {report['metrics']['rtf']:.2f}x")
    print(f"  JSON report: {json_path}")
    print(f"  Markdown report: {markdown_path}")

    if report["top_errors"]["incorrect_reference_words"]:
        top_word = report["top_errors"]["incorrect_reference_words"][0]
        print(f"  Most incorrect word: {top_word['reference_word']} ({top_word['count']} times)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
