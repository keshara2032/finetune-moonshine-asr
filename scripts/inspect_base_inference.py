#!/usr/bin/env python3
"""
Run the repo's base Moonshine inference code on a few dataset samples.

This is intentionally simple:
- defaults to the current English config
- loads the saved local DatasetDict
- grabs a few samples from the requested split
- runs scripts/inference.py's MoonshineInference class
- prints reference text and model transcription
"""

import argparse
import sys
from pathlib import Path

import yaml
from datasets import DatasetDict, load_from_disk


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import MoonshineInference, transcribe_with_sliding_windows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "english_accent_local_no_curriculum.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect base-model transcriptions on a few dataset samples."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config (default: current English config).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to inspect (default: test).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to transcribe (default: 3).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index within the split (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device for inference (default: auto-detect).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 on CUDA.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for generation length. By default uses inference.py's default behavior.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        help="Optional fixed window size for simple sliding-window transcription.",
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        help="Optional stride for sliding-window transcription (default: same as window size).",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def resolve_audio_path(audio_value, dataset_root: Path) -> Path:
    if isinstance(audio_value, dict) and "path" in audio_value:
        audio_value = audio_value["path"]

    if isinstance(audio_value, (str, Path)):
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (dataset_root / audio_path).resolve(strict=False)
        return audio_path

    raise TypeError(f"Unsupported audio value type: {type(audio_value)}")


def resolve_text_column(columns, preferred: str) -> str:
    if preferred in columns:
        return preferred

    for fallback in ("sentence", "text", "transcript", "transcription"):
        if fallback in columns:
            return fallback

    raise ValueError(
        f"Could not find a text column. Preferred '{preferred}', available columns: {list(columns)}"
    )


def load_split(dataset_path: Path, split: str):
    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
        return dataset[split]
    raise ValueError(f"Expected DatasetDict at {dataset_path}, got {type(dataset)}")


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve(strict=False)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_path = resolve_repo_path(config["dataset"]["path"])
    dataset = load_split(dataset_path, args.split)
    text_column = resolve_text_column(
        dataset.column_names,
        config["dataset"].get("text_column", "text"),
    )

    total_samples = len(dataset)
    start_index = max(0, min(args.start_index, max(total_samples - 1, 0)))
    end_index = min(start_index + max(args.num_samples, 1), total_samples)
    sample_indices = list(range(start_index, end_index))

    if not sample_indices:
        raise ValueError(f"No samples available in split '{args.split}'")

    model_name = config["model"]["name"]
    sampling_rate = int(config.get("audio", {}).get("sampling_rate", 16000))

    runner = MoonshineInference(
        model_path=model_name,
        device=args.device,
        fp16=args.fp16,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split}")
    print(f"Model: {model_name}")
    print(f"Indices: {sample_indices}\n")

    for sample_index in sample_indices:
        sample = dataset[sample_index]
        audio_path = resolve_audio_path(sample["audio"], dataset_path)
        reference = str(sample[text_column]).strip()

        if args.window_seconds:
            result = transcribe_with_sliding_windows(
                runner,
                audio_path,
                sampling_rate=sampling_rate,
                window_seconds=args.window_seconds,
                stride_seconds=args.window_stride_seconds,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            result = runner.transcribe(
                audio_path,
                sampling_rate=sampling_rate,
                max_new_tokens=args.max_new_tokens,
            )

        print("=" * 80)
        print(f"Index: {sample_index}")
        print(f"Audio: {audio_path}")
        print(f"Duration: {result['audio_duration']:.2f}s")
        print(f"Inference Time: {result['inference_time']:.2f}s")
        print(f"RTF: {result['rtf']:.3f}")
        print("\nReference:")
        print(reference)
        if "segments" in result:
            print("\nWindowed segments:")
            for segment in result["segments"]:
                print(
                    f"[{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] "
                    f"{segment['text']}"
                )
            print("\nCombined Prediction:")
        else:
            print("\nPrediction:")
        print(result["text"])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
