"""
Data loading utilities for Moonshine ASR fine-tuning.

Supports:
- HuggingFace datasets (Common Voice, LibriSpeech, etc.)
- Custom CSV datasets
- Local audio files
- Local datasets saved with save_to_disk()
"""

from datasets import load_dataset, DatasetDict, Dataset, Audio, load_from_disk
from typing import Optional, Union, Dict, Any
import pandas as pd
from pathlib import Path

from moonshine_ft.utils.preprocessing import normalize_audio


class MoonshineDataLoader:
    """
    Flexible data loader for Moonshine fine-tuning.

    Supports multiple dataset formats and provides unified interface.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sampling_rate: int = 16000
    ):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary (optional)
            sampling_rate: Target audio sampling rate (default: 16000)
        """
        self.config = config or {}
        self.sampling_rate = sampling_rate

    def load_common_voice(
        self,
        language: str,
        version: str = "13_0",
        train_split: str = "train+validation",
        test_split: str = "test",
        cache_dir: Optional[str] = None,
        use_auth_token: bool = True
    ) -> DatasetDict:
        """
        Load Common Voice dataset.

        Args:
            language: Language code (e.g., 'fr', 'en', 'de')
            version: Common Voice version (default: '13_0')
            train_split: Training split specification
            test_split: Test split specification
            cache_dir: Cache directory for downloaded data
            use_auth_token: Use HuggingFace authentication token

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        dataset_name = f"mozilla-foundation/common_voice_{version}"

        print(f"\nLoading Common Voice {version} - {language}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Train split: {train_split}")
        print(f"  Test split: {test_split}")

        common_voice = DatasetDict()

        # Load train split
        common_voice["train"] = load_dataset(
            dataset_name,
            language,
            split=train_split,
            cache_dir=cache_dir,
            token=use_auth_token
        )

        # Load test split
        common_voice["test"] = load_dataset(
            dataset_name,
            language,
            split=test_split,
            cache_dir=cache_dir,
            token=use_auth_token
        )

        # Select only needed columns
        # Common Voice has: audio, sentence, and metadata
        common_voice = common_voice.select_columns(["audio", "sentence"])

        # Cast audio to target sampling rate
        common_voice = common_voice.cast_column(
            "audio",
            Audio(sampling_rate=self.sampling_rate)
        )

        print(f"\nDataset loaded:")
        print(f"  Train samples: {len(common_voice['train']):,}")
        print(f"  Test samples: {len(common_voice['test']):,}")

        # Calculate total duration
        train_duration = sum(
            len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            for sample in common_voice["train"].select(range(min(1000, len(common_voice["train"]))))
        )
        avg_duration = train_duration / min(1000, len(common_voice["train"]))
        estimated_total = avg_duration * len(common_voice["train"]) / 3600

        print(f"  Estimated train duration: {estimated_total:.1f} hours")

        return common_voice

    def load_from_csv(
        self,
        train_csv: str,
        test_csv: str,
        audio_column: str = "audio_path",
        text_column: str = "transcription",
        base_path: Optional[str] = None
    ) -> DatasetDict:
        """
        Load dataset from CSV files.

        CSV format:
            audio_path, transcription
            /path/to/audio1.wav, "hello world"
            /path/to/audio2.wav, "this is a test"

        Args:
            train_csv: Path to training CSV file
            test_csv: Path to test CSV file
            audio_column: Name of column containing audio file paths
            text_column: Name of column containing transcriptions
            base_path: Base path for relative audio paths (optional)

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print(f"\nLoading dataset from CSV:")
        print(f"  Train: {train_csv}")
        print(f"  Test: {test_csv}")

        # Read CSVs
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        audio_column = self._resolve_column_name(
            train_df.columns,
            preferred=audio_column,
            candidates=["audio", "audio_path", "path", "file", "filepath"],
            column_kind="audio"
        )
        text_column = self._resolve_column_name(
            train_df.columns,
            preferred=text_column,
            candidates=["sentence", "transcript", "transcription", "text"],
            column_kind="text"
        )

        # Adjust paths if base_path provided
        if base_path:
            base_path = Path(base_path)
            train_df[audio_column] = train_df[audio_column].apply(
                lambda p: str(base_path / p) if not Path(p).is_absolute() else p
            )
            test_df[audio_column] = test_df[audio_column].apply(
                lambda p: str(base_path / p) if not Path(p).is_absolute() else p
            )

        # Rename columns to standard names
        train_df = train_df.rename(columns={
            audio_column: "audio",
            text_column: "sentence"
        })
        test_df = test_df.rename(columns={
            audio_column: "audio",
            text_column: "sentence"
        })

        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df[["audio", "sentence"]])
        test_dataset = Dataset.from_pandas(test_df[["audio", "sentence"]])

        # Cast audio column to Audio type
        train_dataset = train_dataset.cast_column(
            "audio",
            Audio(sampling_rate=self.sampling_rate)
        )
        test_dataset = test_dataset.cast_column(
            "audio",
            Audio(sampling_rate=self.sampling_rate)
        )

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        print(f"\nDataset loaded:")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")

        return dataset_dict

    def load_librispeech(
        self,
        subset: str = "clean",
        cache_dir: Optional[str] = None
    ) -> DatasetDict:
        """
        Load LibriSpeech dataset.

        Args:
            subset: Subset name ('clean', 'other', 'clean-100', 'clean-360')
            cache_dir: Cache directory for downloaded data

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print(f"\nLoading LibriSpeech - {subset}")

        if subset == "clean":
            train_split = "train.clean.100+train.clean.360"
            test_split = "test.clean"
        elif subset == "other":
            train_split = "train.other.500"
            test_split = "test.other"
        else:
            train_split = f"train.{subset}"
            test_split = "test.clean"

        librispeech = DatasetDict()

        librispeech["train"] = load_dataset(
            "librispeech_asr",
            "clean",
            split=train_split,
            cache_dir=cache_dir
        )

        librispeech["test"] = load_dataset(
            "librispeech_asr",
            "clean",
            split=test_split,
            cache_dir=cache_dir
        )

        # Rename columns to match Common Voice format
        librispeech = librispeech.rename_column("text", "sentence")

        # Cast audio to target sampling rate
        librispeech = librispeech.cast_column(
            "audio",
            Audio(sampling_rate=self.sampling_rate)
        )

        print(f"\nDataset loaded:")
        print(f"  Train samples: {len(librispeech['train']):,}")
        print(f"  Test samples: {len(librispeech['test']):,}")

        return librispeech

    def load_mls(
        self,
        language: str = "french",
        split_train: str = "train",
        split_test: str = "test",
        cache_dir: Optional[str] = None
    ) -> DatasetDict:
        """
        Load Multilingual LibriSpeech (MLS) dataset.

        Args:
            language: Language subset ('french', 'german', 'dutch', 'spanish', 'italian', 'portuguese', 'polish')
            split_train: Training split name
            split_test: Test split name
            cache_dir: Cache directory for downloaded data

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print(f"\nLoading Multilingual LibriSpeech - {language}")

        mls = DatasetDict()

        mls["train"] = load_dataset(
            "facebook/multilingual_librispeech",
            language,
            split=split_train,
            cache_dir=cache_dir
        )

        mls["test"] = load_dataset(
            "facebook/multilingual_librispeech",
            language,
            split=split_test,
            cache_dir=cache_dir
        )

        # Rename columns to match Common Voice format
        mls = mls.rename_column("transcript", "sentence")

        # Cast audio to target sampling rate
        mls = mls.cast_column(
            "audio",
            Audio(sampling_rate=self.sampling_rate)
        )

        print(f"\nDataset loaded:")
        print(f"  Train samples: {len(mls['train']):,}")
        print(f"  Test samples: {len(mls['test']):,}")

        return mls

    def load_local(
        self,
        path: str,
        text_column: str = "transcript"
    ) -> DatasetDict:
        """
        Load dataset from local directory (saved with save_to_disk()).

        Args:
            path: Path to dataset directory
            text_column: Name of text column (default: 'transcript')

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        print(f"\nLoading local dataset from: {path}")

        dataset = load_from_disk(path)

        # Check if it's already a DatasetDict or a single Dataset
        if isinstance(dataset, DatasetDict):
            # Already has train/test splits
            print(f"\nDataset loaded:")
            print(f"  Train samples: {len(dataset['train']):,}")
            print(f"  Test samples: {len(dataset['test']):,}")

            resolved_text_column = self._resolve_column_name(
                dataset["train"].column_names,
                preferred=text_column,
                candidates=["sentence", "transcript", "transcription", "text"],
                column_kind="text"
            )

            # Rename text column if needed
            if resolved_text_column != "sentence":
                dataset = DatasetDict({
                    "train": dataset["train"].rename_column(resolved_text_column, "sentence"),
                    "test": dataset["test"].rename_column(resolved_text_column, "sentence")
                })

        else:
            # Single dataset - should have been split already
            raise ValueError(
                f"Expected DatasetDict with 'train' and 'test' splits, got {type(dataset)}. "
                "Please create train/test split before saving to disk."
            )

        # Cast audio to target sampling rate if needed
        if "audio" in dataset["train"].column_names:
            dataset = dataset.cast_column(
                "audio",
                Audio(sampling_rate=self.sampling_rate)
            )

        return dataset

    def _resolve_column_name(
        self,
        columns,
        preferred: str,
        candidates,
        column_kind: str
    ) -> str:
        """
        Resolve a column name from a preferred value plus common fallbacks.

        This keeps the training path friendly to plain datasets whose columns
        are named ``text``/``transcription`` instead of the repo's internal
        ``sentence`` convention.
        """
        if preferred in columns:
            return preferred

        for candidate in candidates:
            if candidate in columns:
                print(
                    f"  [INFO] Using {column_kind} column '{candidate}' "
                    f"(requested '{preferred}' was not found)"
                )
                return candidate

        raise ValueError(
            f"Could not find a {column_kind} column. "
            f"Tried '{preferred}' and fallbacks {candidates}. "
            f"Available columns: {list(columns)}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MoonshineDataLoader':
        """
        Create data loader from configuration dictionary.

        Args:
            config: Configuration dictionary with 'dataset' key

        Returns:
            Initialized MoonshineDataLoader
        """
        dataset_config = config.get('dataset', {})
        sampling_rate = config.get('audio', {}).get('sampling_rate', 16000)

        return cls(config=config, sampling_rate=sampling_rate)

    def load_dataset(self) -> DatasetDict:
        """
        Load dataset based on configuration.

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        dataset_config = self.config.get('dataset', {})
        dataset_type = dataset_config.get('type', 'common_voice')

        if dataset_type == 'common_voice':
            return self.load_common_voice(
                language=dataset_config.get('language', 'en'),
                version=dataset_config.get('version', '13_0'),
                train_split=dataset_config.get('train_split', 'train+validation'),
                test_split=dataset_config.get('test_split', 'test'),
                cache_dir=dataset_config.get('cache_dir'),
                use_auth_token=dataset_config.get('use_auth_token', True)
            )
        elif dataset_type == 'csv':
            return self.load_from_csv(
                train_csv=dataset_config.get('train_csv'),
                test_csv=dataset_config.get('test_csv'),
                audio_column=dataset_config.get('audio_column', 'audio_path'),
                text_column=dataset_config.get('text_column', 'transcription'),
                base_path=dataset_config.get('base_path')
            )
        elif dataset_type == 'librispeech':
            return self.load_librispeech(
                subset=dataset_config.get('subset', 'clean'),
                cache_dir=dataset_config.get('cache_dir')
            )
        elif dataset_type == 'mls':
            return self.load_mls(
                language=dataset_config.get('language', 'french'),
                split_train=dataset_config.get('split_train', 'train'),
                split_test=dataset_config.get('split_test', 'test'),
                cache_dir=dataset_config.get('cache_dir')
            )
        elif dataset_type == 'local':
            return self.load_local(
                path=dataset_config.get('path'),
                text_column=dataset_config.get('text_column', 'transcript')
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def prepare_dataset(
        self,
        dataset: Dataset,
        processor,
        text_column: str = "sentence"
    ) -> Dataset:
        """
        Prepare dataset for training (feature extraction and tokenization).

        Args:
            dataset: Input dataset with 'audio' and text columns
            processor: HuggingFace processor (feature extractor + tokenizer)
            text_column: Name of text column

        Returns:
            Processed dataset with input_values, labels, and duration
        """
        preprocessing_config = self.config.get('preprocessing', {})
        should_normalize = preprocessing_config.get('normalize_audio', False)
        target_rms = preprocessing_config.get('target_rms', 0.075)

        def prepare_example(batch):
            # Process audio
            audio = batch["audio"]
            audio_array = audio["array"]

            if should_normalize:
                audio_array = normalize_audio(audio_array, target_rms=target_rms)

            inputs = processor(
                audio_array,
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            )
            batch["input_values"] = inputs.input_values[0]

            # Tokenize text (no BOS, add EOS)
            labels = processor.tokenizer(
                batch[text_column],
                add_special_tokens=False
            ).input_ids
            batch["labels"] = labels + [2]  # Add EOS token

            # Store audio duration for curriculum filtering and length bucketing
            batch["duration"] = len(audio_array) / audio["sampling_rate"]
            batch["input_length"] = len(audio_array)

            return batch

        print("\nPreparing dataset (feature extraction + tokenization)...")

        prepared = dataset.map(
            prepare_example,
            remove_columns=dataset.column_names,
            num_proc=self.config.get('preprocessing', {}).get('num_proc', 4)
        )

        print(f"  Processed {len(prepared):,} samples")

        return prepared

    def filter_by_duration(
        self,
        dataset: Dataset,
        max_duration: float = 30.0,
        min_duration: float = 0.1
    ) -> Dataset:
        """
        Filter dataset by audio duration.

        Args:
            dataset: Input dataset with 'duration' column
            max_duration: Maximum duration in seconds
            min_duration: Minimum duration in seconds

        Returns:
            Filtered dataset
        """
        if 'duration' not in dataset.column_names and 'audio_duration' not in dataset.column_names:
            print("\nNo duration column found. Computing durations from audio...")

            def add_duration(example):
                audio = example["audio"]
                example["duration"] = len(audio["array"]) / audio["sampling_rate"]
                return example

            dataset = dataset.map(
                add_duration,
                num_proc=self.config.get('preprocessing', {}).get('num_proc', 4)
            )

        # Determine which duration column exists
        duration_col = 'duration' if 'duration' in dataset.column_names else 'audio_duration'

        def is_valid_duration(duration):
            # When using input_columns, we receive the column value directly
            return min_duration <= duration <= max_duration

        # Use input_columns to avoid decoding audio
        filtered = dataset.filter(is_valid_duration, input_columns=[duration_col])

        print(f"\nFiltering by duration ({min_duration}s - {max_duration}s):")
        print(f"  Original: {len(dataset):,} samples")
        print(f"  Filtered: {len(filtered):,} samples")
        print(f"  Removed: {len(dataset) - len(filtered):,} samples")

        return filtered
