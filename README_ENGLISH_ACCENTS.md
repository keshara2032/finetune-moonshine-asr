# Fine-Tuning Moonshine On English Accent Data

This guide shows the simplest way to fine-tune the base English Moonshine model on a new English dataset with different accents.

It is written for the config in [configs/english_accent_local_no_curriculum.yaml](configs/english_accent_local_no_curriculum.yaml).

## What The Training Expects

Use a Hugging Face `DatasetDict` saved with `save_to_disk()`.

Your saved dataset should have:

- a `train` split
- a `test` split
- an `audio` column
- one text column named one of: `text`, `transcript`, `transcription`, or `sentence`

The loader now handles:

- automatic transcript-column detection
- automatic duration computation
- audio normalization during preprocessing

You do not need to manually create a `duration` column.

## Recommended Folder Layout

```text
/scratch/cjh9fw/moonshine/
├── cache/
├── logs/
│   └── moonshine-en-accents/
└── results/
    └── moonshine-en-accents/
```

```text
/scratch/cjh9fw/finetune-moonshine-asr/
├── dataset_dict.json
├── train/
└── test/
```

The default config already points to:

```yaml
storage:
  root_dir: "/scratch/cjh9fw/moonshine"

dataset:
  type: "local"
  path: "/scratch/cjh9fw/finetune-moonshine-asr"
  text_column: "text"
```

If your dataset lives somewhere else, change `dataset.path` to that absolute path.

## Create The Dataset

### Option 1: From A CSV

If you already have a CSV like:

```csv
audio,text
/absolute/path/audio_0001.wav,hello how are you
/absolute/path/audio_0002.wav,please call me later
```

you can build the dataset like this:

```python
from datasets import Dataset, DatasetDict, Audio
import pandas as pd

df = pd.read_csv("my_english_accents.csv")

dataset = Dataset.from_pandas(df[["audio", "text"]])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"],
})

dataset_dict.save_to_disk("/scratch/cjh9fw/finetune-moonshine-asr")
```

### Option 2: From Audio Files + A Transcript File

If you have `.wav` files and a CSV with filenames and transcripts:

```csv
filename,text
audio_0001.wav,hello how are you
audio_0002.wav,please call me later
```

use:

```python
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

audio_dir = Path("./my_audio")
df = pd.read_csv("transcripts.csv")

df["audio"] = df["filename"].apply(lambda name: str(audio_dir / name))

dataset = Dataset.from_pandas(df[["audio", "text"]])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"],
})

dataset_dict.save_to_disk("/scratch/cjh9fw/finetune-moonshine-asr")
```

## Audio And Transcript Guidelines

For best results:

- keep audio at 16 kHz
- use mono audio when possible
- keep clips roughly between 1 and 20 seconds
- make transcripts match the spoken words closely
- keep transcript formatting consistent across the dataset

Good:

```text
i need to book a flight for tomorrow
```

Less good:

```text
I need to book a flight for tomorrow.
```

Either style can work, but consistency matters more than the exact style.

## Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Fine-Tuning

Use the included English-accent config:

```bash
python train.py --config configs/english_accent_local_no_curriculum.yaml
```

That will:

- load `UsefulSensors/moonshine-tiny`
- load your local dataset
- preprocess audio and text
- cache Hugging Face downloads in `/scratch/cjh9fw/moonshine/cache`
- save checkpoints to `/scratch/cjh9fw/moonshine/results/moonshine-en-accents`
- save the final model to `/scratch/cjh9fw/moonshine/results/moonshine-en-accents/final`

## Evaluate The Model

After training finishes:

```bash
python scripts/evaluate.py \
  --model /scratch/cjh9fw/moonshine/results/moonshine-en-accents/final \
  --dataset /scratch/cjh9fw/finetune-moonshine-asr \
  --split test \
  --text-column text
```

If your original dataset uses `transcript` or `transcription` instead of `text`, change `--text-column`.

For a simpler test-set report with timestamped outputs under `./results/`:

```bash
python scripts/test_model.py --model-source base
python scripts/test_model.py --model-source finetuned
```

That script uses the current English config and test split by default, prints WER/CER plus common word mistakes, and saves JSON + Markdown reports in the repo `results/` folder.

## Run Inference

For a single audio file:

```bash
python scripts/inference.py \
  --model /scratch/cjh9fw/moonshine/results/moonshine-en-accents/final \
  --audio ./sample.wav
```

## Useful Config Knobs

The main file to edit is:

[configs/english_accent_local_no_curriculum.yaml](configs/english_accent_local_no_curriculum.yaml)

Common changes:

- `storage.root_dir`: shared scratch location for caches, logs, and results
- `dataset.path`: absolute path to your saved `DatasetDict`
- `dataset.text_column`: your transcript column name
- `audio.min_duration` / `audio.max_duration`: filter very short or very long clips
- `training.per_device_train_batch_size`: lower this if you run out of GPU memory
- `training.gradient_accumulation_steps`: raise this if you lower batch size
- `training.max_steps`: increase this for larger datasets

## If Training Fails Early

Check these first:

1. Your dataset must be a saved `DatasetDict`, not a single `Dataset`.
2. The `audio` column must be cast with `Audio(sampling_rate=16000)`.
3. Your transcript column must be one of `text`, `transcript`, `transcription`, or `sentence`.
4. Your config path must point to the saved dataset directory, not just the raw audio folder.

## Minimal End-To-End Example

```bash
python train.py --config configs/english_accent_local_no_curriculum.yaml

python scripts/evaluate.py \
  --model /scratch/cjh9fw/moonshine/results/moonshine-en-accents/final \
  --dataset /scratch/cjh9fw/finetune-moonshine-asr \
  --split test \
  --text-column text

python scripts/inference.py \
  --model /scratch/cjh9fw/moonshine/results/moonshine-en-accents/final \
  --audio ./sample.wav
```
