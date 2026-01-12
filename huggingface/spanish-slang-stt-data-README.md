---
language:
- es
license: cc-by-4.0
size_categories:
- 10K<n<100K
task_categories:
- automatic-speech-recognition
task_ids:
- speech-recognition
tags:
- speech
- audio
- spanish
- regional-dialects
- whisper
- fine-tuning
- mexico
- spain
- argentina
- chile
pretty_name: Spanish Regional STT Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: data/splits/train.json
  - split: validation
    path: data/splits/val.json
  - split: test
    path: data/splits/test.json
dataset_info:
  features:
  - name: audio_path
    dtype: string
  - name: transcript
    dtype: string
  - name: duration
    dtype: float64
  - name: language
    dtype: string
  - name: region
    dtype: string
  - name: dataset
    dtype: string
  - name: original_path
    dtype: string
  - name: dialect_markers
    sequence: string
  - name: speaker_id
    dtype: string
  splits:
  - name: train
    num_examples: 31438
  - name: validation
    num_examples: 3929
  - name: test
    num_examples: 3931
---

# Spanish Regional STT Dataset

A speech recognition dataset for fine-tuning Whisper on regional Spanish dialects from Mexico, Spain, Argentina, and Chile.

## Dataset Description

- **Homepage:** [github.com/shraavb/spanish-slang-stt](https://github.com/shraavb/spanish-slang-stt)
- **Repository:** [github.com/shraavb/spanish-slang-stt](https://github.com/shraavb/spanish-slang-stt)
- **Point of Contact:** [Shraavasti Bhat](https://huggingface.co/shraavb)

### Dataset Summary

This dataset contains ~39,000 Spanish speech samples with transcriptions across 4 regional dialects, designed for fine-tuning Whisper and other ASR models on regional Spanish variations.

| Region | Train | Val | Test | Total | Description |
|--------|-------|-----|------|-------|-------------|
| Mexico | 14,133 | 1,813 | 1,779 | 17,725 | Mexican Spanish (CIEMPIESS, Common Voice) |
| Spain | 9,114 | 1,132 | 1,114 | 11,360 | Castilian Spanish (TEDx, Common Voice) |
| Argentina | 4,701 | 569 | 569 | 5,839 | Rioplatense Spanish (OpenSLR) |
| Chile | 3,490 | 415 | 469 | 4,374 | Chilean Spanish (OpenSLR) |
| **Total** | **31,438** | **3,929** | **3,931** | **39,298** | |

## Supported Tasks

- **Automatic Speech Recognition (ASR)**: Fine-tune Whisper or other models for Spanish regional dialects
- **Dialect Classification**: Train models to identify Spanish regional accents
- **Multilingual Spanish ASR**: Build unified models that handle multiple Spanish variants

## Languages

Spanish (`es`) with regional variants:
- `es-MX` - Mexican Spanish
- `es-ES` - Castilian Spanish (Spain)
- `es-AR` - Rioplatense Spanish (Argentina)
- `es-CL` - Chilean Spanish

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | `string` | Relative path to WAV audio file |
| `transcript` | `string` | Text transcription |
| `duration` | `float` | Audio duration in seconds |
| `language` | `string` | Language code (`es`) |
| `region` | `string` | Regional variant: `mexico`, `spain`, `argentina`, `chile` |
| `dataset` | `string` | Source dataset name |
| `original_path` | `string` | Original file path in source |
| `dialect_markers` | `list[string]` | Regional dialect indicators (optional) |
| `speaker_id` | `string` | Speaker identifier (where available) |

### Data Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| `train` | 31,438 | Model training |
| `validation` | 3,929 | Hyperparameter tuning |
| `test` | 3,931 | Final evaluation |

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("shraavb/spanish-slang-stt-data")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Test samples: {len(test)}")
```

### Filtering by Region

```python
from datasets import load_dataset

dataset = load_dataset("shraavb/spanish-slang-stt-data")

# Filter for Mexican Spanish only
mexico_train = dataset["train"].filter(lambda x: x["region"] == "mexico")
print(f"Mexican samples: {len(mexico_train)}")

# Filter for multiple regions
southern_cone = dataset["train"].filter(lambda x: x["region"] in ["argentina", "chile"])
print(f"Argentina + Chile samples: {len(southern_cone)}")
```

### Example Data Instance

```python
{
    "audio_path": "data/processed/mexico/ciempiess/0001F_02MAB_20AGO12.wav",
    "transcript": "buenos días cómo está usted",
    "duration": 3.52,
    "language": "es",
    "region": "mexico",
    "dataset": "ciempiess",
    "original_path": "ciempiess/0001F_02MAB_20AGO12.wav",
    "dialect_markers": [],
    "speaker_id": "0001F"
}
```

### Fine-tuning Whisper

```python
# See full training scripts at:
# https://github.com/shraavb/spanish-slang-stt

from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="es", task="transcribe")

# Fine-tune on regional data
# See scripts/train_regional.py for complete implementation
```

## Source Datasets

| Dataset | Region | Samples | License | URL |
|---------|--------|---------|---------|-----|
| CIEMPIESS | Mexico | ~14,000 | CC BY-NC-SA 4.0 | [ciempiess.org](http://www.ciempiess.org/) |
| TEDx Spanish | Spain | ~6,000 | CC BY-NC-ND 4.0 | [OpenSLR 67](https://openslr.org/67/) |
| Common Voice | Multiple | ~5,000 | CC0 | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/) |
| OpenSLR Argentine | Argentina | ~5,800 | CC BY 4.0 | [OpenSLR 61](https://openslr.org/61/) |
| OpenSLR Chilean | Chile | ~4,300 | CC BY 4.0 | [OpenSLR 71](https://openslr.org/71/) |

## Audio Specifications

- **Sample Rate:** 16,000 Hz
- **Channels:** Mono
- **Format:** WAV
- **Bit Depth:** 16-bit PCM
- **Duration Range:** 0.5s - 30s

## Dataset Creation

### Curation Rationale

This dataset was created to improve Whisper's performance on regional Spanish dialects. Standard Whisper models often struggle with:
- Regional accents and pronunciation variations
- Colloquial expressions and slang
- Fast speech patterns common in casual conversation

### Preprocessing

1. Audio resampled to 16kHz mono WAV
2. Transcripts normalized (lowercase, punctuation standardized)
3. Samples with transcripts exceeding 448 tokens filtered out (Whisper limit)
4. Train/validation/test splits stratified by region

## Considerations

### Biases

- Regional distribution is uneven (Mexico has most samples)
- Spain data includes more formal speech (TEDx talks)
- Some regions have limited speaker diversity

### Licensing

This dataset aggregates data from multiple sources with different licenses. Users should respect the original license terms:
- **CC BY-NC-SA 4.0**: CIEMPIESS (non-commercial use)
- **CC BY-NC-ND 4.0**: TEDx Spanish (non-commercial, no derivatives)
- **CC BY 4.0**: OpenSLR datasets
- **CC0**: Common Voice

## Citation

```bibtex
@misc{spanish-regional-stt-2024,
  title={Spanish Regional Speech-to-Text Dataset},
  author={Bhat, Shraavasti},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/shraavb/spanish-slang-stt-data}
}
```

## More Information

- **Training Scripts:** [github.com/shraavb/spanish-slang-stt](https://github.com/shraavb/spanish-slang-stt)
- **Fine-tuned Models:** Coming soon at [huggingface.co/shraavb](https://huggingface.co/shraavb)
