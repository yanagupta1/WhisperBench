# WhisperBench — Audio Dataset Quality Pipeline + LoRA Fine-Tuning

A production-style pipeline for ingesting raw audio, scoring quality metrics, filtering low-quality samples, and exporting structured datasets for multimodal model training. Built as a lightweight, portable tool for audio training data curation.

---

## Overview

Large-scale audio datasets for model training are noisy by nature — variable SNR, inconsistent durations, low-confidence transcripts, and near-empty utterances all degrade downstream model quality. WhisperBench automates quality scoring and filtering to produce clean, structured datasets ready for training pipelines.

**Tested on:** LibriSpeech test-clean (~2,620 files, ~5.4 hrs audio)

---

## Results

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| Total audio | ~5.4 hrs |
| Rejection rate | 3% |
| Avg SNR (clean) | 26.2 dB |
| Throughput | ~113 files/min (T4 GPU) |
| Baseline WER (Whisper-small) | 16.84% |
| Fine-tuned WER (LoRA, 500 samples) | 16.52% |
| WER improvement | 1.9% |
| Trainable params (LoRA) | 0.4M / 242M (0.2%) |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `WhisperBench.ipynb` | Quality pipeline: transcription → scoring → filtering → Parquet export |
| `WhisperBench_finetune.ipynb` | LoRA fine-tuning on Whisper-small + WER evaluation |

---

## Pipeline

```
Raw Audio (.flac/.wav)
        │
        ▼
 Whisper Transcription
        │
        ▼
 Quality Scoring
  ├── SNR (signal-to-noise ratio)
  ├── Duration (min/max thresholds)
  ├── Transcript confidence (Whisper avg_logprob)
  └── Word count (min utterance length)
        │
        ▼
 Filter: clean / rejected splits
        │
        ▼
 Parquet Export
  ├── whisperbench_clean.parquet
  └── whisperbench_rejected.parquet
```

---

## Fine-Tuning

Whisper-small fine-tuned using LoRA (PEFT) — only 0.2% of parameters trained, avoiding catastrophic forgetting while adapting to domain-specific audio.

**Config:**
- Base model: `openai/whisper-small`
- LoRA rank: 4, alpha: 8
- Target modules: `q_proj`, `v_proj`
- Training samples: 500 (LibriSpeech train-clean-100)
- Epochs: 1
- Learning rate: 1e-4

Trained adapter weights are in `whisper-lora-adapter/`.

---

## Output Schema

Each row in the Parquet output contains:

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Audio filename |
| `duration_s` | float | Duration in seconds |
| `snr_db` | float | Signal-to-noise ratio (dB) |
| `transcript` | string | Whisper transcription |
| `confidence` | float | Avg log-prob across segments |

---

## Setup

```bash
pip install openai-whisper pandas pyarrow tqdm soundfile scipy \
            transformers datasets peft accelerate evaluate
apt-get install ffmpeg
```

**Recommended:** Google Colab with A100 GPU for fine-tuning, T4 for quality pipeline.

---

## Quality Filter Thresholds

| Filter | Default |
|--------|---------|
| Min duration | 1.0s |
| Max duration | 30.0s |
| Min SNR | 10.0 dB |
| Min confidence | -0.8 (log-prob) |
| Min word count | 2 |

All thresholds are configurable in Cell 7 of `WhisperBench.ipynb`.
