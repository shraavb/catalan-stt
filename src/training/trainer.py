"""Whisper fine-tuning trainer."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict
import evaluate

from .config import TrainingConfig
from ..data.loader import AudioSample, AudioDataset, create_data_collator

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper training with proper padding."""

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder_start_token_id if present at beginning
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperTrainer:
    """Fine-tune Whisper for regional Spanish slang and informal speech."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing WhisperTrainer with device: {self.device}")
        logger.info(f"Base model: {config.base_model}")

        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(
            config.base_model,
            language=config.language,
            task=config.task,
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.base_model
        )

        # Configure model for fine-tuning
        self.model.generation_config.language = config.language
        self.model.generation_config.task = config.task
        self.model.generation_config.forced_decoder_ids = None

        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.model.config.use_cache = False

        # Load WER metric
        self.wer_metric = evaluate.load("wer")

        logger.info("Model and processor loaded successfully")

    def prepare_dataset(
        self,
        samples: list[AudioSample],
        max_label_length: int = 448,
    ) -> Dataset:
        """Convert AudioSamples to HuggingFace Dataset format."""
        import librosa

        def process_sample(sample: AudioSample) -> Optional[Dict[str, Any]]:
            # Load and process audio
            audio, _ = librosa.load(sample.audio_path, sr=self.config.sample_rate)

            # Get input features
            input_features = self.processor(
                audio,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_features[0]

            # Tokenize transcript
            labels = self.processor.tokenizer(sample.transcript).input_ids

            # Skip samples with transcripts that are too long for Whisper
            if len(labels) > max_label_length:
                return None

            return {
                "input_features": input_features,
                "labels": labels,
            }

        # Process all samples
        processed = []
        skipped_long = 0
        for sample in samples:
            try:
                result = process_sample(sample)
                if result is not None:
                    processed.append(result)
                else:
                    skipped_long += 1
            except Exception as e:
                logger.warning(f"Failed to process {sample.audio_path}: {e}")

        if skipped_long > 0:
            logger.info(f"Skipped {skipped_long} samples with transcripts exceeding {max_label_length} tokens")

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(processed)

        logger.info(f"Prepared dataset with {len(dataset)} samples")
        return dataset

    def compute_metrics(self, pred) -> Dict[str, float]:
        """Compute WER metric for evaluation."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def train(
        self,
        train_samples: list[AudioSample],
        eval_samples: Optional[list[AudioSample]] = None,
    ) -> str:
        """Run fine-tuning and return path to best checkpoint."""
        logger.info(f"Starting training with {len(train_samples)} samples")

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_samples)

        eval_dataset = None
        if eval_samples:
            eval_dataset = self.prepare_dataset(eval_samples)
            logger.info(f"Using {len(eval_samples)} samples for evaluation")

        # Create data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # Setup training arguments
        training_args = Seq2SeqTrainingArguments(
            **self.config.get_training_args(),
            predict_with_generate=True,
            generation_max_length=225,
            report_to=["tensorboard"],
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            processing_class=self.processor.feature_extractor,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        output_dir = Path(self.config.output_dir)
        trainer.save_model(str(output_dir / "final"))
        self.processor.save_pretrained(str(output_dir / "final"))

        logger.info(f"Training complete. Model saved to {output_dir / 'final'}")

        return str(output_dir / "final")

    def evaluate(
        self,
        test_samples: list[AudioSample],
        model_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            self.processor = WhisperProcessor.from_pretrained(model_path)

        test_dataset = self.prepare_dataset(test_samples)

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="./eval_output",
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            predict_with_generate=True,
            generation_max_length=225,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor.feature_extractor,
        )

        results = trainer.evaluate(test_dataset)

        logger.info(f"Evaluation results: {results}")
        return results
