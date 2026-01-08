#!/usr/bin/env python3
"""Prepare downloaded datasets for training.

This script:
1. Parses different dataset formats (ParlamentParla, Google Crowdsourced, MagicHub)
2. Preprocesses audio (resample to 16kHz, normalize, trim silence)
3. Generates training manifests in JSON format
4. Creates train/val/test splits

Usage:
    python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed
    python scripts/prepare_datasets.py --dataset parlament-clean --input-dir data/raw
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import AudioPreprocessor, AudioMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AudioEntry:
    """Single audio entry for manifest."""
    audio_path: str
    transcript: str
    duration: float
    language: str
    speaker_id: Optional[str] = None
    dataset: Optional[str] = None
    original_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class DatasetParser:
    """Base class for dataset-specific parsers."""

    def __init__(self, dataset_dir: Path, language: str = "ca"):
        self.dataset_dir = dataset_dir
        self.language = language

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        """Yield (audio_path, transcript, speaker_id) tuples."""
        raise NotImplementedError


class ParlamentParlaParser(DatasetParser):
    """Parser for ParlamentParla dataset (OpenSLR 59).

    Structure:
        parlament_v1.0_clean/
        ├── audio/
        │   ├── 00001.wav
        │   └── ...
        └── transcripts/
            └── transcripts.txt  (or similar)
    """

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        # Find the extracted directory
        for subdir in self.dataset_dir.iterdir():
            if subdir.is_dir() and "parlament" in subdir.name.lower():
                root = subdir
                break
        else:
            root = self.dataset_dir

        # Look for audio files and transcripts
        audio_dir = None
        for candidate in ["audio", "wavs", "clips", "."]:
            test_dir = root / candidate
            if test_dir.exists():
                wav_files = list(test_dir.glob("*.wav"))
                if wav_files:
                    audio_dir = test_dir
                    break

        if not audio_dir:
            logger.error(f"No audio directory found in {root}")
            return

        # Look for transcript file
        transcript_file = None
        for pattern in ["*.txt", "*.tsv", "*transcripts*"]:
            files = list(root.glob(f"**/{pattern}"))
            if files:
                transcript_file = files[0]
                break

        # Try to parse transcripts
        transcripts = {}
        if transcript_file and transcript_file.exists():
            logger.info(f"Loading transcripts from {transcript_file}")
            transcripts = self._load_transcripts(transcript_file)

        # Yield audio entries
        for audio_file in sorted(audio_dir.glob("*.wav")):
            file_id = audio_file.stem
            transcript = transcripts.get(file_id, "")

            if not transcript:
                # Try variations
                for key in [file_id, file_id.lstrip("0"), f"0{file_id}"]:
                    if key in transcripts:
                        transcript = transcripts[key]
                        break

            if transcript:
                yield audio_file, transcript, None
            else:
                logger.debug(f"No transcript for {audio_file.name}")

    def _load_transcripts(self, filepath: Path) -> Dict[str, str]:
        """Load transcripts from various formats."""
        transcripts = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Try TSV format (file_id\ttranscript)
                if "\t" in line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        file_id = Path(parts[0]).stem
                        transcripts[file_id] = parts[1]
                        continue

                # Try pipe format (file_id|transcript)
                if "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        file_id = Path(parts[0]).stem
                        transcripts[file_id] = parts[1]
                        continue

                # Try space format (file_id transcript...)
                parts = line.split(None, 1)
                if len(parts) == 2:
                    file_id = Path(parts[0]).stem
                    transcripts[file_id] = parts[1]

        return transcripts


class GoogleCrowdsourcedParser(DatasetParser):
    """Parser for Google Crowdsourced dataset (OpenSLR 69).

    Structure:
        google-catalan-female/
        ├── ca_es_female/
        │   ├── wavs/
        │   │   ├── hash1.wav
        │   │   └── ...
        │   └── line_index_female.tsv
        └── line_index_female.tsv (copy at root)
    """

    def __init__(self, dataset_dir: Path, language: str = "ca", gender: str = "female"):
        super().__init__(dataset_dir, language)
        self.gender = gender

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        # Find index file
        index_file = None
        for pattern in [f"line_index_{self.gender}.tsv", "line_index*.tsv", "*.tsv"]:
            files = list(self.dataset_dir.glob(f"**/{pattern}"))
            if files:
                index_file = files[0]
                break

        if not index_file:
            logger.error(f"No index file found in {self.dataset_dir}")
            return

        # Find audio directory
        audio_dir = None
        for candidate in self.dataset_dir.glob("**/wavs"):
            audio_dir = candidate
            break

        if not audio_dir:
            # Try finding wav files directly
            wav_files = list(self.dataset_dir.glob("**/*.wav"))
            if wav_files:
                audio_dir = wav_files[0].parent

        if not audio_dir:
            logger.error(f"No audio directory found in {self.dataset_dir}")
            return

        logger.info(f"Parsing index: {index_file}")
        logger.info(f"Audio directory: {audio_dir}")

        # Parse TSV index
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    file_id = parts[0]
                    transcript = parts[1]

                    # Find audio file
                    audio_file = audio_dir / f"{file_id}.wav"
                    if not audio_file.exists():
                        audio_file = audio_dir / file_id
                        if not audio_file.exists():
                            continue

                    speaker_id = f"{self.gender}_{file_id[:8]}"
                    yield audio_file, transcript, speaker_id


class SpanishConversationalParser(DatasetParser):
    """Parser for Spanish Conversational Corpus (MagicHub).

    Structure:
        spanish-conversational/
        ├── AUDIO/
        │   ├── conv_001.wav
        │   └── ...
        └── TRANSCRIPTION/
            ├── conv_001.txt
            └── ...
    """

    def __init__(self, dataset_dir: Path, language: str = "es"):
        super().__init__(dataset_dir, language)

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        # Find audio and transcript directories
        audio_dir = None
        transcript_dir = None

        for candidate in ["AUDIO", "audio", "wavs", "wav"]:
            test_dir = self.dataset_dir / candidate
            if test_dir.exists():
                audio_dir = test_dir
                break

        for candidate in ["TRANSCRIPTION", "transcription", "transcripts", "txt"]:
            test_dir = self.dataset_dir / candidate
            if test_dir.exists():
                transcript_dir = test_dir
                break

        if not audio_dir:
            # Try finding wav files directly
            wav_files = list(self.dataset_dir.glob("**/*.wav"))
            if wav_files:
                audio_dir = wav_files[0].parent

        if not audio_dir:
            logger.error(f"No audio directory found in {self.dataset_dir}")
            return

        logger.info(f"Audio directory: {audio_dir}")
        if transcript_dir:
            logger.info(f"Transcript directory: {transcript_dir}")

        # Process audio files
        for audio_file in sorted(audio_dir.glob("*.wav")):
            transcript = ""

            # Try to find matching transcript
            if transcript_dir:
                transcript_file = transcript_dir / f"{audio_file.stem}.txt"
                if transcript_file.exists():
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()

            # Extract speaker from filename if possible
            speaker_id = None
            match = re.search(r"spk(\d+)|speaker(\d+)|s(\d+)", audio_file.stem, re.I)
            if match:
                speaker_id = f"es_speaker_{match.group(1) or match.group(2) or match.group(3)}"

            if transcript:
                yield audio_file, transcript, speaker_id


def get_parser(dataset_id: str, dataset_dir: Path) -> Optional[DatasetParser]:
    """Get appropriate parser for dataset."""
    if "parlament" in dataset_id:
        return ParlamentParlaParser(dataset_dir, language="ca")
    elif "google-catalan-female" in dataset_id:
        return GoogleCrowdsourcedParser(dataset_dir, language="ca", gender="female")
    elif "google-catalan-male" in dataset_id:
        return GoogleCrowdsourcedParser(dataset_dir, language="ca", gender="male")
    elif "spanish" in dataset_id:
        return SpanishConversationalParser(dataset_dir, language="es")
    else:
        logger.warning(f"No parser for dataset: {dataset_id}")
        return None


def process_audio_file(
    args: Tuple[Path, str, Optional[str], str, Path, AudioPreprocessor]
) -> Optional[AudioEntry]:
    """Process a single audio file (for parallel processing)."""
    audio_path, transcript, speaker_id, dataset_id, output_dir, preprocessor = args

    try:
        # Create output path
        output_subdir = output_dir / dataset_id
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{audio_path.stem}.wav"

        # Skip if already processed
        if output_path.exists():
            duration = librosa.get_duration(path=str(output_path))
            return AudioEntry(
                audio_path=str(output_path),
                transcript=transcript,
                duration=round(duration, 3),
                language="ca" if "catalan" in dataset_id or "parlament" in dataset_id else "es",
                speaker_id=speaker_id,
                dataset=dataset_id,
                original_path=str(audio_path),
            )

        # Process audio
        metadata = preprocessor.process_and_save(audio_path, output_path)

        return AudioEntry(
            audio_path=str(output_path),
            transcript=transcript,
            duration=round(metadata.duration_seconds, 3),
            language="ca" if "catalan" in dataset_id or "parlament" in dataset_id else "es",
            speaker_id=speaker_id,
            dataset=dataset_id,
            original_path=str(audio_path),
        )

    except Exception as e:
        logger.error(f"Failed to process {audio_path}: {e}")
        return None


def prepare_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    preprocessor: AudioPreprocessor,
    max_workers: int = 4,
    max_samples: Optional[int] = None,
) -> List[AudioEntry]:
    """Prepare a single dataset."""
    dataset_dir = input_dir / dataset_id

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []

    parser = get_parser(dataset_id, dataset_dir)
    if not parser:
        return []

    logger.info(f"Preparing dataset: {dataset_id}")

    # Collect all entries to process
    entries_to_process = []
    for audio_path, transcript, speaker_id in parser.parse():
        entries_to_process.append((
            audio_path, transcript, speaker_id, dataset_id, output_dir, preprocessor
        ))

        if max_samples and len(entries_to_process) >= max_samples:
            break

    if not entries_to_process:
        logger.warning(f"No valid entries found in {dataset_id}")
        return []

    logger.info(f"Processing {len(entries_to_process)} audio files...")

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_audio_file, args): args for args in entries_to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_id):
            result = future.result()
            if result:
                results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(entries_to_process)} files")
    return results


def create_splits(
    entries: List[AudioEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[AudioEntry], List[AudioEntry], List[AudioEntry]]:
    """Create train/val/test splits."""
    random.seed(seed)

    # Shuffle
    shuffled = entries.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def save_manifest(entries: List[AudioEntry], output_path: Path) -> None:
    """Save entries to JSON manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [entry.to_dict() for entry in entries]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(entries)} entries to {output_path}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare downloaded datasets for training"
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with raw datasets",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed audio",
    )
    parser.add_argument(
        "--manifests-dir", "-m",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for manifest files",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Process specific dataset (default: all found)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Setup preprocessor
    preproc_config = config.get("preprocessing", {})
    preprocessor = AudioPreprocessor(
        target_sample_rate=config.get("data", {}).get("sample_rate", 16000),
        target_db=preproc_config.get("target_db", -20),
        normalize=preproc_config.get("normalize_audio", True),
        trim_silence=preproc_config.get("trim_silence", True),
        silence_threshold_db=preproc_config.get("silence_threshold_db", -40),
    )

    # Find datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        # Find all dataset directories
        datasets = []
        if args.input_dir.exists():
            for subdir in args.input_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    datasets.append(subdir.name)

    if not datasets:
        logger.error(f"No datasets found in {args.input_dir}")
        logger.info("Run download_datasets.py first to download data")
        sys.exit(1)

    logger.info(f"Found datasets: {datasets}")

    # Process each dataset
    all_entries = []

    for dataset_id in datasets:
        entries = prepare_dataset(
            dataset_id=dataset_id,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            preprocessor=preprocessor,
            max_workers=args.workers,
            max_samples=args.max_samples,
        )
        all_entries.extend(entries)

    if not all_entries:
        logger.error("No entries processed")
        sys.exit(1)

    # Create splits
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train, val, test = create_splits(
        all_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
    )

    # Save manifests
    save_manifest(train, args.manifests_dir / "train.json")
    save_manifest(val, args.manifests_dir / "val.json")
    save_manifest(test, args.manifests_dir / "test.json")

    # Save combined manifest
    save_manifest(all_entries, args.manifests_dir / "all.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"Total samples: {len(all_entries)}")
    print(f"  Train: {len(train)} ({100*len(train)/len(all_entries):.1f}%)")
    print(f"  Val:   {len(val)} ({100*len(val)/len(all_entries):.1f}%)")
    print(f"  Test:  {len(test)} ({100*len(test)/len(all_entries):.1f}%)")

    # Per-dataset breakdown
    print("\nPer-dataset breakdown:")
    dataset_counts = {}
    for entry in all_entries:
        ds = entry.dataset or "unknown"
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count}")

    # Total duration
    total_duration = sum(e.duration for e in all_entries)
    print(f"\nTotal duration: {total_duration/3600:.1f} hours")

    print(f"\nManifests saved to: {args.manifests_dir.absolute()}")
    print("\nNext step:")
    print(f"  python scripts/train.py --config {args.config} \\")
    print(f"    --train-manifest {args.manifests_dir}/train.json \\")
    print(f"    --val-manifest {args.manifests_dir}/val.json")


if __name__ == "__main__":
    main()
