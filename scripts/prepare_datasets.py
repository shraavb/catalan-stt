#!/usr/bin/env python3
"""Prepare downloaded datasets for training.

This script:
1. Parses different dataset formats (MagicHub Spanish Conversational, Common Voice, etc.)
2. Preprocesses audio (resample to 16kHz, normalize, trim silence)
3. Generates training manifests in JSON format with region metadata
4. Creates train/val/test splits (optionally per-region)

Usage:
    python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed
    python scripts/prepare_datasets.py --dataset spanish-conversational --input-dir data/raw
    python scripts/prepare_datasets.py --region mexico --input-dir data/raw --output-dir data/processed
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


# Region to dataset mapping
DATASET_REGIONS = {
    "spanish-conversational": "spain",
    "common-voice-spain": "spain",
    "common-voice-mexico": "mexico",
    "common-voice-argentina": "argentina",
    "ciempiess": "mexico",
    "untref": "argentina",
    "openslr-argentina": "argentina",
    "chilean-spanish": "chile",
    "tedx-spanish": "spain",
}

# Slang markers by region (for detection)
SLANG_MARKERS = {
    "spain": ["tio", "tia", "mola", "guay", "currar", "pasta", "flipar", "chaval", "colega", "quedarse"],
    "mexico": ["chido", "padre", "neta", "chale", "onda", "guey", "wey", "morro", "chingon", "fresa"],
    "argentina": ["che", "boludo", "pibe", "mina", "laburo", "copado", "morfar", "chabon", "quilombo", "guita"],
    "chile": ["cachai", "polola", "pololo", "fome", "bacán", "bacan", "al tiro", "po", "weon", "cuático"],
}


@dataclass
class AudioEntry:
    """Single audio entry for manifest."""
    audio_path: str
    transcript: str
    duration: float
    language: str
    region: str = "general"
    dialect_markers: Optional[List[str]] = None
    speaker_id: Optional[str] = None
    dataset: Optional[str] = None
    original_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def detect_dialect_markers(transcript: str, region: str = "general") -> List[str]:
    """Detect slang markers in transcript based on region."""
    transcript_lower = transcript.lower()
    markers = []

    # Check region-specific markers first
    if region in SLANG_MARKERS:
        for marker in SLANG_MARKERS[region]:
            if marker in transcript_lower:
                markers.append(marker)

    # Also check all markers if no specific region
    if region == "general":
        for region_markers in SLANG_MARKERS.values():
            for marker in region_markers:
                if marker in transcript_lower and marker not in markers:
                    markers.append(marker)

    return markers


class DatasetParser:
    """Base class for dataset-specific parsers."""

    def __init__(self, dataset_dir: Path, language: str = "es"):
        self.dataset_dir = dataset_dir
        self.language = language

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        """Yield (audio_path, transcript, speaker_id) tuples."""
        raise NotImplementedError


class SpanishConversationalParser(DatasetParser):
    """Parser for Spanish Conversational Corpus (MagicHub).

    Structure (handles both flat and nested):
        spanish-conversational/
        ├── Spanish_Conversational_Speech_Corpus/  (optional nested dir)
        │   ├── WAV/
        │   │   └── *.wav
        │   └── TXT/
        │       └── *.txt
        OR:
        ├── AUDIO/
        │   └── *.wav
        └── TRANSCRIPTION/
            └── *.txt
    """

    def __init__(self, dataset_dir: Path, language: str = "es"):
        super().__init__(dataset_dir, language)

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        # Find audio and transcript directories
        audio_dir = None
        transcript_dir = None

        # Check for nested structure first (e.g., Spanish_Conversational_Speech_Corpus/)
        search_dirs = [self.dataset_dir]
        for subdir in self.dataset_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                search_dirs.append(subdir)

        # Search for audio directory
        audio_candidates = ["WAV", "AUDIO", "audio", "wavs", "wav"]
        for search_dir in search_dirs:
            for candidate in audio_candidates:
                test_dir = search_dir / candidate
                if test_dir.exists() and list(test_dir.glob("*.wav")):
                    audio_dir = test_dir
                    break
            if audio_dir:
                break

        # Search for transcript directory (look in same parent as audio)
        if audio_dir:
            audio_parent = audio_dir.parent
            transcript_candidates = ["TXT", "TRANSCRIPTION", "transcription", "transcripts", "txt"]
            for candidate in transcript_candidates:
                test_dir = audio_parent / candidate
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


class CommonVoiceParser(DatasetParser):
    """Parser for Common Voice data (reads from train_manifest.json).

    Structure:
        common-voice-{region}/
        ├── audio/
        │   └── cv_*.wav
        └── train_manifest.json
    """

    def __init__(self, dataset_dir: Path, language: str = "es"):
        super().__init__(dataset_dir, language)

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        import json

        # Look for manifest file
        manifest_path = self.dataset_dir / "train_manifest.json"
        if not manifest_path.exists():
            logger.error(f"No train_manifest.json found in {self.dataset_dir}")
            return

        logger.info(f"Loading manifest from {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest:
            audio_path = Path(entry.get("audio_path", ""))
            transcript = entry.get("transcript", "")
            speaker_id = entry.get("speaker_id")

            # Handle both absolute and relative paths
            if not audio_path.is_absolute():
                # Try path as-is from project root first
                if not audio_path.exists():
                    # Try relative to dataset directory
                    audio_path = self.dataset_dir / audio_path.name

            if audio_path.exists() and transcript:
                yield audio_path, transcript, speaker_id
            elif audio_path.exists() and not transcript:
                logger.debug(f"Skipping {audio_path.name} - no transcript")
            else:
                logger.debug(f"Audio file not found: {audio_path}")


class GenericParser(DatasetParser):
    """Generic parser for datasets with audio + transcripts."""

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        # Find audio directory (case-insensitive)
        audio_dir = None
        for candidate in ["audio", "AUDIO", "wavs", "clips", "speech", "."]:
            test_dir = self.dataset_dir / candidate
            if test_dir.exists():
                wav_files = list(test_dir.glob("*.wav"))
                if wav_files:
                    audio_dir = test_dir
                    break

        if not audio_dir:
            logger.error(f"No audio directory found in {self.dataset_dir}")
            return

        logger.info(f"Audio directory: {audio_dir}")

        # Look for transcript file (prioritize transcripts.tsv over generic .txt)
        transcript_file = None
        for pattern in ["transcripts.tsv", "transcripts.txt", "*transcripts*", "*.tsv", "*.txt"]:
            files = [f for f in self.dataset_dir.glob(f"**/{pattern}")
                     if f.name.lower() not in ("license.txt", "readme.txt", "about.html")]
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


class CIEMPIESSParser(DatasetParser):
    """Parser for CIEMPIESS Mexican Spanish Corpus.

    Structure:
        ciempiess/
        └── data/
            ├── train/
            │   └── */*/*.wav
            ├── test/
            │   └── */*/*.wav
            └── transcriptions/
                ├── CIEMPIESS_train.fileids
                ├── CIEMPIESS_train.transcription
                ├── CIEMPIESS_test.fileids
                └── CIEMPIESS_test.transcription
    """

    def parse(self) -> Iterator[Tuple[Path, str, Optional[str]]]:
        data_dir = self.dataset_dir / "data"
        if not data_dir.exists():
            logger.error(f"No data directory found in {self.dataset_dir}")
            return

        transcription_dir = data_dir / "transcriptions"
        if not transcription_dir.exists():
            logger.error(f"No transcriptions directory found in {data_dir}")
            return

        # Load transcripts from both train and test sets
        transcripts = {}
        for split in ["train", "FULL_TRAIN", "test"]:
            trans_file = transcription_dir / f"CIEMPIESS_{split}.transcription"
            if trans_file.exists():
                transcripts.update(self._load_ciempiess_transcripts(trans_file))

        if not transcripts:
            logger.error("No transcripts loaded from CIEMPIESS")
            return

        logger.info(f"Loaded {len(transcripts)} transcripts from CIEMPIESS")

        # Find all wav files in train and test directories
        for split_dir in ["train", "test"]:
            split_path = data_dir / split_dir
            if not split_path.exists():
                continue

            for audio_file in sorted(split_path.glob("**/*.wav")):
                file_id = audio_file.stem
                if file_id in transcripts:
                    transcript = transcripts[file_id]
                    # Clean up transcript (remove <s>, </s>, <sil>, ++dis++, normalize case)
                    transcript = self._clean_transcript(transcript)
                    if transcript:
                        yield audio_file, transcript, None
                else:
                    logger.debug(f"No transcript for {file_id}")

    def _load_ciempiess_transcripts(self, filepath: Path) -> Dict[str, str]:
        """Load transcripts from CIEMPIESS .transcription format."""
        transcripts = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: <s> transcript </s> (file_id)
                match = re.match(r"<s>\s*(.*?)\s*</s>\s*\((\S+)\)", line)
                if match:
                    transcript = match.group(1)
                    file_id = match.group(2)
                    transcripts[file_id] = transcript
        return transcripts

    def _clean_transcript(self, transcript: str) -> str:
        """Clean CIEMPIESS transcript format."""
        # Remove silence markers
        transcript = re.sub(r"<sil>", "", transcript)
        # Remove disfluency markers
        transcript = re.sub(r"\+\+dis\+\+", "", transcript)
        # Remove other special markers
        transcript = re.sub(r"\+\+\w+\+\+", "", transcript)
        # Normalize case (CIEMPIESS uses uppercase for stressed vowels)
        transcript = transcript.lower()
        # Clean up whitespace
        transcript = " ".join(transcript.split())
        return transcript


def get_parser(dataset_id: str, dataset_dir: Path) -> Optional[DatasetParser]:
    """Get appropriate parser for dataset."""
    if "common-voice" in dataset_id:
        return CommonVoiceParser(dataset_dir, language="es")
    elif "ciempiess" in dataset_id:
        return CIEMPIESSParser(dataset_dir, language="es")
    elif dataset_id == "spanish-conversational":
        return SpanishConversationalParser(dataset_dir, language="es")
    else:
        # Try generic parser for all other datasets
        return GenericParser(dataset_dir, language="es")


def process_audio_file(
    args: Tuple[Path, str, Optional[str], str, str, Path, AudioPreprocessor]
) -> Optional[AudioEntry]:
    """Process a single audio file (for parallel processing)."""
    audio_path, transcript, speaker_id, dataset_id, region, output_dir, preprocessor = args

    try:
        # Create output path (organized by region if specified)
        if region and region != "general":
            output_subdir = output_dir / region / dataset_id
        else:
            output_subdir = output_dir / dataset_id
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{audio_path.stem}.wav"

        # Detect dialect markers in transcript
        dialect_markers = detect_dialect_markers(transcript, region)

        # Skip if already processed
        if output_path.exists():
            duration = librosa.get_duration(path=str(output_path))
            return AudioEntry(
                audio_path=str(output_path),
                transcript=transcript,
                duration=round(duration, 3),
                language="es",
                region=region,
                dialect_markers=dialect_markers if dialect_markers else None,
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
            language="es",
            region=region,
            dialect_markers=dialect_markers if dialect_markers else None,
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
    region: Optional[str] = None,
    max_workers: int = 4,
    max_samples: Optional[int] = None,
) -> List[AudioEntry]:
    """Prepare a single dataset.

    Args:
        dataset_id: ID of the dataset
        input_dir: Directory containing raw datasets
        output_dir: Directory for processed audio
        preprocessor: Audio preprocessor instance
        region: Region override (spain, mexico, argentina, etc.)
        max_workers: Number of parallel workers
        max_samples: Maximum samples to process

    Returns:
        List of AudioEntry objects
    """
    dataset_dir = input_dir / dataset_id

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []

    parser = get_parser(dataset_id, dataset_dir)
    if not parser:
        return []

    # Determine region from dataset ID if not provided
    effective_region = region or DATASET_REGIONS.get(dataset_id, "general")
    logger.info(f"Preparing dataset: {dataset_id} (region: {effective_region})")

    # Collect all entries to process
    entries_to_process = []
    for audio_path, transcript, speaker_id in parser.parse():
        entries_to_process.append((
            audio_path, transcript, speaker_id, dataset_id, effective_region, output_dir, preprocessor
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
    parser.add_argument(
        "--region", "-r",
        type=str,
        choices=["spain", "mexico", "argentina", "general"],
        default=None,
        help="Region override for all datasets (spain, mexico, argentina)",
    )
    parser.add_argument(
        "--per-region-splits",
        action="store_true",
        help="Create separate train/val/test splits per region",
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
            region=args.region,
            max_workers=args.workers,
            max_samples=args.max_samples,
        )
        all_entries.extend(entries)

    if not all_entries:
        logger.error("No entries processed")
        sys.exit(1)

    # Create splits
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if args.per_region_splits:
        # Group entries by region and create per-region splits
        entries_by_region: Dict[str, List[AudioEntry]] = {}
        for entry in all_entries:
            region = entry.region or "general"
            if region not in entries_by_region:
                entries_by_region[region] = []
            entries_by_region[region].append(entry)

        # Create splits for each region
        for region, region_entries in entries_by_region.items():
            train, val, test = create_splits(
                region_entries,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=test_ratio,
                seed=args.seed,
            )

            # Save region-specific manifests
            region_dir = args.manifests_dir / region
            save_manifest(train, region_dir / "train.json")
            save_manifest(val, region_dir / "val.json")
            save_manifest(test, region_dir / "test.json")
            save_manifest(region_entries, region_dir / "all.json")

            logger.info(f"Region {region}: {len(train)} train, {len(val)} val, {len(test)} test")

        # Also save combined manifests
        train, val, test = create_splits(
            all_entries,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
    else:
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

    # Per-region breakdown
    print("\nPer-region breakdown:")
    region_counts: Dict[str, int] = {}
    region_with_markers: Dict[str, int] = {}
    for entry in all_entries:
        region = entry.region or "general"
        region_counts[region] = region_counts.get(region, 0) + 1
        if entry.dialect_markers:
            region_with_markers[region] = region_with_markers.get(region, 0) + 1

    for region, count in sorted(region_counts.items()):
        markers = region_with_markers.get(region, 0)
        pct = 100 * markers / count if count > 0 else 0
        print(f"  {region}: {count} ({markers} with slang markers, {pct:.1f}%)")

    # Total duration
    total_duration = sum(e.duration for e in all_entries)
    print(f"\nTotal duration: {total_duration/3600:.1f} hours")

    print(f"\nManifests saved to: {args.manifests_dir.absolute()}")
    if args.per_region_splits:
        print("\nPer-region manifests saved to:")
        for region in region_counts.keys():
            print(f"  {args.manifests_dir / region}")

    print("\nNext step:")
    print(f"  python scripts/train.py --config {args.config} \\")
    print(f"    --train-manifest {args.manifests_dir}/train.json \\")
    print(f"    --val-manifest {args.manifests_dir}/val.json")


if __name__ == "__main__":
    main()
