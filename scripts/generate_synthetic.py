#!/usr/bin/env python3
"""Generate synthetic training data using ElevenLabs TTS."""

import argparse
import logging
import json
from pathlib import Path

from src.data import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample Catalan-Spanish phrases for testing
SAMPLE_PHRASES = [
    # Greetings
    "Hola, ¿qué tal, nen?",
    "Buenos días, ¿cómo estás?",
    "Apa, ¿qué haces por aquí?",

    # Common expressions
    "Ostras, qué bien!",
    "Home, no me lo puedo creer!",
    "Vale, vale, lo entiendo.",

    # Conversational
    "¿Vamos a tomar algo?",
    "Mira, te explico una cosa.",
    "Pos sí, tienes razón.",

    # Questions
    "¿Dónde está la estación?",
    "¿Me puedes ayudar, por favor?",
    "¿Cuánto cuesta esto?",

    # Directions
    "Gira a la derecha y sigue todo recto.",
    "Está al lado de la plaza.",
    "No está muy lejos de aquí.",

    # Shopping
    "Me gustaría comprar esto, por favor.",
    "¿Tienen algo más barato?",
    "¿Aceptan tarjeta de crédito?",

    # Dining
    "Una mesa para dos, por favor.",
    "¿Qué me recomiendas?",
    "La cuenta, por favor.",

    # Catalan-influenced
    "Nen, ven aquí un momento.",
    "Home, no seas borde!",
    "Apa, vamos que llegamos tarde!",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio data using ElevenLabs TTS"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSON file with phrases (optional, uses sample phrases if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate",
    )
    parser.add_argument(
        "--alternate-voices",
        action="store_true",
        default=True,
        help="Alternate between different Spanish voices",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds",
    )

    args = parser.parse_args()

    # Load phrases
    if args.input:
        logger.info(f"Loading phrases from {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            if isinstance(data[0], dict):
                phrases = [d.get("text", d.get("transcript", "")) for d in data]
            else:
                phrases = data
        else:
            raise ValueError("Input JSON must be a list of phrases or objects with 'text' field")
    else:
        logger.info("Using sample Catalan-Spanish phrases")
        phrases = SAMPLE_PHRASES

    if args.max_samples:
        phrases = phrases[:args.max_samples]

    logger.info(f"Generating audio for {len(phrases)} phrases")

    # Generate
    try:
        generator = SyntheticDataGenerator(
            api_key=args.api_key,
            output_dir=args.output_dir,
        )

        results = generator.synthesize_batch(
            texts=phrases,
            delay_seconds=args.delay,
            alternate_voices=args.alternate_voices,
        )

        # Create manifest
        manifest_path = generator.create_manifest(results)

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        print("\n" + "=" * 50)
        print("SYNTHETIC DATA GENERATION COMPLETE")
        print("=" * 50)
        print(f"Total phrases: {len(phrases)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {args.output_dir}")
        print(f"Manifest: {manifest_path}")
        print("=" * 50)

    except ValueError as e:
        logger.error(f"Error: {e}")
        print("\nTo use this script, you need an ElevenLabs API key.")
        print("Set the ELEVENLABS_API_KEY environment variable or use --api-key.")
        print("\nGet your API key at: https://elevenlabs.io")


if __name__ == "__main__":
    main()
