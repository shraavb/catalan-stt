#!/usr/bin/env python3
"""Mine slang expressions from Spanish subtitles.

Extracts regional slang vocabulary and phrases from downloaded subtitles,
creating a structured dictionary for STT training and SpeakEasy integration.

Usage:
    python scripts/slang_miner.py --input-dir data/subtitles
    python scripts/slang_miner.py --input-dir data/subtitles --region mexico
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Comprehensive regional slang dictionaries
# These serve as seed terms - the miner will find additional examples in context
REGIONAL_SLANG = {
    "mexico": {
        "words": {
            # Common slang
            "chido": {"meaning": "cool, awesome", "formality": "informal"},
            "padre": {"meaning": "cool, great (adj)", "formality": "informal"},
            "neta": {"meaning": "truth, really", "formality": "informal"},
            "chale": {"meaning": "damn, bummer", "formality": "informal"},
            "onda": {"meaning": "vibe, what's up", "formality": "informal"},
            "güey": {"meaning": "dude, bro", "formality": "very_informal"},
            "wey": {"meaning": "dude, bro (alt spelling)", "formality": "very_informal"},
            "morro": {"meaning": "kid, young person", "formality": "informal"},
            "morra": {"meaning": "girl, young woman", "formality": "informal"},
            "chingón": {"meaning": "awesome, badass", "formality": "vulgar"},
            "chingar": {"meaning": "to bother, mess up", "formality": "vulgar"},
            "naco": {"meaning": "tacky, low-class", "formality": "offensive"},
            "fresa": {"meaning": "preppy, snobby", "formality": "informal"},
            "cuate": {"meaning": "buddy, friend", "formality": "informal"},
            "carnal": {"meaning": "close friend, bro", "formality": "informal"},
            "chamba": {"meaning": "work, job", "formality": "informal"},
            "lana": {"meaning": "money", "formality": "informal"},
            "bronca": {"meaning": "problem, fight", "formality": "informal"},
            "chela": {"meaning": "beer", "formality": "informal"},
            "cruda": {"meaning": "hangover", "formality": "informal"},
            "gacho": {"meaning": "bad, ugly, unfair", "formality": "informal"},
            "órale": {"meaning": "alright, wow, come on", "formality": "informal"},
            "ándale": {"meaning": "hurry up, come on", "formality": "informal"},
            "simón": {"meaning": "yes, yeah", "formality": "informal"},
            "nel": {"meaning": "no, nope", "formality": "informal"},
            "mande": {"meaning": "pardon? (polite)", "formality": "formal"},
            "ahorita": {"meaning": "right now (or later)", "formality": "neutral"},
            "tantito": {"meaning": "a little bit", "formality": "informal"},
        },
        "phrases": {
            "qué onda": "what's up",
            "qué pedo": "what's up (vulgar)",
            "no manches": "no way, come on",
            "no mames": "no way (vulgar)",
            "a huevo": "hell yeah",
            "de volada": "quickly, right away",
            "al chile": "for real, honestly",
            "está cañón": "it's intense/difficult",
            "me vale": "I don't care",
            "qué padre": "how cool",
            "qué chido": "how cool",
            "ni modo": "oh well, no way around it",
            "de plano": "definitely, simply",
            "qué rollo": "what's going on",
        },
    },
    "argentina": {
        "words": {
            "che": {"meaning": "hey, buddy", "formality": "informal"},
            "boludo": {"meaning": "dude, idiot", "formality": "very_informal"},
            "boluda": {"meaning": "dude (female), idiot", "formality": "very_informal"},
            "pibe": {"meaning": "kid, guy", "formality": "informal"},
            "piba": {"meaning": "girl, woman", "formality": "informal"},
            "mina": {"meaning": "woman, girl", "formality": "informal"},
            "chabón": {"meaning": "dude, guy", "formality": "informal"},
            "laburo": {"meaning": "work, job", "formality": "informal"},
            "laburar": {"meaning": "to work", "formality": "informal"},
            "guita": {"meaning": "money", "formality": "informal"},
            "morfar": {"meaning": "to eat", "formality": "informal"},
            "afanar": {"meaning": "to steal", "formality": "informal"},
            "quilombo": {"meaning": "mess, chaos", "formality": "vulgar"},
            "copado": {"meaning": "cool, great", "formality": "informal"},
            "piola": {"meaning": "cool, chill", "formality": "informal"},
            "groso": {"meaning": "awesome, great", "formality": "informal"},
            "trucho": {"meaning": "fake, sketchy", "formality": "informal"},
            "chorro": {"meaning": "thief", "formality": "informal"},
            "bondi": {"meaning": "bus", "formality": "informal"},
            "birra": {"meaning": "beer", "formality": "informal"},
            "fiaca": {"meaning": "laziness", "formality": "informal"},
            "garpar": {"meaning": "to pay", "formality": "informal"},
            "chamuyo": {"meaning": "smooth talk, BS", "formality": "informal"},
            "chamuyar": {"meaning": "to sweet talk", "formality": "informal"},
            "bancar": {"meaning": "to support, tolerate", "formality": "informal"},
            "bardear": {"meaning": "to insult, mess with", "formality": "informal"},
            "flashear": {"meaning": "to imagine, hallucinate", "formality": "informal"},
            "rescatarse": {"meaning": "to calm down", "formality": "informal"},
        },
        "phrases": {
            "che boludo": "hey dude",
            "qué onda": "what's up",
            "de una": "for sure, absolutely",
            "ni en pedo": "no way, not a chance",
            "al pedo": "for nothing, pointless",
            "en pedo": "drunk",
            "qué sé yo": "I don't know",
            "dale": "okay, go ahead",
            "todo bien": "all good",
            "la posta": "the truth",
            "a full": "at full speed, a lot",
            "re copado": "really cool",
            "qué bajón": "what a bummer",
            "está bueno": "it's good",
            "no da": "not cool, inappropriate",
        },
    },
    "spain": {
        "words": {
            "tío": {"meaning": "dude, guy", "formality": "informal"},
            "tía": {"meaning": "girl, woman", "formality": "informal"},
            "mola": {"meaning": "it's cool", "formality": "informal"},
            "molar": {"meaning": "to be cool", "formality": "informal"},
            "guay": {"meaning": "cool", "formality": "informal"},
            "currar": {"meaning": "to work", "formality": "informal"},
            "curro": {"meaning": "work, job", "formality": "informal"},
            "pasta": {"meaning": "money", "formality": "informal"},
            "flipar": {"meaning": "to freak out, be amazed", "formality": "informal"},
            "flipante": {"meaning": "amazing, crazy", "formality": "informal"},
            "rollo": {"meaning": "thing, vibe, story", "formality": "informal"},
            "chaval": {"meaning": "kid, young guy", "formality": "informal"},
            "chavala": {"meaning": "young girl", "formality": "informal"},
            "colega": {"meaning": "buddy, mate", "formality": "informal"},
            "quedarse": {"meaning": "to stay, be left", "formality": "neutral"},
            "tronco": {"meaning": "dude, buddy", "formality": "informal"},
            "mazo": {"meaning": "a lot, very", "formality": "informal"},
            "mogollón": {"meaning": "a ton, lots", "formality": "informal"},
            "majo": {"meaning": "nice, friendly", "formality": "informal"},
            "maja": {"meaning": "nice (female)", "formality": "informal"},
            "cutre": {"meaning": "cheap, tacky", "formality": "informal"},
            "chungo": {"meaning": "bad, sketchy", "formality": "informal"},
            "petarse": {"meaning": "to crash, break", "formality": "informal"},
            "pasada": {"meaning": "amazing thing", "formality": "informal"},
            "caña": {"meaning": "small beer", "formality": "neutral"},
            "botellón": {"meaning": "outdoor drinking party", "formality": "informal"},
            "movida": {"meaning": "scene, situation", "formality": "informal"},
            "quedar": {"meaning": "to meet up", "formality": "neutral"},
        },
        "phrases": {
            "qué guay": "how cool",
            "mola mucho": "it's really cool",
            "qué rollo": "what a drag",
            "buen rollo": "good vibes",
            "mal rollo": "bad vibes",
            "de puta madre": "awesome (vulgar)",
            "ir de cañas": "to go for beers",
            "qué pasada": "how amazing",
            "flipas": "you're crazy / no way",
            "eso mola": "that's cool",
            "qué fuerte": "how intense/crazy",
            "no me jodas": "don't mess with me",
            "vaya tela": "wow, unbelievable",
            "me mola": "I like it",
            "quedamos": "let's meet up",
        },
    },
    "chile": {
        "words": {
            "cachai": {"meaning": "you know? understand?", "formality": "informal"},
            "cachar": {"meaning": "to understand, catch", "formality": "informal"},
            "pololo": {"meaning": "boyfriend", "formality": "informal"},
            "polola": {"meaning": "girlfriend", "formality": "informal"},
            "pololear": {"meaning": "to date", "formality": "informal"},
            "fome": {"meaning": "boring, lame", "formality": "informal"},
            "bacán": {"meaning": "cool, awesome", "formality": "informal"},
            "po": {"meaning": "emphatic particle (pues)", "formality": "informal"},
            "weón": {"meaning": "dude, idiot", "formality": "very_informal"},
            "wea": {"meaning": "thing, stuff", "formality": "very_informal"},
            "weá": {"meaning": "thing (alt)", "formality": "very_informal"},
            "cuático": {"meaning": "crazy, intense", "formality": "informal"},
            "carrete": {"meaning": "party", "formality": "informal"},
            "carretear": {"meaning": "to party", "formality": "informal"},
            "copete": {"meaning": "alcoholic drink", "formality": "informal"},
            "luca": {"meaning": "1000 pesos", "formality": "informal"},
            "pega": {"meaning": "work, job", "formality": "informal"},
            "al tiro": {"meaning": "right away", "formality": "informal"},
            "tiro": {"meaning": "immediately", "formality": "informal"},
            "piola": {"meaning": "chill, low-key", "formality": "informal"},
            "latero": {"meaning": "annoying, boring", "formality": "informal"},
            "filete": {"meaning": "great, excellent", "formality": "informal"},
            "chancho": {"meaning": "pig, lucky", "formality": "informal"},
        },
        "phrases": {
            "cachai po": "you know?",
            "ya po": "come on, already",
            "dale po": "go ahead",
            "pucha": "darn",
            "qué lata": "what a drag",
            "la raja": "awesome (vulgar)",
            "buena onda": "good vibes",
            "mala onda": "bad vibes",
            "estar choreado": "to be fed up",
            "qué fome": "how boring",
            "está filete": "it's great",
            "a la pinta": "dressed up",
            "al lote": "hastily, carelessly",
        },
    },
}


def load_subtitles(input_dir: Path, region: Optional[str] = None) -> list[dict]:
    """Load subtitle manifests from directory."""
    subtitles = []

    if not input_dir.exists():
        logger.warning(f"Subtitle directory not found: {input_dir}")
        return subtitles

    if region:
        manifest_path = input_dir / region / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                subtitles.extend(json.load(f))
    else:
        for region_dir in input_dir.iterdir():
            if region_dir.is_dir():
                manifest_path = region_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, encoding="utf-8") as f:
                        subtitles.extend(json.load(f))

    return subtitles


def extract_slang_occurrences(
    subtitles: list[dict],
    region: str,
) -> dict:
    """Extract slang occurrences with context from subtitles."""
    slang_config = REGIONAL_SLANG.get(region, {})
    slang_words = set(slang_config.get("words", {}).keys())
    slang_phrases = set(slang_config.get("phrases", {}).keys())

    # Results
    word_occurrences = defaultdict(list)
    phrase_occurrences = defaultdict(list)
    new_candidates = Counter()

    for sub in subtitles:
        if sub.get("region") != region:
            continue

        title = sub.get("title", "Unknown")

        for entry in sub.get("entries", []):
            text = entry.get("text", "").lower()

            # Check for known slang words
            for word in slang_words:
                if re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE):
                    word_occurrences[word].append({
                        "text": entry.get("text"),
                        "source": title,
                        "timestamp": entry.get("timestamp"),
                    })

            # Check for known phrases
            for phrase in slang_phrases:
                if phrase.lower() in text:
                    phrase_occurrences[phrase].append({
                        "text": entry.get("text"),
                        "source": title,
                        "timestamp": entry.get("timestamp"),
                    })

            # Find potential new slang (words with unusual frequency patterns)
            words = re.findall(r'\b[a-záéíóúüñ]+\b', text, re.IGNORECASE)
            for word in words:
                if len(word) > 3 and word not in slang_words:
                    new_candidates[word] += 1

    return {
        "word_occurrences": dict(word_occurrences),
        "phrase_occurrences": dict(phrase_occurrences),
        "candidates": new_candidates.most_common(100),
    }


def build_slang_dictionary(
    subtitles: list[dict],
    region: str,
    min_occurrences: int = 2,
) -> dict:
    """Build comprehensive slang dictionary for a region."""
    slang_config = REGIONAL_SLANG.get(region, {})
    occurrences = extract_slang_occurrences(subtitles, region)

    dictionary = {
        "region": region,
        "words": {},
        "phrases": {},
        "statistics": {},
    }

    # Process words
    for word, info in slang_config.get("words", {}).items():
        examples = occurrences["word_occurrences"].get(word, [])
        dictionary["words"][word] = {
            **info,
            "occurrences": len(examples),
            "examples": examples[:5],  # Top 5 examples
        }

    # Process phrases
    for phrase, meaning in slang_config.get("phrases", {}).items():
        examples = occurrences["phrase_occurrences"].get(phrase, [])
        dictionary["phrases"][phrase] = {
            "meaning": meaning,
            "occurrences": len(examples),
            "examples": examples[:5],
        }

    # Add statistics
    total_words = sum(len(occ) for occ in occurrences["word_occurrences"].values())
    total_phrases = sum(len(occ) for occ in occurrences["phrase_occurrences"].values())

    dictionary["statistics"] = {
        "total_word_occurrences": total_words,
        "total_phrase_occurrences": total_phrases,
        "unique_words_found": len(occurrences["word_occurrences"]),
        "unique_phrases_found": len(occurrences["phrase_occurrences"]),
        "potential_new_slang": [
            {"word": word, "count": count}
            for word, count in occurrences["candidates"][:20]
        ],
    }

    return dictionary


def create_training_vocabulary(dictionaries: dict[str, dict]) -> dict:
    """Create unified vocabulary for STT training."""
    vocabulary = {
        "all_slang_words": set(),
        "all_slang_phrases": set(),
        "by_region": {},
    }

    for region, dictionary in dictionaries.items():
        vocabulary["all_slang_words"].update(dictionary.get("words", {}).keys())
        vocabulary["all_slang_phrases"].update(dictionary.get("phrases", {}).keys())
        vocabulary["by_region"][region] = {
            "words": list(dictionary.get("words", {}).keys()),
            "phrases": list(dictionary.get("phrases", {}).keys()),
        }

    # Convert sets to sorted lists
    vocabulary["all_slang_words"] = sorted(vocabulary["all_slang_words"])
    vocabulary["all_slang_phrases"] = sorted(vocabulary["all_slang_phrases"])

    return vocabulary


def main():
    parser = argparse.ArgumentParser(
        description="Mine slang expressions from Spanish subtitles"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/subtitles"),
        help="Directory containing downloaded subtitles (default: data/subtitles)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/slang"),
        help="Output directory for slang dictionaries (default: data/slang)",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["mexico", "argentina", "spain", "chile", "all"],
        default="all",
        help="Region to process (default: all)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update configs/default.yaml with expanded slang markers",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load subtitles
    logger.info(f"Loading subtitles from {args.input_dir}")
    subtitles = load_subtitles(args.input_dir, args.region if args.region != "all" else None)

    if not subtitles:
        logger.warning("No subtitles found. Using base slang dictionaries.")
        subtitles = []

    # Process regions
    regions = list(REGIONAL_SLANG.keys()) if args.region == "all" else [args.region]
    dictionaries = {}

    for region in regions:
        logger.info(f"\nProcessing region: {region}")

        dictionary = build_slang_dictionary(subtitles, region)
        dictionaries[region] = dictionary

        # Save regional dictionary
        output_path = args.output_dir / f"{region}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {output_path}")

        # Print summary
        stats = dictionary.get("statistics", {})
        print(f"\n{region.upper()} Summary:")
        print(f"  Words found: {stats.get('unique_words_found', 0)}")
        print(f"  Phrases found: {stats.get('unique_phrases_found', 0)}")
        print(f"  Total occurrences: {stats.get('total_word_occurrences', 0) + stats.get('total_phrase_occurrences', 0)}")

    # Create unified vocabulary
    vocabulary = create_training_vocabulary(dictionaries)
    vocab_path = args.output_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved unified vocabulary: {vocab_path}")

    # Update config if requested
    if args.update_config:
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Update slang markers
            config["slang_markers"] = {
                "lexical": vocabulary["all_slang_words"][:50],  # Top 50
                "regional": vocabulary["by_region"],
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            logger.info(f"Updated config: {config_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("SLANG MINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique words: {len(vocabulary['all_slang_words'])}")
    print(f"Total unique phrases: {len(vocabulary['all_slang_phrases'])}")
    print(f"\nDictionaries saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review potential new slang in each {args.output_dir}/<region>.json")
    print(f"  2. Use vocabulary.json for STT training")
    print(f"  3. Integrate with SpeakEasy slang learning module")


if __name__ == "__main__":
    main()
