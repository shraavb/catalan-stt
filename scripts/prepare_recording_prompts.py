#!/usr/bin/env python3
"""Prepare recording prompts from existing dialogue data."""

import json
import pandas as pd
from pathlib import Path
import random


def main():
    data_dir = Path("data/transcripts")
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dialogues
    dialogues_path = data_dir / "selected_dialogues_for_recording.json"

    if dialogues_path.exists():
        with open(dialogues_path, "r", encoding="utf-8") as f:
            dialogues = json.load(f)
    else:
        # Fallback to CSV
        csv_path = data_dir / "spanish_slang_phrases.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dialogues = df.to_dict("records")
        else:
            print("No dialogue data found!")
            return

    # Extract unique phrases for recording
    phrases = []
    seen = set()

    for d in dialogues:
        # Handle different data formats
        if isinstance(d, dict):
            text = d.get("text", d.get("prompt", d.get("response", "")))
        else:
            text = str(d)

        # Clean and dedupe
        text = text.strip()
        if text and text not in seen and len(text) > 5:
            seen.add(text)
            phrases.append({
                "id": len(phrases) + 1,
                "text": text,
                "recorded": False,
            })

    # Limit and shuffle
    random.seed(42)
    random.shuffle(phrases)
    phrases = phrases[:200]  # Start with 200 prompts

    # Save recording prompts
    prompts_path = output_dir / "recording_prompts.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)

    print(f"Created {len(phrases)} recording prompts at {prompts_path}")

    # Also create a simple text file for easy reading while recording
    txt_path = output_dir / "recording_prompts.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in phrases:
            f.write(f"{p['id']:03d}. {p['text']}\n\n")

    print(f"Created text file at {txt_path}")

    # Print sample
    print("\nSample prompts to record:")
    print("-" * 50)
    for p in phrases[:10]:
        print(f"  {p['id']:03d}. {p['text']}")


if __name__ == "__main__":
    main()
