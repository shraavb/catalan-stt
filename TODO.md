# TODO

## Approach 2: Regional Fine-tuning

### Run Training on RunPod
```bash
# SSH into RunPod
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Navigate to repo
cd /workspace/spanish-slang-stt

# Run regional fine-tuning for all 4 regions
export PYTHONPATH=/workspace/spanish-slang-stt:$PYTHONPATH
python scripts/train_regional.py --all-regions \
  --train-manifest data/splits/train.json \
  --val-manifest data/splits/val.json \
  --base-model openai/whisper-small
```

Regions to train:
- [ ] Spain (9,114 samples)
- [ ] Mexico (14,133 samples)
- [ ] Argentina (4,701 samples)
- [ ] Chile (3,490 samples)

### Upload Models to HuggingFace
```bash
python scripts/upload_model.py \
  --model-path models/spanish-slang-whisper-<region>/final \
  --repo-id shraavb/spanish-slang-whisper-<region> \
  --region <region>
```

- [ ] Upload spain model
- [ ] Upload mexico model
- [ ] Upload argentina model
- [ ] Upload chile model

### Evaluate Regional Models
- [ ] Run evaluation against baseline Whisper
- [ ] Compare WER across regions
- [ ] Document results

## Notes
- Data archive: `spanish-stt-data.tar` (6.3GB, ~38.6k audio files)
- Data already uploaded to RunPod at `/workspace/spanish-slang-stt/`
- Remember to rename `scripts/evaluate.py` to `scripts/evaluate_model.py` on RunPod to avoid HuggingFace evaluate package conflict
