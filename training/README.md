# Training a language-specific sequence recognizer

The temporal model in `backend/recognition/temporal_model.py` predicts CTC gloss sequences and frame-level boundary classes from timestamped landmarks. It is an architecture baseline, not a trained translator. The output is a sign/gloss sequence; a separate language-specific realization layer is still required to produce Bengali or English sentences.

## Data contract

Each JSONL row in the training manifest contains:

```json
{"sample_id":"s01-clip001","signer_id":"s01","sequence_id":"clip001","language":"en-US-ASL","feature_file":"features/s01-clip001.npy","gloss_ids":[4,17],"split":"train"}
```

For model training, include a `boundary_file` path to an int64 NumPy array with one label per feature timestep: 0=outside a sign, 1=sign onset, 2=sign interior, 3=sign offset. Boundary labels must come from annotations or a documented alignment process, not frame heuristics. Vocabulary IDs are contiguous from 1; CTC blank is reserved at 0.

Feature arrays are float32 `[time, 2212]`: left hand (21×4), right hand (21×4), upper pose (33×4), face (478×4). The four channels are x, y, z, visibility. This exact layout is also implemented by `backend/app/features.py`. The vocabulary reserves CTC blank at ID 0; gloss IDs start at 1. Record the dataset license, source, annotation policy, signer, recording session, and split manifest alongside the data.

## Leakage controls

Assign signer IDs to train/validation/test before extracting overlapping windows or augmenting samples. A signer must occur in one split only. Keep all clips from one recording session together. Fit normalization statistics on training data only. Do not augment validation/test. Report sample and signer counts per split. For PHOENIX or any other domain-specific corpus, describe the domain and do not imply general conversational coverage.

Create an initial deterministic signer split from a metadata JSONL file with:

```powershell
python training/split_by_signer.py data/raw/manifest.jsonl data/processed/manifest.split.jsonl --seed 2026
```

The input must contain unique `sample_id` and `signer_id` fields and must not already contain `split` assignments. Review the emitted signer/sample counts and preserve the output manifest with the experiment.

## Training requirements

Use legally obtained data with sequence-level gloss annotations and signer metadata. Start with a small vocabulary from one sign language, train the CTC gloss model, tune on validation signers, and report held-out signer WER/sign error rate plus latency and hardware. Save vocabulary, feature layout/version, language ID, data/split hashes, preprocessing configuration, training code revision, and metrics in every checkpoint. Add ONNX export only after validating numerical parity on held-out examples.

Bangla Sign Language needs its own annotated dataset, vocabulary, signer-independent evaluation and Bengali realization rules. ASL checkpoints must fail language compatibility checks.

Train the baseline from the repository root (install `training/requirements.txt` in a dedicated virtual environment first):

```powershell
python training/train_ctc.py data/processed/manifest.split.jsonl data/processed/vocabulary.json models/asl-temporal.pt --language en-US-ASL --epochs 50
```

The script checks signer split isolation, reports validation gloss WER each epoch, saves the best validation checkpoint, then evaluates that checkpoint once on the held-out test split. These numbers describe the supplied dataset only. The local API can load the checkpoint with the manifest below; sentence realization remains subsequent implementation work.

To opt a trained checkpoint into local gloss inference, save a manifest beside the checkpoint (for the example command, `models/asl-temporal.manifest.json`):

```json
{
  "model_id": "asl-landmark-ctc",
  "model_version": "1",
  "sign_language": "en-US-ASL",
  "checkpoint": "asl-temporal.pt",
  "feature_layout": "hands-left-right-21x4_pose-33x4_face-478x4_v1",
  "window_size": 96
}
```

Set the environment variable before starting the API:

```powershell
$env:SIGNFLOW_MODEL_MANIFEST = "models/asl-temporal.manifest.json"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The service verifies the language, checkpoint metadata, vocabulary, and feature layout before loading. It then emits temporal gloss hypotheses only. Natural-language sentence generation and speech remain unavailable until a separately evaluated language realization layer is configured. The model's softmax confidence is an uncalibrated score; calibrate it on held-out signers before using it as a probability.
