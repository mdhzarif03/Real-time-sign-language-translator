# Training a language-specific sequence recognizer

The spatial graph encoders in `backend/recognition/temporal_model.py` encode each hand's joint topology and upper-body pose graph, pool facial landmarks, then use a temporal Transformer with CTC gloss and boundary heads. It is an architecture baseline, not a trained translator. The output is a sign/gloss sequence; a separate language-specific realization layer is still required to produce Bengali or English sentences.

## Dataset selection and use rights

Dataset availability does not imply production-use rights. Confirm the source license and any signer consent restrictions before downloading, training, or distributing derived weights. The currently documented corpora do not provide an obvious, verified, commercially deployable continuous corpus for both target languages:

- [How2Sign](https://how2sign.github.io/) provides continuous ASL with English translations and signer splits, but is research-only and licensed CC BY-NC 4.0. It is not suitable for a commercial deployment without separate permission.
- [WLASL](https://github.com/krazyjoy/WLASL) is word-level ASL, governed by its C-UDA, and its maintainers explicitly prohibit commercial use. It can inform isolated-sign experiments, not establish continuous sentence translation.
- [BdSLW60](https://arxiv.org/abs/2402.08635) is a 60-word, 9,307-trial BdSL dataset with 18 signers. The paper reports 75.1% testing accuracy for its attention BiLSTM baseline; this is a word-level benchmark, not an end-to-end sentence translator. Obtain and review the dataset's actual terms before use.
- [Ban-Sign-Sent-9K-V1](https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1) describes 1,922 continuous BdSL sentences and 9,610 videos, but its card currently has no license metadata. Do not use or redistribute it until the rights and signer split are verified with its maintainers.

This repository intentionally does not fetch or bundle these datasets. Until a suitable, licensed corpus is selected and signer-independent evaluation is completed, no language checkpoint should be marked production-ready.

## Data contract

Raw videos are not bundled. A corpus-specific adapter must provide trusted per-sign gloss timestamps and keep every source video/signers within the dataset license. `prepare_landmarks.py` performs video decoding, 12.5 Hz sampling, holistic feature extraction, and boundary-target generation using the same 2,212-float layout as runtime. It requires Python 3.12 plus the training dependencies.

First, write metadata JSONL rows like the following. `start_ms` and `end_ms` must be corpus annotations reviewed by a fluent sign-language annotator; do not synthesize them from sentence duration or model predictions:

```json
{"sample_id":"clip-0001","signer_id":"signer-01","session_id":"recording-2025-01","language":"en-US-ASL","video_path":"videos/clip-0001.mp4","sentence_end_ms":1710,"segments":[{"gloss":"GO","start_ms":480,"end_ms":920},{"gloss":"SCHOOL","start_ms":1040,"end_ms":1710}]}
```

Assign signer splits before feature extraction, define a language-specific gloss vocabulary, then extract features:

```powershell
python training/split_by_signer.py data/raw/annotations.jsonl data/processed/annotations.split.jsonl --seed 2026
python training/prepare_landmarks.py data/processed/annotations.split.jsonl data/processed/asl-vocabulary.json data/raw data/processed --model frontend/src/vision-assets/models/holistic_landmarker.task --sample-hz 12.5
```

The second command creates `manifest.features.jsonl` with `.npy` arrays and boundary targets. It rejects unknown glosses, missing signer splits, traversal outside the dataset root, and segments that are too short at the selected sampling rate.

Each JSONL row in the training manifest contains:

```json
{"sample_id":"s01-clip001","signer_id":"s01","session_id":"recording-s01-01","sequence_id":"clip001","language":"en-US-ASL","feature_file":"features/s01-clip001.npy","gloss_ids":[4,17],"split":"train"}
```

For model training, include a `boundary_file` path to an int64 NumPy array with one label per feature timestep: 0=outside a sign, 1=sign onset, 2=sign interior, 3=non-final sign offset, 4=sentence end. The final gloss end must align with the manually annotated `sentence_end_ms`; the runtime commits a sentence only from class 4, not from every within-sentence sign boundary. Boundary labels must come from annotations or a documented alignment process, not frame heuristics. Vocabulary IDs are contiguous from 1; CTC blank is reserved at 0.

Feature arrays are float32 `[time, 2212]`: left hand (21×4), right hand (21×4), upper pose (33×4), face (478×4). The four channels are x, y, z, visibility. This exact layout is also implemented by `backend/app/features.py`. The vocabulary reserves CTC blank at ID 0; gloss IDs start at 1. Record the dataset license, source, annotation policy, signer, recording session, and split manifest alongside the data.

## Leakage controls

Assign signer IDs to train/validation/test before extracting overlapping windows or augmenting samples. A signer must occur in one split only. Keep all clips from one recording session together. Fit normalization statistics on training data only. Do not augment validation/test. Report sample and signer counts per split. For PHOENIX or any other domain-specific corpus, describe the domain and do not imply general conversational coverage.

Create an initial deterministic signer split from a metadata JSONL file with:

```powershell
python training/split_by_signer.py data/raw/manifest.jsonl data/processed/manifest.split.jsonl --seed 2026
```

The input must contain unique `sample_id`, `signer_id`, and `session_id` fields and must not already contain `split` assignments. The splitter keeps both signers and recording sessions disjoint, including sessions containing more than one signer. Review the emitted signer/sample counts and preserve the output manifest with the experiment.

## Training requirements

Use legally obtained data with sequence-level gloss annotations and signer metadata. Start with a small vocabulary from one sign language, train the CTC gloss model, tune on validation signers, and report held-out signer WER/sign error rate plus latency and hardware. Save vocabulary, feature layout/version, language ID, data/split hashes, preprocessing configuration, training code revision, and metrics in every checkpoint. Add ONNX export only after validating numerical parity on held-out examples.

Bangla Sign Language needs its own annotated dataset, vocabulary, signer-independent evaluation and Bengali realization rules. ASL checkpoints must fail language compatibility checks.

Train the baseline from the repository root (install `training/requirements.txt` in a dedicated virtual environment first):

```powershell
python training/train_ctc.py data/processed/manifest.split.jsonl data/processed/vocabulary.json models/asl-temporal.pt --language en-US-ASL --epochs 50
```

The script checks signer split isolation, reports validation gloss WER each epoch, saves the best validation checkpoint, then evaluates that checkpoint once on the held-out test split. These numbers describe the supplied dataset only. The local API can load the checkpoint with the manifest below; sentence realization remains subsequent implementation work.

For a reproducible standalone held-out report (sequence WER, sentence error rate, token precision/recall/F1, edit counts, signer/sample counts, hashes, hardware/software, and model-only latency percentiles), run:

```powershell
python training/evaluate.py data/processed/manifest.features.jsonl models/asl-temporal.pt --language en-US-ASL --output reports/asl-heldout.json
```

Model latency excludes video decoding and landmark extraction; do not present it as end-to-end latency. The held-out test set should only be evaluated after model/configuration decisions are frozen.

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
