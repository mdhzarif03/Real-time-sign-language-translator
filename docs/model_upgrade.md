# Model upgrade notes

## What changed

The project moved from `spatiotemporal-landmark-ctc-v3` to `spatiotemporal-landmark-ctc-v4-gru`.

1. **Coordinate normalization**: hand/body/face coordinates are root-centered and scale-normalized while preserving orientation, reducing sensitivity to camera distance and user size.
2. **Fast spatial encoding**: hand and pose graphs remain topology-aware; the 478-point face stream now uses attention pooling instead of a 478×478 message matrix.
3. **Temporal model**: a compact GRU consumes the fused landmark embedding plus frame-to-frame differences. This is causal within each sliding window and avoids quadratic Transformer attention.
4. **Training augmentation**: training-only landmark jitter, anatomical stream dropout, and mild temporal frame dropout improve detector-miss and timing robustness.
5. **Optimization**: AdamW, gradient clipping, CUDA mixed precision, ReduceLROnPlateau, and early stopping are enabled.
6. **Runtime cadence**: the browser samples landmarks more frequently and the API performs recognition every two observations instead of every four, reducing visible update delay.

## Why this direction matches the supplied research

The uploaded 2026 IJRIAS paper describes MediaPipe + CNN as a practical real-time baseline, but also lists lighting, fast motion, user variation, continuous text generation, and real-time processing as limitations. It reports roughly 15–20 FPS on a standard laptop for its implementation. The 2025 comparative paper reports that its 3D-CNN reached higher offline accuracy than its LSTM but with substantially higher inference cost, while the LSTM was more responsive on low-resource hardware. These are dataset-specific experimental results, not universal guarantees.

The upgraded model therefore targets the project's actual bottleneck: efficient temporal inference from landmarks rather than raw-video 3D convolutions.

## Data is still the critical missing component

The source tree does not contain training videos. The papers also do not supply a training corpus that can simply be bundled into this project. A real checkpoint requires a language-specific corpus, verified rights, gloss/translation annotations, and signer-independent splits.

For ASL research, relevant public resources include:

- **ASL Citizen**: about 83,399 isolated videos, 2,731 signs, 52 signers, with signer-independent train/validation/test splits. The Microsoft project page says commercial use requires contacting the project and provides the dataset download separately.
- **How2Sign**: more than 80 hours of continuous ASL with gloss annotations and English translations. It is research-only under CC BY-NC 4.0.
- **YouTube-ASL**: about 984 hours, 11,093 videos and 610,193 English captions. It is useful for continuous ASL research, subject to the dataset's source/copyright terms.
- **WLASL/MS-ASL**: large isolated/word-level ASL resources useful for pretraining or isolated recognition, but WLASL and MS-ASL are governed by C-UDA terms and are not suitable as unrestricted production data.

Bangla Sign Language must be trained separately. An ASL checkpoint must never be presented as a BdSL model.

## Required training sequence

1. Select one sign language and document the corpus license.
2. Preserve signer and recording-session identity.
3. Assign signer-disjoint train/validation/test splits before feature extraction.
4. Extract the v2-normalized 2,212-value landmark features.
5. Train the v4 GRU/CTC model with augmentation only on the training split.
6. Tune architecture and decoding on validation signers.
7. Freeze decisions and evaluate the held-out test signers exactly once.
8. Report gloss WER/sign error rate, sentence error rate, precision/recall/F1, signer counts, and model-only p50/p95 latency.
9. Only then wire the checkpoint into the local runtime.
10. Add a separately evaluated gloss-to-natural-language realization layer. Gloss output is not automatically a grammatical English or Bengali translation.
