# System architecture and Phase 1 baseline

## Product boundary

The target is continuous sign-language translation from a live camera to text and interruptible speech. Sign languages are separate linguistic systems: each recognizer, label inventory, training corpus, and realization layer is explicitly configured. ASL data or output is never represented as Bangla Sign Language (BdSL).

The current repository is a clean baseline, not a working translator. Until a suitable trained checkpoint exists, recognition status is `model_unavailable`; no guessed text is emitted.

## Data flow

```text
Browser camera (local; no recording)
  → adaptive frame sampling in a worker
  → hand + upper-body pose + face landmarks and visibility
  → stable, timestamped signer stream
  → temporal recognizer + boundary/sequence decoder
  → confidence-bearing sign/semantic hypotheses
  → language-specific conservative realization
  → stable partial/final transcript
  → cancellable phrase-level TTS queue
```

The default wire format carries landmarks and metadata, not image or video bytes. Camera permission, model status, and local/cloud processing mode are visible states. Network loss, denied camera access, and model load errors must be isolated and explained in the UI.

## Components and interfaces

| Component | Responsibility | Initial implementation choice |
| --- | --- | --- |
| Camera client | Permission, device selection, preview, timestamps | Browser MediaDevices, React/TypeScript |
| Landmark adapter | Hands, pose, face, visibility and handedness | MediaPipe Holistic Landmarker, local runtime |
| Tracker/stream | Stable identities, sequence ordering, bounded latest-frame queue | Client worker; monotonic timestamps |
| Recognition service | Temporal decoding and confidence/boundary estimates | Python/FastAPI WebSocket; PyTorch training; ONNX Runtime deployment |
| Language realization | Convert sign/semantic hypotheses to target-language text conservatively | Language-specific adapter, no generic auto-completion |
| TTS | Local fallback, voice/rate/volume, cancellation and bounded queue | Browser SpeechSynthesis first |
| Evaluation | Signer-independent metrics and latency reports | Python training/evaluation tooling, added with training phase |

The WebSocket accepts only versioned JSON events validated against `contracts/stream-event.schema.json`. The schema intentionally excludes raw frames. Production transport also needs message-size limits, rate limits, origin checks, and bounded queues.

### Temporal recognition design

The main model is not a frame classifier. The planned model consumes a sliding timestamped sequence with left/right hand identity, per-landmark visibility, pose and face features. A spatial/relational encoder feeds a temporal Transformer or temporal convolution encoder; a CTC or boundary-aware sequence decoder emits a sequence with timestamps and calibrated confidence. Architecture selection is finalized against the chosen language dataset and target hardware. Tracking and temporal smoothing handle short dropouts, variable speed, and pose holds; a held pose cannot by itself produce repeated signs.

The recognition adapter contract must support model ID/version, sign-language ID, health/loading state, partial and final hypotheses, uncertainty, and latency. Model absence or incompatible language must fail closed.

### Language and uncertainty

Recognition output is not equivalent to a spoken sentence. A language-specific adapter maps sign/semantic hypotheses to target-language text and attaches provenance/confidence to edits. It must not add unsupported content. Low-confidence signs remain marked uncertain or uncommitted; context may resolve only when supported by decoder evidence. BdSL-to-Bengali realization requires BdSL data and linguistic expertise; ASL-to-English models do not satisfy it.

### Speech behavior

Speak only stable phrase/sentence commits, not every partial update. Keep the transcript independent of speech state. Use one cancellable queue, discard stale queued phrases when new signing begins, and provide stop/retry controls. Browser speech is a local fallback; voice quality and startup latency vary by browser/device and must be measured.

## Dataset and training strategy

Dataset adapters must preserve source/license, language, signer, session, sequence boundaries, and annotation provenance. Candidate datasets include WLASL/MS-ASL for ASL and PHOENIX-Weather for its specific task/domain; they are not interchangeable. BdSL requires a legally usable, appropriately annotated BdSL corpus or a consented data-collection effort.

Split at signer level before clip/window generation and augmentation. Keep train/validation/test signers disjoint where dataset size permits. Record preprocessing and split manifests; augment only training data. Evaluate sequence recognition with WER, sign error rate, precision/recall/F1 where applicable, signer counts, sample counts, model version, hardware, and latency. Never describe a benchmark as real-world performance. No accuracy is claimed until measured.

Training lifecycle: ingest and validate metadata → signer split → preprocess/landmarks → train temporal model → tune/calibrate on validation → evaluate once on held-out test → checkpoint/version → export ONNX → runtime parity and latency check.

## Real-time and privacy requirements

- Camera capture and inference stay off the rendering path; use workers and asynchronous service calls.
- Use bounded queues with latest-frame semantics under load; drop stale frames rather than accumulating delay.
- Start with adaptive sampling and tracking; reduce sampling rate when end-to-end latency exceeds the configured budget.
- Reuse buffers and batch only when batching does not increase interactive latency.
- Report camera FPS, landmark/inference/recognition latency, TTS start delay, confidence, model state, and CPU/GPU metrics only when available.
- Do not log/store raw video. Keep datasets and checkpoints out of version control. Cloud processing is explicit opt-in.

## Failure states

Camera denial → explain browser permission/device selection. Missing or incompatible model → keep app usable and display recognition unavailable. TTS failure → retain text and expose retry. Slow service → drop stale work and expose degraded status. WebSocket disconnect → reconnect with bounded backoff without retaining camera frames. Each subsystem failure must not crash the rest of the interface.

## Planned repository layout

```text
contracts/       Versioned stream/event schemas
configs/         Safe defaults; no secrets or model weights
docs/            Architecture, data, training, deployment guidance
frontend/        React/TypeScript camera and accessible UI (Phase 2)
backend/         FastAPI lifecycle, validation, streaming (Phase 3)
vision/          Landmark adapters and tracking (Phase 3)
recognition/     Temporal model interface, decoder, runtime (Phase 4+)
language/        Language-specific realization (Phase 6)
tts/             Speech adapters and queue (Phase 6)
training/        Dataset adapters, training, evaluation, export (Phase 5+)
models/          Locally supplied versioned checkpoints; ignored
data/            Local datasets; ignored
```

Implemented foundations now include the React/Vite camera client and local landmark worker in `frontend/`, a loopback-only FastAPI status/landmark WebSocket in `backend/app/`, a fixed landmark feature layout, a temporal CTC/Transformer model baseline, and signer-split/training entry points under `training/`. The local service deliberately returns `model_unavailable`; model manifest parsing alone never activates inference.

## Development phases

1. **Foundation:** architecture decisions, privacy/capability rules, versioned stream contract, and safe configuration. Complete.
2. **Camera/UI:** accessible React shell, device/permission states, worker-based capture, local-processing indicator, and metrics layout. Implemented; needs device-level visual validation.
3. **Vision/transport:** local landmark adapter, temporal sampling, WebSocket validation, health/error states, and no-video transport. Implemented as a local landmark/status path; needs browser/device validation.
4. **Recognition runtime:** a local PyTorch temporal checkpoint loader, language/feature compatibility checks, CTC gloss decoder, confidence/timing events, and raw-gloss UI are implemented. Sequence stability/calibration and model evaluation remain.
5. **Language-specific model:** signer-split tooling and a training baseline exist. A legally usable, annotated dataset and trained checkpoint are still required; no inference quality is claimed.
6. **Language/TTS:** UI controls and browser speech fallback controls are present, but sentence realization and recognition-fed phrase speech remain to implement.
7. **Hardening:** accessibility review, failure recovery, security limits, device benchmarks, dataset/model documentation, and browser camera verification remain.

## Hardware assumptions and limits

Phase 2 camera/UI can run on a modern browser-capable laptop with a webcam. Landmark inference may run on CPU at reduced sampling or use supported GPU acceleration. Training a useful temporal model generally requires substantially more compute and, more importantly, an adequate language-specific labeled dataset. Exact FPS, latency, and accuracy cannot be specified without a target device, checkpoint, and evaluation protocol. Facial/non-manual cues, occlusion, signer variation, and regional variants require data and validation; landmarks alone do not guarantee understanding.
