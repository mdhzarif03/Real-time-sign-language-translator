# Real-time sign language translator

Privacy-first foundation for continuous sign-language translation. The live camera and local hand/pose/face landmark pipeline are implemented. No trained sign-language recognition checkpoint is included, so the app does not fabricate translations or claim accuracy.

## Current state

- The web client captures camera video locally and runs MediaPipe Holistic Landmarker for face, pose, and both hands in a dedicated worker.
- Local recognizer WebSocket failures retry with capped exponential backoff; reconnects receive a new stream ID, and stopping capture cancels pending retries.
- Camera frames are not uploaded or recorded. Local vision task files are downloaded once into the frontend's public assets.
- Translation and speech output are unavailable until a compatible temporal recognition model and language realization layer are supplied.
- Architecture and stream contracts: [`docs/architecture.md`](docs/architecture.md), [`contracts/stream-event.schema.json`](contracts/stream-event.schema.json).

## Run the client

Requirements: Node.js 20.19+ or 22.12+, npm, a webcam, and a browser with camera and module-worker support. For the local Python API use Python 3.12. Run from the repository root:

```powershell
cd frontend
npm install
npm run setup:vision
npm run dev
```

Open the localhost URL printed by Vite. `setup:vision` downloads official task models into the local, ignored `frontend/src/vision-assets/models` directory; a SHA-256 manifest is written alongside them. Vite bundles those models and the pinned WASM runtime into local static assets. Camera video remains local and no model/CDN downloads happen while the app is running.

Run the local API in a second PowerShell window from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Install the API test client before running the full test suite:

```powershell
python -m pip install -r tests/requirements.txt
```

The API accepts validated landmark messages only from loopback clients, enforces message/rate limits, and reports recognition unavailable by default. To load a trained checkpoint, follow the manifest instructions in [`training/README.md`](training/README.md). Loaded models emit raw gloss hypotheses; natural-language realization is not implemented.

## Architecture choices

- React + TypeScript client for camera permissions, responsive controls, and transcript UI.
- MediaPipe Tasks Vision in a module worker for hand, pose, and face landmark extraction. It is not a sign-language translator.
- Future recognition service: FastAPI WebSocket, PyTorch for training, ONNX Runtime for deployment.
- Browser SpeechSynthesis is the intended local TTS fallback after stable translated text exists.

ASL and Bangla Sign Language are distinct languages requiring independent training data, models, and evaluation. No model is configured in `configs/default.yaml`. Do not treat landmarks as sign predictions.

The temporal training baseline and signer-split tooling are documented in [`training/README.md`](training/README.md). Training requires a suitable annotated dataset; no sample dataset or pretrained sign model is bundled.

## Checks

Run the repository checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm test
npm run build
```

To run the PyTorch model and model-forward tests, create a dedicated Python 3.12 environment and install `training/requirements.txt`; keep it separate from the lightweight API environment. See [`training/README.md`](training/README.md) for dataset preparation, training, held-out evaluation, and checkpoint loading.

## Privacy and limitations

Local processing is the default. No raw frames are persisted, logged, or sent to a server. The current version visualizes extracted landmarks only. It cannot yet interpret sign sequences or speak translations. Accuracy, latency, and generalization have not been benchmarked.
