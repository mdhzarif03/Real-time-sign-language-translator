# Real-time sign language translator

An extensible, privacy-first foundation for continuous sign-language translation. Recognition is language-specific and temporal. This repository currently contains architecture and wire/configuration contracts only; it does not contain a trained recognition model or claim translation accuracy.

## Current state

- Repository was initially empty apart from `.gitattributes`.
- Architecture and initial configuration/contracts are recorded in [`docs/architecture.md`](docs/architecture.md).
- No camera capture, landmark inference, translation, or speech implementation has been added yet.
- No model checkpoint is configured. The application must report this state instead of generating predictions.

## Architecture decisions

- **Client:** React + TypeScript + Vite for camera permission, preview, accessible controls, and transcript. Capture and rendering stay in the browser; compute-heavy work must not block the UI thread.
- **Vision:** MediaPipe Tasks is the initial candidate for local hand, pose, and face landmarks. It extracts visual features; it is not a sign-language translator. The adapter remains replaceable.
- **Service:** Python + FastAPI WebSocket for local streaming inference and health/status. Transmit timestamped landmarks rather than video by default.
- **ML:** PyTorch for training and experiment tracking; ONNX Runtime for deployment when target hardware supports it. Temporal encoder and sequence decoder are behind a language-specific recognizer interface.
- **Language/TTS:** Independent language realization with conservative confidence handling; browser SpeechSynthesis as the first local TTS adapter, replaceable by a streaming provider.

These are implementation choices for the next phases, not dependencies installed by this foundation commit.

## Planned local development commands

The repository has not yet been initialized as a frontend or Python package. When Phase 2 begins, use these commands to add the client in its own directory and establish a Python environment without replacing the repository:

```powershell
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
cd ..
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic-settings
```

Do not install a framework-specific ML runtime until the target platform and supported model are selected. Dataset and checkpoint files belong outside version control.

## Privacy and capability policy

Local processing is the default. Do not persist raw camera frames or log them. Any future cloud mode requires an explicit opt-in and a clear UI indicator. Until a language-specific checkpoint has been trained and evaluated on signer-independent data, the product must show that recognition is unavailable; it must not substitute hard-coded gestures, random predictions, or an unrelated sign-language model.
