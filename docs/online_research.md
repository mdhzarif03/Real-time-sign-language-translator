# Online research used for the v4 upgrade

Checked 2026-10-01.

## Relevant datasets

- Microsoft ASL Citizen: https://www.microsoft.com/en-us/research/project/asl-citizen/
  - About 83,399 videos, 2,731 signs, 52 signers.
  - Signer-independent train/validation/test splits are provided.
  - The project page says commercial use requires contacting the dataset team.
  - The official download is about 42.8 GB.
- How2Sign: https://how2sign.github.io/
  - More than 80 hours of continuous ASL with English translations, gloss annotations and multiple modalities.
  - Research-only, CC BY-NC 4.0.
  - The downloadable video/keypoint material is much larger than a normal source repository.
- YouTube-ASL: https://arxiv.org/abs/2306.15162
  - About 984 hours, 11,093 videos and 610,193 English captions in the released corpus description.
  - Useful for continuous ASL research, but source-video and copyright terms must be respected.
- WLASL: https://github.com/dxli94/WLASL
  - 2,000 word-level ASL classes.
  - C-UDA; academic/computational use only according to the dataset card.
- MS-ASL: https://arxiv.org/abs/1812.01053
  - More than 25,000 annotated videos, 1,000 signs and more than 200 signers.
  - Useful for isolated/word-level ASL research and signer-independent benchmarking.

## Relevant implementation guidance

Google's MediaPipe documentation confirms that Holistic Landmarker exposes hand, pose and face landmarks and supports image/video/live-stream modes. The live-stream API is explicitly designed to lower latency by allowing the task to drop inputs when necessary. The repository continues to process camera data locally.

## Model decision

The uploaded 2026 paper describes a MediaPipe + CNN real-time ASL prototype and lists lighting, fast motion, signer variation, continuous text generation and processing speed as practical problems. The uploaded 2025 comparison reports 92.4% accuracy for its 3D CNN and 86.7% for its LSTM, but also reports about 65 ms versus 20 ms inference time and substantially larger resource requirements for the 3D CNN. Those are results on that paper's dataset and setup, not universal performance claims.

The v4 design therefore uses:

- landmark normalization instead of raw-pixel video;
- hand/pose graph encoding;
- attention pooling for the face instead of a large face graph;
- a compact GRU for temporal modeling;
- explicit frame differences for motion;
- training-only augmentation;
- mixed precision, gradient clipping, learning-rate reduction and early stopping;
- more frequent runtime updates with a small sliding window.

## Important limitation

No real sign-language training corpus was uploaded with this project. The ZIP contains code and tests, but no videos or landmark arrays. Therefore no claim of a production-accurate ASL or BdSL model can be made from the supplied material alone. A synthetic smoke-training run was used only to validate the training loop; its checkpoint is deliberately not shipped as a translator model.
