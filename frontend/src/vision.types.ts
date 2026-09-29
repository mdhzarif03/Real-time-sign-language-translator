export type Point = { x: number; y: number; z: number; visibility: number };

export type LandmarkFrame = {
  hands: Array<{ handedness: 'Left' | 'Right'; points: Point[] }>;
  pose: Point[];
  face: Point[];
  inferenceMs: number;
  timestampMs: number;
};

export type VisionWorkerMessage =
  | { type: 'ready'; delegate: 'GPU' | 'CPU' }
  | { type: 'result'; frame: LandmarkFrame }
  | { type: 'error'; message: string };

export type VisionWorkerCommand =
  | { type: 'init' }
  | { type: 'frame'; bitmap: ImageBitmap; timestampMs: number };
