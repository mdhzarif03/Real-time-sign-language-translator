/// <reference lib="webworker" />
import {
  FilesetResolver,
  HolisticLandmarker,
} from '@mediapipe/tasks-vision';
import visionWasmLoader from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_module_internal.js?url';
import visionWasmBinary from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_module_internal.wasm?url';
import visionWasmNoSimdLoader from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js?url';
import visionWasmNoSimdBinary from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm?url';
import holisticModel from './vision-assets/models/holistic_landmarker.task?url';
import type { LandmarkFrame, Point, VisionWorkerCommand } from './vision.types';

declare const self: DedicatedWorkerGlobalScope;

let holisticTask: HolisticLandmarker | undefined;

function points(values: Array<{ x: number; y: number; z: number; visibility?: number }> | undefined): Point[] {
  return (values ?? []).map(({ x, y, z, visibility }) => ({ x, y, z, visibility: visibility ?? 1 }));
}

async function createTasks(delegate: 'GPU' | 'CPU') {
  const files = await FilesetResolver.forVisionTasks('', true);
  const simd = await FilesetResolver.isSimdSupported(true);
  const runtimeFiles = {
    ...files,
    wasmLoaderPath: simd ? visionWasmLoader : visionWasmNoSimdLoader,
    wasmBinaryPath: simd ? visionWasmBinary : visionWasmNoSimdBinary,
  };
  holisticTask = await HolisticLandmarker.createFromOptions(runtimeFiles, {
    baseOptions: { delegate, modelAssetPath: holisticModel },
    runningMode: 'VIDEO',
    minFaceDetectionConfidence: 0.45,
    minFacePresenceConfidence: 0.45,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minHandLandmarksConfidence: 0.45,
    // Face landmarks carry the non-manual features needed by the recognizer.
    // Blendshape inference adds a second model graph that is not supported by
    // every WebGL delegate/browser combination.
    outputFaceBlendshapes: false,
  });
}

self.onmessage = async ({ data }: MessageEvent<VisionWorkerCommand>) => {
  if (data.type === 'init') {
    try {
      // Holistic combines several graphs. MediaPipe's WebGL delegate can
      // initialize successfully yet fail during face/pose inference in some
      // browser drivers, so use the portable CPU delegate until capability
      // probing can verify the entire graph, not merely task creation.
      await createTasks('CPU');
      self.postMessage({ type: 'ready', delegate: 'CPU' });
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Vision model initialization failed.';
      self.postMessage({ type: 'error', message: detail });
    }
    return;
  }
  if (!holisticTask) {
    data.bitmap.close();
    return;
  }
  const start = performance.now();
  try {
    const result = holisticTask.detectForVideo(data.bitmap, data.timestampMs);
    const leftHand = result.leftHandLandmarks[0];
    const rightHand = result.rightHandLandmarks[0];
    const hands = [
      ...(leftHand ? [{ handedness: 'Left' as const, points: points(leftHand) }] : []),
      ...(rightHand ? [{ handedness: 'Right' as const, points: points(rightHand) }] : []),
    ];
    const frame: LandmarkFrame = {
      hands,
      pose: points(result.poseLandmarks[0]),
      face: points(result.faceLandmarks[0]),
      inferenceMs: performance.now() - start,
      timestampMs: data.timestampMs,
    };
    self.postMessage({ type: 'result', frame });
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'Landmark inference failed.';
    self.postMessage({ type: 'error', message: detail });
  } finally {
    data.bitmap.close();
  }
};
