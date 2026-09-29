import { mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const modelsRoot = join(root, 'src', 'vision-assets', 'models');
await mkdir(modelsRoot, { recursive: true });

const models = {
  'holistic_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task',
};
const manifest = { source: 'MediaPipe official model storage', tasksVisionVersion: '1.0.1', files: {} };

for (const [filename, url] of Object.entries(models)) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${filename}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.byteLength < 100_000) throw new Error(`Downloaded ${filename} is unexpectedly small`);
  await writeFile(join(modelsRoot, filename), bytes);
  manifest.files[filename] = { url, bytes: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex') };
  console.log(`Downloaded ${filename} (${(bytes.byteLength / 1_000_000).toFixed(1)} MB)`);
}
await writeFile(join(modelsRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log('Local vision assets are ready. Camera frames remain in the browser.');
