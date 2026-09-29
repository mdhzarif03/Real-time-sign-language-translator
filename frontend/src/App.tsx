import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AudioLines, Camera, CameraOff, Check, ChevronDown, CircleHelp, Expand, Hand, Languages,
  Mic2, Pause, Play, RotateCcw, Settings2, ShieldCheck, Sparkles, Square, Volume2,
} from 'lucide-react';
import type { LandmarkFrame, VisionWorkerMessage } from './vision.types';

type CameraState = 'idle' | 'starting' | 'running' | 'paused' | 'error';

const HAND_LINKS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9],
  [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const POSE_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];

function drawLandmarks(canvas: HTMLCanvasElement, video: HTMLVideoElement, frame: LandmarkFrame, visible: boolean) {
  const ctx = canvas.getContext('2d');
  if (!ctx || !video.videoWidth || !video.videoHeight) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!visible) return;

  const scale = Math.max(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const offsetX = (rect.width - video.videoWidth * scale) / 2;
  const offsetY = (rect.height - video.videoHeight * scale) / 2;
  const xy = (p: { x: number; y: number }) => [offsetX + p.x * video.videoWidth * scale, offsetY + p.y * video.videoHeight * scale] as const;
  const drawChain = (points: LandmarkFrame['pose'], pairs: number[][], color: string) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineCap = 'round';
    for (const [a, b] of pairs) {
      if (!points[a] || !points[b]) continue;
      const [ax, ay] = xy(points[a]); const [bx, by] = xy(points[b]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
  };
  drawChain(frame.pose, POSE_LINKS, 'rgba(119, 221, 190, .76)');
  for (const hand of frame.hands) {
    drawChain(hand.points, HAND_LINKS, hand.handedness === 'Left' ? '#85e2c4' : '#a8b6ff');
    hand.points.forEach((point, index) => {
      const [x, y] = xy(point);
      ctx.fillStyle = index % 4 === 0 ? '#fff' : (hand.handedness === 'Left' ? '#4ac6a1' : '#8295ff');
      ctx.beginPath(); ctx.arc(x, y, index === 0 ? 3.2 : 2.2, 0, Math.PI * 2); ctx.fill();
    });
  }
  if (frame.face.length) {
    ctx.fillStyle = 'rgba(255, 211, 132, .7)';
    frame.face.filter((_, index) => index % 12 === 0).forEach((point) => {
      const [x, y] = xy(point); ctx.beginPath(); ctx.arc(x, y, 1.2, 0, Math.PI * 2); ctx.fill();
    });
  }
}

function friendlyCameraError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') return 'Camera permission was denied. Allow camera access in your browser settings and try again.';
    if (error.name === 'NotFoundError') return 'No camera was found. Connect a camera and try again.';
    if (error.name === 'NotReadableError') return 'The camera is busy in another app. Close that app and try again.';
  }
  return error instanceof Error ? error.message : 'Could not start the camera.';
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const socketRef = useRef<WebSocket | undefined>(undefined);
  const streamIdRef = useRef('');
  const streamLanguageRef = useRef('en-US-ASL');
  const sequenceRef = useRef(0);
  const previousHypothesisRef = useRef<Array<{ text: string; confidence: number; uncertain: boolean }>>([]);
  const busyRef = useRef(false);
  const epochRef = useRef(0);
  const fpsWindowRef = useRef({ startedAt: 0, frames: 0 });
  const loopRef = useRef(0);
  const lastSpeechRef = useRef('');
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState('');
  const [serviceStatus, setServiceStatus] = useState('Local recognizer not connected');
  const [visionState, setVisionState] = useState<'not-ready' | 'loading' | 'ready' | 'error'>('not-ready');
  const [visionError, setVisionError] = useState('');
  const [delegate, setDelegate] = useState<'GPU' | 'CPU'>('CPU');
  const [frame, setFrame] = useState<LandmarkFrame>();
  const [visualization, setVisualization] = useState(true);
  const [fps, setFps] = useState(0);
  const [cameraId, setCameraId] = useState('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [signLanguage, setSignLanguage] = useState('ASL');
  const signLanguageRef = useRef(signLanguage);
  signLanguageRef.current = signLanguage;
  const [outputLanguage, setOutputLanguage] = useState('en-US');
  const [threshold, setThreshold] = useState(70);
  const [speechRate, setSpeechRate] = useState(1);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [glossTokens, setGlossTokens] = useState<Array<{ text: string; confidence: number; uncertain: boolean }>>([]);
  const [glossConfidence, setGlossConfidence] = useState<number | undefined>();
  const [glossFinal, setGlossFinal] = useState(false);
  const workerErrorSeen = useRef(false);

  useEffect(() => {
    const updateVoices = () => setVoices(window.speechSynthesis?.getVoices() ?? []);
    updateVoices();
    window.speechSynthesis?.addEventListener('voiceschanged', updateVoices);
    return () => window.speechSynthesis?.removeEventListener('voiceschanged', updateVoices);
  }, []);

  const connectLocalRecognizer = useCallback((language: string) => {
    socketRef.current?.close(1000, 'Starting a fresh signer stream');
    streamIdRef.current = crypto.randomUUID(); sequenceRef.current = 0;
    streamLanguageRef.current = language;
    previousHypothesisRef.current = [];
    const socket = new WebSocket(`ws://127.0.0.1:8000/api/v1/stream/${streamIdRef.current}`);
    socketRef.current = socket;
    setServiceStatus('Connecting to local recognizer…');
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as { type?: string; state?: string; message?: string; confidence?: number; is_final?: boolean; tokens?: Array<{ text: string; confidence: number; uncertain: boolean }> };
        if (message.type === 'status') setServiceStatus(message.message || message.state || 'Local service ready');
        else if (message.type === 'hypothesis') {
          setServiceStatus('Temporal recognition active');
          const incoming = message.tokens ?? [];
          if (message.is_final) {
            setGlossTokens(incoming); setGlossFinal(true);
            previousHypothesisRef.current = incoming;
          } else {
            const previous = previousHypothesisRef.current;
            let stableLength = 0;
            while (stableLength < previous.length && stableLength < incoming.length && previous[stableLength].text === incoming[stableLength].text) stableLength += 1;
            setGlossTokens(incoming.slice(0, stableLength)); setGlossFinal(false);
            previousHypothesisRef.current = incoming;
          }
          setGlossConfidence(message.confidence);
        }
      } catch {
        setServiceStatus('Local service returned an invalid status');
      }
    };
    socket.onerror = () => setServiceStatus('Local recognizer unavailable; landmarks remain on this device');
    socket.onclose = () => {
      if (streamRef.current && socketRef.current === socket) setServiceStatus('Local recognizer disconnected; landmarks remain on this device');
    };
    return socket;
  }, []);

  const onWorkerMessage = useCallback((event: MessageEvent<VisionWorkerMessage>) => {
    const message = event.data;
    if (message.type === 'ready') {
      setDelegate(message.delegate); setVisionState('ready'); setVisionError('');
    } else if (message.type === 'error') {
      setVisionState('error'); setVisionError(message.message);
      if (streamRef.current) setCameraState('paused');
      busyRef.current = false;
    } else {
      busyRef.current = false;
      setFrame(message.frame);
      const socket = socketRef.current;
      const video = videoRef.current;
      if (socket?.readyState === WebSocket.OPEN && video && socket.bufferedAmount < 512_000) {
        const observation = {
          schema_version: 1,
          type: 'observation',
          stream_id: streamIdRef.current,
          sequence: sequenceRef.current++,
          timestamp_ms: message.frame.timestampMs,
          sign_language: streamLanguageRef.current,
          image_width: video.videoWidth,
          image_height: video.videoHeight,
          landmarks: {
            left_hand: message.frame.hands.find((hand) => hand.handedness === 'Left')?.points ?? null,
            right_hand: message.frame.hands.find((hand) => hand.handedness === 'Right')?.points ?? null,
            pose: message.frame.pose.length ? message.frame.pose : null,
            face: message.frame.face.length ? message.frame.face : null,
          },
        };
        socket.send(JSON.stringify(observation));
      }
      const now = performance.now();
      if (!fpsWindowRef.current.startedAt) fpsWindowRef.current.startedAt = now;
      fpsWindowRef.current.frames += 1;
      const elapsed = now - fpsWindowRef.current.startedAt;
      if (elapsed >= 1000) {
        setFps(Math.min(60, fpsWindowRef.current.frames * 1000 / elapsed));
        fpsWindowRef.current = { startedAt: now, frames: 0 };
      }
    }
  }, []);

  const stopCapture = useCallback((nextState: CameraState = 'idle') => {
    epochRef.current += 1;
    cancelAnimationFrame(loopRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = undefined;
    if (videoRef.current) videoRef.current.srcObject = null;
    workerRef.current?.terminate(); workerRef.current = undefined;
    socketRef.current?.close(1000, 'Capture stopped'); socketRef.current = undefined;
    setServiceStatus('Local recognizer not connected');
    busyRef.current = false; fpsWindowRef.current = { startedAt: 0, frames: 0 };
    setFrame(undefined); setFps(0); setCameraState(nextState);
    if (canvasRef.current) canvasRef.current.getContext('2d')?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  const startCapture = useCallback(async (requestedCameraId?: string) => {
    stopCapture('starting');
    setCameraError(''); setVisionError(''); setVisionState('loading'); workerErrorSeen.current = false;
    const epoch = epochRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { deviceId: (requestedCameraId ?? cameraId) ? { exact: requestedCameraId ?? cameraId } : undefined, facingMode: (requestedCameraId ?? cameraId) ? undefined : 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
      });
      if (epoch !== epochRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview is unavailable.');
      video.srcObject = stream;
      await video.play();
      const devices = await navigator.mediaDevices.enumerateDevices();
      setCameras(devices.filter((device) => device.kind === 'videoinput'));
      const activeId = stream.getVideoTracks()[0]?.getSettings().deviceId;
      if (activeId) setCameraId(activeId);
      connectLocalRecognizer(signLanguageRef.current === 'ASL' ? 'en-US-ASL' : 'bn-BD-BdSL');
      const worker = new Worker(new URL('./vision.worker.ts', import.meta.url), { type: 'module', name: 'local-vision' });
      workerRef.current = worker;
      worker.onmessage = onWorkerMessage;
      worker.onerror = (event) => {
        if (workerErrorSeen.current) return;
        workerErrorSeen.current = true; setVisionState('error');
        setVisionError(`Local landmark worker failed: ${event.message || 'check that vision assets are installed.'}`);
        setCameraState('paused');
      };
      worker.postMessage({ type: 'init' });
      setCameraState('running');
    } catch (error) {
      setCameraError(friendlyCameraError(error)); setCameraState('error'); setVisionState('not-ready');
    }
  }, [cameraId, connectLocalRecognizer, onWorkerMessage, stopCapture]);

  useEffect(() => {
    if (cameraState !== 'running' || visionState !== 'ready') return;
    const tick = async (now: number) => {
      const video = videoRef.current;
      const worker = workerRef.current;
      if (video && worker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !busyRef.current && now - (frame?.timestampMs ?? 0) > 80) {
        busyRef.current = true;
        try {
          const bitmap = await createImageBitmap(video);
          worker.postMessage({ type: 'frame', bitmap, timestampMs: Math.max(now, (frame?.timestampMs ?? 0) + 1) }, [bitmap]);
        } catch {
          busyRef.current = false;
        }
      }
      loopRef.current = requestAnimationFrame(tick);
    };
    loopRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(loopRef.current);
  }, [cameraState, frame, visionState]);

  useEffect(() => {
    const video = videoRef.current; const canvas = canvasRef.current;
    if (video && canvas && frame) drawLandmarks(canvas, video, frame, visualization);
  }, [frame, visualization]);

  useEffect(() => () => {
    cancelAnimationFrame(loopRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    workerRef.current?.terminate();
    window.speechSynthesis?.cancel();
  }, []);

  const stopSpeech = () => window.speechSynthesis?.cancel();
  const retrySpeech = () => {
    if (!lastSpeechRef.current || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(lastSpeechRef.current);
    utterance.lang = outputLanguage; utterance.rate = speechRate;
    const voice = voices.find((item) => item.voiceURI === selectedVoice);
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
  };

  const handCount = frame?.hands.length ?? 0;
  const visionLabel = visionState === 'ready' ? 'Vision active' : visionState === 'loading' ? 'Loading local models' : visionState === 'error' ? 'Vision unavailable' : 'Vision paused';
  const cameraLabel = cameraState === 'running' ? 'Camera live' : cameraState === 'paused' ? 'Camera paused' : cameraState === 'starting' ? 'Connecting camera' : 'Camera off';

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#main" aria-label="SignFlow home"><span className="brand-mark"><Hand size={19} strokeWidth={2.3} /></span><span>signflow<span className="brand-period">.</span></span></a>
        <nav className="topnav" aria-label="Main"><a className="active" href="#main">Translator</a><a href="#settings">Settings</a></nav>
        <div className="privacy-pill"><span className="privacy-dot" /><ShieldCheck size={15} /><span>LOCAL PROCESSING</span></div>
      </header>

      <main id="main" className="workspace">
        <section className="intro-row">
          <div><div className="eyebrow"><Sparkles size={14} /> ACCESSIBLE COMMUNICATION</div><h1>Sign, and be understood.</h1><p className="intro-copy">A private workspace for signing. Camera analysis stays on this device.</p></div>
          <div className="model-badge"><span className="badge-dot" /><span>{serviceStatus}</span><CircleHelp size={15} aria-label="The local model status is shown here" /></div>
        </section>

        <div className="main-grid">
          <section className="camera-card card" aria-labelledby="camera-title">
            <div className="card-heading"><div><div className="section-kicker">LIVE INPUT</div><h2 id="camera-title">Camera</h2></div><button className={`state-chip ${cameraState === 'running' ? 'is-live' : ''}`} aria-live="polite"><span className="chip-dot" />{cameraLabel}</button></div>
            <div className="camera-stage-wrap">
              <div className="camera-stage" aria-label="Live camera preview">
                <video ref={videoRef} autoPlay muted playsInline aria-label="Your live camera view" />
                <canvas ref={canvasRef} className="landmark-canvas" aria-hidden="true" />
                {cameraState !== 'running' && <div className="camera-placeholder"><div className="camera-placeholder-icon"><Camera size={25} /></div><strong>{cameraState === 'paused' ? 'Camera paused' : cameraState === 'error' ? 'Camera unavailable' : 'Your camera preview'}</strong><span>{cameraState === 'error' ? cameraError : 'Start when you are ready. Nothing is recorded.'}</span>{cameraState !== 'error' && <span className="placeholder-local"><ShieldCheck size={13} /> Processed on this device</span>}</div>}
                {cameraState === 'running' && <><div className="camera-corner top-left" /><div className="camera-corner top-right" /><div className="camera-corner bottom-left" /><div className="camera-corner bottom-right" /><div className="camera-overlay-label"><span className="record-dot" /> LIVE</div><div className="camera-overlay-metrics"><span>{fps.toFixed(0)} <small>FPS</small></span><i /><span>{frame?.inferenceMs.toFixed(0) ?? '—'} <small>MS</small></span></div></>}
              </div>
            </div>
            {(visionError || cameraError) && <div className="inline-error" role="alert">{visionError || cameraError}{visionState === 'error' && <> Check that setup:vision completed and reload the page.</>}</div>}
            <div className="vision-status-row"><div className={`vision-status ${visionState === 'ready' ? 'good' : visionState === 'error' ? 'bad' : ''}`}><span className="status-dot" />{visionLabel}</div><span className="status-detail">{visionState === 'ready' ? `MediaPipe · ${delegate} · ${handCount} ${handCount === 1 ? 'hand' : 'hands'} detected` : 'Hand · pose · face landmarks'}</span></div>
            <div className="camera-controls">
              {cameraState === 'running' ? <button className="button button-secondary" onClick={() => { cancelAnimationFrame(loopRef.current); setCameraState('paused'); }}><Pause size={16} /> Pause</button> : cameraState === 'paused' && streamRef.current ? <button className="button button-secondary" onClick={() => setCameraState('running')}><Play size={16} /> Resume</button> : <button className="button button-primary" onClick={() => void startCapture()} disabled={cameraState === 'starting'}><Camera size={16} />{cameraState === 'starting' ? 'Starting…' : 'Start camera'}</button>}
              {streamRef.current && <button className="button button-quiet" onClick={() => stopCapture()}><CameraOff size={16} /> Stop camera</button>}
              <span className="control-spacer" />
              <label className="select-wrap camera-select-label"><span className="sr-only">Camera device</span><select value={cameraId} onChange={(event) => { const id = event.target.value; setCameraId(id); if (streamRef.current) void startCapture(id); }} disabled={cameras.length < 2}><option value="">Default camera</option>{cameras.map((camera, index) => <option value={camera.deviceId} key={camera.deviceId}>{camera.label || `Camera ${index + 1}`}</option>)}</select><ChevronDown size={14} /></label>
            </div>
          </section>

          <section className="translation-card card" aria-labelledby="translation-title">
            <div className="card-heading"><div><div className="section-kicker">LIVE SIGN SEQUENCE</div><h2 id="translation-title">Recognized glosses</h2></div><button className="icon-button" title="Clear transcript" aria-label="Clear transcript" onClick={() => { lastSpeechRef.current = ''; setGlossTokens([]); setGlossConfidence(undefined); setGlossFinal(false); previousHypothesisRef.current = []; stopSpeech(); }}><RotateCcw size={16} /></button></div>
            {glossTokens.length ? <div className="raw-sequence"><div className="raw-sequence-label"><span>RAW SIGN SEQUENCE · GLOSS</span><span>{glossFinal ? 'STABLE' : 'PARTIAL'}</span></div><p aria-live="polite">{glossTokens.map((token, index) => <span className={token.uncertain ? 'gloss-token uncertain' : 'gloss-token'} title={`Model score ${(token.confidence * 100).toFixed(0)}%${token.uncertain ? ' · uncertain' : ''}`} key={`${index}-${token.text}`}>{token.uncertain && <b aria-label="uncertain">?</b>}{token.text}</span>)}</p><div className="raw-sequence-note"><CircleHelp size={14} /> Gloss output is not a natural-language translation and will not be spoken.</div></div> : <div className="translation-empty"><div className="wave-icon"><AudioLines size={22} /></div><h3>Translation is not available yet</h3><p>This workspace can track visual landmarks. A trained {signLanguage} sequence model is required before it can recognize signs or generate a sentence.</p><div className="model-notice"><CircleHelp size={16} /><span>No compatible recognition model is loaded. No words will be guessed.</span></div></div>}
            <div className="confidence-row"><div><span className="muted-label">MODEL SCORE · UNCALIBRATED</span><strong>{glossConfidence === undefined ? '—' : `${(glossConfidence * 100).toFixed(0)}%`}</strong></div><div className="confidence-track"><span style={{ width: `${(glossConfidence ?? 0) * 100}%` }} /></div><span className="confidence-unavailable">{glossConfidence === undefined ? 'Unavailable' : glossConfidence * 100 < threshold ? 'Uncertain' : 'Model score'}</span></div>
            <div className="speech-row"><div className="speech-identity"><div className="speech-icon"><Volume2 size={17} /></div><div><strong>Speech output</strong><span>Available after a translation model is added</span></div></div><div className="speech-actions"><button className="icon-button" aria-label="Stop speech" title="Stop speech" onClick={stopSpeech}><Square size={14} fill="currentColor" /></button><button className="icon-button" aria-label="Retry last speech" title="Retry last speech" onClick={retrySpeech} disabled={!lastSpeechRef.current}><RotateCcw size={15} /></button></div></div>
            <div className="transcript-foot"><span><Mic2 size={14} /> Partial transcript</span><span>Waiting for a trained model</span></div>
          </section>
        </div>

        <section id="settings" className="settings-card card" aria-labelledby="settings-title">
          <div className="settings-heading"><div className="settings-title"><div className="settings-icon"><Settings2 size={17} /></div><div><h2 id="settings-title">Session settings</h2><p>Configure your signing session and output preferences.</p></div></div><button className={`toggle-control ${visualization ? 'on' : ''}`} role="switch" aria-checked={visualization} onClick={() => setVisualization((value) => !value)}><span className="toggle-thumb" /><span>Landmark overlay</span></button></div>
          <div className="settings-grid">
            <label className="field"><span><Languages size={14} /> Sign language</span><div className="select-wrap"><select value={signLanguage} onChange={(event) => { const value = event.target.value; setSignLanguage(value); if (streamRef.current) connectLocalRecognizer(value === 'ASL' ? 'en-US-ASL' : 'bn-BD-BdSL'); }}><option value="ASL">American Sign Language (ASL)</option><option value="BdSL">Bangla Sign Language (BdSL)</option></select><ChevronDown size={14} /></div><small>No trained model available</small></label>
            <label className="field"><span><AudioLines size={14} /> Output language</span><div className="select-wrap"><select value={outputLanguage} onChange={(event) => setOutputLanguage(event.target.value)}><option value="en-US">English</option><option value="bn-BD">বাংলা (Bengali)</option></select><ChevronDown size={14} /></div><small>For speech output</small></label>
            <label className="field"><span><Volume2 size={14} /> Voice</span><div className="select-wrap"><select value={selectedVoice} onChange={(event) => setSelectedVoice(event.target.value)}><option value="">System default</option>{voices.filter((voice) => voice.lang.toLowerCase().startsWith(outputLanguage.slice(0, 2).toLowerCase())).map((voice) => <option key={voice.voiceURI} value={voice.voiceURI}>{voice.name}</option>)}</select><ChevronDown size={14} /></div><small>Browser voices · on-device where available</small></label>
            <label className="field range-field"><span><span><Check size={14} /> Confidence threshold</span><b>{threshold}%</b></span><input type="range" min="50" max="95" step="5" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /><small>Low-confidence signs will remain uncertain</small></label>
            <label className="field range-field"><span><span><Volume2 size={14} /> Speech speed</span><b>{speechRate.toFixed(1)}×</b></span><input type="range" min="0.7" max="1.3" step="0.1" value={speechRate} onChange={(event) => setSpeechRate(Number(event.target.value))} /><small>Adjust spoken translation pace</small></label>
          </div>
        </section>

        <footer className="footer-note"><span><ShieldCheck size={14} /> Your camera feed stays on this device.</span><span><Expand size={13} /> Designed for keyboard and screen reader access</span></footer>
      </main>
    </div>
  );
}
