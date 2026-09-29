import { afterEach, describe, expect, it, vi } from 'vitest';
import { SpeechQueue } from './SpeechQueue';

class FakeUtterance {
  lang = '';
  rate = 1;
  volume = 1;
  voice?: SpeechSynthesisVoice;
  onstart: (() => void) | null = null;
  onend: (() => void) | null = null;
  onerror: ((event: { error: string }) => void) | null = null;

  constructor(readonly text: string) {}
}

const utterances: FakeUtterance[] = [];

function speechEngine() {
  const engine = {
    cancel: vi.fn(),
    speak: vi.fn((utterance: FakeUtterance) => utterances.push(utterance)),
  };
  return engine as unknown as SpeechSynthesis;
}

afterEach(() => { utterances.length = 0; });

describe('SpeechQueue', () => {
  it('speaks sentence-sized chunks in order and reports start latency', () => {
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance);
    const states: string[] = [];
    const latencies: number[] = [];
    const engine = speechEngine();
    const queue = new SpeechQueue(engine, (state) => states.push(state), (ms) => latencies.push(ms));
    const voice = { voiceURI: 'test-voice' } as SpeechSynthesisVoice;

    queue.replace('First sentence. দ্বিতীয় বাক্য!', { language: 'bn-BD', rate: 1.1, volume: 0.7, voice });

    expect(engine.cancel).toHaveBeenCalledOnce();
    expect(utterances.map((item) => item.text)).toEqual(['First sentence.']);
    expect(utterances[0]).toMatchObject({ lang: 'bn-BD', rate: 1.1, volume: 0.7, voice });
    expect(engine.speak).toHaveBeenCalledOnce();
    utterances[0].onstart?.();
    expect(latencies).toHaveLength(1);
    utterances[0].onend?.();
    expect(engine.speak).toHaveBeenCalledTimes(2);
    expect(utterances[1].text).toBe('দ্বিতীয় বাক্য!');
    utterances[1].onend?.();
    expect(states.at(-1)).toBe('idle');
  });

  it('replaces stale speech and ignores cancellation errors from old utterances', () => {
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance);
    const states: string[] = [];
    const engine = speechEngine();
    const queue = new SpeechQueue(engine, (state) => states.push(state));
    const settings = { language: 'en-US', rate: 1, volume: 1 };

    queue.replace('Old sentence.', settings);
    const oldUtterance = utterances[0];
    queue.replace('New sentence.', settings);
    oldUtterance.onerror?.({ error: 'interrupted' });

    expect(engine.cancel).toHaveBeenCalledTimes(2);
    expect(utterances.map((item) => item.text)).toEqual(['Old sentence.', 'New sentence.']);
    expect(states).not.toContain('error');
    expect(queue.lastTranslation).toBe('New sentence.');
  });

  it('stop cancels queued speech and makes late events inert', () => {
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance);
    const states: string[] = [];
    const engine = speechEngine();
    const queue = new SpeechQueue(engine, (state) => states.push(state));
    queue.replace('One. Two.', { language: 'en-US', rate: 1, volume: 1 });
    const active = utterances[0];

    queue.stop();
    active.onend?.();

    expect(engine.cancel).toHaveBeenCalledTimes(2);
    expect(engine.speak).toHaveBeenCalledOnce();
    expect(states.at(-1)).toBe('stopped');
  });

  it('fails closed when speech synthesis is unavailable', () => {
    const states: string[] = [];
    const queue = new SpeechQueue(undefined, (state) => states.push(state));
    queue.replace('Do not speak unverified data.', { language: 'en-US', rate: 1, volume: 1 });

    expect(states).toEqual(['unavailable', 'unavailable']);
    expect(queue.lastTranslation).toBe('');
  });
});
