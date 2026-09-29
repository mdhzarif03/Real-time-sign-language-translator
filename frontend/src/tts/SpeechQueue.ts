export type SpeechSettings = {
  language: string;
  rate: number;
  volume: number;
  voice?: SpeechSynthesisVoice;
};

export type SpeechState = 'unavailable' | 'idle' | 'queued' | 'speaking' | 'stopped' | 'error';

/** Single-engine, bounded TTS queue. New translations replace stale speech. */
export class SpeechQueue {
  private queue: SpeechSynthesisUtterance[] = [];
  private generation = 0;
  private lastText = '';
  private submittedAt = 0;

  constructor(
    private readonly synth: SpeechSynthesis | undefined,
    private readonly onState: (state: SpeechState) => void,
    private readonly onStartLatency: (milliseconds: number) => void = () => undefined,
  ) {
    onState(synth ? 'idle' : 'unavailable');
  }

  get lastTranslation() {
    return this.lastText;
  }

  replace(text: string, settings: SpeechSettings) {
    const phrases = text.match(/[^.!?。！？]+[.!?。！？]?/gu)?.map((phrase) => phrase.trim()).filter(Boolean) ?? [];
    if (!this.synth || phrases.length === 0) {
      if (!this.synth) this.onState('unavailable');
      return;
    }
    this.lastText = text;
    this.submittedAt = performance.now();
    const currentGeneration = ++this.generation;
    this.synth.cancel();
    this.queue = phrases.map((phrase) => {
      const utterance = new SpeechSynthesisUtterance(phrase);
      utterance.lang = settings.language;
      utterance.rate = settings.rate;
      utterance.volume = settings.volume;
      if (settings.voice) utterance.voice = settings.voice;
      utterance.onstart = () => {
        if (currentGeneration === this.generation) {
          this.onStartLatency(performance.now() - this.submittedAt);
          this.onState('speaking');
        }
      };
      utterance.onend = () => {
        if (currentGeneration === this.generation) this.advance(currentGeneration);
      };
      utterance.onerror = (event) => {
        if (currentGeneration !== this.generation || event.error === 'canceled' || event.error === 'interrupted') return;
        this.queue = [];
        this.onState('error');
      };
      return utterance;
    });
    this.onState('queued');
    this.advance(currentGeneration);
  }

  retry(settings: SpeechSettings) {
    if (this.lastText) this.replace(this.lastText, settings);
  }

  stop() {
    ++this.generation;
    this.queue = [];
    this.synth?.cancel();
    this.onState(this.synth ? 'stopped' : 'unavailable');
  }

  dispose() {
    ++this.generation;
    this.queue = [];
    this.synth?.cancel();
  }

  private advance(generation: number) {
    if (generation !== this.generation) return;
    const next = this.queue.shift();
    if (next && this.synth) this.synth.speak(next);
    else this.onState('idle');
  }
}
