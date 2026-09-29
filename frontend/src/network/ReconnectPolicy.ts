/** Exponential retry delay with a ceiling; a live camera session retries until stopped. */
export class ReconnectPolicy {
  private attempts = 0;

  constructor(private readonly baseDelayMs = 500, private readonly maxDelayMs = 30_000) {
    if (baseDelayMs < 1 || maxDelayMs < baseDelayMs) throw new RangeError('Invalid reconnect delay bounds');
  }

  nextDelayMs() {
    const delay = Math.min(this.maxDelayMs, this.baseDelayMs * 2 ** this.attempts);
    this.attempts += 1;
    return delay;
  }

  reset() {
    this.attempts = 0;
  }
}
