import { describe, expect, it } from 'vitest';
import { ReconnectPolicy } from './ReconnectPolicy';

describe('ReconnectPolicy', () => {
  it('backs off exponentially and caps the wait', () => {
    const policy = new ReconnectPolicy(250, 1000);
    expect([policy.nextDelayMs(), policy.nextDelayMs(), policy.nextDelayMs(), policy.nextDelayMs(), policy.nextDelayMs()])
      .toEqual([250, 500, 1000, 1000, 1000]);
  });

  it('resets after the connection has been stable', () => {
    const policy = new ReconnectPolicy(300, 2400);
    policy.nextDelayMs();
    policy.nextDelayMs();
    policy.reset();
    expect(policy.nextDelayMs()).toBe(300);
  });

  it('rejects invalid delay bounds', () => {
    expect(() => new ReconnectPolicy(0, 10)).toThrow(RangeError);
    expect(() => new ReconnectPolicy(20, 10)).toThrow(RangeError);
  });
});
