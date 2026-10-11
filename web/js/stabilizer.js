// Converts noisy per-frame detections into deliberate, debounced key presses.
// A sign must be held steadily before it commits, and the same sign only
// repeats after the hand is lowered (or held for a long time).

export class Stabilizer {
  constructor({ holdMs = 600, repeatMs = 1800, releaseMs = 500 } = {}) {
    this.holdMs = holdMs;
    this.repeatMs = repeatMs;
    this.releaseMs = releaseMs;
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.candidateSince = 0;
    this.lastCommitted = null;
    this.lastCommitTime = 0;
    this.lastSeen = 0;
    this.progress = 0;
  }

  // label: string | null. Returns the label to commit, or null.
  update(label, now) {
    if (label === null) {
      this.candidate = null;
      this.progress = 0;
      if (now - this.lastSeen > this.releaseMs) this.lastCommitted = null;
      return null;
    }
    this.lastSeen = now;

    if (label !== this.candidate) {
      this.candidate = label;
      this.candidateSince = now;
      this.progress = 0;
      return null;
    }

    const held = now - this.candidateSince;
    const needed = label === this.lastCommitted ? this.repeatMs : this.holdMs;
    this.progress = Math.min(1, held / needed);
    if (held < needed) return null;

    this.lastCommitted = label;
    this.lastCommitTime = now;
    this.candidateSince = now;
    this.progress = 0;
    return label;
  }
}
