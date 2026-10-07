/**
 * Gauge Component — Semi-circular SVG verdict gauge.
 * Renders a score from 0-100% with color gradient.
 */
class VerdictGauge {
  constructor(containerEl) {
    this.container = containerEl;
  }

  render(score, verdictText) {
    const pct = Math.round(score * 100);

    // Determine color
    let color, cssClass;
    if (score < 0.35) {
      color = '#22C55E';
      cssClass = 'safe';
    } else if (score < 0.65) {
      color = '#EAB308';
      cssClass = 'warning';
    } else {
      color = '#EF4444';
      cssClass = 'danger';
    }

    // SVG arc params
    const radius = 60;
    const circumference = Math.PI * radius; // half circle
    const offset = circumference * (1 - score);

    this.container.innerHTML = `
      <div class="gauge-container">
        <svg class="gauge-svg" viewBox="0 0 150 90">
          <path class="gauge-bg" 
            d="M 15 80 A 60 60 0 0 1 135 80" />
          <path class="gauge-fill"
            d="M 15 80 A 60 60 0 0 1 135 80"
            stroke="${color}"
            stroke-dasharray="${circumference}"
            stroke-dashoffset="${offset}" />
          <text class="gauge-value" x="75" y="72" text-anchor="middle">${pct}%</text>
        </svg>
      </div>
      <div class="verdict-label ${cssClass}">${verdictText || this._defaultVerdict(score)}</div>
    `;
  }

  _defaultVerdict(score) {
    if (score < 0.35) return 'LIKELY REAL';
    if (score < 0.65) return 'UNCERTAIN';
    return 'LIKELY FAKE';
  }

  clear() {
    this.container.innerHTML = '';
  }
}

// Signal bar renderer
function renderSignals(containerEl, signals, title) {
  if (!signals || Object.keys(signals).length === 0) {
    containerEl.innerHTML = '';
    return;
  }

  const rows = Object.entries(signals).map(([key, value]) => {
    const pct = Math.round(value * 100);
    let color;
    if (value < 0.35) color = 'var(--safe)';
    else if (value < 0.65) color = 'var(--warning)';
    else color = 'var(--danger)';

    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    return `
      <div class="signal-row">
        <span class="signal-name">${label}</span>
        <div class="signal-bar-wrap">
          <div class="signal-bar">
            <div class="signal-bar-fill" style="width:${pct}%; background:${color};"></div>
          </div>
          <span class="signal-value">${pct}%</span>
        </div>
      </div>
    `;
  }).join('');

  containerEl.innerHTML = `
    ${title ? `<div class="signals-title">${title}</div>` : ''}
    ${rows}
  `;
}

// Metadata renderer
function renderMetadata(containerEl, details) {
  if (!details || Object.keys(details).length === 0) {
    containerEl.innerHTML = '';
    return;
  }

  const skipKeys = new Set([
    'efficientnet_score', 'classical_score', 'face_manipulation_score',
    'ai_generated_score', 'wav2vec2_score',
  ]);

  const rows = Object.entries(details)
    .filter(([k]) => !skipKeys.has(k))
    .map(([key, value]) => {
      const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      return `
        <div class="metadata-row">
          <span class="metadata-key">${label}</span>
          <span class="metadata-value">${value}</span>
        </div>
      `;
    }).join('');

  containerEl.innerHTML = rows;
}

// Color for score
function scoreColor(score) {
  if (score < 0.35) return 'var(--safe)';
  if (score < 0.65) return 'var(--warning)';
  return 'var(--danger)';
}
