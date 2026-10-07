/**
 * Upload Zone Component
 * Handles drag-and-drop file uploads with preview, validation, and scanning overlay.
 *
 * Usage:
 *   const zone = new UploadZone('image-upload-zone', 'image-file-input', {
 *     accept: ['image/jpeg', 'image/png', 'image/webp'],
 *     maxSizeMB: 10,
 *     onFile: (file, dataUrl) => { ... },
 *     onRemove: () => { ... }
 *   });
 */

class UploadZone {
  /**
   * @param {string} zoneId - Upload zone container element ID
   * @param {string} inputId - Hidden file input element ID
   * @param {Object} options
   * @param {string[]} [options.accept] - Accepted MIME types
   * @param {number} [options.maxSizeMB=10] - Max file size in MB
   * @param {Function} [options.onFile] - Callback (file, dataUrl)
   * @param {Function} [options.onRemove] - Callback when file removed
   */
  constructor(zoneId, inputId, options = {}) {
    this.zone = document.getElementById(zoneId);
    this.input = document.getElementById(inputId);
    this.accept = options.accept || ['image/jpeg', 'image/png', 'image/webp'];
    this.maxSizeMB = options.maxSizeMB || 10;
    this.onFile = options.onFile || (() => {});
    this.onRemove = options.onRemove || (() => {});

    this.file = null;
    this.dataUrl = null;

    this._bindEvents();
  }

  _bindEvents() {
    // Click to browse
    this.zone.addEventListener('click', (e) => {
      if (e.target.closest('.upload-zone__remove')) return;
      if (!this.file) this.input.click();
    });

    // Keyboard accessible
    this.zone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!this.file) this.input.click();
      }
    });

    // File input change
    this.input.addEventListener('change', () => {
      if (this.input.files.length > 0) {
        this._handleFile(this.input.files[0]);
      }
    });

    // Drag & drop
    this.zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.zone.classList.add('dragover');
    });

    this.zone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      this.zone.classList.remove('dragover');
    });

    this.zone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.zone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        this._handleFile(e.dataTransfer.files[0]);
      }
    });
  }

  /**
   * Validate and process the dropped/selected file
   * @param {File} file
   */
  _handleFile(file) {
    // Validate type
    if (!this.accept.includes(file.type)) {
      this._showError(`Unsupported format. Please use: ${this.accept.map(t => t.split('/')[1]).join(', ')}`);
      return;
    }

    // Validate size
    if (file.size > this.maxSizeMB * 1024 * 1024) {
      this._showError(`File too large. Maximum size: ${this.maxSizeMB}MB`);
      return;
    }

    this.file = file;

    // Read as data URL for preview
    const reader = new FileReader();
    reader.onload = (e) => {
      this.dataUrl = e.target.result;
      this._showPreview();
      this.onFile(this.file, this.dataUrl);
    };
    reader.readAsDataURL(file);
  }

  /**
   * Display image preview inside the upload zone
   */
  _showPreview() {
    this.zone.classList.add('has-file');
    this.zone.innerHTML = `
      <div class="upload-zone__preview">
        <img src="${this.dataUrl}" alt="Uploaded image preview" id="preview-image">
        <canvas class="upload-zone__overlay-canvas" id="bbox-overlay-canvas"></canvas>
        <button class="upload-zone__remove" aria-label="Remove image" id="remove-image-btn">✕</button>
      </div>
    `;

    // Bind remove button
    const removeBtn = this.zone.querySelector('#remove-image-btn');
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.reset();
      this.onRemove();
    });
  }

  /**
   * Show scanning overlay animation on the upload zone
   */
  showScanning() {
    const existing = this.zone.querySelector('.scanning-overlay');
    if (existing) return;

    const overlay = document.createElement('div');
    overlay.className = 'scanning-overlay';
    overlay.innerHTML = `
      <div class="scanning-line"></div>
      <div class="spinner"></div>
    `;
    this.zone.querySelector('.upload-zone__preview')?.appendChild(overlay);
  }

  /**
   * Remove scanning overlay
   */
  hideScanning() {
    const overlay = this.zone.querySelector('.scanning-overlay');
    if (overlay) overlay.remove();
  }

  /**
   * Get the preview image element (for drawing bounding boxes)
   * @returns {HTMLImageElement|null}
   */
  getPreviewImage() {
    return this.zone.querySelector('#preview-image');
  }

  /**
   * Get the overlay canvas element
   * @returns {HTMLCanvasElement|null}
   */
  getOverlayCanvas() {
    return this.zone.querySelector('#bbox-overlay-canvas');
  }

  /**
   * Reset to initial empty state
   */
  reset() {
    this.file = null;
    this.dataUrl = null;
    this.input.value = '';
    this.zone.classList.remove('has-file');
    this.zone.innerHTML = `
      <input type="file" class="upload-zone__input" id="${this.input.id}" accept="${this.accept.join(',')}">
      <div class="upload-zone__icon" aria-hidden="true">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      </div>
      <p class="upload-zone__text">Drop image here or click to browse</p>
      <p class="upload-zone__hint">Supports JPG, PNG, WebP — Max ${this.maxSizeMB}MB</p>
    `;

    // Re-bind the new input
    this.input = document.getElementById(this.input.id);
    this._bindEvents();
  }

  /**
   * Show error toast
   * @param {string} message
   */
  _showError(message) {
    // Use the global toast if available
    if (typeof showToast === 'function') {
      showToast(message, 'error');
    } else {
      console.error('[UploadZone]', message);
    }
  }
}
