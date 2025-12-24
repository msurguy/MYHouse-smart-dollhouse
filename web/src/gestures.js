/**
 * Gesture Data Collection and Processing
 * Captures mouse movements and converts them to feature vectors
 */

export class GestureCollector {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.isDrawing = false;
        this.points = [];
        this.lastPoint = null;
        this.lastTime = null;

        // Configuration matching Python implementation
        this.config = {
            numFeatures: 6,        // Features per datapoint
            numDatapoints: 40,     // Same as Python: 40 datapoints per gesture
            inputSize: 240,        // 40 * 6 = 240
            sampleInterval: 50     // ~50ms between samples (20Hz like Python)
        };

        // Drawing style
        this.strokeColor = '#00d4ff';
        this.strokeWidth = 3;

        this.setupCanvas();
    }

    setupCanvas() {
        // Set up event listeners
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));

        // Touch support
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));

        this.clear();
    }

    clear() {
        this.ctx.fillStyle = '#0a0a15';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        this.ctx.strokeStyle = '#1a1a2e';
        this.ctx.lineWidth = 1;
        const gridSize = 40;

        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        // Draw center crosshair
        this.ctx.strokeStyle = '#333';
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2, 0);
        this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height / 2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.stroke();

        this.points = [];
        this.lastPoint = null;
        this.lastTime = null;
    }

    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    onMouseDown(e) {
        this.isDrawing = true;
        this.points = [];
        this.lastPoint = null;
        this.lastTime = performance.now();

        const coords = this.getCanvasCoords(e);
        this.addPoint(coords.x, coords.y);
    }

    onMouseMove(e) {
        if (!this.isDrawing) return;

        const coords = this.getCanvasCoords(e);
        this.addPoint(coords.x, coords.y);
    }

    onMouseUp(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.onGestureComplete();
        }
    }

    onTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.isDrawing = true;
        this.points = [];
        this.lastPoint = null;
        this.lastTime = performance.now();

        const rect = this.canvas.getBoundingClientRect();
        const x = (touch.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (touch.clientY - rect.top) * (this.canvas.height / rect.height);
        this.addPoint(x, y);
    }

    onTouchMove(e) {
        e.preventDefault();
        if (!this.isDrawing) return;

        const touch = e.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        const x = (touch.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (touch.clientY - rect.top) * (this.canvas.height / rect.height);
        this.addPoint(x, y);
    }

    onTouchEnd(e) {
        e.preventDefault();
        if (this.isDrawing) {
            this.isDrawing = false;
            this.onGestureComplete();
        }
    }

    addPoint(x, y) {
        const currentTime = performance.now();

        // Draw the point
        this.ctx.strokeStyle = this.strokeColor;
        this.ctx.lineWidth = this.strokeWidth;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        if (this.lastPoint) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
            this.ctx.lineTo(x, y);
            this.ctx.stroke();
        }

        // Calculate features for this point
        const point = { x, y, time: currentTime };

        if (this.lastPoint) {
            const dx = x - this.lastPoint.x;
            const dy = y - this.lastPoint.y;
            const dt = (currentTime - this.lastTime) / 1000; // Convert to seconds
            const distance = Math.sqrt(dx * dx + dy * dy);
            const speed = dt > 0 ? distance / dt : 0;
            const angle = Math.atan2(dy, dx);

            // Normalize values to be similar to accelerometer/gyroscope data range
            // The Python data had values roughly in range [-3, 3]
            const normalizedDx = dx / 100;  // Normalize by canvas scale
            const normalizedDy = dy / 100;
            const normalizedSpeed = speed / 500;  // Normalize speed
            const normalizedAngle = angle / Math.PI;  // Normalize angle to [-1, 1]
            const pressure = 1.0;  // Constant for mouse (could be variable for stylus)
            const normalizedDt = Math.min(dt * 10, 2);  // Normalize time delta

            point.features = [
                normalizedDx,
                normalizedDy,
                normalizedSpeed,
                normalizedAngle,
                pressure,
                normalizedDt
            ];

            this.points.push(point);
        }

        this.lastPoint = { x, y };
        this.lastTime = currentTime;
    }

    /**
     * Convert raw points to fixed-length feature vector
     * Same as Python: 40 datapoints × 6 features = 240 values
     */
    extractFeatures() {
        const numDatapoints = this.config.numDatapoints;
        const numFeatures = this.config.numFeatures;
        const features = new Array(numDatapoints * numFeatures).fill(0);

        if (this.points.length === 0) {
            return features;
        }

        // Resample to exactly 40 datapoints
        const resampledPoints = this.resamplePoints(numDatapoints);

        // Fill in features
        for (let i = 0; i < resampledPoints.length; i++) {
            const point = resampledPoints[i];
            if (point.features) {
                for (let j = 0; j < numFeatures; j++) {
                    features[i * numFeatures + j] = point.features[j];
                }
            }
        }

        return features;
    }

    /**
     * Resample points to a fixed number using linear interpolation
     */
    resamplePoints(targetCount) {
        if (this.points.length === 0) {
            return [];
        }

        if (this.points.length === 1) {
            // Pad with the same point
            return Array(targetCount).fill(this.points[0]);
        }

        if (this.points.length >= targetCount) {
            // Downsample: select evenly spaced points
            const result = [];
            for (let i = 0; i < targetCount; i++) {
                const index = Math.floor(i * (this.points.length - 1) / (targetCount - 1));
                result.push(this.points[index]);
            }
            return result;
        }

        // Upsample: interpolate between points
        const result = [];
        const step = (this.points.length - 1) / (targetCount - 1);

        for (let i = 0; i < targetCount; i++) {
            const floatIndex = i * step;
            const lowerIndex = Math.floor(floatIndex);
            const upperIndex = Math.min(lowerIndex + 1, this.points.length - 1);
            const t = floatIndex - lowerIndex;

            const lowerPoint = this.points[lowerIndex];
            const upperPoint = this.points[upperIndex];

            // Interpolate features
            const interpolatedFeatures = [];
            if (lowerPoint.features && upperPoint.features) {
                for (let j = 0; j < this.config.numFeatures; j++) {
                    const value = lowerPoint.features[j] * (1 - t) + upperPoint.features[j] * t;
                    interpolatedFeatures.push(value);
                }
            }

            result.push({
                x: lowerPoint.x * (1 - t) + upperPoint.x * t,
                y: lowerPoint.y * (1 - t) + upperPoint.y * t,
                features: interpolatedFeatures.length > 0 ? interpolatedFeatures : lowerPoint.features
            });
        }

        return result;
    }

    /**
     * Called when gesture drawing is complete
     * Override this in the app to handle the gesture
     */
    onGestureComplete() {
        // This will be overridden
        console.log('Gesture complete, points:', this.points.length);
    }

    /**
     * Draw a preview of resampled points
     */
    drawResampledPreview() {
        const resampled = this.resamplePoints(this.config.numDatapoints);

        this.ctx.fillStyle = '#ff6b6b';
        for (const point of resampled) {
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
}


/**
 * Data Store for collected gesture samples
 */
export class GestureDataStore {
    constructor() {
        this.samples = [];
        this.storageKey = 'gestureData';
        this.load();
    }

    /**
     * Add a new sample
     */
    addSample(label, features) {
        const sample = {
            id: Date.now(),
            label,
            features,
            timestamp: new Date().toISOString()
        };

        this.samples.push(sample);
        this.save();

        return sample;
    }

    /**
     * Remove a sample by ID
     */
    removeSample(id) {
        this.samples = this.samples.filter(s => s.id !== id);
        this.save();
    }

    /**
     * Get samples by label
     */
    getSamplesByLabel(label) {
        return this.samples.filter(s => s.label === label);
    }

    /**
     * Get sample counts per class
     */
    getSampleCounts() {
        const counts = {};
        for (let i = 0; i < 7; i++) {
            counts[i] = this.samples.filter(s => s.label === i).length;
        }
        return counts;
    }

    /**
     * Get total sample count
     */
    getTotalCount() {
        return this.samples.length;
    }

    /**
     * Get all samples
     */
    getAllSamples() {
        return [...this.samples];
    }

    /**
     * Get recent samples
     */
    getRecentSamples(count = 10) {
        return [...this.samples].reverse().slice(0, count);
    }

    /**
     * Clear all data
     */
    clear() {
        this.samples = [];
        this.save();
    }

    /**
     * Save to localStorage
     */
    save() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.samples));
        } catch (e) {
            console.error('Failed to save data:', e);
        }
    }

    /**
     * Load from localStorage
     */
    load() {
        try {
            const data = localStorage.getItem(this.storageKey);
            if (data) {
                this.samples = JSON.parse(data);
            }
        } catch (e) {
            console.error('Failed to load data:', e);
            this.samples = [];
        }
    }

    /**
     * Export data as JSON
     */
    exportJSON() {
        const data = JSON.stringify(this.samples, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `gesture-data-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();

        URL.revokeObjectURL(url);
    }

    /**
     * Import data from JSON file
     */
    async importJSON(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    if (Array.isArray(data)) {
                        // Validate data structure
                        const validSamples = data.filter(s =>
                            typeof s.label === 'number' &&
                            Array.isArray(s.features) &&
                            s.features.length === 240
                        );

                        // Add unique IDs if missing
                        validSamples.forEach(s => {
                            if (!s.id) s.id = Date.now() + Math.random();
                        });

                        this.samples = [...this.samples, ...validSamples];
                        this.save();
                        resolve(validSamples.length);
                    } else {
                        reject(new Error('Invalid data format'));
                    }
                } catch (e) {
                    reject(e);
                }
            };

            reader.onerror = () => reject(reader.error);
            reader.readAsText(file);
        });
    }

    /**
     * Split data into training and validation sets
     */
    splitData(validationRatio = 0.2) {
        // Shuffle samples
        const shuffled = [...this.samples].sort(() => Math.random() - 0.5);

        const splitIndex = Math.floor(shuffled.length * (1 - validationRatio));

        return {
            training: shuffled.slice(0, splitIndex),
            validation: shuffled.slice(splitIndex)
        };
    }
}
