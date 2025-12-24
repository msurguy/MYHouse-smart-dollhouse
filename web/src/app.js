/**
 * Main Application - Gesture Recognition Training & Testing
 */

import './styles.css';
import { GestureModel } from './model.js';
import { GestureCollector, GestureDataStore } from './gestures.js';

class GestureApp {
    constructor() {
        // Initialize components
        this.model = new GestureModel();
        this.dataStore = new GestureDataStore();

        // Gesture labels
        this.gestureLabels = {
            0: 'No Gesture',
            1: 'TV Poke',
            2: 'Shutter (Swipe)',
            3: 'Letter M',
            4: 'Letter Y',
            5: 'Fan Slow (1 circle)',
            6: 'Fan Fast (2 circles)'
        };

        // Test statistics
        this.testStats = {
            total: 0,
            correct: 0,
            byClass: {}
        };

        // Charts
        this.lossChart = null;
        this.accuracyChart = null;

        // Initialize UI
        this.initTabs();
        this.initCollectTab();
        this.initTrainTab();
        this.initTestTab();

        // Check for saved model
        this.checkSavedModel();
    }

    // ==================== TAB MANAGEMENT ====================

    initTabs() {
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active tab
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update active content
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));

                const targetId = tab.dataset.tab + '-tab';
                document.getElementById(targetId).classList.add('active');
            });
        });
    }

    // ==================== COLLECT TAB ====================

    initCollectTab() {
        const canvas = document.getElementById('collect-canvas');
        this.collectCollector = new GestureCollector(canvas);

        // Override gesture complete handler
        this.collectCollector.onGestureComplete = () => {
            const features = this.collectCollector.extractFeatures();
            const label = parseInt(document.getElementById('gesture-select').value);

            // Add sample to data store
            this.dataStore.addSample(label, features);

            // Update UI
            this.updateSampleCounts();
            this.updateRecentList();

            // Show feedback
            this.showCollectFeedback(label);

            // Clear canvas after a short delay
            setTimeout(() => {
                this.collectCollector.clear();
            }, 500);
        };

        // Buttons
        document.getElementById('clear-data-btn').addEventListener('click', () => {
            if (confirm('Clear all collected data? This cannot be undone.')) {
                this.dataStore.clear();
                this.updateSampleCounts();
                this.updateRecentList();
            }
        });

        document.getElementById('export-data-btn').addEventListener('click', () => {
            this.dataStore.exportJSON();
        });

        document.getElementById('import-data-btn').addEventListener('click', () => {
            document.getElementById('import-file').click();
        });

        document.getElementById('import-file').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    const count = await this.dataStore.importJSON(file);
                    alert(`Imported ${count} samples successfully!`);
                    this.updateSampleCounts();
                    this.updateRecentList();
                } catch (err) {
                    alert('Failed to import data: ' + err.message);
                }
                e.target.value = '';
            }
        });

        // Initial update
        this.updateSampleCounts();
        this.updateRecentList();
    }

    showCollectFeedback(label) {
        const canvas = document.getElementById('collect-canvas');
        const ctx = canvas.getContext('2d');

        // Draw feedback text
        ctx.fillStyle = 'rgba(0, 212, 255, 0.8)';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`✓ ${this.gestureLabels[label]}`, canvas.width / 2, 30);
    }

    updateSampleCounts() {
        const counts = this.dataStore.getSampleCounts();
        const container = document.getElementById('sample-counts');

        container.innerHTML = Object.entries(counts).map(([label, count]) => `
            <div class="sample-count">
                <span class="label">${this.gestureLabels[label]}</span>
                <span class="count">${count}</span>
            </div>
        `).join('');
    }

    updateRecentList() {
        const recent = this.dataStore.getRecentSamples(10);
        const container = document.getElementById('recent-list');

        if (recent.length === 0) {
            container.innerHTML = '<p style="color: #666; font-size: 0.9rem;">No samples yet</p>';
            return;
        }

        container.innerHTML = recent.map(sample => `
            <div class="recent-item">
                <span class="gesture-name">${this.gestureLabels[sample.label]}</span>
                <span class="time">${new Date(sample.timestamp).toLocaleTimeString()}</span>
                <button class="delete-btn" data-id="${sample.id}">×</button>
            </div>
        `).join('');

        // Add delete handlers
        container.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.dataStore.removeSample(parseInt(btn.dataset.id));
                this.updateSampleCounts();
                this.updateRecentList();
            });
        });
    }

    // ==================== TRAIN TAB ====================

    initTrainTab() {
        // Train button
        document.getElementById('train-btn').addEventListener('click', () => {
            this.startTraining();
        });

        // Stop button
        document.getElementById('stop-train-btn').addEventListener('click', () => {
            this.model.stop();
            document.getElementById('stop-train-btn').disabled = true;
        });

        // Save model
        document.getElementById('save-model-btn').addEventListener('click', async () => {
            try {
                await this.model.saveModel('gesture-model');
                this.updateModelStatus('Model saved!', true);
            } catch (err) {
                alert('Failed to save model: ' + err.message);
            }
        });

        // Load model
        document.getElementById('load-model-btn').addEventListener('click', () => {
            document.getElementById('load-model-file').click();
        });

        document.getElementById('load-model-file').addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                try {
                    const success = await this.model.loadModel(files);
                    if (success) {
                        this.updateModelStatus('Model loaded!', true);
                        document.getElementById('save-model-btn').disabled = false;
                    } else {
                        alert('Failed to load model');
                    }
                } catch (err) {
                    alert('Failed to load model: ' + err.message);
                }
                e.target.value = '';
            }
        });

        // Initialize charts
        this.initCharts();
    }

    initCharts() {
        // Loss chart
        const lossCanvas = document.getElementById('loss-chart');
        const lossCtx = lossCanvas.getContext('2d');
        this.lossChart = {
            ctx: lossCtx,
            canvas: lossCanvas,
            data: { loss: [], valLoss: [] }
        };

        // Accuracy chart
        const accCanvas = document.getElementById('accuracy-chart');
        const accCtx = accCanvas.getContext('2d');
        this.accuracyChart = {
            ctx: accCtx,
            canvas: accCanvas,
            data: { acc: [], valAcc: [] }
        };
    }

    drawChart(chart, title) {
        const { ctx, canvas, data } = chart;
        const padding = 40;
        const width = canvas.width - padding * 2;
        const height = canvas.height - padding * 2;

        // Clear
        ctx.fillStyle = '#0a0a15';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Title
        ctx.fillStyle = '#888';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(title, canvas.width / 2, 15);

        // Get all values to determine scale
        const allValues = [...(data.loss || []), ...(data.valLoss || []),
                           ...(data.acc || []), ...(data.valAcc || [])];

        if (allValues.length === 0) return;

        const maxVal = Math.max(...allValues, 1);
        const minVal = Math.min(...allValues, 0);

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();

        // Draw lines
        const drawLine = (values, color) => {
            if (values.length < 2) return;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < values.length; i++) {
                const x = padding + (i / (values.length - 1)) * width;
                const y = canvas.height - padding - ((values[i] - minVal) / (maxVal - minVal)) * height;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();
        };

        if (data.loss) drawLine(data.loss, '#00d4ff');
        if (data.valLoss) drawLine(data.valLoss, '#ff6b6b');
        if (data.acc) drawLine(data.acc, '#00ff88');
        if (data.valAcc) drawLine(data.valAcc, '#ffaa00');

        // Legend
        ctx.font = '10px sans-serif';
        let legendX = padding + 10;

        if (data.loss) {
            ctx.fillStyle = '#00d4ff';
            ctx.fillText('● Train', legendX, padding + 15);
            legendX += 50;
        }
        if (data.valLoss || data.valAcc) {
            ctx.fillStyle = data.valLoss ? '#ff6b6b' : '#ffaa00';
            ctx.fillText('● Val', legendX, padding + 15);
        }
    }

    async startTraining() {
        const samples = this.dataStore.getAllSamples();

        if (samples.length < 10) {
            alert('Please collect at least 10 samples before training');
            return;
        }

        // Get training options
        const epochs = parseInt(document.getElementById('epochs-input').value);
        const learningRate = parseFloat(document.getElementById('lr-input').value);
        const batchSize = parseInt(document.getElementById('batch-input').value);
        const validationSplit = parseFloat(document.getElementById('val-split-input').value);
        const addNoise = document.getElementById('noise-augment').checked;

        // Update UI
        document.getElementById('train-btn').disabled = true;
        document.getElementById('stop-train-btn').disabled = false;
        document.getElementById('total-epochs').textContent = epochs;
        document.getElementById('training-progress').style.width = '0%';

        // Reset charts
        this.lossChart.data = { loss: [], valLoss: [] };
        this.accuracyChart.data = { acc: [], valAcc: [] };

        try {
            await this.model.train(
                samples,
                { epochs, learningRate, batchSize, validationSplit, addNoise },
                // onEpochEnd callback
                (epoch, logs) => {
                    // Update progress
                    document.getElementById('current-epoch').textContent = epoch + 1;
                    document.getElementById('current-loss').textContent = logs.loss.toFixed(4);
                    document.getElementById('train-accuracy').textContent =
                        (logs.acc * 100).toFixed(1) + '%';

                    if (logs.val_acc !== undefined) {
                        document.getElementById('val-accuracy').textContent =
                            (logs.val_acc * 100).toFixed(1) + '%';
                    }

                    document.getElementById('training-progress').style.width =
                        ((epoch + 1) / epochs * 100) + '%';

                    // Update charts
                    this.lossChart.data.loss.push(logs.loss);
                    if (logs.val_loss !== undefined) {
                        this.lossChart.data.valLoss.push(logs.val_loss);
                    }
                    this.drawChart(this.lossChart, 'Loss');

                    this.accuracyChart.data.acc.push(logs.acc);
                    if (logs.val_acc !== undefined) {
                        this.accuracyChart.data.valAcc.push(logs.val_acc);
                    }
                    this.drawChart(this.accuracyChart, 'Accuracy');
                },
                // onTrainingEnd callback
                (history) => {
                    this.updateModelStatus('Training complete!', true);
                    document.getElementById('save-model-btn').disabled = false;
                }
            );
        } catch (err) {
            alert('Training failed: ' + err.message);
            console.error(err);
        } finally {
            document.getElementById('train-btn').disabled = false;
            document.getElementById('stop-train-btn').disabled = true;
        }
    }

    updateModelStatus(message, trained = false) {
        const status = document.getElementById('model-status');
        status.textContent = message;
        status.classList.toggle('trained', trained);
    }

    async checkSavedModel() {
        const exists = await this.model.checkSavedModel('gesture-model');
        if (exists) {
            this.updateModelStatus('Saved model found. Click "Load Model" to use it.');
        }
    }

    // ==================== TEST TAB ====================

    initTestTab() {
        const canvas = document.getElementById('test-canvas');
        this.testCollector = new GestureCollector(canvas);

        // Override gesture complete handler
        this.testCollector.onGestureComplete = () => {
            this.testGesture();
        };

        // Clear button
        document.getElementById('clear-test-btn').addEventListener('click', () => {
            this.testCollector.clear();
            document.getElementById('prediction-result').innerHTML =
                '<span class="prediction-label">Draw a gesture...</span>';
            document.getElementById('confidence-bars').innerHTML = '';
        });

        // Initialize confidence bars
        this.initConfidenceBars();
    }

    initConfidenceBars() {
        const container = document.getElementById('confidence-bars');
        container.innerHTML = Object.entries(this.gestureLabels).map(([id, name]) => `
            <div class="confidence-row" data-class="${id}">
                <span class="label">${name}</span>
                <div class="bar-container">
                    <div class="bar" style="width: 0%"></div>
                </div>
                <span class="value">0%</span>
            </div>
        `).join('');
    }

    testGesture() {
        if (!this.model.isTrained) {
            document.getElementById('prediction-result').innerHTML =
                '<span class="prediction-label" style="color: #ff6b6b;">Model not trained!</span>';
            return;
        }

        const features = this.testCollector.extractFeatures();

        try {
            const prediction = this.model.predict(features);

            // Update prediction display
            const resultEl = document.getElementById('prediction-result');
            resultEl.innerHTML = `
                <span class="prediction-label">${prediction.displayName}</span>
                <span class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</span>
            `;

            // Update confidence bars
            const bars = document.querySelectorAll('.confidence-row');
            bars.forEach((row, i) => {
                const probability = prediction.probabilities[i];
                const bar = row.querySelector('.bar');
                const value = row.querySelector('.value');

                bar.style.width = (probability * 100) + '%';
                value.textContent = (probability * 100).toFixed(1) + '%';

                // Highlight highest
                row.style.opacity = i === prediction.class ? 1 : 0.6;
            });

            // Update test stats
            this.updateTestStats(prediction);

        } catch (err) {
            console.error('Prediction failed:', err);
            document.getElementById('prediction-result').innerHTML =
                '<span class="prediction-label" style="color: #ff6b6b;">Prediction failed!</span>';
        }
    }

    updateTestStats(prediction) {
        this.testStats.total++;

        // For now, just track predictions (no ground truth in test mode)
        if (!this.testStats.byClass[prediction.class]) {
            this.testStats.byClass[prediction.class] = 0;
        }
        this.testStats.byClass[prediction.class]++;

        // Update display
        const statsEl = document.getElementById('test-stats-content');
        statsEl.innerHTML = `
            <p><strong>Total predictions:</strong> ${this.testStats.total}</p>
            <p><strong>Predictions by class:</strong></p>
            <ul style="margin-left: 20px; list-style: disc;">
                ${Object.entries(this.testStats.byClass)
                    .map(([cls, count]) =>
                        `<li>${this.gestureLabels[cls]}: ${count} (${(count/this.testStats.total*100).toFixed(1)}%)</li>`
                    ).join('')}
            </ul>
        `;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GestureApp();
    console.log('Gesture Recognition App initialized');
});
