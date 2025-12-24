/**
 * TensorFlow.js Model for Gesture Recognition
 * Converted from Python MXNet implementation
 */

import * as tf from '@tensorflow/tfjs';

export class GestureModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };

        // Model configuration matching Python implementation
        this.config = {
            numFeatures: 6,           // dx, dy, speed, angle, pressure, time_delta
            numDatapoints: 40,        // 40 datapoints per gesture (same as Python)
            inputSize: 240,           // 40 * 6 = 240 features
            hiddenUnits: 128,         // Same as Python
            numClasses: 7,            // 7 gesture classes
            dropoutRate: 0.5          // Same as Python
        };

        // Gesture class mapping (same as Python)
        this.gestureClasses = {
            0: 'no-gesture',
            1: 'tv-poke',
            2: 'shutter-right-left',
            3: 'letter-m',
            4: 'letter-y',
            5: 'fan-one-circle',
            6: 'fan-two-circles'
        };

        this.gestureNames = {
            'no-gesture': 'No Gesture',
            'tv-poke': 'TV Poke',
            'shutter-right-left': 'Shutter (Swipe)',
            'letter-m': 'Letter M',
            'letter-y': 'Letter Y',
            'fan-one-circle': 'Fan Slow (1 circle)',
            'fan-two-circles': 'Fan Fast (2 circles)'
        };
    }

    /**
     * Create the neural network architecture
     * Matches the Python MXNet model:
     * - Input: 240 features
     * - Dense(128, relu) + Dropout(0.5)
     * - Dense(128, relu) + Dropout(0.5)
     * - Dense(7, softmax)
     */
    createModel() {
        this.model = tf.sequential({
            layers: [
                // First hidden layer
                tf.layers.dense({
                    units: this.config.hiddenUnits,
                    activation: 'relu',
                    inputShape: [this.config.inputSize],
                    kernelInitializer: 'glorotNormal' // Xavier initialization
                }),
                tf.layers.dropout({ rate: this.config.dropoutRate }),

                // Second hidden layer
                tf.layers.dense({
                    units: this.config.hiddenUnits,
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal'
                }),
                tf.layers.dropout({ rate: this.config.dropoutRate }),

                // Output layer
                tf.layers.dense({
                    units: this.config.numClasses,
                    activation: 'softmax'
                })
            ]
        });

        console.log('Model created:');
        this.model.summary();

        return this.model;
    }

    /**
     * Compile the model with optimizer and loss function
     */
    compile(learningRate = 0.05) {
        if (!this.model) {
            this.createModel();
        }

        this.model.compile({
            optimizer: tf.train.sgd(learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log(`Model compiled with learning rate: ${learningRate}`);
    }

    /**
     * Add noise augmentation to training data (same as Python)
     * noise = 0.1 * mean(abs(data)) * random_normal
     */
    augmentWithNoise(data) {
        return tf.tidy(() => {
            const meanAbs = tf.mean(tf.abs(data));
            const noise = tf.randomNormal(data.shape).mul(0.1).mul(meanAbs);
            return data.add(noise);
        });
    }

    /**
     * Prepare data for training
     * @param {Array} samples - Array of gesture samples
     * @param {boolean} shuffle - Whether to shuffle the data
     * @returns {Object} - { xs: Tensor, ys: Tensor }
     */
    prepareData(samples, shuffle = true) {
        if (samples.length === 0) {
            return null;
        }

        // Shuffle samples if requested
        if (shuffle) {
            samples = [...samples].sort(() => Math.random() - 0.5);
        }

        // Extract features and labels
        const features = samples.map(s => s.features);
        const labels = samples.map(s => s.label);

        // Create tensors
        const xs = tf.tensor2d(features);

        // One-hot encode labels
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.config.numClasses);

        return { xs, ys };
    }

    /**
     * Train the model
     * @param {Array} trainSamples - Training samples
     * @param {Object} options - Training options
     * @param {Function} onEpochEnd - Callback for epoch end
     * @param {Function} onTrainingEnd - Callback for training end
     */
    async train(trainSamples, options = {}, onEpochEnd = null, onTrainingEnd = null) {
        const {
            epochs = 100,
            batchSize = 32,
            validationSplit = 0.2,
            learningRate = 0.05,
            addNoise = true
        } = options;

        // Reset training history
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };

        // Create and compile model
        this.createModel();
        this.compile(learningRate);

        // Prepare data
        const data = this.prepareData(trainSamples);
        if (!data) {
            throw new Error('No training data available');
        }

        let { xs, ys } = data;

        // Add noise augmentation if enabled
        if (addNoise) {
            xs = this.augmentWithNoise(xs);
        }

        this.stopTraining = false;

        try {
            // Train the model
            await this.model.fit(xs, ys, {
                epochs,
                batchSize,
                validationSplit,
                shuffle: true,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Store history
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.accuracy.push(logs.acc);
                        if (logs.val_loss !== undefined) {
                            this.trainingHistory.valLoss.push(logs.val_loss);
                            this.trainingHistory.valAccuracy.push(logs.val_acc);
                        }

                        // Call user callback
                        if (onEpochEnd) {
                            onEpochEnd(epoch, logs);
                        }

                        // Check for stop signal
                        if (this.stopTraining) {
                            this.model.stopTraining = true;
                        }

                        // Allow UI updates
                        await tf.nextFrame();
                    },
                    onTrainEnd: () => {
                        this.isTrained = true;
                        if (onTrainingEnd) {
                            onTrainingEnd(this.trainingHistory);
                        }
                    }
                }
            });
        } finally {
            // Cleanup tensors
            xs.dispose();
            ys.dispose();
        }

        return this.trainingHistory;
    }

    /**
     * Stop training
     */
    stop() {
        this.stopTraining = true;
    }

    /**
     * Make a prediction on a single gesture
     * @param {Array} features - 240 feature values
     * @returns {Object} - { class: number, className: string, confidence: number, probabilities: Array }
     */
    predict(features) {
        if (!this.model || !this.isTrained) {
            throw new Error('Model not trained');
        }

        return tf.tidy(() => {
            // Reshape to [1, 240]
            const input = tf.tensor2d([features]);

            // Get predictions
            const predictions = this.model.predict(input);
            const probabilities = predictions.dataSync();

            // Get class with highest probability
            const classIndex = predictions.argMax(1).dataSync()[0];
            const confidence = probabilities[classIndex];

            return {
                class: classIndex,
                className: this.gestureClasses[classIndex],
                displayName: this.gestureNames[this.gestureClasses[classIndex]],
                confidence,
                probabilities: Array.from(probabilities)
            };
        });
    }

    /**
     * Evaluate model on test data
     * @param {Array} testSamples - Test samples
     * @returns {Object} - { loss, accuracy, confusionMatrix }
     */
    evaluate(testSamples) {
        if (!this.model || !this.isTrained) {
            throw new Error('Model not trained');
        }

        const data = this.prepareData(testSamples, false);
        if (!data) {
            throw new Error('No test data available');
        }

        const { xs, ys } = data;

        try {
            // Evaluate model
            const result = this.model.evaluate(xs, ys);
            const [loss, accuracy] = result.map(t => t.dataSync()[0]);

            // Generate confusion matrix
            const predictions = this.model.predict(xs);
            const predClasses = predictions.argMax(1).dataSync();
            const trueClasses = ys.argMax(1).dataSync();

            const confusionMatrix = this.computeConfusionMatrix(
                Array.from(trueClasses),
                Array.from(predClasses)
            );

            // Cleanup
            result.forEach(t => t.dispose());
            predictions.dispose();

            return { loss, accuracy, confusionMatrix };
        } finally {
            xs.dispose();
            ys.dispose();
        }
    }

    /**
     * Compute confusion matrix
     */
    computeConfusionMatrix(trueLabels, predictedLabels) {
        const matrix = Array(this.config.numClasses).fill(null)
            .map(() => Array(this.config.numClasses).fill(0));

        for (let i = 0; i < trueLabels.length; i++) {
            matrix[trueLabels[i]][predictedLabels[i]]++;
        }

        return matrix;
    }

    /**
     * Save model to local storage / download
     */
    async saveModel(name = 'gesture-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }

        // Save to downloads
        await this.model.save(`downloads://${name}`);

        // Also save to localStorage for quick access
        await this.model.save(`localstorage://${name}`);

        console.log(`Model saved as ${name}`);
        return true;
    }

    /**
     * Load model from files or localStorage
     */
    async loadModel(files = null, name = 'gesture-model') {
        try {
            if (files) {
                // Load from uploaded files
                this.model = await tf.loadLayersModel(
                    tf.io.browserFiles(files)
                );
            } else {
                // Try to load from localStorage
                this.model = await tf.loadLayersModel(`localstorage://${name}`);
            }

            // Recompile the loaded model
            this.model.compile({
                optimizer: tf.train.sgd(0.05),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            this.isTrained = true;
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            return false;
        }
    }

    /**
     * Check if a model exists in localStorage
     */
    async checkSavedModel(name = 'gesture-model') {
        try {
            const models = await tf.io.listModels();
            return `localstorage://${name}` in models;
        } catch {
            return false;
        }
    }

    /**
     * Get model summary
     */
    getSummary() {
        if (!this.model) {
            return 'No model created';
        }

        let summary = '';
        this.model.summary(undefined, undefined, (line) => {
            summary += line + '\n';
        });
        return summary;
    }

    /**
     * Dispose of the model and free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isTrained = false;
        }
    }
}
