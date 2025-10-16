// gru.js
class GRUModel {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            val_loss: [],
            val_accuracy: []
        };
    }

    createModel(inputShape, outputSize) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: inputShape
                }),
                tf.layers.dropout({ rate: 0.2 }),
                
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                
                tf.layers.dense({
                    units: outputSize,
                    activation: 'sigmoid'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        console.log('GRU Model created:');
        this.model.summary();
        
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 100, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not created. Call createModel() first.');
        }

        this.isTraining = true;
        this.trainingHistory = { loss: [], accuracy: [], val_loss: [], val_accuracy: [] };

        try {
            const history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: 32,
                validationData: [X_test, y_test],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.accuracy.push(logs.acc);
                        this.trainingHistory.val_loss.push(logs.val_loss);
                        this.trainingHistory.val_accuracy.push(logs.val_acc);

                        if (callbacks.onEpochEnd) {
                            callbacks.onEpochEnd(epoch, logs);
                        }

                        // Prevent memory leaks
                        await tf.nextFrame();
                    },
                    onTrainEnd: () => {
                        this.isTraining = false;
                        if (callbacks.onTrainEnd) {
                            callbacks.onTrainEnd();
                        }
                    }
                }
            });

            return history;
        } catch (error) {
            this.isTraining = false;
            throw error;
        }
    }

    stopTraining() {
        this.isTraining = false;
        // Note: TensorFlow.js doesn't have direct stop training method
        // This flag will be checked in training loop
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        return this.model.predict(X);
    }

    async evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        
        const result = this.model.evaluate(X_test, y_test);
        const loss = await result[0].data();
        const accuracy = await result[1].data();
        
        result[0].dispose();
        result[1].dispose();
        
        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    async computePerStockAccuracy(predictions, y_test, symbols, predictionDays = 3) {
        const predData = await predictions.array();
        const trueData = await y_test.array();
        
        const stockAccuracies = {};
        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            
            for (let sample = 0; sample < predData.length; sample++) {
                for (let day = 0; day < predictionDays; day++) {
                    const predIdx = stockIdx * predictionDays + day;
                    const predicted = predData[sample][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueData[sample][predIdx];
                    
                    if (predicted === actual) {
                        correct++;
                    }
                    total++;
                }
            }
            
            stockAccuracies[symbol] = correct / total;
        });
        
        return stockAccuracies;
    }

    async getPredictionTimeline(predictions, y_test, symbols, predictionDays = 3) {
        const predData = await predictions.array();
        const trueData = await y_test.array();
        
        const timeline = {};
        symbols.forEach((symbol, stockIdx) => {
            timeline[symbol] = [];
            
            for (let sample = 0; sample < predData.length; sample++) {
                const sampleResult = [];
                for (let day = 0; day < predictionDays; day++) {
                    const predIdx = stockIdx * predictionDays + day;
                    const predicted = predData[sample][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueData[sample][predIdx];
                    
                    sampleResult.push({
                        predicted,
                        actual,
                        correct: predicted === actual
                    });
                }
                timeline[symbol].push(sampleResult);
            }
        });
        
        return timeline;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('indexeddb://multi-stock-gru-model');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://multi-stock-gru-model');
            console.log('Model loaded from storage');
            return true;
        } catch (error) {
            console.log('No saved model found');
            return false;
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GRUModel;
} else {
    window.GRUModel = GRUModel;
}
