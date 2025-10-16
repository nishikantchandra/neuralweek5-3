// app.js
import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictorApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.charts = {};
        this.isInitialized = false;
        
        this.initializeUI();
    }

    initializeUI() {
        // File upload handling
        document.getElementById('loadDataBtn').addEventListener('click', () => this.loadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('stopTrainBtn').addEventListener('click', () => this.stopTraining());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluateModel());

        this.updateUIState('initial');
        this.isInitialized = true;
    }

    async loadData() {
        const fileInput = document.getElementById('csvFile');
        const statusDiv = document.getElementById('dataStatus');
        
        if (!fileInput.files.length) {
            statusDiv.innerHTML = '<span class="error">Please select a CSV file</span>';
            return;
        }

        try {
            statusDiv.innerHTML = 'Loading CSV file...';
            
            await this.dataLoader.loadCSV(fileInput.files[0]);
            this.dataLoader.prepareData();
            
            const summary = this.dataLoader.getDataSummary();
            statusDiv.innerHTML = `<span class="success">
                Loaded ${summary.symbols.length} stocks: ${summary.symbols.join(', ')}<br>
                ${summary.trainSamples} training samples, ${summary.testSamples} test samples
            </span>`;
            
            this.updateUIState('data_loaded');
            
        } catch (error) {
            statusDiv.innerHTML = `<span class="error">Error loading data: ${error.message}</span>`;
            console.error('Data loading error:', error);
        }
    }

    async startTraining() {
        if (!this.dataLoader.X_train) {
            alert('Please load data first');
            return;
        }

        this.updateUIState('training');
        
        const progressDiv = document.getElementById('trainingProgress');
        const progressBar = document.getElementById('trainingBar');
        
        try {
            // Create model
            const inputShape = [this.dataLoader.X_train.shape[1], this.dataLoader.X_train.shape[2]];
            const outputSize = this.dataLoader.y_train.shape[1];
            this.model.createModel(inputShape, outputSize);

            // Start training
            await this.model.train(
                this.dataLoader.X_train, 
                this.dataLoader.y_train,
                this.dataLoader.X_test,
                this.dataLoader.y_test,
                100, // epochs
                {
                    onEpochEnd: (epoch, logs) => {
                        progressDiv.textContent = 
                            `Epoch: ${epoch + 1}/100 - Loss: ${logs.loss.toFixed(4)} - Acc: ${logs.acc.toFixed(4)} - Val Loss: ${logs.val_loss.toFixed(4)}`;
                        progressBar.value = ((epoch + 1) / 100) * 100;
                    },
                    onTrainEnd: () => {
                        this.updateUIState('trained');
                        progressDiv.textContent += ' - Training completed!';
                    }
                }
            );
            
        } catch (error) {
            console.error('Training error:', error);
            progressDiv.innerHTML = `<span class="error">Training failed: ${error.message}</span>`;
            this.updateUIState('data_loaded');
        }
    }

    stopTraining() {
        this.model.stopTraining();
        document.getElementById('trainingProgress').textContent += ' - Training stopped';
        this.updateUIState('data_loaded');
    }

    async evaluateModel() {
        if (!this.model.model) {
            alert('Please train the model first');
            return;
        }

        const resultsContainer = document.getElementById('resultsContainer');
        resultsContainer.innerHTML = 'Evaluating model...';

        try {
            // Get predictions
            const predictions = await this.model.predict(this.dataLoader.X_test);
            
            // Compute overall accuracy
            const evaluation = await this.model.evaluate(this.dataLoader.X_test, this.dataLoader.y_test);
            
            // Compute per-stock accuracy
            const stockAccuracies = await this.model.computePerStockAccuracy(
                predictions, 
                this.dataLoader.y_test, 
                this.dataLoader.symbols
            );
            
            // Get prediction timeline
            const timeline = await this.model.getPredictionTimeline(
                predictions,
                this.dataLoader.y_test,
                this.dataLoader.symbols
            );

            // Display results
            this.displayResults(evaluation, stockAccuracies, timeline);
            
            // Clean up
            predictions.dispose();
            
        } catch (error) {
            console.error('Evaluation error:', error);
            resultsContainer.innerHTML = `<span class="error">Evaluation failed: ${error.message}</span>`;
        }
    }

    displayResults(evaluation, stockAccuracies, timeline) {
        const resultsContainer = document.getElementById('resultsContainer');
        
        // Sort stocks by accuracy
        const sortedStocks = Object.entries(stockAccuracies)
            .sort(([, accA], [, accB]) => accB - accA);
        
        let html = `
            <h4>Overall Model Performance</h4>
            <p>Test Loss: ${evaluation.loss.toFixed(4)} | Test Accuracy: ${(evaluation.accuracy * 100).toFixed(2)}%</p>
            
            <h4>Stock Performance Ranking</h4>
            <div class="chart-container">
                <canvas id="accuracyChart"></canvas>
            </div>
            
            <h4>Prediction Timeline Examples</h4>
            <div id="timelineCharts"></div>
        `;
        
        resultsContainer.innerHTML = html;
        
        // Create accuracy bar chart
        this.createAccuracyChart(sortedStocks);
        
        // Create timeline charts for top 3 stocks
        this.createTimelineCharts(sortedStocks.slice(0, 3), timeline);
    }

    createAccuracyChart(sortedStocks) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        if (this.charts.accuracy) {
            this.charts.accuracy.destroy();
        }
        
        this.charts.accuracy = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedStocks.map(([symbol]) => symbol),
                datasets: [{
                    label: 'Prediction Accuracy',
                    data: sortedStocks.map(([, accuracy]) => accuracy * 100),
                    backgroundColor: sortedStocks.map(([, accuracy]) => 
                        accuracy > 0.6 ? 'rgba(75, 192, 192, 0.8)' : 
                        accuracy > 0.5 ? 'rgba(255, 205, 86, 0.8)' : 'rgba(255, 99, 132, 0.8)'
                    ),
                    borderColor: sortedStocks.map(([, accuracy]) => 
                        accuracy > 0.6 ? 'rgb(75, 192, 192)' : 
                        accuracy > 0.5 ? 'rgb(255, 205, 86)' : 'rgb(255, 99, 132)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Accuracy: ${context.raw.toFixed(2)}%`
                        }
                    }
                }
            }
        });
    }

    createTimelineCharts(topStocks, timeline) {
        const container = document.getElementById('timelineCharts');
        container.innerHTML = '';
        
        topStocks.forEach(([symbol, accuracy], idx) => {
            const stockTimeline = timeline[symbol];
            const sampleCount = Math.min(10, stockTimeline.length);
            
            const chartContainer = document.createElement('div');
            chartContainer.className = 'timeline-container';
            chartContainer.innerHTML = `<h5>${symbol} - Sample Predictions (Accuracy: ${(accuracy * 100).toFixed(2)}%)</h5>
                                      <canvas id="timelineChart${idx}"></canvas>`;
            container.appendChild(chartContainer);
            
            const ctx = document.getElementById(`timelineChart${idx}`).getContext('2d');
            this.createStockTimelineChart(ctx, symbol, stockTimeline.slice(0, sampleCount));
        });
    }

    createStockTimelineChart(ctx, symbol, timelineData) {
        const datasets = [];
        const labels = Array.from({length: timelineData[0].length}, (_, i) => `Day ${i + 1}`);
        
        // Add sample lines
        timelineData.forEach((sample, sampleIdx) => {
            const correctPoints = [];
            const incorrectPoints = [];
            
            sample.forEach((pred, dayIdx) => {
                if (pred.correct) {
                    correctPoints.push({x: dayIdx, y: sampleIdx});
                } else {
                    incorrectPoints.push({x: dayIdx, y: sampleIdx});
                }
            });
            
            if (correctPoints.length > 0) {
                datasets.push({
                    label: `Sample ${sampleIdx + 1} Correct`,
                    data: correctPoints,
                    pointBackgroundColor: 'green',
                    pointBorderColor: 'green',
                    showLine: false,
                    pointRadius: 6
                });
            }
            
            if (incorrectPoints.length > 0) {
                datasets.push({
                    label: `Sample ${sampleIdx + 1} Wrong`,
                    data: incorrectPoints,
                    pointBackgroundColor: 'red',
                    pointBorderColor: 'red',
                    showLine: false,
                    pointRadius: 6
                });
            }
        });
        
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Prediction Day'
                        },
                        min: -0.5,
                        max: 2.5,
                        ticks: {
                            stepSize: 1
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Sample Index'
                        },
                        reverse: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const sampleIdx = context.parsed.y;
                                const dayIdx = context.parsed.x;
                                const pred = timelineData[sampleIdx][dayIdx];
                                return `Sample ${sampleIdx + 1}, Day ${dayIdx + 1}: ${pred.correct ? 'Correct' : 'Wrong'} (Pred: ${pred.predicted}, Actual: ${pred.actual})`;
                            }
                        }
                    }
                }
            }
        });
    }

    updateUIState(state) {
        const loadBtn = document.getElementById('loadDataBtn');
        const trainBtn = document.getElementById('trainBtn');
        const stopBtn = document.getElementById('stopTrainBtn');
        const evalBtn = document.getElementById('evaluateBtn');
        
        switch (state) {
            case 'initial':
                loadBtn.disabled = false;
                trainBtn.disabled = true;
                stopBtn.disabled = true;
                evalBtn.disabled = true;
                break;
                
            case 'data_loaded':
                loadBtn.disabled = false;
                trainBtn.disabled = false;
                stopBtn.disabled = true;
                evalBtn.disabled = true;
                break;
                
            case 'training':
                loadBtn.disabled = true;
                trainBtn.disabled = true;
                stopBtn.disabled = false;
                evalBtn.disabled = true;
                break;
                
            case 'trained':
                loadBtn.disabled = false;
                trainBtn.disabled = false;
                stopBtn.disabled = true;
                evalBtn.disabled = false;
                break;
        }
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockApp = new StockPredictorApp();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StockPredictorApp;
} else {
    window.StockPredictorApp = StockPredictorApp;
}
