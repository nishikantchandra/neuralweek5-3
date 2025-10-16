// data-loader.js
class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.symbols = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.dateIndex = [];
        this.featureNames = [];
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(`CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`));
                        return;
                    }
                    this.rawData = results.data.filter(row => row.Date && row.Symbol);
                    resolve(this.rawData);
                },
                error: (error) => reject(error)
            });
        });
    }

    prepareData() {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('No data loaded');
        }

        // Extract unique symbols and sort dates
        this.symbols = [...new Set(this.rawData.map(row => row.Symbol))];
        const allDates = [...new Set(this.rawData.map(row => row.Date))].sort();
        
        // Create pivoted data structure
        const pivotedData = {};
        this.symbols.forEach(symbol => {
            pivotedData[symbol] = {
                Open: new Array(allDates.length).fill(null),
                Close: new Array(allDates.length).fill(null)
            };
        });

        // Fill pivoted data
        const dateToIndex = new Map(allDates.map((date, idx) => [date, idx]));
        
        this.rawData.forEach(row => {
            const dateIdx = dateToIndex.get(row.Date);
            if (dateIdx !== undefined && pivotedData[row.Symbol]) {
                pivotedData[row.Symbol].Open[dateIdx] = row.Open;
                pivotedData[row.Symbol].Close[dateIdx] = row.Close;
            }
        });

        // Handle missing data (forward fill)
        this.symbols.forEach(symbol => {
            ['Open', 'Close'].forEach(field => {
                let lastValue = null;
                for (let i = 0; i < allDates.length; i++) {
                    if (pivotedData[symbol][field][i] !== null) {
                        lastValue = pivotedData[symbol][field][i];
                    } else if (lastValue !== null) {
                        pivotedData[symbol][field][i] = lastValue;
                    }
                }
            });
        });

        // Normalize data per stock (Min-Max scaling)
        const normalizedData = {};
        this.symbols.forEach(symbol => {
            normalizedData[symbol] = { Open: [], Close: [] };
            
            ['Open', 'Close'].forEach(field => {
                const values = pivotedData[symbol][field].filter(val => val !== null);
                const min = Math.min(...values);
                const max = Math.max(...values);
                
                normalizedData[symbol][field] = pivotedData[symbol][field].map(val => 
                    val !== null ? (val - min) / (max - min) : 0
                );
            });
        });

        // Create feature matrix and prepare samples
        const WINDOW_SIZE = 12;
        const PREDICTION_DAYS = 3;
        
        const samples = [];
        const labels = [];
        this.dateIndex = [];

        for (let i = WINDOW_SIZE; i < allDates.length - PREDICTION_DAYS; i++) {
            const sample = [];
            
            // Create input sequence (last 12 days)
            for (let j = i - WINDOW_SIZE; j < i; j++) {
                const timeStep = [];
                this.symbols.forEach(symbol => {
                    timeStep.push(normalizedData[symbol].Open[j], normalizedData[symbol].Close[j]);
                });
                sample.push(timeStep);
            }

            // Create output labels (next 3 days for each stock)
            const label = [];
            this.symbols.forEach(symbol => {
                const currentClose = normalizedData[symbol].Close[i-1];
                for (let k = 0; k < PREDICTION_DAYS; k++) {
                    const futureClose = normalizedData[symbol].Close[i + k];
                    label.push(futureClose > currentClose ? 1 : 0);
                }
            });

            samples.push(sample);
            labels.push(label);
            this.dateIndex.push(allDates[i]);
        }

        // Split chronologically (80% train, 20% test)
        const splitIndex = Math.floor(samples.length * 0.8);
        
        this.X_train = tf.tensor3d(samples.slice(0, splitIndex));
        this.y_train = tf.tensor2d(labels.slice(0, splitIndex));
        this.X_test = tf.tensor3d(samples.slice(splitIndex));
        this.y_test = tf.tensor2d(labels.slice(splitIndex));

        this.featureNames = this.symbols.flatMap(symbol => [`${symbol}_Open`, `${symbol}_Close`]);
        this.processedData = normalizedData;

        console.log(`Created ${samples.length} samples (${splitIndex} train, ${samples.length - splitIndex} test)`);
        console.log(`Input shape: [${samples.length}, ${WINDOW_SIZE}, ${this.symbols.length * 2}]`);
        console.log(`Output shape: [${samples.length}, ${this.symbols.length * PREDICTION_DAYS}]`);
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }

    getDataSummary() {
        return {
            symbols: this.symbols,
            trainSamples: this.X_train ? this.X_train.shape[0] : 0,
            testSamples: this.X_test ? this.X_test.shape[0] : 0,
            windowSize: this.X_train ? this.X_train.shape[1] : 0,
            features: this.X_train ? this.X_train.shape[2] : 0,
            outputSize: this.y_train ? this.y_train.shape[1] : 0
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataLoader;
} else {
    window.DataLoader = DataLoader;
}
