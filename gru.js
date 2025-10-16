/* gru.js
 * Multi-output GRU for 10-stock × 3-day binary classification
 *  – input shape: [samples, 12, 20]  (12 days, 10 stocks × 2 features)
 *  – output shape: [samples, 30]     (10 stocks × 3 future days, sigmoid)
 *  – binaryCrossentropy + binaryAccuracy
 *  – optional bidirectional stack
 *  – per-stock accuracy helper
 *  – browser-only (TensorFlow.js CDN)
 */

class GRUModel {
  constructor(inputShape, outputSize, biDir = true) {
    this.model       = null;
    this.inputShape  = inputShape;   // [timeSteps, featuresPerStep]
    this.outputSize  = outputSize;   // 30 (10 stocks × 3 horizons)
    this.biDir       = biDir;
    this.history     = null;
  }

  /* ---------- build ---------- */
  buildModel() {
    const layers = [];

    /* first GRU (return sequences) */
    if (this.biDir) {
      layers.push(tf.layers.bidirectional({
        layer: tf.layers.gru({ units: 64, returnSequences: true }),
        inputShape: this.inputShape
      }));
    } else {
      layers.push(tf.layers.gru({
        units: 64, returnSequences: true, inputShape: this.inputShape
      }));
    }
    layers.push(tf.layers.dropout({ rate: 0.2 }));

    /* second GRU (return false) */
    layers.push(tf.layers.gru({ units: 32, returnSequences: false }));
    layers.push(tf.layers.dropout({ rate: 0.2 }));

    /* output */
    layers.push(tf.layers.dense({
      units: this.outputSize,
      activation: 'sigmoid'
    }));

    this.model = tf.sequential({ layers });
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return this.model;
  }

  /* ---------- train ---------- */
  async train(X_train, y_train, X_test, y_test,
              epochs = 50, batchSize = 32,
              onEpochEnd = null)   // optional callback
  {
    if (!this.model) this.buildModel();

    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        const pct = ((epoch + 1) / epochs * 100).toFixed(1);
        const msg = `Epoch ${epoch + 1}/${epochs} – ` +
                    `loss:${logs.loss.toFixed(4)} acc:${logs.binaryAccuracy.toFixed(4)} – ` +
                    `val_loss:${logs.val_loss.toFixed(4)} val_acc:${logs.val_binaryAccuracy.toFixed(4)}`;
        if (onEpochEnd) onEpochEnd(epoch + 1, epochs, logs, pct);
        console.log(msg);
        await tf.nextFrame(); // keep UI responsive
      }
    };

    this.history = await this.model.fit(X_train, y_train, {
      epochs, batchSize, validationData: [X_test, y_test], callbacks
    });
    return this.history;
  }

  /* ---------- predict ---------- */
  async predict(X) {
    if (!this.model) throw new Error('Model not trained');
    return this.model.predict(X);
  }

  /* ---------- per-stock accuracy ---------- */
  evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
    const yTrueArr = yTrue.arraySync();
    const yPredArr = yPred.arraySync();
    const stockAcc = {}, stockPred = {};

    symbols.forEach((sym, idx) => {
      let correct = 0, total = 0;
      const preds = [];
      for (let s = 0; s < yTrueArr.length; s++) {
        for (let h = 0; h < horizon; h++) {
          const tgt = idx * horizon + h;
          const trueVal = yTrueArr[s][tgt];
          const predVal = yPredArr[s][tgt] > 0.5 ? 1 : 0;
          if (trueVal === predVal) correct++;
          total++;
          preds.push({ true: trueVal, pred: predVal, correct: trueVal === predVal });
        }
      }
      stockAcc[sym] = correct / total;
      stockPred[sym] = preds;
    });
    return { stockAccuracies: stockAcc, stockPredictions: stockPred };
  }

  /* ---------- save / load weights (browser localStorage example) ---------- */
  async saveWeights(key = 'gru-weights') {
    const weights = await this.model.save(`localstorage://${key}`);
    console.log('Weights saved to localStorage under', key);
    return weights;
  }

  async loadWeights(key = 'gru-weights') {
    if (!this.model) this.buildModel();
    await this.model.load(`localstorage://${key}`);
    console.log('Weights loaded from localStorage under', key);
  }

  /* ---------- memory ---------- */
  dispose() {
    if (this.model) this.model.dispose();
  }
}

export default GRUModel;
