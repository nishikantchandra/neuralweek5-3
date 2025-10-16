/* app.js
 * UI controller + visualisation layer
 *  – hooks file-input → DataLoader → GRUModel
 *  – trains in-browser, shows live progress
 *  – after predict: ranks stocks by accuracy, draws horizontal bar + per-stock timeline
 *  – 100 % client-side, ES6 modules, GitHub-Pages ready
 */

import DataLoader from './data-loader.js';
import GRUModel  from './gru.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model      = null;
    this.isTraining = false;
    this.currentEval= null;   // evaluation object from model
    this.accuracyChart = null;
    this.initUI();
  }

  initUI() {
    const fileIn   = document.getElementById('csvFile');
    const trainBtn = document.getElementById('trainBtn');
    const predBtn  = document.getElementById('predictBtn');

    fileIn .addEventListener('change', e => this.handleFile(e));
    trainBtn.addEventListener('click', () => this.trainModel());
    predBtn .addEventListener('click', () => this.runPredict());
  }

  async handleFile(evt) {
    const file = evt.target.files[0];
    if (!file) return;
    try {
      this.setStatus('Loading CSV…');
      await this.dataLoader.loadCSV(file);
      this.setStatus('Pre-processing…');
      this.dataLoader.createSequences(12,3); // 12-day window, 3-day horizon
      document.getElementById('trainBtn').disabled = false;
      this.setStatus('Data ready – press Train Model');
    } catch (err) {
      this.setStatus(`Load error: ${err.message}`);
      console.error(err);
    }
  }

  async trainModel() {
    if (this.isTraining) return;
    this.isTraining = true;
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('predictBtn').disabled = true;

    try {
      const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader;
      this.model = new GRUModel([12, symbols.length*2], symbols.length*3); // 12×20 → 30 outputs

      this.setStatus('Training…');
      await this.model.train(X_train, y_train, X_test, y_test,
                             50,   // epochs
                             32,   // batch
                             (epoch, total, logs, prog) => this.updateProgress(epoch, total, logs, prog));

      document.getElementById('predictBtn').disabled = false;
      this.setStatus('Training complete – press Run Prediction');
    } catch (err) {
      this.setStatus(`Training error: ${err.message}`);
      console.error(err);
    } finally {
      this.isTraining = false;
    }
  }

  async runPredict() {
    if (!this.model) { alert('Train model first'); return; }
    try {
      this.setStatus('Predicting…');
      const { X_test, y_test, symbols } = this.dataLoader;
      const preds = await this.model.predict(X_test);
      this.currentEval = this.model.evaluatePerStock(y_test, preds, symbols);

      this.visualise(this.currentEval, symbols);
      preds.dispose();
      this.setStatus('Prediction complete – charts below');
    } catch (err) {
      this.setStatus(`Prediction error: ${err.message}`);
      console.error(err);
    }
  }

  /* ---------- visualisation ---------- */
  visualise(evalObj, symbols) {
    this.drawAccuracyBars(evalObj.stockAccuracies, symbols);
    this.drawTimelines(evalObj.stockPredictions, symbols);
  }

  drawAccuracyBars(accuracies) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    if (this.accuracyChart) this.accuracyChart.destroy();

    const sorted = Object.entries(accuracies)
                         .sort(([,a],[,b])=>b-a); // best → worst
    const labels = sorted.map(([s])=>s);
    const data   = sorted.map(([,a])=>a*100);

    this.accuracyChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets:[{
          label:'Accuracy %',
          data,
          backgroundColor: data.map(v=> v>60?'rgba(75,192,192,0.8)'
                                           : v>50?'rgba(255,205,86,0.8)'
                                                  :'rgba(255,99,132,0.8)'),
          borderWidth:1
        }]
      },
      options: {
        indexAxis:'y',
        scales:{x:{min:0,max:100,title:{display:true,text:'Accuracy %'}}},
        plugins:{legend:{display:false}}
      }
    });
  }

  drawTimelines(predictions) {
    const container = document.getElementById('timelineContainer');
    container.innerHTML = '';

    // show top-3 most accurate stocks
    const top = Object.entries(predictions)
                      .sort(([,a],[,b])=>b.filter(p=>p.correct).length/a.length - a.filter(p=>p.correct).length/b.length)
                      .slice(0,3);

    top.forEach(([sym, preds]) => {
      const wrap = document.createElement('div');
      wrap.className = 'stock-chart';
      wrap.innerHTML = `<h4>${sym} Timeline</h4><canvas id="tl-${sym}"></canvas>`;
      container.appendChild(wrap);

      const ctx = document.getElementById(`tl-${sym}`).getContext('2d');
      const sample = preds.slice(0,50); // first 50 preds for clarity
      const labels = sample.map((_,i)=>`P${i+1}`);
      const correctFlag = sample.map(p=>p.correct?1:0);

      new Chart(ctx, {
        type:'line',
        data:{
          labels,
          datasets:[{
            label:'Correct (1) / Wrong (0)',
            data: correctFlag,
            borderColor:'#4bc0c0',
            backgroundColor:'rgba(75,192,192,0.2)',
            pointBackgroundColor: correctFlag.map(f=>f?'#4bc0c0':'#ff6384'),
            fill:true,
            tension:0.3
          }]
        },
        options:{
          scales:{y:{min:0,max:1, ticks:{callback:v=>v===1?'Correct':'Wrong'}}},
          plugins:{tooltip:{callbacks:{label:(ctx)=>{
            const p = sample[ctx.dataIndex];
            return `Pred:${p.pred?'Up':'Down'} | Actual:${p.true?'Up':'Down'}`;
          }}}}
        }
      });
    });
  }

  /* ---------- helpers ---------- */
  setStatus(msg){ document.getElementById('status').textContent = msg; }
  updateProgress(epoch, total, logs, prog){
    const pct = (epoch/total)*100;
    document.getElementById('trainingProgress').value = pct;
    this.setStatus(`Epoch ${epoch}/${total} – loss:${logs.loss.toFixed(4)} acc:${logs.binaryAccuracy.toFixed(4)}`);
  }

  /* clean-up on page unload (optional) */
  dispose(){
    if(this.dataLoader) this.dataLoader.dispose();
    if(this.model)      this.model.dispose();
    if(this.accuracyChart) this.accuracyChart.destroy();
  }
}

/* bootstrap once DOM ready */
document.addEventListener('DOMContentLoaded', () => new StockPredictionApp());
