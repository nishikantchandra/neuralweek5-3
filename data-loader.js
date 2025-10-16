/* data-loader.js
 * Browser-only CSV → TensorFlow.js pipeline
 *  – parses a local CSV (Date,Symbol,Open,Close,High,Low,Volume)
 *  – keeps only Open/Close per stock
 *  – Min-Max normalises per stock
 *  – builds sliding windows: 12-day input → 3-day-ahead binary up/down labels
 *  – chronological 80 / 20 train/test split
 *  – exposes X_train, y_train, X_test, y_test, symbols (all JS tensors)
 *  – memory-safe (dispose() method)
 */

class DataLoader {
  constructor() {
    this.rawCSV = null;          // original text
    this.pivot = null;           // {date:{symbol:{Open,Close}}}
    this.symbols = [];           // alphabetical list of tickers
    this.dates = [];             // sorted date strings
    this.minMax = {};            // per-stock MinMax scaler params
    this.X_train = null;
    this.y_train = null;
    this.X_test = null;
    this.y_test = null;
  }

  /* PUBLIC ENTRY POINT */
  async loadCSV(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = e => {
        try {
          this.rawCSV = e.target.result;
          this.parseAndPivot();
          this.computeMinMax();
          resolve();
        } catch (err) { reject(err); }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  /* 1. Parse CSV → pivot table: date → symbol → {Open,Close} */
  parseAndPivot() {
    const lines = this.rawCSV.trim().split('\n');
    if (lines.length < 2) throw new Error('CSV too short');

    const headers = lines[0].split(',').map(h => h.trim());
    const dateIdx = headers.indexOf('Date');
    const symIdx  = headers.indexOf('Symbol');
    const openIdx = headers.indexOf('Open');
    const closeIdx= headers.indexOf('Close');

    [dateIdx, symIdx, openIdx, closeIdx].forEach(i => {
      if (i < 0) throw new Error('Missing required column');
    });

    const pivot = {};            // date -> symbol -> {Open,Close}
    const symSet = new Set();

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      const date   = cols[dateIdx];
      const symbol = cols[symIdx];
      const open   = parseFloat(cols[openIdx]);
      const close  = parseFloat(cols[closeIdx]);

      if (!pivot[date]) pivot[date] = {};
      pivot[date][symbol] = { Open: open, Close: close };
      symSet.add(symbol);
    }

    this.pivot  = pivot;
    this.dates  = Object.keys(pivot).sort();
    this.symbols= Array.from(symSet).sort();

    console.log(`Loaded ${this.symbols.length} stocks × ${this.dates.length} dates`);
  }

  /* 2. Compute Min-Max per stock (fit on whole history) */
  computeMinMax() {
    this.minMax = {};
    this.symbols.forEach(s => {
      let minO = Infinity, maxO = -Infinity;
      let minC = Infinity, maxC = -Infinity;
      this.dates.forEach(d => {
        const row = (this.pivot[d] || {})[s];
        if (!row) return;
        minO = Math.min(minO, row.Open);
        maxO = Math.max(maxO, row.Open);
        minC = Math.min(minC, row.Close);
        maxC = Math.max(maxC, row.Close);
      });
      this.minMax[s] = { Open: { min: minO, max: maxO },
                         Close:{ min: minC, max: maxC } };
    });
  }

  /* 3. Normalise a single value */
  norm(val, min, max) {
    if (max === min) return 0;
    return (val - min) / (max - min);
  }

  /* 4. Build supervised samples
   *    inputLen  = 12  (look-back window)
   *    outputLen = 3   (D+1, D+2, D+3)
   */
  createSequences(inputLen = 12, outputLen = 3) {
    const X = [], y = [];

    /* we need at least inputLen past + outputLen future */
    for (let i = inputLen; i <= this.dates.length - outputLen - 1; i++) {
      const currentDate = this.dates[i];

      /* ---------- build input tensor ---------- */
      const seq = [];
     for (let i = inputLen; i <= this.dates.length - outputLen - 1; i++) {
  const currentDate = this.dates[i];

  /* ---------- build input tensor ---------- */
  const seq = [];
  let ok = true;                   // ✅ moved outside
  for (let j = inputLen - 1; j >= 0; j--) {
    const day = this.dates[i - j];
    const stepVec = [];
    ...
  }
  if (!ok || seq.length !== inputLen) continue;  // ✅ now safe   // incomplete window

      /* ---------- build 3-day binary labels ---------- */
      const baseClose = {};               // Close on currentDate
      this.symbols.forEach(sym => {
        baseClose[sym] = (this.pivot[currentDate] || {})[sym]?.Close;
        if (baseClose[sym] == null) { ok = false; }
      });
      if (!ok) continue;

      const labels = [];
      for (let off = 1; off <= outputLen; off++) {
        const futureDate = this.dates[i + off];
        this.symbols.forEach(sym => {
          const futClose = (this.pivot[futureDate] || {})[sym]?.Close;
          labels.push(futClose > baseClose[sym] ? 1 : 0);
        });
      }
      if (labels.length !== this.symbols.length * outputLen) continue;

      X.push(seq);
      y.push(labels);
    }

    if (X.length === 0) throw new Error('No valid sequences generated');

    /* 80 % chronological train / 20 % test */
    const split = Math.floor(X.length * 0.8);
    this.X_train = tf.tensor3d(X.slice(0, split));
    this.y_train = tf.tensor2d(y.slice(0, split));
    this.X_test  = tf.tensor3d(X.slice(split));
    this.y_test  = tf.tensor2d(y.slice(split));

    console.log(`Sequences: total=${X.length}, train=${this.X_train.shape[0]}, test=${this.X_test.shape[0]}`);
    return { X_train: this.X_train, y_train: this.y_train,
             X_test:  this.X_test,  y_test:  this.y_test,
             symbols: this.symbols };
  }

  /* 5. Memory clean-up */
  dispose() {
    [this.X_train, this.y_train, this.X_test, this.y_test]
      .forEach(t => t && t.dispose());
  }
}

export default DataLoader;
