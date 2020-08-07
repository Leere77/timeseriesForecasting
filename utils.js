class utils {
    static async loadCsv(file, col, csv) {
        if (file == 'sin') { // sin function (few harmonics) plus linear trend
            let noise = tf.truncatedNormal([1, 300], 0, .25 , 'float32', 1).dataSync()
            let fx = x => (Math.sin(x /5) + Math.sin(x / 3)) / 4 + x / 50
            let fx1 = x => x % 20+ x / 50
            let genData = (size, f) => [...new Array(size)].map((_, x) => ({x, y: f(x) + noise[x]}))

            return genData(300, fx)
        }

        let csvDataset = this.CSVToArray(csv).slice(1)

        let data = csvDataset
            .map(val => Number(val[col]))
            .filter(Boolean)
            .map(Math.log)
            .map((y, x) => ({x, y}))

        return data
    }

    static y = ({y}) => y
    static x = ({x}) => x
    static sum = (a, b) => a + b 
    static range = (length, offset = 0) => Array(length).fill().map((_, idx) => idx + offset)

    static dataSplit(data, ratio) {
        let NUM_TRAIN = Math.floor(ratio * data[0].length)
        let train = []
        let test = []
    
        for (let ts of data) {
            train.push(ts.slice(0, NUM_TRAIN ))
            test.push(ts.slice(NUM_TRAIN))
        }

        return { train, test }
    }

    static genWindows(data, st) {
        let res = []
   st++
        for (let i = 0; i <= data.length - st; i++)
            res.push(data.slice(i, i + st).map(this.y))
    
        return res
    }

    static convertToTensor(data, st) {
        return tf.tidy(() => {
            let labels = []

            for (let i = 0; i < data[0].length; i++) {
                labels.push([])
                for (let j = 0; j < data.length; j++) {
                    labels[i].push(data[j][i].pop())
                }
            }
            tf.tensor(data).print()
            let res = data[0].map(seq => seq.map(val => ([val])))
    
            for (let i = 1; i < data.length; i++)
                for (let seq = 0; seq < res.length; seq++)
                    for (let seqIdx = 0; seqIdx < res[0].length; seqIdx++)
                        res[seq][seqIdx].push(data[i][seq][seqIdx])

            let inputTensor = tf.tensor(res)
            let labelTensor = tf.tensor2d(labels, [labels.length, labels[0].length]);
            console.log(inputTensor.shape)
            console.log((labelTensor.shape))

            return {
                inputs: inputTensor,
                labels: labelTensor
            }
        })
    }

    // multistep forecasting (doesn't perform better than the one step solution)
    // static genWindows1(data, st) {
    //     let res = []

    //     for (let i = 0; i <= data.length - st; i++)
    //         res.push(data.slice(i, i + st).map(this.y))
            
    //     return res
    // }

    // static convertToTensor1(data, st = 5) { 
    //     data = data.map(ts => this.genWindows1(ts, st))
    //     let x = tf.tensor(data[0].slice(0, -st))
    //     let y = tf.tensor(data[0].slice(st))
    //     x.print()
    //     y.print()
    //     for (let i = 1; i < data.length; i++) {
    //         x = x.concat(tf.tensor(data[i].slice(0, -st)), [1])
    //         y = y.concat(tf.tensor(data[i].slice(st)), [1])
    //     }
    //     x.print()
    //     y.print()
    //     return { inputs: x, labels: y }
    // }

    static extrema(ts, thr = 1) {
        let a = thr * 3 // threshold depends on input noise variance
        let firstMean = ts.slice(0, thr * a).reduce((a, b) => ({y: a.y + b.y})).y/ (thr * a)
        let locMax = [{x: 0, y: firstMean}]
        let locMin = [{x: 0, y: firstMean}]

        for (let i = 1; i < ts.length - 1; i++) {
            let tl = ts.slice(i >= thr ? i - thr : 0, i).map(({y}) => y)
            let tr = ts.slice(i + 1, i + thr + 1).map(({y}) => y)
            let t = ts[i].y
            
            if (Math.max(...tl) < t && t > Math.max(...tr))
                locMax.push({x: ts[i].x, y: t})
    
            if (Math.min(...tl) > t && t < Math.min(...tr))
                locMin.push({x: ts[i].x, y: t})
            
        }
        let lastMean = ts.slice(-thr * a).reduce((a, b) => ({y: a.y + b.y})).y/ (thr * a)
        locMax.push({x: ts.length, y: lastMean})
        locMin.push({x: ts.length, y: lastMean})

        return {locMax, locMin}
    }
//EMD doesn't work correct yet
    static emd(ts, threshold = 1) {
        const {locMax, locMin} = this.extrema(ts, threshold)
        const splineMax = new Spline(locMax.map(({x}) => x), locMax.map(({y}) => y))
        const splineMin = new Spline(locMin.map(({x}) => x), locMin.map(({y}) => y))

        let interpMax = []
        let interpMin = []
    
        for (let x = 0; x < ts.length; x++) {
            interpMax.push({x, y: splineMax.at(x)})
            interpMin.push({x, y: splineMin.at(x)})
        }

        let h = interpMax.map(({y, x}) => ({x, y: (y + interpMin[x].y) / 2}))

        if(locMax.length < 6) 
            return [h]

        return [h, ...this.emd(h)]
    }

    static isIMF(ts, lMax, lMin) {
        const {locMax, locMin} = this.extrema(ts)

        const splineMax = new Spline(locMax.map(({x}) => x), locMax.map(({y}) => y))
        const splineMin = new Spline(locMin.map(({x}) => x), locMin.map(({y}) => y))

        let interpMax = []
        let interpMin = []
    
        for (let x = 0; x < ts.length; x++) {
            interpMax.push({x, y: splineMax.at(x)})
            interpMin.push({x, y: splineMin.at(x)})
        }

        let h = interpMax.map(({y, x}) => ({x, y: (y + interpMin[x].y) / 2}))

        let zeros = 4
        let mean = 0

        for (let i = 1; i < ts.length - 1; i++) 
            if (ts[i - 1].y > 0 && ts[i].y < 0 ||  ts[i - 1].y < 0 && ts[i].y > 0)
                zeros++
        
        mean = tf.tensor1d(h.map(this.y)).mean().dataSync()[0]
        zeros = Math.abs(zeros - lMax - lMin) 
        
        if (zeros > 1)
            zeros = false
        else
            zeros = true

        return (zeros && Math.abs(mean) < 0.075) ? true : false
    }

    static emd2(ts) {
        let c = []
        let r = []
        let h = ts
        let w = ts

        for (let i = 0; i < 4; i++) {
        
            while (true) {
                const {locMax, locMin} = this.extrema(h)
                const splineMax = new Spline(locMax.map(({x}) => x), locMax.map(({y}) => y))
                const splineMin = new Spline(locMin.map(({x}) => x), locMin.map(({y}) => y))
        
                let interpMax = []
                let interpMin = []
            
                for (let x = 0; x < ts.length; x++) {
                    interpMax.push({x, y: splineMax.at(x)})
                    interpMin.push({x, y: splineMin.at(x)})
                }
        
                h = interpMax.map(({y, x}) => ({x, y: w[x].y - (y + interpMin[x].y) / 2}))

                //if (this.isIMF(h, locMax.length, locMin.length, interpMax.map(({y, x}) => ({x, y: (y + interpMin[x].y) / 2}))))
                    break
            }

            c.push(h)

            r.push(ts.map((val, idx) => ({x: ts[idx].x, y: w[idx].y - h[idx].y})))
            mean = Infinity
            w = r[r.length - 1]
            
            c.forEach(cts => this.drawPlot('c'+i, this.toPlotly(cts)))
        }

        let res = c[0].map(({x, y}) => ({x, y}))

        for (let i = 1; i < c.length; i++)
            for (let j = 0; j < c[0].length; j++)
                res[j].y += c[i][j].y
        
        res.forEach((val, idx) => (val.y += r[r.length - 1][idx].y))
        this.drawPlot('r', this.toPlotly(res))
        return [c, r[r.length - 1], res]
    }

    static emd(ts) {
        let isMonotonic = ts => 
            ts.every((e, i, a) => i ? e.y >= a[i-1].y : true) || ts.every((e, i, a) => i ? e.y <= a[i-1].y : true)
        
        function isIMF(ts, extremaLength) {
            let zeros = 0
            for (let i = 1; i < ts.length - 1; i++) 
                if (ts[i - 1].y > 0 && ts[i].y < 0 ||  ts[i - 1].y < 0 && ts[i].y > 0)
                    zeros++

            return Math.abs(zeros - extremaLength + 4) < 2 ? true : false
        }

        let c = []
        let r = []
        let m = []
        let h = ts.slice()
        let w = utils.toTfvis(utils.toPlotly(ts))

        for (let i = 0; i < 40; i++) {
            while (true) {
                const {locMax, locMin} = utils.extrema(h)
                const splineMax = new Spline(locMax.map(({x}) => x), locMax.map(({y}) => y))
                const splineMin = new Spline(locMin.map(({x}) => x), locMin.map(({y}) => y))
                
                let interpMax = []
                let interpMin = []
            
                for (let x = 0; x < ts.length; x++) {
                    interpMax.push({x, y: splineMax.at(x)})
                    interpMin.push({x, y: splineMin.at(x)})
                }
                m.push(interpMax.map(({x, y}) => ({x, y: (y + interpMin[x].y) / 2})))
                h = interpMax.map(({x, y}) => ({x, y: h[x].y - (y + interpMin[x].y) / 2}))
                utils.drawPlot('current', [ts, interpMax, interpMin, h].map(utils.toPlotly), {height: 800, width: 800})
                
                if (isIMF(h, locMax.length + locMin.length))
                    break
            }
            
            c.push(h)

            for (let j = 0; j < ts.length; j++)
                w[j].y -= c[c.length - 1][j].y

            c.forEach(cts => utils.drawPlot('c'+i, [utils.toPlotly(cts)]))
            utils.drawPlot('w'+i, [utils.toPlotly(w)])
            r.push(utils.toTfvis(utils.toPlotly(w)))

            if (isMonotonic(w))
                break
            else 
                h = w

            //c.forEach(cts => utils.drawPlot('c'+i, [utils.toPlotly(cts)]))
        }

        let res = w//[0].map(({x, y}) => ({x, y}))

        for (let i = 1; i < c.length; i++)
            for (let j = 0; j < c[0].length; j++)
                res[j].y += c[i][j].y
        
        //utils.drawPlot('res', [res, ts].map(utils.toPlotly), {height: 800})
        
        return m
    }

    static diff(data) {
        let diffs = []
    
        for (let ts of data) {
            let diff = []

            for (let i = 1; i < ts.length; i++)
                diff.push({
                    x: i,
                    y: ts[i].y - ts[i - 1].y
                })
    
            diffs.push(diff)
        }
    
        return diffs
    }

    static inverseDiff({diffs, firstVal}) {
        let data = []

        for (let tsIdx = 0; tsIdx < diffs.length; tsIdx++) {
            let ts = []
            
            ts.push(firstVal[tsIdx])
            
            for (let i = 1; i < diffs[tsIdx].length; i++)
                ts.push({
                    x: ts[0].x + i,
                    y: ts[i - 1].y + diffs[tsIdx][i].y
                })
                
            data.push(ts)
        }

        return data
    }

    static inverseDiff1({diffs, fv, diffFeatures}) {
        let data = []

        for (let tsIdx = 0; tsIdx < diffFeatures; tsIdx++) {
            let ts = []
            
            ts.push(fv ? fv : diffs[tsIdx][0]) // was y: firstVal[tsIdx]
            
            for (let i = 1; i < diffs[tsIdx].length; i++)
                ts.push({
                    x: ts[0].x + i,
                    y: ts[i - 1].y + diffs[tsIdx][i].y
                })
                
            data.push(ts)
        }

        return data.concat(diffs.slice(diffFeatures))
    }

    static standardize(data) {
        let mean = []
        let std = []

        for (let ts of data) {
            let t = tf.moments(tf.tensor(ts.map(v=>v.y)))
            mean.push(t.mean.dataSync()[0])
            std.push(Math.sqrt(t.variance.dataSync()[0]))
        }

        for (let idx of data.keys())
            data[idx].forEach(val => val.y = (val.y - mean[idx]) / std[idx])

        return { data, mean, std }
    }

    static standardizeInv({data, mean, std}) {
        for (let idx of data.keys())
            data[idx].forEach(val => val.y = (val.y * std[idx] + mean[idx]))

        return data
    }

    static scaler(data) {
        let max = []
        let min = []
    
        for (let ts of data) {
            max.push(Math.max(...ts.map(v=>v.y)))
            min.push(Math.min(...ts.map(v=>v.y)))
        }
    
        for (let idx of data.keys())
            data[idx].forEach(val => val.y = (val.y - min[idx]) / (max[idx] - min[idx]))
    
        return { data, min, max }
    }

    static scalerInv({data, min, max}) {
        for (let idx of data.keys())
            data[idx].forEach(val => val.y = val.y * (max[idx] - min[idx]) + min[idx])
    
        return data
    }

    static corr(ts1, ts2) {
        let ts1T = tf.tensor(ts1.map(this.y))
        let ts2T = tf.tensor(ts2.map(this.y))
        let ts1M = tf.moments(ts1T)
        let ts2M = tf.moments(ts2T)
        let mean1 = ts1M.mean.dataSync()[0]
        let mean2 = ts2M.mean.dataSync()[0]
        let var1 = ts1M.variance.dataSync()[0]
        let var2 = ts2M.variance.dataSync()[0]

        return tf.tensor([...Array(ts1.length)]
            .map((_, idx) => (ts1[idx].y - mean1) * (ts2[idx].y - mean2))).sum().dataSync()[0]
             / (Math.sqrt(var1) * Math.sqrt(var2)) / ts1.length
    }

    static autoCorr(data) {
        let corrs = []
        let ts = data.map(this.y)//(val=>val.y
        let tsLen = ts.length
        let mean = ts.reduce(this.sum) / tsLen
    
        for (let k = 0; k < tsLen - k; k++) {
            let h = 0
            let b = 0
    
            for (let t = 1; t < tsLen - k; t++)
                h += (ts[t] - mean) * (ts[t + k] - mean)
    
            for (let t = 1; t < tsLen; t++)
                b += (ts[t] - mean) ** 2
    
            corrs.push({
                x: k,
                y: h / b
            })
        }
    
        return corrs.slice(0, 100)
    }

    static simpleSmooth(data, p) {
        let res = []
        let ts
    
        for (let tsIdx of data.keys()) {
            res.push([])
            ts = data[tsIdx]
    
            for (let i = p; i < ts.length - p; i++) {
                let z_m = 0
                    for (let j = -p; j <= p; j++) 
                        z_m += ts[i + j].y / (2 * p + 1)

                res[tsIdx].push({x: i - p, y: z_m})
            }
        }
    
        return res
    }

    static algProp(d, delta = 7) {
        let res = []

        for (let t = delta; t < d.length - delta; t++) 
            if (d[t].y) res.push((d[t - delta].y + d[t + delta].y) / (d[t].y * 2))

        return res.map((y, x) => ({x, y}))
    }

    static geometricProp(d, delta = 15) {
        let res = []

        for (let t = delta; t < d.length - delta; t++) 
            if (d[t].y) res.push((d[t - delta].y * d[t + delta].y) / (d[t].y ** 2))

        return res.map((y, x) => ({x, y}))
    }

    static shearFunction(d, p = 1) {
        let res = []

        for (let tau = 1; tau < d.length / 2; tau++) {
            let temp = 0

            for (let t = 0; t < d.length - tau; t++) 
                temp += Math.abs(d[t + tau].y - d[t].y)**p

            res.push(Math.pow(temp, 1/p) / (d.length - tau))
        }

        return res.map((y, idx) => ({x: idx + 1, y}))
    }

    static diffCentral(d) {
        let diffs = []
        let firstVal = []
    
        for (let ts of d) {
            let dTs = []
            firstVal.push([ts[0], ts[1]])
    
            for (let i = 1; i < ts.length - 1; i++) 
                dTs.push({
                    x: i - 1,
                    y: (ts[i + 1].y - ts[i - 1].y) 
                })
    
            diffs.push(dTs)
        }
        
        return {diffs, firstVal}
    }
    
    static reverseDiffCentral( {diffs, firstVal} ) {
        let res = []
    
        for (let tsIdx= 0 ; tsIdx < diffs.length; tsIdx++) {
            let ts = [firstVal[tsIdx][0], firstVal[tsIdx][1]]
    
            for (let x = 0; x < diffs[tsIdx].length; x++)
                ts.push({x: x + 2, y: ts[ts.length - 2].y + diffs[tsIdx][x].y })
            
            res.push(ts)
        }
    
        return res
    }

    static noiseE(ts) {
        let res = 0

        for (let i = 2; i < ts.length - 2; i++) 
            res += (ts[i + 2] - 2 * ts[i + 1]  + 2 * ts[i - 1] - ts[i - 2]) ** 2

        res /= ((ts.length - 4) * 10)

        // for (let i = 2; i < ts.length - 2; i++) 
        //     res += (ts[i - 2] - 4 * ts[i - 1] + 6*ts[i]  - 4 * ts[i + 1] + ts[i + 2]) ** 2

        // res /= ((ts.length - 4) * 70)

        return res
    }

    static matrixA(size) {
        size -= 4
        let res = []
        let arr = len => Array(len).fill(0)
        let a0 = [
            [1, -4, 6, -4, 1],
            [-4, 17, -28, 22, -8, 1],
            [6, -28, 53, -52, 28, -8, 1],
            [-4, 22, -52, 69, -56, 28, -8, 1]
        ]
        
        for (let [idx, val] of a0.entries())
            res.push(val.concat(arr(size - idx - 1)))

        for (let i = 0; i < size - 4; i++) 
            res.push(arr(i).concat([1, -8, 28, -56, 70, -56, 28, -8, 1]).concat(arr(size - i - 5)))

        for (let [idx, val] of a0.reverse().entries())
            res.push(arr(size + idx - 4).concat(val.reverse()))

        return res
    }

    static cetralDiff(ts) {
        let res = [[], []]

        for (let i = 2; i < ts.length - 2; i++) {
            res[0].push((ts[i + 1] - ts[i - 1]) / 2)
            res[1].push((ts[i + 2] - ts[i - 2]) / 4)
        }

        return [math.matrix(res[0]), math.matrix(res[1])]
    }

    static solve(ts, t, A, I) {
        let tI = math.multiply(I, t * 4)
        let AplustI = math.add(A, tI)
        let b = math.multiply(A, ts)
        let S = math.lusolve(AplustI, b)
        
        let sumS = S._data.map(s => s ** 2).reduce(this.sum)

        return [S._data, sumS]
    }

    static equation(ts) {
        ts = ts.map(({y}) => y)
        let original = ts.slice()
        
        let size = ts.length, noise

        let A = math.multiply(math.matrix(this.matrixA(size)), 0.25)
        let I = math.identity(size)

        let t0 = 0.1, t1, r, S, sumS

        for (let i = 0; i < 2;i++) { // iterations count depends on epsilon value
            noise = this.noiseE(ts) * size
            r = this.solve(ts, t0, A, I)
            S = r[0]
            sumS = r[1]

            while (true) {
                if (sumS > noise) {
                    while (true) {
                        [S, sumS] = this.solve(ts, t0 * 2, A, I)
                        
                        if (sumS < noise) {
                            t1 = t0 * 2
                            break
                        } else t0 = t0 * 2
                    }
                } else {
                    while (true) {
                        [S, sumS] = this.solve(ts, t0 / 2, A, I)
                        
                        if (sumS > noise) {
                            t1 = t0 / 2
                            break
                        } else t0 = t0 / 2
                    }
                }

                let t = (t0 + t1) / 2
                let r = this.solve(ts, t, A, I)

                S = r[0]
                sumS = r[1]
                console.log(Math.abs(sumS - noise))
                
                if (Math.abs(sumS - noise) < 0.0005)
                    break

                if (sumS < noise)
                    t0 = t
                else 
                    t1 = t
        }
        ts = ts.map((val, idx) => val - S[idx])
        
        this.drawPlot('noise1', [{y: original, name: 'Input'}, {y: ts, name: 'Output'}], {height: 800, width: 1300})
    }

        return ts.map((y, x) => ({x, y}))
    }

    static CSVToArray(strData, strDelimiter) {
        //taken from https://gist.github.com/luishdez/644215
        strDelimiter = (strDelimiter || ",") 

        var objPattern = new RegExp(
            (
                "(\\" + strDelimiter + "|\\r?\\n|\\r|^)" +
                "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +
                "([^\"\\" + strDelimiter + "\\r\\n]*))"
            ), "gi");
        
        var arrData = [[]];
        var arrMatches = null;

        while (arrMatches = objPattern.exec( strData )){
            var strMatchedDelimiter = arrMatches[ 1 ];
            var strMatchedValue;

            if (strMatchedDelimiter.length && strMatchedDelimiter !== strDelimiter)
                arrData.push( [] );

            if (arrMatches[ 2 ])
                strMatchedValue = arrMatches[ 2 ].replace(new RegExp( "\"\"", "g" ), "\"");
            else 
                strMatchedValue = arrMatches[ 3 ];
            
            arrData[ arrData.length - 1 ].push( strMatchedValue );
        }
        
        return( arrData );
    }

    static drawPlot(id, series, l) {
        let layout = {
            title: id,
            width: 1400,
            height: 300,
            yaxis: {title:  {text: 'Значение'}},
            xaxis: {title:  {text: 'Время'}}
        }

        layout = {...layout, ...l}
    
        let el = document.createElement('div')
        el.id = id
        document.querySelector('.plots').appendChild(el)
        
        Plotly.newPlot(id, [...series], layout)
    }

    static toPlotly(ts) {
        let res = {x: [], y: []}
        
        ts.forEach(({x, y}) => {
            res.x.push(x)
            res.y.push(y)
        })

        return res
    }

    static toTfvis(ts) {
        let res = []

        for (let i = 0; i < ts.y.length; i++)
            res.push({x: ts.x[i], y: ts.y[i]})

        return res
    }
}