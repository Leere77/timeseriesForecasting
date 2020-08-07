console.clear()
tf.setBackend('cpu')
Math.seedrandom('d')

const epochCounter = document.querySelector('.epochs')
const width = window.innerWidth
let cache = {}
let stop = false
let dataset 
let redraw = [false, false]

if (!config.local) {
    document.querySelector('.start').disabled = false
    document.querySelector('.start').hidden = false
}

csv_file.addEventListener('change', function () {
    const reader = new FileReader()
    reader.onload = function fileReadCompleted() {
        cache.csv = reader.result
        reader.result.split('\n')[0].split(',')
            .forEach(col => {
                document.querySelector('.cols').innerHTML += 
                `<input type="checkbox" name="${col}">
                <label for="${col}"> ${col}</label> <br>`
            })
            document.querySelector('.start').disabled = false
            document.querySelector('.start').hidden = false
    }
    
    reader.readAsText(this.files[0])
    cache.title = this.files[0].name.split('.')[0]
  })

document.addEventListener('keydown', e => {
    if (e.code == 'KeyS') 
        stop = true
})

async function init(config) {
    let rawValues = []
    let { datasets, ratio, smooth: {enabled}, features } = config
    let tsName = config.datasets.filter(ts => ts.inUse).map(({name, col}) => name + col)

    //load series
    for (let ts of datasets)
        if (ts.inUse) 
            rawValues.push(await utils.loadCsv(ts.name, ts.col, cache.csv))
           
    // apply min-max scaler
    let { data, ...scales } = utils.scaler(rawValues)
    
    //smooth data
    if (enabled) 
        data = data.map(ts => utils.equation(ts))  

    // split data
    let { train, test } = utils.dataSplit(data, ratio)
    
    for (let i = 1; i < features; i++)
        console.log(utils.corr(train[0], train[i]))
    
    let series = []

    for (let i = 0; i < features; i++) 
        series.push({
            ...utils.toPlotly(train[i]),
            name: tsName[i] + " (обучающая)"
        })

    for (let i = 0; i < features; i++)
        series.push({
            ...utils.toPlotly(test[i]),
            name: tsName[i] + ' (тестовая)'
        })
        
    let layout = {
        title: 'Временные ряды',
        width,
        height: 800,
        yaxis: {title:  {text: 'Значение'}},
        xaxis: {title:  {text: 'Время'}}
    }
    
    Plotly.newPlot('ts_', series, layout)
    
    const d = utils.diff(train)
    
    cache.datasets = data
    cache.data = utils.dataSplit(data, ratio)
    let algProp = utils.algProp(cache.datasets[0])//utils.algProp(cache.datasets[0])
    
    let shear = utils.shearFunction(algProp)
    let locMin = utils.extrema(shear).locMin.slice(0, -1).filter(({x}) => (x > 5 && x < 80))
    
    utils.drawPlot('shear', 
        [
            {...utils.toPlotly(shear), name: 'Сдвиговая функция'},
            {...utils.toPlotly(locMin),
                mode: 'markers',
                type: 'scatter', name: 'locmin'}
        ], {height: 800, width, yaxis: {title:  {text: 'Значение'}},
        xaxis: {title:  {text: 'Период'}}})
    config.st = document.querySelector('.st_state').checked ? locMin.sort((a, b) => (a.y - b.y))[0].x : +stVal.value
    
    Plotly.newPlot('corr',
    //[{ y: utils.autoCorr(algProp).map(v => v.y)}],
    [{ y: utils.autoCorr(d[0]).map(v => v.y) }],
    {...layout,
        title: 'Автокорреляция',
        yaxis: {title:  {text: 'Коэф. корр-ии'}},
        xaxis: {title: {text: 'Lag'}}
    })

    return d
}

async function start(config) {
    const train = await init(config)

    const trainWindows = train.map(ts => utils.genWindows(ts, config.st))
    //const testWindows = test.map(ts => utils.genWindows(ts, config.st))
    
    await fit(trainWindows, config)
}

async function main() { 
    let datasets = [...document.querySelector('.cols').querySelectorAll('input')]
        .map((col, idx) => ({name: cache.title, inUse: col.checked, col: idx}))
        
    if (config.local && !datasets.filter(ts => ts.inUse).length) {
        alert('Необходимо выбрать хотя бы один ряд')
        return
    }
    
    let currentConfig = {
        ...config,
        ratio: config.local ? +ratioSlider.value / 100 : config.ratio,
        datasets: config.local ? datasets : config.datasets,
        features: datasets.filter(ts => ts.inUse).length,
        smooth: {...config.smooth, enabled: document.querySelector('.smooth').checked}
    }
    
    let model = await start(currentConfig)
}

async function fit(trainW, config) {
    const { st, epochs, batchSize, units, features } = config 

    const { inputs, labels } = utils.convertToTensor(trainW)
    //const { inputs: inputsTest, labels: labelsTest } = utils.convertToTensor(testW)

    const model = tf.sequential()
    let inputShape = [st, features]
    let regularizer = tf.regularizers.l1l2()
    let dropout = 0.5
    let kernerInitializer = tf.initializers.glorotNormal({seed: 13})
    let biasInitializer = tf.initializers.glorotNormal({seed: 13})

    model.add(tf.layers.reshape({inputShape, targetShape: [st * features]}))

    model.add(tf.layers.dense({
        units: units,
        kernerInitializer,
        activation: 'tanh',
        useBias: 0,
        kernel_regularizer: regularizer
    }))

    model.add(tf.layers.dropout(dropout))

    model.add(tf.layers.dense({
        units: units * features,
        kernerInitializer,
        activation: 'tanh',
        useBias: 0,
        kernel_regularizer: regularizer
    }))

    model.add(tf.layers.dropout(dropout))

    model.add(tf.layers.dense({
        units: units ,
        kernerInitializer,
        activation: 'sigmoid',
        useBias: 1,
        biasInitializer,
        kernel_regularizer: regularizer
    }))

    model.add(tf.layers.dense({units: features, activation: 'tanh'}))

    let optimizer = [0.01, 0.95, 1e-07]
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    })

    let history = [{x: [], y: [], name: 'train'}, {x: [], y: [], name: 'val'}]

    let layout = {
        title: 'Обучение',
        width: 800,
        height: 400,
        yaxis: {title:  {text: 'MSE'}},
        xaxis: {title:  {text: 'Номер итерации'}}
    }

    function append(epoch, {val_mse, mse}) {
        if(history[0].y.length > 100) {
            history[0].x = history[0].x.slice(1)
            history[0].y = history[0].y.slice(1)
            history[1].x = history[1].x.slice(1)
            history[1].y = history[1].y.slice(1)
        }

        if (stop) {
            model.stopTraining = true
            stop = false
        }

        epochCounter.innerText = (epoch/epochs * 100).toFixed() + "%"

        currentErr = testModel(model, inputs, config)
        history[0].x.push(epoch)
        history[1].x.push(epoch)
        history[0].y.push(mse)
        history[1].y.push(val_mse)

        Plotly.newPlot('train', history, layout)
    }

    await model.fit(inputs, labels, 
        {
            batchSize,
            epochs,
            shuffle: config.shuffle,
            callbacks: { onEpochEnd: async (epoch, logs) => append(epoch, logs) },
            validationSplit: 0.1
        })

    predict(model, config)
}

function predict(model, config) {
    let { st, features } = config
    let data = cache.datasets
    let tsName = config.datasets.filter(ts => ts.inUse).map(({name, col}) => name + col)

    // for (let ts of data)
    //     data = data.concat(utils.emd(ts).slice(0, IMFs))
    
    const diffs = utils.diff(data)
    let a = diffs.map(ts => ts.slice(-st).map(({y}) => y))
    let seq = tf.tensor(a).transpose().reshape([1, st, features])
    const predStart = data[0].length

    let predictedPoints = tf.tidy(() => {
        let preds = []

        for (let i = 0; i < st * 2; i++) {
            let pred = model.predict(seq).dataSync()
            let predTensor = tf.tensor(pred, [1, 1, features])
            preds.push(pred)
            seq = seq.concat(predTensor, 1).slice([0, 1])
        }
        
        return Array(features).fill()
            .map((_, tsIdx) => preds.map((val, idx) => ({x: idx + predStart, y: val[tsIdx]})))
    })
    
    predictedPoints = utils.inverseDiff({
        diffs: predictedPoints,
        firstVal: data.map(ts => ts.slice(-1)).map(val => val[0])}
    )
    
    let series = []
  
    for (let i = 0; i < predictedPoints.length; i++) {
        series.push({
            ...utils.toPlotly(predictedPoints[i]),
            name: tsName[i] + " (прогноз)"
        })

        series.push({
            ...utils.toPlotly(data[i]),
            name: tsName[i]
        })
    }
    
    let range = [0, Math.max(Math.max(...series[0].y), Math.max(...series[1].y))]
    
    let layout = {
        title: `Прогноз, Ширина окна ${st}`,
        width,
        height: 800,
        yaxis: {
            title: {text: 'Значение'},
            showspikes : true,
            range
        },
        xaxis: {title:  {text: 'Время'}, showspikes : true},
        hovermode: 'closest',
    }

    layout.shapes = [{
        type: 'line',
        x0: predStart,
        y0: range[0],
        x1: predStart,
        y1: range[1],
        line:{
            color: 'rgb(0, 0, 0)',
            width: 2,
            dash:'dot'
        }
    }]

    Plotly.newPlot('predict', series, layout)
    document.querySelector("#predict").scrollIntoView()
}

function testModel(model, inputs, config) {
    let { st, features } = config
    let {train, test} = cache.data

    const predStart = test[0][0].x

    let tsName = config.datasets.filter(ts => ts.inUse).map(({name, col}) => name + col)

    let predictedPoints = tf.tidy(() => {
        let preds = []
        let seq = inputs.slice([inputs.shape[0] - 1], 1)

        for (let i = 0; i < test[0].length; i++) {
            let pred = model.predict(seq).dataSync()
            let predTensor = tf.tensor(pred, [1, 1, features])
            preds.push(pred)
            seq = seq.concat(predTensor, 1).slice([0, 1])
        }
        
        return Array(features).fill()
            .map((_, tsIdx) => preds.map((val, idx) => ({x: idx + predStart, y: val[tsIdx]})))
    })

    predictedPoints = utils.inverseDiff({
            diffs: predictedPoints, firstVal: train.map(ts => ts.slice(-1)).map(val => val[0]),
        })

    let err = 0

    for (let i = 0; i < test[0].length; i++) 
        if (test[0][i].y > 0) 
            err += Math.abs(test[0][i].y - predictedPoints[0][i].y) / 
            (Math.abs(test[0][i].y) + Math.abs(predictedPoints[0][i].y))  * 100 / test[0].length

    let series = []
  
    for (let i = 0; i < predictedPoints.length; i++) {
        series.push({
            ...utils.toPlotly(train[i]),
            name: tsName[i] + " (обучающая)"
        })

        series.push({
            ...utils.toPlotly(predictedPoints[i]),
            name: tsName[i] + " (прогноз)"
        })

        series.push({
            ...utils.toPlotly(test[i]),
            name: tsName[i] + " (тестовая)"
        })
    }
    
    let range = [Math.min(...series[series.length-1].y) - 0.2, Math.max(...series[series.length-1].y) + 0.2]
    
    let layout = {
        title: `Прогноз, Ширина окна ${st} <br> Ошибка:  ${err.toFixed(2)}`,
        width,
        height: 800,
        yaxis: {
            title: {text: 'Значение'},
            showspikes : true,
            range
        },
        xaxis: {title:  {text: 'Время'}, showspikes : true},
        hovermode: 'closest',
    }

    layout.shapes = [{
        type: 'line',
        x0: train[0].length,
        y0: range[0],
        x1: train[0].length,
        y1: range[1],
        line:{
            color: 'rgb(0, 0, 0)',
            width: 2,
            dash:'dot'
        }
    }]

    if (!redraw[0]) {
        redraw[0] = true
        Plotly.newPlot('forecast', series, layout)
    } else 
        Plotly.react('forecast', series, layout)
    
    return err
}