const config = {
    st: 20, // step size
    ratio: 0.85,
    batchSize: 200, 
    epochs: 1050,
    predLength: 60, 
    lstmLayer: 0,
    diff: 1,
    shuffle: 0,
    units: 15, // neurons in each layer
    stateful: 0,
    datasets: [ 
        {name: 'airline-passengers', col: 1, inUse: 0},
        {name: 'sin', col: 1, inUse: 1}, 
    ],
    local: 1,
    smooth: {
        p: 1,
        enabled: true
    }
}