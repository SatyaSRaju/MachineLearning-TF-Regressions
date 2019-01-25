require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})
//
const regression = new LinearRegression(features, labels, {
    learningRate: 0.1, //trail and error
    iterations: 3,
    batchSize: 10
})

regression.train();
const rSquared = regression.test(testFeatures, testLabels );

// plot({
//     x: regression.bHistory,
//     y: regression.mseHistory.reverse(),
//     xLabel: 'Value of B',
//     yLabel: 'Mean Squared Error'
// })

plot({
   
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
})

//console.log('MSE History : ', regression.mseHistory)
console.log('The Value of rSquared is' , rSquared)

// console.log (
//     'Updated M is ', regression.weights.get(1,0), 
//     "Updated B Value is ", regression.weights.get(0,0)
//     )

regression.predict([
    [120, 2, 380]
]).print()