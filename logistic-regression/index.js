require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: [
        'horsepower',
        'displacement',
        'weight'
    ],
    labelColumns : ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
           return value === 'TRUE' ? 1:0
        }
    }
})

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iteration: 100,
    batchSize: 10,
    decisionBoundary: 0.5
})

regression.train();

console.log( "Finding the Accuracy %  for Test Features and Predicted Labels")
console.log(regression.test(testFeatures, testLabels))

console.log ("The Prediction for values  (Horsepower , Displacement , Weight)  [130, 307,1.75],  [88, 97, 1.065] is " )
regression.predict(
  [  
    [130, 307,1.75],
    [88, 97, 1.065]
  ]
).print()

plot({
    x: regression.costHistory.reverse()
})