const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogisticRegression {
    constructor(features, labels, options) {

        /* Lodash Implementation ~ Tensor Implementation with Tensor Weights
         *  this.m = 0
         *  this.b = 0
         * 
         */


        this.features = this.processFeatures(features)
        this.labels= tf.tensor(labels)
        this.costHistory =[]
        this.bHistory = []
       
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 }, options)

        //Create a Zero based Tensor with m and b values as Zero.
        this.weights = tf.zeros([this.features.shape[1],1])
    }

    gradientDescent(features, labels) {

        /* Implementaion with lodash */
            // const currentGuessForMPG =  this.features.map(row => {
            //     return this.m*row[0]+this.b
            // })

            // const bSlope = _.sum(currentGuessForMPG.map((guess, i) =>{
            //        return guess - this.labels[i][0]

            // })) * 2/this.features.length

            // const mSlope = _.sum(currentGuessForMPG.map((guess, i) => {
            //      return -1 * this.features[i][0] * (this.labels[i][0] - guess )
            // })) * 2 / this.features.length

            // this.m = this.m - mSlope * this.options.learningRate
            // this.b = this.b - bSlope * this.options.learningRate


        /* Refactor with Tensors */

        //Slope of MSE wrt M and B =  Features * ((Features * Weights) - Labels)/n
         
        //mx+b  = Features * Weights
        const currentGuesses = features.matMul(this.weights).sigmoid() 
        //(Features * Weights) - Labels
        const differences = currentGuesses.sub(labels) 

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))

    }
    train() {

        const batchQuantity = Math.floor(this.features.shape[0]/this.options.batchSize)


        for (let i = 0; i < this.options.iterations; i++ ) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j*this.options.batchSize
                const {batchSize} =  this.options
                 //Slice the Feature Tensor into Batch Sizes
                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                )
                //Slice the Label Tensor into Batch Sizes
                const labelSlice = this.labels.slice (
                    [startIndex, 0],
                    [batchSize, -1]
                )    
                this.gradientDescent(featureSlice, labelSlice)
            }    
            this.bHistory.push(this.weights.get(0,0))
            
            this.recordCost()
            this.updateLearnigRate()
        }
    }

    predict(observations) {
        return this.processFeatures(observations)
                    .matMul(this.weights)
                    .sigmoid()
                    .greater(this.options.decisionBoundary)
                    .cast('float32')
    }

    test( testFeatures, testLabels ) {
        const predictions =  this.predict(testFeatures)
        testLabels = tf.tensor(testLabels)
        const incorrect = predictions.sub (testLabels) .abs().sum().get()
        return  (predictions.shape[0] - incorrect)/predictions.shape[0]
    }

    processFeatures(features) {
        features = tf.tensor(features)

        //Generate One Column with 1's with feature set row size. Concat Tensor of One's with Features Tensor
        //Concat horizontaly with Axis 1
        features = tf.ones([features.shape[0],1]).concat(features,1)

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else {
            features = this.standardizeFeatures(features)
        }

        return features
    }
    standardizeFeatures(features) {
        const { mean, variance } = tf.moments(features, 0)
        this.mean=mean
        this.variance=variance
        return features.sub(mean).div(variance.pow(.5))
    }

    recordCost() {
        //Cross Entropy
        const guesses = this.features.matMul(this.weights).sigmoid()

        // -(1/m) * (Actual^T * log(guesses) + (1 - Actual)^T log (1 - guesses) )

        const termOne = this.labels
            .transpose()
            .matMul(guesses.log())
         
         const termTwo =  this.labels
            .mul(-1)
            .add(1)   
            .transpose()
            .matMul(
                guesses
                    .mul(-1)
                    .add(1)
                    .log()
            );
          const cost = termOne.add(termTwo) 
                .div(this.features.shape[0])
                .mul(-1)
                .get(0,0)
            this.costHistory.unshift(cost)
    }

    //
    updateLearnigRate() {
        if (this.costHistory.length <  2 ) {
            return
        }
        // if MSE Went *UP*, decrease Learning Rate Or if MSW Went *DOWN*, Increase LR by x%
       if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate/= 2
       }  else {
           this.options.learningRate *= 1.05
       }
    }
}

module.exports = LogisticRegression;