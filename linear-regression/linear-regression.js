const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {

        /* Lodash Implementation ~ Tensor Implementation with Tensor Weights
         *  this.m = 0
         *  this.b = 0
         * 
         */


        this.features = this.processFeatures(features)
        this.labels= tf.tensor(labels)
        this.mseHistory =[]
        this.bHistory = []
       
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options)

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
        const currentGuesses = features.matMul(this.weights) 
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
            
            this.recordMSE()
            this.updateLearnigRate()
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights)
    }

    test( testFeatures, testLabels ) {
        testFeatures  = this.processFeatures(testFeatures)
        testLabels = tf.tensor(testLabels)
        
        const predictions = testFeatures.matMul(this.weights)

        //Sum Of Squared of Residuals n = 1..n ∑(Label - Predicted)^2
        const res = testLabels.sub(predictions).pow(2).sum().get()
        //Total Sum of Squares n = 1..n ∑(Label - Average)^2
        const tot = testLabels.sub( testLabels.mean()).pow(2).sum().get()
        return 1 - (res/tot)
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

    recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get()

            this.mseHistory.unshift(mse)
    }

    //
    updateLearnigRate() {
        if (this.mseHistory.length <  2 ) {
            return
        }
        // if MSE Went *UP*, decrease Learning Rate Or if MSW Went *DOWN*, Increase LR by x%
       if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate/= 2
       }  else {
           this.options.learningRate *= 1.05
       }
    }
}

module.exports = LinearRegression;