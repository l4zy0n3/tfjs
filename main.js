import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({
    units: 8,
    inputShape: [2]
}));
model.add(tf.layers.dense({
    units: 1,
    inputShape: [8]
}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1, 0],
                        [0, 1],
                        [0, 0],
                        [1, 1]], [4, 2]);
const ys = tf.tensor2d([1, 1, 0, 1], [4, 1]);

// Train the model using the data.
model.fit(xs, ys).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([0, 0], [1, 2])).print();
});