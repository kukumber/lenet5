#!/usr/bin/python3

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Load trainig and eval data

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""

    # Input layer
    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    # Convolutional Layer #1
    # Input 28x28x1
    # Output 28x28x6
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.tanh)

    # Pooling Layer #1
    # Output 14x14x6
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolutional Layer #2
    # Output 10x10x16
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[5,5],
        padding='valid',
        activation=tf.nn.tanh)

    # Pooling Layer #2
    # Output 5x5x16
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # Dense layers
    pool2_flat = tf.reshape(pool2, [-1, 16 * 5 * 5])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=120,
                             activation=tf.nn.tanh)
    dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.tanh)

    # Logits layer
    logits = tf.layers.dense(inputs=dense2, units=10)
    
    predictions = {
        # Generate predictions (for both TRAIN and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),

        # Add 'softmax-tensor to the graph. It is used for PREDICT and by
        # 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def run_model():

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                          model_dir='/tmp/lenet5')

    # Setup a logging hook for predictions
    tensor_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log,
                                          every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True)

    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 100000,
        hooks = [logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

if __name__ == "__main__":
    run_model()
