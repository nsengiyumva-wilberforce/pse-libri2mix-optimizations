import tensorflow as tf

import keras

from keras import layers

import numpy as np

import time
tf.profiler.experimental.server.start(6009)

# sample model
tf.profiler.experimental.start('logdir')


inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instance optimizer
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

# instantiate a loss function
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# prepare the metrics
train_acc_metric = keras.metrics.SparseCategoricalCrossentropy()
val_acc_metric = keras.metrics.SparseCategoricalCrossentropy()

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10, 000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # run the forward pass of the layer.
        # the operations that the layer applies
        # to its inputs are going to be recorded
        # on the the GradientTaper.
        logits = model(x, training=True) # Logits for this minibatch

        # compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)

    # use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss,
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # updating training metric.
    train_acc_metric.update_state(y_batch_train, logits)

    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x_batch_val, training=False)

        # update val metrics
    val_acc_metric.update_state(y_batch_val, val_logits)




epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" %(epoch,))
    start_time = time.time()


    # iterate over batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            # print("\nstating step %d" %(step, ))
            # open GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            loss_value = train_step(x_batch_train, y_batch_train)
            # log every 200 batches
            if step % 200 == 0:
                print(
                    "training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc), ))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_state()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print("Validation acc: %.4f" % (float(val_acc), ))
    print("Time taken: %.2fs"% (time.time() - start_time))
    tf.profiler.experimental.client.trace('localhost:6009',
                                      'logdir', 2000)

    tf.profiler.experimental.stop()
