import tensorflow as tf
import os
import gzip
import numpy as np
import _pickle as cPickle
import random


def load_data():
    datafile = os.getenv('DATA', '/home/anton/Programming/ml/preserve/datasets') + '/MNIST/mnist.pkl.gz'
    f = gzip.open(datafile, 'rb')
    tr_d, va_d, te_d = cPickle.load(f)
    f.close()
    training_inputs = [np.reshape(x, (784,)) for x in tr_d[0]]
    training_results = [to_one_hot(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784,)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784,)) for x in te_d[0]]
    test_results = [to_one_hot(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return list(training_data), list(validation_data), list(test_data)


def to_one_hot(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e


def evaluate(session, layer_out, tests):
    input_data, output_given = zip(*[(inp, out) for inp, out in tests])
    output_guess = session.run(layer_out, feed_dict={input_layer: input_data, labels_given: output_given})
    output_given_int = np.argmax(output_given, axis=1)
    output_guess_int = np.argmax(output_guess, axis=1)

    score = sum([given == guess for given, guess in zip(output_given_int, output_guess_int)])

    return score


precision_type = tf.float32
layer_sizes = [784, 200, 10]
epochs = 100
batch_size = 20

weights = []
biases = []
layers = []

input_layer = tf.placeholder(precision_type, shape=(None, layer_sizes[0]))
layers.append(input_layer)

layers_params = [(before, after) for before, after in zip(layer_sizes[:-1], layer_sizes[1:])]
for idx, layer_params in enumerate(layers_params):
    weights.append(tf.Variable(
        initial_value=tf.random_normal(shape=layer_params, stddev=0.1),
        dtype=precision_type,
        trainable=True)
    )
    biases.append(tf.Variable(
        initial_value=tf.random_normal(shape=(1, layer_params[1]), stddev=0.1),
        dtype=precision_type,
        trainable=True)
    )

    if idx is len(layers_params)-1:
        layers.append(tf.matmul(layers[-1], weights[-1]) + biases[-1])  # for the last layer, we don't normalize
    else:
        layers.append(tf.sigmoid(tf.matmul(layers[-1], weights[-1]) + biases[-1]))

labels_given = tf.placeholder(dtype=precision_type, shape=(None, layer_sizes[-1]))
labels_guess = layers[-1]

costf = tf.nn.softmax_cross_entropy_with_logits(logits=labels_guess, labels=labels_given)

trainer = tf.train.GradientDescentOptimizer(0.5).minimize(costf)

training_data, _, test_data = load_data()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        random.shuffle(training_data)
        batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
        for batch in batches:
            input_data, output_given = zip(*[(inp, out) for inp, out in batch])
            session.run(trainer, feed_dict={input_layer: input_data, labels_given: output_given})
        print('##################')
        print('Epoch {epoch_num}'.format(epoch_num=epoch))
        print('Score = {score} / {score_max}'
              .format(score=evaluate(session, layers[-1], test_data), score_max=len(test_data)))
        print('')

