import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers, optimizers

import numpy as np

from absl import flags
from absl import app

from models import NP
from data import MNISTDataGen

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 64, "Batch size for training steps")
flags.DEFINE_integer("iter", int(2e5), "Number of iterations for training")
flags.DEFINE_integer("x_dims", 1, "Dimensionality of the feature vector")
flags.DEFINE_integer("y_dims", 1, "Dimensionality of the output vector")
flags.DEFINE_integer("max_num_context", 500, "Biggest context set")
flags.DEFINE_integer("latent_dim", 128, "The size of latent dimension")

flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_float("amplitude", 1.0, "Amplitude of the Square-Exp GP kernel")
flags.DEFINE_float("length_scale", 0.4,
                   "Length scale of the Square-Exp GP kernel")

flags.DEFINE_list("latent_encoder_arch", [
                  128, 128, 128], "Encoder architecture")
flags.DEFINE_list("deterministic_encoder_arch", None,
                  "Deterministic encoder architecture")
flags.DEFINE_list(
    "decoder_arch", [128, 128, 128, 128, 2], "Decoder architecture")

flags.DEFINE_string("from_weights", None, "Weights file to initialize from")

flags.DEFINE_string("weights_file", None, "File to save the model weights in")
flags.DEFINE_string("flags_file", None, "File to save the flags in")


def train_one_step(model, optimizer, batch):
    (context_x, context_y), (target_x, target_y) = batch

    with tf.GradientTape() as tape:
        mu, sigma, _, _, loss = model(
            context_x, context_y, target_x, target_y=target_y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


@tf.function
def train(model, optimizer, iterations, dataset_train, dataset_test):
    for it in range(iterations):
        batch = dataset_train.make_batch()
        loss = train_one_step(model, optimizer, batch)
        if it % 100 == 0:
            (context_x, context_y), (target_x, target_y) = dataset_test.make_batch()

            mu, sigma, _, _, loss = model(
                context_x, context_y, target_x, target_y=target_y)
            tf.print("Iteration", it, "loss", loss)
            #plot_functions(target_x, target_y, context_x, context_y, mu, sigma)


def main(argv):
    del argv
    iterations = tf.constant(FLAGS.iter)

    dataset_train = MNISTDataGen(FLAGS.batch_size, FLAGS.max_num_context)
    dataset_test = MNISTDataGen(
        FLAGS.batch_size, FLAGS.max_num_context, testing=True)

    latent_encoder_sizes = [int(x) for x in FLAGS.latent_encoder_arch]

    if FLAGS.deterministic_encoder_arch is None:
        deterministic_encoder_sizes = None
    else:
        deterministic_encoder_sizes = [
            int(x) for x in FLAGS.deterministic_encoder_arch]

    decoder_sizes = [int(x) for x in FLAGS.decoder_arch]

    assert(decoder_sizes[-1] == 2 * FLAGS.y_dims)

    model = NP(FLAGS.latent_dim, latent_encoder_sizes, FLAGS.decoder_arch,
               deterministic_encoder_sizes=deterministic_encoder_sizes)

    if FLAGS.from_weights is not None:
        (context_x, context_y), (target_x, target_y) = dataset_train.make_batch()
        _ = model(context_x, context_y, target_x)

        model.load_weights(FLAGS.from_weights)

    optimizer = optimizers.Adam(FLAGS.lr)

    train(model, optimizer, iterations, dataset_train, dataset_test)

    if FLAGS.weights_file is not None:
        model.save_weights(FLAGS.weights_file)

    if FLAGS.flags_file is not None:
        FLAGS.append_flags_into_file(FLAGS.flags_file)


if __name__ == "__main__":
    app.run(main)
