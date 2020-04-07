import tensorflow as tf
from tensorflow_probability import distributions
import numpy as np

from tensorflow.keras import layers


class Aggregator(tf.keras.Model):
    def __init__(self):
        super(Aggregator, self).__init__()

    def call(self, representations):
        return tf.reduce_mean(representations, axis=1)


class Encoder(tf.keras.Model):
    def __init__(self, sizes):
        super(Encoder, self).__init__()

        self._layers = [layers.Dense(size, activation='relu')
                        for size in sizes[:-1]]

        self._layers.append(layers.Dense(sizes[-1]))

    def call(self, x, y):
        out = tf.concat([x, y], axis=-1)

        for i in range(len(self._layers)):
            out = self._layers[i](out)

        return out


class Decoder(tf.keras.Model):
    def __init__(self, sizes):
        super(Decoder, self).__init__()

        self._layers = [layers.Dense(size, activation='relu')
                        for size in sizes[:-1]]
        self._layers.append(layers.Dense(sizes[-1]))

    def call(self, representation, x):
        representation = tf.tile(tf.expand_dims(
            representation, 1), [1, tf.shape(x)[1], 1])
        out = tf.concat([representation, x], axis=-1)

        for i in range(len(self._layers)):
            out = self._layers[i](out)

        mu, log_sigma = tf.split(out, 2, axis=-1)

        sigma = 0.1 + 0.9 * tf.math.softplus(log_sigma)

        return mu, sigma


class NP(tf.keras.Model):
    def __init__(self, latent_dim, latent_encoder_sizes, decoder_sizes, deterministic_encoder_sizes=None):
        super(NP, self).__init__()

        if deterministic_encoder_sizes is None:
            self.deterministic_encoder = None
        else:
            self.deterministic_encoder = Encoder(deterministic_encoder_sizes)

        self.latent_encoder = Encoder(latent_encoder_sizes)
        self.decoder = Decoder(decoder_sizes)

        self.latent_dim = latent_dim

        self.latent_hidden = layers.Dense(latent_dim, activation='relu')

        self.latent_mu = layers.Dense(latent_dim)
        self.latent_log_sigma = layers.Dense(latent_dim)

        self.aggregator = Aggregator()

    def latent_dist(self, x_values, y_values):
        latent_reps = self.latent_encoder(x_values, y_values)
        latent_rep = self.aggregator(latent_reps)

        hidden = self.latent_hidden(latent_rep)

        latent_mu = self.latent_mu(hidden)
        latent_log_sigma = self.latent_log_sigma(hidden)

        latent_sigma = 0.1 + 0.9 * tf.sigmoid(latent_log_sigma)

        return distributions.Normal(
            loc=latent_mu, scale=latent_sigma)

    def call(self, context_x, context_y, target_x, target_y=None):

        q_context = self.latent_dist(context_x, context_y)

        if target_y is not None:
            q_target = self.latent_dist(target_x, target_y)
            z = q_target.sample()
        else:
            z = q_context.sample()

        if self.deterministic_encoder is not None:
            deterministic_reps = self.deterministic_encoder(
                context_x, context_y)
            deterministic_rep = self.aggregator(deterministic_reps)

            representation = tf.concat(
                [deterministic_rep, z], axis=-1)
        else:
            representation = z

        mu, sigma = self.decoder(representation, target_x)
        dist = distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=sigma)

        if target_y is not None:
            log_p = dist.log_prob(target_y)

            kl = tf.reduce_sum(distributions.kl_divergence(
                q_target, q_context), axis=-1, keepdims=True)

            num_targets = tf.shape(target_x)[1]

            kl = tf.tile(kl, [1, num_targets])

            loss = - tf.reduce_mean(log_p - kl /
                                    tf.cast(num_targets, tf.float32))
        else:
            kl = None
            log_p = None
            loss = None
            loss_ = None

        return mu, sigma, kl, log_p, loss
