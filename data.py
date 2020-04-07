import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets


class BanditDataGen():
    def __init__(self, num_episodes, num_context, num_target, batch_size=16, delta_val=None):
        self.num_episodes = num_episodes
        self.num_context = num_context
        self.num_target = num_target
        self.delta_val = delta_val
        self.num_points = num_context + num_target

        if delta_val is None:
            self.deltas = np.random.rand(num_episodes)
        else:
            self.deltas = [delta_val] * num_episodes

        radii = np.random.rand(num_episodes, self.num_points)
        angles = 2 * np.pi * \
            np.random.rand(num_episodes, self.num_points)

        X1 = radii * np.cos(angles)
        X2 = radii * np.sin(angles)
        X = np.concatenate([X1[:, :, None], X2[:, :, None]], axis=-1)

        actions = np.random.randint(5, size=(num_episodes, self.num_points))
        rewards = np.zeros((num_episodes, self.num_points))
        rewards -= 1

        optimal_actions = np.zeros(
            (num_episodes, self.num_points), dtype=np.int64)

        q1 = angles <= np.pi / 2
        q2 = (angles > np.pi / 2) & (angles <= np.pi)
        q3 = (angles > np.pi) & (angles <= 3 * np.pi / 2)
        q4 = angles > 3 * np.pi / 2

        outside = radii > self.deltas[:, None]

        optimal_actions[outside & q1] = 1
        optimal_actions[outside & q2] = 2
        optimal_actions[outside & q3] = 3
        optimal_actions[outside & q4] = 4

        optimal = optimal_actions == actions

        scale = 0.01

        def set_rewards(mask, loc):
            rewards[mask] = np.random.normal(
                loc=loc, scale=scale, size=np.sum(mask))

        set_rewards((~outside & optimal), 1.2)
        set_rewards((~outside & ~optimal), 1)
        set_rewards((outside & optimal), 50)
        set_rewards((outside & ~optimal & actions > 0), 1)
        set_rewards((outside & actions == 0), 1.2)

        self.X = X
        self.actions = actions
        self.rewards = rewards


class ImageDataGen():
    def __init__(self, batch_size, max_num_context, shape, X):
        self.X = tf.constant(X)
        self.shape = shape

        self.x_values = np.indices(shape).transpose(
            [1, 2, 0]).reshape((-1, 2)).astype("float32")
        self.x_values = np.tile(self.x_values, (batch_size, 1, 1))
        self.x_values = tf.constant(self.x_values)
        self.x_values = tf.constant(self.x_values)

        self.batch_size = batch_size
        self.max_num_context = max_num_context

    def make_batch(self):
        img_idxs = tf.random.uniform(
            (self.batch_size,), minval=0, maxval=self.X.shape[0], dtype=tf.int32)

        y_values = tf.gather(self.X, img_idxs, axis=0)

        num_context = tf.random.uniform(
            (), minval=3, maxval=self.max_num_context, dtype=tf.int32)

        idxs = tf.random.shuffle(np.arange(self.shape[0] * self.shape[1]))

        context_x = tf.gather(self.x_values, idxs[:num_context], axis=1)
        context_y = tf.gather(y_values, idxs[:num_context], axis=1)

        return (context_x, context_y), (self.x_values, y_values)


class ImageTopRowsGen(ImageDataGen):
    def __init__(self, batch_size, rows, shape, X):
        super(ImageTopRowsGen, self).__init__(
            batch_size, rows * shape[1], shape, X)

        self.rows = rows
        self.idxs = np.arange(self.max_num_context)

    def make_batch(self):
        img_idxs = tf.random.uniform(
            (self.batch_size,), minval=0, maxval=self.X.shape[0], dtype=tf.int32)

        y_values = tf.gather(self.X, img_idxs, axis=0)

        context_x = tf.gather(self.x_values, self.idxs, axis=1)
        context_y = tf.gather(y_values, idxs[:num_context], axis=1)

        return (context_x, context_y), (self.x_values, y_values)


class MNISTDataGen(ImageDataGen):
    def __init__(self, batch_size, max_num_context, testing=False):
        (X_train, _), (X_test, _) = datasets.mnist.load_data()

        if testing:
            X = X_test
        else:
            X = X_train

        X = X.astype("float32").reshape(-1, 28 * 28, 1) / 255.

        super(MNISTDataGen, self).__init__(
            batch_size, max_num_context, (28, 28), X)


class GPDataGen():

    def __init__(
            self,
            batch_size,
            max_num_context,
            kernel,
            x_size=1,
            y_size=1,
            testing=False):

        self.max_num_context = max_num_context
        self.batch_size = batch_size
        self.kernel = kernel
        self.x_size = x_size
        self.y_size = y_size
        self.testing = testing

    def make_batch(self):
        num_context = tf.random.uniform(
            (), minval=3, maxval=self.max_num_context, dtype=tf.int32)

        context_x = tf.random.uniform(
            (self.batch_size, num_context, self.x_size), -2, 2)

        if not self.testing:
            num_target = tf.random.uniform(
                (), minval=2, maxval=self.max_num_context, dtype=tf.int32)

            target_x = tf.random.uniform(
                (self.batch_size, num_target, self.x_size), -2, 2)
        else:
            num_target = 400

            target_x = tf.range(-2., 2., 0.01, dtype=tf.float32)
            target_x = tf.tile(target_x[None, :], [self.batch_size, 1])
            target_x = target_x[:, :, None]

        num_total_points = num_context + num_target

        x_values = tf.concat([context_x, target_x], axis=1)

        matrix = self.kernel.matrix(
            x_values, x_values) + 1e-4 * tf.eye(num_total_points)

        cholesky = tf.cast(tf.linalg.cholesky(
            tf.cast(matrix, tf.float64)), tf.float32)
        y_values = tf.matmul(
            tf.tile(tf.expand_dims(cholesky, 1), (1, self.y_size, 1, 1)),
            tf.random.normal((self.batch_size, self.y_size, num_total_points, 1)))

        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        context_y = y_values[:, :num_context, :]
        target_y = y_values[:, num_context:num_total_points, :]

        return (context_x, context_y), (target_x, target_y)


class GPThompsonSampling():

    def __init__(
            self,
            number_total_points,
            kernel,
            x_size=1,
            y_size=1,
    ):

        self.number_total_points = number_total_points
        self.kernel = kernel
        self.x_size = x_size
        self.y_size = y_size

    def draw_function(self):
        x_values = tf.range(-2., 2., 4 /
                            self.number_total_points, dtype=tf.float32)
        x_values = tf.expand_dims(x_values, 0)
        x_values = tf.expand_dims(x_values, 2)
        matrix = self.kernel.matrix(
            x_values, x_values) + 1e-4 * tf.eye(self.number_total_points)

        cholesky = tf.cast(tf.linalg.cholesky(
            tf.cast(matrix, tf.float64)), tf.float32)
        y_values = tf.matmul(
            tf.tile(tf.expand_dims(cholesky, 1), (1, self.y_size, 1, 1)),
            tf.random.normal((1, self.y_size, self.number_total_points, 1)))
        
        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])
        print('tf.math.argmin(y_values)',tf.math.argmin(y_values,1))
        print('tf.math.argmin(y_values)[0,0]',tf.math.argmin(y_values,1)[0,0])
        print('-'*40)
        x_min = tf.expand_dims(x_values[:,tf.math.argmin(y_values,1)[0,0],:], 0)
        y_min = tf.expand_dims(tf.reduce_min(y_values), 0)
        print('x_min',x_min)
        print('y_min',y_min)
        #TODO fix this 
        #print('xmin',x_min)
        #print('ymin',y_min)
        return ((x_values, y_values), (x_min, y_min))
