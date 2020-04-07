import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers, optimizers

import numpy as np

from absl import flags
from absl import app

from models import NP
from data import GPDataGen, GPThompsonSampling

FLAGS = flags.FLAGS

flags.DEFINE_integer("x_dims", 1, "Dimensionality of the feature vector")
flags.DEFINE_integer("y_dims", 1, "Dimensionality of the output vector")


flags.DEFINE_float("amplitude", 1.0, "Amplitude of the Square-Exp GP kernel")
flags.DEFINE_float("length_scale", 0.4,
                   "Length scale of the Square-Exp GP kernel")
flags.DEFINE_integer("x_values_per_function", 401, "How finely we sample the function")

flags.DEFINE_integer("latent_dim", 128, "The size of latent dimension")
flags.DEFINE_list("latent_encoder_arch", [
                  128, 128, 128, 128], "Encoder architecture")
flags.DEFINE_float("deterministic_encoder_arch", None, "Deterministic encoder architecture")
flags.DEFINE_list("decoder_arch", [128, 128, 2], "Decoder architecture")

flags.DEFINE_integer("max_iterations", 20, "Maximum number of iterations allowed to find minimum")
flags.DEFINE_float("tolerance", 0.05, "Allowed tolerance for final y value of minimum")
flags.DEFINE_integer("number_functions", 1, "Number of functions for which minimum is found")




def thompson_sample(model, function, minimum):
    tf.random.set_seed(1234)
    checked_indices = tf.Variable(False,dtype=bool,shape=(FLAGS.x_values_per_function))
    
    index = tf.random.uniform(shape=(),minval=0, maxval=FLAGS.x_values_per_function, dtype=tf.int64)
    
    delta = tf.SparseTensor([index], [tf.Constant(True)],shape=(FLAGS.x_values_per_function))
    
    checked_indices = tf.math.logical_and(checked_indices,delta)
                  
    x_values , y_values = function
    x_min , y_min = minimum
    context_x = x_values[0,index,:]
    context_y = y_values[0,index,:]
    context_x = tf.expand_dims(context_x, 0)
    context_x = tf.expand_dims(context_x, 2)
    context_y = tf.expand_dims(context_y, 0)
    context_y = tf.expand_dims(context_y, 2)
    
    print(checked_indices)
    print('xmin',x_min)
    print('ymin',y_min)
    for it in range(2,FLAGS.max_iterations+1):
        mu, _, _, _, _ = model(context_x, context_y, x_values)
        
        index_minvalue = tf.argmin(mu, 1)[0,0]
        
        # check if value in tensor
        #if index_minvalue 
        
        
        new_x = x_values[0,index_minvalue,:]
        new_y = y_values[0,index_minvalue,:]

        context_x = tf.concat([context_x,tf.expand_dims(tf.expand_dims(new_x,0),2)],1)
        context_y = tf.concat([context_y,tf.expand_dims(tf.expand_dims(new_y,0),2)],1)
        indices = tf.concat([indices, tf.expand_dims(index_minvalue,0)],0)
        print('new_x','new_y')
        print(new_x,new_y)
        print('tf.abs(new_x - x_min)',tf.abs(new_x - x_min))
        if tf.abs(new_x - x_min) < FLAGS.tolerance:
            break;
    print(context_x)
    print(context_y)
    print('indices',indices)
    return it

def random_sample(function, minimum):
    pass


def load_model():

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=FLAGS.amplitude, length_scale=FLAGS.length_scale)
    
    latent_encoder_sizes = [int(x) for x in FLAGS.latent_encoder_arch]
    if FLAGS.deterministic_encoder_arch is None:
        deterministic_encoder_sizes = None
    else:
        deterministic_encoder_sizes = [int(x)
                                   for x in FLAGS.deterministic_encoder_arch]
    decoder_sizes = [int(x) for x in FLAGS.decoder_arch]

    assert(decoder_sizes[-1] == 2 * FLAGS.y_dims)

    model = NP(FLAGS.latent_dim, latent_encoder_sizes, FLAGS.decoder_arch,
               deterministic_encoder_sizes=deterministic_encoder_sizes)
    
    dataset_test = GPDataGen(1, 10, kernel, testing=True)
    
    (context_x, context_y), (target_x, target_y) = dataset_test.make_batch()
    mu, sigma, _, _, loss = model(context_x, context_y, target_x, target_y=target_y)
    
    model.load_weights('regression_weights.h5')
    
    return model
    

def main(argv):
                  
    del argv

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=FLAGS.amplitude, length_scale=FLAGS.length_scale)

    model = load_model()
    GP = GPThompsonSampling(400,kernel)
    
    #Store the number of function evaluations needed for all tested function
    #using thompson sampling and random sampling
    
    evaluations = tf.zeros([FLAGS.number_functions,2],tf.float32)
    for i in range(FLAGS.number_functions):
        (x_values, y_values),(x_min, y_min) = GP.draw_function()       
        evaluations[i,0] = thompson_sample(model, (x_values,y_values), (x_min,y_min))
        #evaluations[i,1] = random_sample((x_values,y_values), (x_min,y_min))
    
    
    print(evaluations)


if __name__ == "__main__":
    app.run(main)
