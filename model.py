from keras import layers
import keras
import keras_preprocessing
import keras.backend as K


def plot_model(encoder, to_file, show_shapes):
    pass

class GeneExpressionLevel():
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def __int__(self, directory_image, directory_mask, input_shape, result_directory, batch_size, epoch):
        self.directory_image = directory_image
        self.directory_mask = directory_mask
        self.input_shape = input_shape
        self.result_directory = result_directory
        self.batch_size = batch_size
        self.epoch = epoch
        self.original_dim = input_shape[0] * input_shape[1] * input_shape[2]

    def train(self):
        x = layers.Input(shape=self.input_shape, name='encoder_input')
        self.CoreModel(inputs=x)

    def CoreModel(self, inputs=None):
        intermediate_dim = 512
        latent_dim = 2

        # VAE model = encoder + decoder
        # build encoder model
        x = layers.Dense(intermediate_dim, activation='relu')(inputs)
        self.z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

        h = layers.Dense(intermediate_dim, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(h)
        self.z_log_sigma = layers.Dense(latent_dim)(h)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = keras.Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(self.input_shape, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = keras.Model(inputs, outputs, name='vae_mlp')
        return vae

