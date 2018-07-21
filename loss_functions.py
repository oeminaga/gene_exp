
import keras
import keras.backend as K

def loss_function(self, y_true, y_predict):
    reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_predict)
    reconstruction_loss *= self.original_dim
    kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)