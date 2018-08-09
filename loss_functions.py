import keras
import keras.backend as K
import metrics
import tensorflow as tf_X

z_mean = 0
z_log_var = 0
original_dim = (1,1)
def loss_binary_crossentropy(y_true, y_predict):
    reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_predict)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)

def dice_coef_loss(y_true, y_pred):
    return -metrics.dice_coef(y_true, y_pred)

def iou_loss(y_true, y_pred):
    '''
    Eq. (1) The intersection part - tf_X.mul is element-wise,
    if logits were also binary then tf_X.reduce_sum would be like a bitcount here.
    '''
    y_true_tmp = K.flatten(y_true)
    y_pred_tmp = K.flatten(y_pred)
    inter = tf_X.reduce_sum(tf_X.multiply(y_pred_tmp, y_true_tmp))

    '''
    Eq. (2) The union part - element-wise sum and multiplication, then vector sum
    '''
    union = tf_X.reduce_sum(tf_X.subtract(tf_X.add(y_pred_tmp, y_true_tmp), tf_X.multiply(y_pred_tmp, y_true_tmp)))

    # Eq. (4)
    loss = tf_X.subtract(tf_X.constant(1.0, dtype=tf_X.float32), tf_X.div(inter, union))
    return loss

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
