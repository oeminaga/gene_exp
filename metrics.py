import keras.backend as K
import tensorflow as tf_X
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

# Metrics
def IOU(self, y_true, y_pred):
    score, up_opt = tf_X.metrics.mean_iou(y_true, y_pred, self.nb_class)
    K.get_session().run(tf_X.local_variables_initializer())
    with tf_X.control_dependencies([up_opt]):
        score = tf_X.identity(score)
        return score

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def count_possitve_true(y_true, y_pred):
    y_negative = K.cast(y_true == 0, 'float32')
    y_positive = K.cast(y_true == 1, 'float32')

    N_negative = K.sum(y_negative)
    N_positive = K.sum(y_positive)
    y_true_f = N_positive / (N_negative + N_positive)
    return y_true_f

def loss_sigmoid_cross_entropy_with_logits(y_true, y_pred):
    y_true_ff = K.flatten(y_true)
    y_pred_ff = K.flatten(y_pred)
    return tf_X.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf_X.metrics.auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf_X.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf_X.add_to_collection(tf_X.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf_X.control_dependencies([update_op]):
        value = tf_X.identity(value)
        return value

def error_rate(y_true, y_pred):
    if y_true is None:
        return 0
    if y_pred is None:
        return 0

    score_neg, up_opt_neg = tf_X.metrics.false_negatives(y_true, y_pred)
    score_pos, up_opt_pos = tf_X.metrics.false_positives(y_true, y_pred)
    K.get_session().run(tf_X.local_variables_initializer())
    batch_size = K.int_shape(y_true)[0]
    fn = 0
    fp = 0
    with tf_X.control_dependencies([up_opt_neg]):
        fn = tf_X.identity(score_neg)

    with tf_X.control_dependencies([up_opt_pos]):
        fp = tf_X.identity(score_pos)

    total_f = (fp + fn) / batch_size
    return total_f

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P

def auc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ptas = K.stack([self.binary_PTA(y_true_f, y_pred_f, k) for k in np.linspace(0, 1, 2)], axis=0)
    pfas = K.stack([self.binary_PFA(y_true_f, y_pred_f, k) for k in np.linspace(0, 1, 2)], axis=0)
    pfas = K.concatenate([K.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """

    if y_pred is None:
        return 0

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    if y_true is None:
        return 0

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision