import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Input,AveragePooling2D, Flatten, Dropout

tf.enable_eager_execution()


# =====================
#    Define the layer
# =====================

# ---- Conv + Batch normalize ----

def Conv2D_BN(input_layer, filters, kernel_size, strides, padding, activation=None):
    x = Conv2D(filters, kernel_size, strides, padding)(input_layer)
    x = BatchNormalization(axis=-1)(x)
    return x

# ---- Inception structure ----
def Inception(input_layer, IV_filter):
    (branch1, branch2, branch3, branch4) = IV_filter

    # 1*1
    branch1x1 = Conv2D(branch1[0], (1, 1), strides=(1, 1), padding='same')(input_layer)

    # 1*1 --> 3*3
    branch3x3 = Conv2D(branch2[0], (1, 1), strides=(1, 1), padding='same')(input_layer)
    branch3x3 = Conv2D(branch2[1], (3, 3), strides=(1, 1), padding='same')(branch3x3)

    # 1*1 --> 3*3 --> 3*3
    branch3x3_db = Conv2D(branch3[0], (1, 1), strides=(1, 1), padding='same')(input_layer)
    branch3x3_db = Conv2D(branch3[1], (5, 5), strides=(1, 1), padding='same')(branch3x3_db)
    branch3x3_db = Conv2D(branch3[1], (5, 5), strides=(1, 1), padding='same')(branch3x3_db)

    # Max pooling --> 1*1
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch_pool = Conv2D(branch4[0], (1, 1), strides=(1, 1), padding='same')(branch_pool)

    return concatenate([branch1x1, branch3x3, branch3x3_db, branch_pool], axis=-1)


# =====================
#    Build the model
# =====================

def Googlenet(width, height, depth, classes):
    with tf.name_scope("Input_layer"):
        inputs = Input(shape=(width, height, depth))

    with tf.name_scope("Conv_Mp_1"):
        x = Conv2D_BN(inputs, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    with tf.name_scope("Conv_Mp_2"):
        x = Conv2D_BN(x, 128, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    with tf.name_scope("Inception_layer_1"):
        x = Inception(x, [(64,), (64, 96), (32, 64), (32,)])
        x = Inception(x, [(64,), (64, 128), (16, 32), (32,)])
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Inception(x, [(192,), (96, 208), (16, 48), (64,)])

    with tf.name_scope("FCN"):
        x = Dense(1000, activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model

Googlenet_model = Googlenet(100,100,3,10)
Googlenet_model.summary()