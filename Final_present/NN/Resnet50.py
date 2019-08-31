from keras.layers import add, Dense, Conv2D, MaxPooling2D, BatchNormalization, \
                         GlobalAveragePooling2D, Input, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load oxflower17 dataset
# import tflearn.datasets.oxflower17 as oxflower17
# from sklearn.model_selection import train_test_split

# x, y = oxflower17.load_data(one_hot=True)

# Split train and test data
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Data augumentation with Keras tools
img_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# Define convolution with batch normalization
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=-1, name=bn_name)(x)
    return x


# Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(input_model, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    filter1, filter2, filter3 = nb_filter

    x = Conv2d_BN(input_model, nb_filter=filter1, kernel_size=(1, 1), strides=strides, padding='same')
    x = Activation('relu')(x)
    x = Conv2d_BN(x, nb_filter=filter2, kernel_size=kernel_size, strides=strides, padding='same')
    x = Activation('relu')(x)
    x = Conv2d_BN(x, nb_filter=filter3, kernel_size=(1, 1), strides=strides, padding='same')

    # need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model, nb_filter=filter3, kernel_size=(1, 1), strides=strides, padding='same')
        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x
    else:
        x = add([x, input_model])
        x = Activation('relu')(x)
        return x


# Built ResNet34
def ResNet50(width, height, depth, classes):
    Img = Input(shape=(width, height, depth))

    # First stage conv1_x
    x = Conv2d_BN(Img, 64, (7, 7), strides=(2, 2), padding='same')
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Residual conv2_x ouput
    x = Residual_Block(x, nb_filter=(64, 64, 256), kernel_size=(3, 3), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=(64, 64, 256), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(64, 64, 256), kernel_size=(3, 3))

    # Residual conv3_x ouput
    x = Residual_Block(x, nb_filter=(128, 128, 512), kernel_size=(3, 3), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=(128, 128, 512), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(128, 128, 512), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(128, 128, 512), kernel_size=(3, 3))

    # Residual conv4_x ouput
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(256, 256, 1024), kernel_size=(3, 3))

    # Residual conv5_x ouput
    x = Residual_Block(x, nb_filter=(512, 512, 2048), kernel_size=(3, 3), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=(512, 512, 2048), kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=(512, 512, 2048), kernel_size=(3, 3))

    # Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input=Img, output=x)
    return model


# ResNet50_model = ResNet50(512, 512, 1, 2)
# ResNet50_model.summary()

# ResNet50_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#                        loss='categorical_crossentropy',
#                        metrics=['accuracy'])
#
# ResNet50_model.fit_generator(img_gen.flow(X_train * 255, y_train, batch_size=16),
#                              steps_per_epoch=len(X_train) / 16,
#                              validation_data=(X_test, y_test),
#                              epochs=30)
#
# ResNet50_model.save('model/Resnet_50.h5')