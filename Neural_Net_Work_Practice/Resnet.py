import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import add,Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,GlobalAveragePooling2D,concatenate,Input, concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam

# Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

# Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

# Define convolution with batch normalization
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x


# Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(input_model, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(input_model, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

    # need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, input_model])
        return x


# Built ResNet34
def ResNet34(width, height, depth, classes):
    Img = Input(shape=(width, height, depth))

    x = Conv2d_BN(Img, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Residual conv2_x ouput 56x56x64
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))

    # Residual conv3_x ouput 28x28x128
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2),
                       with_conv_shortcut=True)  # need do convolution to add different channel
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))

    # Residual conv4_x ouput 14x14x256
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2),
                       with_conv_shortcut=True)  # need do convolution to add different channel
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))

    # Residual conv5_x ouput 7x7x512
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3))

    # Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input=Img, output=x)
    return model

ResNet34_model = ResNet34(224,224,3,17)
ResNet34_model.summary()

ResNet34_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
ResNet34_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

# Save the model
ResNet34_model.save('model/Resnet_34.h5')

# evaluate the model
scores = ResNet34_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (ResNet34_model.metrics_names[1], scores[1]*100))