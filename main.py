# Convolutional Neural Networks
import numpy as np

import tensorflow as tf

# for preprocessing data
from keras.preprocessing.image import ImageDataGenerator

# for prediction
from keras.preprocessing import image

def cnn():
    # print(tf.__version__) # must be over 2.0

    # step1-1:
    # preprocessing train dataset
    # apply transform images to avcid over feeding and get more accuracy in the end
    # transform images is zoom in/out, landscape, portrait and so on on an image
    # this is image augmentation. transform images -> not over learn and over train

    # instantiate ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, # rescaling images == feature scaling (normalization)
        shear_range=0.2, # transform images
        zoom_range=0.2, # transform images
        horizontal_flip=True)

    # import to dataset
    train_dataset = train_datagen.flow_from_directory(
        'dataset/training_set', # path
        target_size=(64, 64), # final size -> smaller is fast: (150, 150)  original
        batch_size=32, # regular batch size
        class_mode='binary') # 'cat' or 'dog' -> binary

    # step1 -2:
    # preprocessing test dataset

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # rescale=1. / 255 -> feature scaling. train applied this one, so this test does too.
    # BUT, do not transform images as we want to compare these and train

    test_dataset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64), # same size with train
        batch_size=32, # same size with train
        class_mode='binary')

    # step2
    # Building CNN

    # initializing CNN
    cnn = tf.keras.models.Sequential()

    # step2-1: Convolution
    # adding Conv2D layers each image: need at least 4 args 
    # filters is how many feature detector (32, 512)
    # kernel size is feature detector size NxM matrix
    # activation: we want to use 'Rectifier' because of images
    # input_shape [64, 64, 3] must follow the train/test size, 3 is color, 1 is black and white: the first layer must have this
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=[64, 64, 3]))

    # step2-2: Pooling (Max)
    # tf.keras.layers.MaxPooling2D() creates pooled feature maps: need 2 args at least
    # pool_size is a frame size for feature map
    # strides: steps for frame
    # padding is when a frame moves and capture empty areas, they create fake cells with 0 (valid)
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    # adding second Convolutional layer
    # second layer does not need input_shape[]
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    # step2-3: flattening (make arrays for ANN input)
    # Flatten: create 1d array
    cnn.add(tf.keras.layers.Flatten())

    # step2-4: full connection 
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # step2-5: output 
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # dog or cat (binary -> sigmoid); dog, cat, or fish (multiple -> softmax)
    
    # step2-6: compile CNN for training 
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # step3-1: training dataset
    cnn.fit(x=train_dataset, validation_data=test_dataset, epochs=25)
    # validation_data must have it, otherwise it thinks as y -> error
    # no batch size as already mentioned with train_dataset/test_dataset

    # step4-1: prediction
    # importing an image: at least 2 args
    # Path
    # target_size must be the same as train and test
    pred_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))

    # transform (np format) the image into array for CNN
    pred_img = image.img_to_array(pred_img)
    # we set batch size for training, even this is a single image, we must add extra dimensions for batch
    pred_img = np.expand_dims(pred_img, axis = 0) # axis is where we want to add extra dimensions: axis = 0 is add into the first dimension
    # print(pred_img)
    
    result = cnn.predict(pred_img)
    # which number is for dog? 0=dog? 1=dog?
    train_dataset.class_indices # labeling categorical values

    if(result[0][0]==1): # pred_img[first batch][batch rows where cat_or_dog_1.jpg has]
        print('\nThat\'s a Dog')
    else:
        print('\nThat\'s a Cat')


if __name__ == '__main__':
    cnn()

