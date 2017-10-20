from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


CNN = Sequential()


CNN.add(Convolution2D(32, kernel_size=(7,7),input_shape=(128, 128 ,3), activation='relu'))
# should u consider making the pooling overlapping like in ConvNet, kernel 3x3, step-size=2
CNN.add(MaxPooling2D(pool_size=(2,2)))

# 2nd layer
CNN.add(Convolution2D(32, kernel_size=(7,7), activation='relu'))
# should u consider making the pooling overlapping like in ConvNet, kernel 3x3, step-size=2
# maybe try getting rid of this layer bc with it you only put around 8 units into the network...without it around 16 which is more info
CNN.add(MaxPooling2D(pool_size=(2,2) ))

#3rd layer
CNN.add(Convolution2D(32, kernel_size=(7,7), activation='relu'))
# should u consider making the pooling overlapping like in ConvNet, kernel 3x3, step-size=2
CNN.add(MaxPooling2D(pool_size=(2,2) ))

#4th layer
#CNN.add(Convolution2D(32, kernel_size=(3,3), activation='relu'))
# should u consider making the pooling overlapping like in ConvNet, kernel 3x3, step-size=2
#CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Flatten())
#First fully connected layer
CNN.add(Dense(units = 128, activation='relu'))
#Second fully connected layer
CNN.add(Dense(units = 128, activation='relu'))

CNN.add(Dense(units = 1, activation='sigmoid'))

CNN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=128,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=128,
        class_mode='binary')

CNN.fit_generator(
        train_datagen,
        steps_per_epoch=8000//128,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=2000//128)
