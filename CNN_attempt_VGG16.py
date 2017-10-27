from keras import applications
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

batch_size1 = 10
np.set_printoptions(threshold=np.nan)
def _setup_the_model_():
    datagen = ImageDataGenerator(rescale=1./255)
    model = applications.VGG16(include_top=False, weights='imagenet')
    #training_generator = datagen.flow_from_directory(directory='dataset - Copy\\training_set',
    #                                                 target_size=(150,150),
    #                                                 batch_size= batch_size1,
    #                                                 #class_mode=None,
    #                                                 shuffle=False)

    #bottleneck_features_train = model.predict_generator(
    #    training_generator, 8000 // batch_size1)
    #print(bottleneck_features_train)
    #np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        directory='dataset - Copy\\test_set',
        target_size=(150, 150),
        batch_size=batch_size1,
        #class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, 2000 // batch_size1)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

    print(bottleneck_features_validation) # prints out a softmax-esque array of probability of each group being in picture. want argmax() prob

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array([0] * 4000 + [1] * 4000)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * 1000 + [1] * 1000)
    #print(train_data.shape[1:])
    #print(train_data[0])

    #print(validation_data.shape)
    #print(validation_data[0])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=50,
              batch_size=batch_size1,
              validation_data=(validation_data, validation_labels))
    model.save_weights('bottleneck_fc_model.h5')
#_setup_the_model_()
train_top_model()