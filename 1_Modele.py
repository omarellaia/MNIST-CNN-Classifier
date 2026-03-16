# **************************************************************************
# Ce modèle est un classifieur (un CNN) entrainé sur l'ensemble de données MNIST afin de distinguer entre les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 4 000 images | classe '7': images 4 000 images
# validation   : classe '2': 1 000 images | classe '7': images 1 000 images
# test         : classe '2': 1 000 images | classe '7': images 1 000 images 
# ------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import time

# GPU SETUP
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# VARIABLES
mainDataPath = "/content/mnist"  
trainPath = "/content/mnist/entrainement"
validationPath = "/content/mnist/validation"
testPath = "/content/mnist/test"
modelsPath = "Model.hdf5"

training_batch_size = 8000
validation_batch_size = 2000
image_scale = 28
image_channels = 1
images_color_mode = "grayscale"
image_shape = (image_scale, image_scale, image_channels)
fit_batch_size = 32
fit_epochs = 10

input_layer = Input(shape=image_shape)

def feature_extraction(input):
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def fully_connected(encoded):
    x = Flatten()(encoded)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)
    sortie = Activation('sigmoid')(x)
    return sortie

model = Model(input_layer, fully_connected(feature_extraction(input_layer)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)
training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=training_batch_size,
    class_mode="binary",
    shuffle=True)
validation_generator = validation_data_generator.flow_from_directory(
    validationPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=validation_batch_size,
    class_mode="binary",
    shuffle=True)

print(training_generator.class_indices)
print(validation_generator.class_indices)

(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

# Entraînement du modèle
start_time = time.time()
modelcheckpoint = ModelCheckpoint(filepath=modelsPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
classifier = model.fit(x_train, y_train, epochs=fit_epochs, batch_size=fit_batch_size, validation_data=(x_val, y_val), verbose=1, callbacks=[modelcheckpoint], shuffle=True)
end_time = time.time()

# Affichage du temps d'exécution
print(f"Temps d'exécution: {end_time - start_time} secondes")

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(classifier.history['accuracy'], label='Training Accuracy')
plt.plot(classifier.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(classifier.history['loss'], label='Training Loss')
plt.plot(classifier.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
# **************************************************************************
# INF5081
# Travail pratique 2 
# ===========================================================================
# #===========================================================================
# Ce modèle est un classifieur (un CNN) entrainé sur l'ensemble de données MNIST afin de distinguer entre les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 4 000 images | classe '7': images 4 000 images
# validation   : classe '2': 1 000 images | classe '7': images 1 000 images
# test         : classe '2': 1 000 images | classe '7': images 1 000 images 
# ------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import time

# GPU SETUP
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# VARIABLES
mainDataPath = "/content/mnist"  
trainPath = "/content/mnist/entrainement"
validationPath = "/content/mnist/validation"
testPath = "/content/mnist/test"
modelsPath = "Model.hdf5"

training_batch_size = 8000
validation_batch_size = 2000
image_scale = 28
image_channels = 1
images_color_mode = "grayscale"
image_shape = (image_scale, image_scale, image_channels)
fit_batch_size = 32
fit_epochs = 10

input_layer = Input(shape=image_shape)

def feature_extraction(input):
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def fully_connected(encoded):
    x = Flatten()(encoded)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)
    sortie = Activation('sigmoid')(x)
    return sortie

model = Model(input_layer, fully_connected(feature_extraction(input_layer)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)
training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=training_batch_size,
    class_mode="binary",
    shuffle=True)
validation_generator = validation_data_generator.flow_from_directory(
    validationPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=validation_batch_size,
    class_mode="binary",
    shuffle=True)

print(training_generator.class_indices)
print(validation_generator.class_indices)

(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

# Entraînement du modèle
start_time = time.time()
modelcheckpoint = ModelCheckpoint(filepath=modelsPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
classifier = model.fit(x_train, y_train, epochs=fit_epochs, batch_size=fit_batch_size, validation_data=(x_val, y_val), verbose=1, callbacks=[modelcheckpoint], shuffle=True)
end_time = time.time()

# Affichage du temps d'exécution
print(f"Temps d'exécution: {end_time - start_time} secondes")

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(classifier.history['accuracy'], label='Training Accuracy')
plt.plot(classifier.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(classifier.history['loss'], label='Training Loss')
plt.plot(classifier.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
