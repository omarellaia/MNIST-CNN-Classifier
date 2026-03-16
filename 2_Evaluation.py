# **************************************************************************
# ÉVALUATION
# ===========================================================================

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.models import load_model

# GPU SETUP
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# CHARGEMENT DU MODÈLE
model_path = "Model.hdf5"
Classifier = load_model(model_path)

# VARIABLES
mainDataPath = "/content/mnist/"
testPath = mainDataPath + "test"

number_images = 2000
number_images_class_0 = 1000
number_images_class_1 = 1000
image_scale = 28
images_color_mode = "grayscale"

# CHARGEMENT DES IMAGES DE TEST
test_data_generator = ImageDataGenerator(rescale=1. / 255)
test_itr = test_data_generator.flow_from_directory(
    testPath,
    target_size=(image_scale, image_scale),
    class_mode="binary",
    shuffle=False,
    batch_size=1,
    color_mode=images_color_mode)

# ÉVALUATION
y_true = np.array([0] * number_images_class_0 + [1] * number_images_class_1)
test_eval = Classifier.evaluate(test_itr, steps=number_images, verbose=1)
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# PRÉDICTION DES CLASSES
predicted_classes = Classifier.predict(test_itr, steps=number_images, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes)

# CALCUL DES IMAGES BIEN ET MAL CLASSÉES
correct = np.where(predicted_classes.reshape(-1) == y_true)[0]
incorrect = np.where(predicted_classes.reshape(-1) != y_true)[0]

print(f"> {len(correct)} étiquettes bien classées")
print(f"> {len(incorrect)} étiquettes mal classées")

# MATRICE DE CONFUSION
cm = confusion_matrix(y_true, predicted_classes)
print("Matrice de confusion :")
print(cm)

# EXTRACTION ET AFFICHAGE D'UNE IMAGE MAL CLASSÉE POUR CHAQUE CLASSE
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for idx, cls in enumerate([0, 1]):
    incorrect_indices = incorrect[np.where(y_true[incorrect] == cls)[0]]
    if len(incorrect_indices) > 0:
        first_incorrect = incorrect_indices[0]
        image_data = test_itr[first_incorrect][0]
        ax[idx].imshow(image_data.reshape(image_scale, image_scale), cmap='gray')
        ax[idx].set_title(f"Mal classé {cls}, Prédit {int(predicted_classes[first_incorrect])}")
        ax[idx].axis('off')
plt.show()
