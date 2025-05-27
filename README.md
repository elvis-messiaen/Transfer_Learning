# Projet de Transfer Learning pour la Classification de Radiographies Pulmonaires

Ce projet implémente plusieurs modèles de deep learning pour la classification de radiographies pulmonaires, en utilisant la technique du transfer learning.

## Prérequis

### Configuration Système
- Python 3.10
- Environnement virtuel
- Système d'exploitation : Windows, macOS ou Linux

### Installation

1. **Création de l'environnement virtuel**

#### Windows
```bash
# Créer un nouvel environnement avec Python 3.10
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate
```

#### macOS/Linux
```bash
# Créer un nouvel environnement avec Python 3.10
python3.10 -m venv .venv

# Activer l'environnement
source .venv/bin/activate
```

2. **Installation des dépendances**

#### Windows
```bash
# Mise à jour de pip
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Installation des packages principaux
pip install tensorflow
pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

#### macOS (Intel)
```bash
# Mise à jour de pip
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Installation des packages principaux
pip install tensorflow
pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

#### macOS (Apple Silicon - M1/M2/M4)
```bash
# Mise à jour de pip
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Installation des packages principaux
pip install tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

#### Linux
```bash
# Mise à jour de pip
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Installation des packages principaux
pip install tensorflow
pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

## Structure du Projet

Le projet contient trois modèles principaux :
1. VGG16
2. ResNet50
3. Xception

## Utilisation

### 1. Préparation des Données
Les données doivent être organisées dans la structure suivante :
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### 2. Exécution des Modèles

#### VGG16
```python
# Imports nécessaires
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Configuration du modèle
img_shape = (224, 224, 3)
model_VGG16 = VGG16(input_shape=img_shape, include_top=False, weights='imagenet')
model_VGG16.trainable = False

# Ajout des couches de classification
x = Flatten()(model_VGG16.output)
prediction = Dense(1, activation='softmax')(x)
model = Model(inputs=model_VGG16.input, outputs=prediction)
```

#### ResNet50
```python
# Imports nécessaires
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Configuration du modèle
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Ajout des couches de classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)
```

#### Xception
```python
# Imports nécessaires
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Configuration du modèle
base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Ajout des couches de classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)
```

### 3. Entraînement des Modèles

Pour chaque modèle, utilisez le code suivant :
```python
# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)
```

### 4. Évaluation des Modèles

```python
# Évaluation sur les données de test
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64)

# Affichage des résultats
print(f"Accuracy sur données de test : {test_accuracy * 100:.2f}%")
print(f"Loss sur données de test : {test_loss:.4f}")
```

## Visualisation des Résultats

Le projet inclut des visualisations pour :
- Courbes d'apprentissage (accuracy et loss)
- Matrice de confusion
- Exemples de prédictions

## Notes Importantes

1. **Prétraitement des Images**
   - Toutes les images sont redimensionnées à 224x224 pixels
   - Les valeurs des pixels sont normalisées entre 0 et 1
   - Les images sont converties du format BGR (OpenCV) à RGB

2. **Configuration du Modèle**
   - Les couches de base sont gelées (non entraînables)
   - Des couches de dropout sont ajoutées pour éviter le surapprentissage
   - L'activation finale est sigmoïde pour la classification binaire

3. **Optimisation**
   - Utilisation de l'optimiseur Adam
   - Taille de batch de 32
   - 40 époques d'entraînement

## Dépannage

### Windows
Si vous rencontrez des problèmes avec les imports, assurez-vous que :
1. Votre environnement virtuel est activé
2. Toutes les dépendances sont correctement installées
3. Vous avez installé Visual C++ Redistributable pour Windows

### macOS
Si vous rencontrez des problèmes avec les imports, assurez-vous que :
1. Votre environnement virtuel est activé
2. Toutes les dépendances sont correctement installées
3. Si vous utilisez une puce Apple Silicon, vous avez installé les versions spécifiques de TensorFlow

### Linux
Si vous rencontrez des problèmes avec les imports, assurez-vous que :
1. Votre environnement virtuel est activé
2. Toutes les dépendances sont correctement installées
3. Vous avez les bibliothèques système nécessaires installées :
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip python3-venv
   sudo apt-get install libgl1-mesa-glx
   ```

## Contribution

N'hésitez pas à contribuer au projet en :
- Ajoutant de nouveaux modèles
- Améliorant la documentation
- Corrigeant des bugs
- Optimisant les performances
