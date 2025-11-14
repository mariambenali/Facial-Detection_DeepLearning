# Facial-Detection_DeepLearning
The SaaS application aims to detect faces and classify emotions using Convolutional Neural Networks (CNNs)



---

## 📝 Description

Ce projet utilise **Deep Learning** pour détecter les visages dans des images et prédire les émotions associées. Il combine **OpenCV**, **TensorFlow/Keras**, et **FastAPI** pour créer une API capable de traiter des images en temps réel.

### Fonctionnalités principales

- Détection de visages avec **Haar Cascade**.  
- Prédiction des émotions (joie, tristesse, colère, surprise…) avec un **modèle CNN pré-entraîné**.  
- API REST avec **FastAPI** pour recevoir des images et renvoyer les émotions détectées.  
- Tests automatisés avec **pytest**.  
- Linting et formatage avec **flake8** et **black**.  
- CI/CD avec **GitHub Actions**.

---

## 📂 Structure du projet
```bash
Facial-Detection_DeepLearning/
│
├─ app/
│ ├─ schema.py 
│ ├─ main.py 
│ └─ detect_and_predict.py 
│
├─ tests/ 
│
├─ requirements.txt 
├─ .github/workflows/ 
├─ myvenv/ 
└─ README.md
```
---

## 💻 Installation

1. **Cloner le projet :**  
```bash
git clone https://github.com/mariambenali/Facial-Detection_DeepLearning.git
cd Facial-Detection_DeepLearning

````

2. **Créer un environnement virtuel et l’activer :**  

```
python3 -m venv myvenv
source myvenv/bin/activate  # Mac/Linux
# myvenv\Scripts\activate    # Windows
```

3. **Installer les dépendances :**  

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Lancer l’API

```
uvicorn app.main:app --reload
```
L’API sera disponible sur : http://127.0.0.1:8000

Endpoints disponibles :

##### . POST /predict_emotion : Recevoir une image et prédire l’émotion.

##### . GET /emotions : Lister toutes les prédictions stockées en base.


---

## 🧩 Exécution des tests
```
pytest tests/ --maxfail=1 --disable-warnings -v
```

Vérifie que toutes les fonctions principales fonctionnent correctement.

Les tests doivent être placés dans le dossier tests/ et nommés test_*.py.

---
## 📦 Dépendances principales:

##### .Python 3.11

##### .TensorFlow

##### .OpenCV (opencv-python-headless)

##### .FastAPI

##### .SQLAlchemy

##### .Pytest

##### .Flake8, Black

---
## 🛠 GitHub Actions (CI/CD)

##### . Tests automatiques à chaque push ou pull request sur main, master ou DeepLearning.

##### . Installation automatique de Python et des dépendances.

##### . Linting et formatage du code.

##### . Exemple du workflow : .github/workflows/python-tests.yml
#

```
name: Facial Detection CI
on:
  push:
    branches: [ main, master, DeepLearning ]
  pull_request:
    branches: [ main, master, DeepLearning ]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: pip install tensorflow opencv-python-headless pytest flake8 black
      - run: pytest tests/ --maxfail=1 --disable-warnings -v
      - run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      - run: black --check .
```

---

## 🔗 Liens utiles

##### . Documentation FastAPI

##### . TensorFlow

##### . OpenCV

##### . Pytest