# Guide d'Installation - OmniVision AI

Bienvenue dans le guide d'utilisation et d'installation du projet **OmniVision AI**, votre application de détection d'objets en temps réel pour salles de classe.

## Étape 1 : Installer Python 3.10
1. Téléchargez Python 3.10 depuis le site officiel : [python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)
2. Lors de l'installation, **cochez impérativement la case** `Add Python 3.10 to PATH`.
3. Vérifiez l'installation en ouvrant un terminal (Invite de commandes ou PowerShell) et tapez :
   ```bash
   python --version
   ```
   *Le résultat doit afficher une version 3.10.x*

## Étape 2 : Créer un environnement virtuel
Il est indispensable d'isoler les dépendances du projet :
1. Ouvrez votre terminal et placez-vous dans le dossier racine du projet `OmniVision`.
2. Créez l'environnement virtuel avec la commande :
   ```bash
   python -m venv venv
   ```
3. Activez l'environnement :
   - Sur **Windows (PowerShell)** : 
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Sur **Windows (CMD)** :
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - Sur **Mac/Linux** :
     ```bash
     source venv/bin/activate
     ```

## Étape 3 : Installer les dépendances
Avec l'environnement virtuel bien activé (un préfixe `(venv)` doit apparaitre dans votre terminal), lancez la commande d'installation :
```bash
pip install -r requirements.txt
```

## Étape 4 : Téléchargement du modèle IA
Vous n'avez **aucune étape manuelle de téléchargement**. Lors de l'exécution, la librairie `ultralytics` vérifiera la présence du poids `yolov8n.pt`. S'il est absent, il sera téléchargé automatiquement sur votre machine. Assurez-vous simplement d'avoir une connexion internet active lors du premier lancement.

## Étape 5 : Lancer le projet

Le projet dispose de **deux modes** fonctionnels :

### 🚀 Mode 1 : Script Temps Réel (via `main.py`)
Ce mode lance une fenêtre de votre webcam avec les algorithmes de machine learning qui analysent la classe à plus de 25 FPS :
```bash
python main.py
```
*(💡 Appuyez sur la touche `Q` pour stopper la boucle infinie et fermer la fenêtre)*

### 🌐 Mode 2 : Interface Graphique (via `app.py`)
Ce mode lance un tableau de bord web interactif utilisant Streamlit, autorisant l'upload d'images statiques ou l'utilisation du flux de la caméra intégrée au panel.
```bash
streamlit run app.py
```
Votre navigateur par défaut s'ouvrira de lui-même à l'adresse `http://localhost:8501`.

---
**Félicitations, votre IA de vision par ordinateur prend vie ! 👁️**
