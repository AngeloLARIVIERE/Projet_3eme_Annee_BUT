ROCHER Quentin 
LARIVIERE Angelo

# SAÉ: Modélisation mathématique<br>Reconnaissance faciale en temps réel

<img src="moi.png" width="400" height="auto" />


## Description du projet

Le but de cette SAÉ est de construire un système de reconnaissance faciale qui fonctionne en temps réel.

Le système utilise la caméra de votre PC (ou alors une caméra externe) pour capturer des images de visages. Ainsi, on pourra construire un **dataset** contenant des visages de différentes personnes associées à leurs noms.

Par la suite, on entraînera un **algorithme de classification** permettant de prédire le nom de la personne qui se trouve devant la caméra en fonction de son visage.

## Implémentation

Les 2 notebooks `1_dataset.ipynb`et `2_ago.ipynb` permettent d'implémenter un tel **système de reconnaissance faciale en temps réel** basé sur l'algorithme des **$k$ plus proches voisins ($k$-NN)**.

Le premier notebook implémente la création du **dataset** et le second code l'algorithme **$k$-NN** et son intégration dans un système de reconnaissance faciale en temps réel.

## Consignes

1. **Comprenez les 2 notebooks** en détails, et, si besoin, débuggez-les jusqu'à ce qu'ils fonctionnent correctement (chez moi, ils marchent).
2. **Adaptez le 2ème notebook** à d'autres algorithmes de classification de votre choix, tels que la régression logistique, les arbres de décision, etc.
3. Essayer d'adapter votre système pour le **cas d'utilisation de reconnaissance binaire** suivant: le système devra répondre `admis` ou `non admis` suivant que le visage détecté est le votre ou non.
4. Documentez-vous sur les **réseaux de neurones** et essayez d'implémenter et d'intégrer un algorithme de classification par réseaux de neurones.
5. Si vous avez le temps, documentez-vous sur les **réseaux de neurones convolutionnels** (qui sont spécialisés dans le traitement des images) et essayez d'implémenter et d'intégrer un algorithme de classification par réseaux de neurones convolutionnels.
6. Rendez votre projet sous la forme d'un **répertoire GitHub** .  Votre  repo contiendra un **fichier README** (md file) et plusieurs **notebooks jupyter** (ipynb files) propres et commentés  qui présentent votre projet.

## Instalation des librairis pour le réseau de neuronnes
````shell
  python.exe -m pip install --upgrade pip
  python.exe -m pip install keras 
  python.exe -m pip install tensorflow 
````

# Ecplication du code
Nous avons faits un fichier jupiter pour chaque modèle que nous avons utilisé. Mais nous avons aussi fait un fichir **py**
qui synthétise tous notre code pour éviter de trop le dupliquer. Dans celui-ci on peut choisir quelle méthode nous voulons utiliser
ainsi que de pouvoir determiner qui parmis les gens inscrit dans le dataset sont admis ou non.

# Documentation de la classe DenseNetClassifier
Pour cette classe nous avonc fait le choix de faire un modèle de résaux de neuronnes  dense avec 3 couche de neuronnes et une couche de sortie avec un nombre de neurone égale au nombre de nom de personne différente enregistrer.   
la première couche possède 512 neuronne, la deuxième en contient 64, puis la troisième en contient 32.
Cette classe contient trois fonctions, la première **init** nous permet d'initialiser notre réseau de neuronnes dense. La deuxième fonction 
**fit** permet d'entrainer notre réseau de neuronnes. Et enfin la fonction **predict** qui nous permet de prédire à qui sont les visages.


# Documentation de la classe ConvNetClassifier
Pour cette classe nous avons fait le choix de faire un modèle de réseaux de neuronnes avec deux couches. La première couche 
contient 32 neuronnes et la deuxième en contient 16. Il est important de comprendre qui la dernière couche du réseaux de neuronnes
n'est autre que qu'une couche de type DenseNetClassifier qui contient 32 neuronnes. Dans ce réseau de neuronnes nous utilisons aussi la Régression linéaire.

Cette classe contient trois fonctions, la première **init** nous permet d'initialiser notre réseau de neuronnes. La deuxième fonction 
**fit** permet d'entrainer notre réseau de neuronnes. Et enfin la fonction **predict** qui nous permet de prédire à qui sont les visages.


# Teste de fonctionnement
Lors de nos différents tests, nous avons pu voir que le réseau de neurones convolutionnel donnait de meilleurs résultats avec un entraînement de 500 epochs ainsi qu'avec un jeu de données d'au moins 40 photos par personne. Cela permet au réseau de mieux s'entraîner sur chaque individu.

Nous avons également remarqué, pour les différentes méthodes autres que les réseaux de neurones, qu'elles étaient assez efficaces pour prédire avec précision deux personnes différentes. Cependant, lorsque nous ajoutons plus de deux personnes dans le jeu de données, cela ne lui permet pas de bien prédire les individus.
# Annexes
Voici les liens qui nous on permis de réaliser notre réseau de neuronnes à l'aide de la librairi keras.
- [La base de la librairi keras](https://keras.io/examples/vision/image_classification_from_scratch/)
- [application de la librairi avec la reconnaissance d'image](https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/)
- [Problème rencontré par Angelo lors de l'instalation de tenserflow](https://stackoverflow.com/a/76085534)
