
import pickle
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from algo_nn import ConvNetClassifier, DenseNetClassifier

with open('data/visages.pkl', 'rb') as fh:
    visages = pickle.load(fh)

with open('data/noms.pkl', 'rb') as fh:
    noms = pickle.load(fh)

print('Shape of visages matrix --> ', visages.shape)

# Attention, pour la régression logistique, les images doivent être applaties (flattened) en dimension 1
N = len(noms)


# Attention, pour la régression logistique, les images doivent être applaties (flattened) en dimension 1
N = len(noms)


while True:
    choix = input("Rentrer 1,2,3,4 ou 5 : \n" +"1. algo de logistique regression\n" + "2. algo d'abres de decision\n" + "3. algo de k-NN \n" + "4. Résaux de neuronne Dense\n" + "5. réseaux de neurones convolutionnels\n" )
    try:
        choix = int(choix)
    except ValueError:
        print('Choix invalide !')
    if 1 <= choix <= 5:
        break
    print('Choix invalide !')

if choix <= 4:
    visages = visages.reshape(N, -1)

if choix == 1 :
    algores = LogisticRegression(max_iter=1000)
    algores.fit(visages, noms)
    titre = "Regression logistique"
      
elif choix == 2 :
    algores = DecisionTreeClassifier(random_state=0)
    algores.fit(visages, noms)
    titre = "Arbres de decision"
    
elif choix == 3 :
    algores = KNeighborsClassifier(n_neighbors=4)
    algores.fit(visages, noms)
    titre = "k-NN"

elif choix == 4:
    algores = DenseNetClassifier(noms)
    algores.fit(visages, noms)
    titre = "DenseNet"

elif choix == 5:
    algores = ConvNetClassifier(noms)
    algores.fit(visages, noms)
    titre = "ConvNet"


cascade_visage = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0) # 0 pour 'built-in' caméra, 1 pour caméra externe
user = {}
test = "\n"
setNom = list(set(noms))
for i in range(len(setNom)):
    user[i] = setNom[i]
    test += str(i) + "." + setNom[i] + "\n"

numAdmin = list(map(int, input("Qui sont admis : " + test).split()))
admin = []
for num in numAdmin:
    admin.append(user[num])
while True:
    
    ret, trame = camera.read()
    if ret == True:
        
        gris = cv2.cvtColor(trame, cv2.COLOR_BGR2GRAY)
        coordonnees_visage = cascade_visage.detectMultiScale(gris, 1.3, 5)

        for (x, y, l, h) in coordonnees_visage:
            
            visage = trame[y:y + h, x:x + l, :]
            visage_redimensionne = cv2.resize(visage, (50, 50))
            if choix <= 4:
                visage_redimensionne = visage_redimensionne.flatten().reshape(1,-1)
            elif choix == 5:
                visage_redimensionne = visage_redimensionne.reshape(1, 50, 50, 3)
            texte = algores.predict(visage_redimensionne)
            
            if texte[0] in admin:
                data = texte[0] + " admis"
                col = (0, 255, 0)
            else:
                col = (255, 0, 0)
                data = texte[0] + " non admis"
            
            cv2.putText(trame, data, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(trame, (x, y), (x + l, y + l), (0, 0, 255), 2)

        cv2.imshow(titre, trame)
        
        if cv2.waitKey(1) == 27:
            break
            
    else:
        
        print("erreur")
        break

cv2.destroyAllWindows()
camera.release()