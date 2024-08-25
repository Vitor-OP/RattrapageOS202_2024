import numpy as np 
import numpy.linalg as linalg
import time
import labels

def loi_jointe(data, v1, v2, v3=None):
    X1 = data[:,v1] # Echantillons variable v1
    X2 = data[:,v2] # Echantillons variable v2
    modalX1 = np.unique(X1) # Modalités contenus dans X1
    modalX2 = np.unique(X2) # Modalités contenus dans X2

    if v3 is None:
        return np.array([ [np.mean((X1 == e1)*(X2 == e2)) for e2 in modalX2] for e1 in modalX1])
    X3 = data[:,v3]
    modalX3 = np.unique(X3) # Les modalités de la variable v3
    return np.array([ [ [np.mean((X1 == e1)*(X2 == e2)*(X3 == e3)) for e3 in modalX3 ] for e2 in modalX2] for e1 in modalX1])

def sont_independantes( pxy ):
    """
    Vérifie si les variables v1, v2 sont indépendantes dans le tableau de loi de probabilité pxy
    """
    epsilon = 1.E-8
    # Calcul des marginales :
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    return np.max(np.abs(pxy - np.array([px]).T @ np.array([py]))) < epsilon

# Classification de Bayes à partir d'un prédicteur qualitatif 
def classif_Bayes(train,test,v):
    values = np.unique(train[:,v])
    pxpy = loi_jointe(train,labels.MixProdElec,v)
    X = test[:,v]
    prev = np.empty(len(X), dtype=np.int32)
    for i in range(len(X)):
        # np.argmax(values==x.iloc[i]) donne la position de la valeur x dans le tableau values
        prev[i] = np.argmax(pxpy[:,np.argmax(values == X[i])])
    return prev

def classif_Bayes2(train,test,v1, v2):
    values_per_var = (np.unique(train[:,v1]),np.unique(train[:,v2]))
    # La ligne en dessous forme toutes les paires possibles de valeurs entre les deux variables :
    # values = np.array(np.meshgrid(values_per_var[0], values_per_var[1])).T.reshape(-1,2)
    pxpypz = loi_jointe(train, labels.MixProdElec, v1, v2)

    X = (test[:,v1],test[:,v2])
    n : int = X[0].shape[0]
    loc1 = np.array([np.argmax(values_per_var[1] == X[1][i]) for i in range(n)])
    loc0 = np.array([np.argmax(values_per_var[0] == X[0][j]) for j in range(n)])
    prev = np.empty(n, dtype=np.int32)
    for i in range(n):  
        prev[i] = np.argmax(pxpypz[:,loc0[i], loc1[i]])
    return prev

def naif_Bayes2(train,test,v1, v2):
    values_per_var = (np.unique(train[:,v1]),np.unique(train[:,v2]))
    pxpy = loi_jointe(train, labels.MixProdElec, v1)
    pxpz = loi_jointe(train, labels.MixProdElec, v2)

    X = (test[:,v1],test[:,v2])
    n : int = X[0].shape[0]
    loc1 = np.array([np.argmax(values_per_var[1] == X[1][i]) for i in range(n)])
    loc0 = np.array([np.argmax(values_per_var[0] == X[0][j]) for j in range(n)])
    loi_naive = pxpy[:,loc0] * pxpz[:,loc1]
    prev = np.empty(n, dtype=np.int32)
    for i in range(n):  
        prev[i] = np.argmax(loi_naive[:,i])
    return prev

if __name__ == "__main__":
    import time
    data = np.loadtxt("train.npy")
    np.random.seed(2024)
    msk = np.random.rand(data.shape[0]) < 0.75
    train = data[ msk]
    test  = data[~msk]

    deb = time.time()
    pred  = classif_Bayes(train, test, labels.Jour)
    fin = time.time()
    # On vérifie la précision des prédictions faites par rapport à ce qu'on aurait dû trouver :
    print(f"Précision de la prédiction jour: {100*np.sum(pred == test[:,labels.MixProdElec])/pred.shape[0]}%")
    print(f"Temps bayes une variable : {fin-deb} secondes")

    deb = time.time()
    pred = classif_Bayes(train, test, labels.Mois)
    fin = time.time()
    print(f"Précision de la prédiction mois: {100*np.sum(pred == test[:,labels.MixProdElec])/pred.shape[0]}%")
    print(f"Temps bayes une variable : {fin-deb} secondes")

    deb = time.time()
    pred = classif_Bayes2(train, test, labels.Jour, labels.Mois)
    fin = time.time()
    print(f"Précision de la prédiction jour/mois: {100*np.sum(pred == test[:,labels.MixProdElec])/pred.shape[0]}%")
    print(f"Temps bayes deux variables : {fin-deb} secondes")

    deb = time.time()
    pxy = loi_jointe(train, labels.Jour, labels.Mois)
    fin = time.time()
    print(f"Calcul loi jointe : {fin-deb} secondes")
    if sont_independantes(pxy): # Loi indépendante ?
        print("Jour et Mois sont bien indépendants : on test l'algorithme de Bayes naïf")
    else:
        print("Désolé, vos deux variables ne sont pas indépendantes. On fait le bayes naïf, mais risque d'erreur")
    deb = time.time()
    pred = naif_Bayes2(train, test, labels.Jour, labels.Mois)
    fin = time.time()
    print(f"Précision de la prédiction jour/mois: {100*np.sum(pred == test[:,labels.MixProdElec])/pred.shape[0]}%")
    print(f"Temps bayes naïf deux variables : {fin-deb} secondes")
