import numpy as np 
import numpy.linalg as linalg
import time
import labels
from multiprocessing import Pool, cpu_count, Array
import ctypes

def parallel_classify(func, train, test, *args):
    n_processes = 8
    # n_processes = cpu_count()
    chunk_size = len(test) // n_processes
    chunks = [test[i:i + chunk_size] for i in range(0, len(test), chunk_size)]
    with Pool(n_processes) as pool:
        results = pool.starmap(func, [(train, chunk, *args) for chunk in chunks])
    
    return np.concatenate(results)

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
    
    # SEQUENTIAL one variable Bayes
    print("Bayes one variable jour:")
    deb_seq = time.time()
    pred_seq = classif_Bayes(train, test, labels.Jour)
    fin_seq = time.time()
    acc_seq = 100 * np.sum(pred_seq == test[:, labels.MixProdElec]) / pred_seq.shape[0]
    print(f"Sequential Précision de la prédiction jour: {acc_seq}%")
    print(f"Sequential Temps bayes une variable : {fin_seq-deb_seq} secondes")
    
    # PARALLEL one variable Bayes
    deb_par = time.time()
    pred_par = parallel_classify(classif_Bayes, train, test, labels.Jour)
    fin_par = time.time()
    acc_par = 100 * np.sum(pred_par == test[:, labels.MixProdElec]) / pred_par.shape[0]
    print(f"Parallel Précision de la prédiction jour: {acc_par}%")
    print(f"Parallel Temps bayes une variable : {fin_par-deb_par} secondes")
    
    speedup_bayes1 = (fin_seq - deb_seq) / (fin_par - deb_par)
    print(f"Speedup for Bayes one variable: {speedup_bayes1:.2f}\n")
    print()
    
    # SEQUENTIAL one variable Bayes
    print("Bayes one variable mois:")
    deb_seq = time.time()
    pred_seq = classif_Bayes(train, test, labels.Mois)
    fin_seq = time.time()
    acc_seq = 100 * np.sum(pred_seq == test[:, labels.MixProdElec]) / pred_seq.shape[0]
    print(f"Sequential Précision de la prédiction mois: {acc_seq}%")
    print(f"Sequential Temps bayes une variable : {fin_seq-deb_seq} secondes")
    
    # PARALLEL one variable Bayes
    deb_par = time.time()
    pred_par = parallel_classify(classif_Bayes, train, test, labels.Mois)
    fin_par = time.time()
    acc_par = 100 * np.sum(pred_par == test[:, labels.MixProdElec]) / pred_par.shape[0]
    print(f"Parallel Précision de la prédiction mois: {acc_par}%")
    print(f"Parallel Temps bayes une variable : {fin_par-deb_par} secondes")
    
    speedup_bayes1 = (fin_seq - deb_seq) / (fin_par - deb_par)
    print(f"Speedup for Bayes one variable: {speedup_bayes1:.2f}\n")
    print()

    # SEQUENTIAL two variables Bayes
    print("Bayes two variables jour/mois:")
    deb_seq = time.time()
    pred_seq = classif_Bayes2(train, test, labels.Jour, labels.Mois)
    fin_seq = time.time()
    acc_seq = 100 * np.sum(pred_seq == test[:, labels.MixProdElec]) / pred_seq.shape[0]
    print(f"Sequential Précision de la prédiction jour/mois: {acc_seq}%")
    print(f"Sequential Temps bayes deux variables : {fin_seq-deb_seq} secondes")

    # PARALLEL two variables Bayes
    deb_par = time.time()
    pred_par = parallel_classify(classif_Bayes2, train, test, labels.Jour, labels.Mois)
    fin_par = time.time()
    acc_par = 100 * np.sum(pred_par == test[:, labels.MixProdElec]) / pred_par.shape[0]
    print(f"Parallel Précision de la prédiction jour/mois: {acc_par}%")
    print(f"Parallel Temps bayes deux variables : {fin_par-deb_par} secondes")

    speedup_bayes2 = (fin_seq - deb_seq) / (fin_par - deb_par)
    print(f"Speedup for Bayes two variables: {speedup_bayes2:.2f}\n")
    print()

    # deb = time.time()
    # pxy = loi_jointe(train, labels.Jour, labels.Mois)
    # fin = time.time()
    # print(f"Calcul loi jointe : {fin-deb} secondes")
    # if sont_independantes(pxy): # Loi indépendante ?
    #     print("Jour et Mois sont bien indépendants : on test l'algorithme de Bayes naïf")
    # else:
    #     print("Désolé, vos deux variables ne sont pas indépendantes. On fait le bayes naïf, mais risque d'erreur")
    # deb = time.time()
    # pred = naif_Bayes2(train, test, labels.Jour, labels.Mois)
    # fin = time.time()
    # print(f"Précision de la prédiction jour/mois: {100*np.sum(pred == test[:,labels.MixProdElec])/pred.shape[0]}%")
    # print(f"Temps bayes naïf deux variables : {fin-deb} secondes")

    deb = time.time()
    pxy = loi_jointe(train, labels.Jour, labels.Mois)
    fin = time.time()
    print(f"Calcul loi jointe : {fin-deb} secondes")
    if sont_independantes(pxy):
        print("Jour et Mois sont bien indépendants : on teste l'algorithme de Bayes naïf")
    else:
        print("Désolé, vos deux variables ne sont pas indépendantes. On fait le Bayes naïf, mais risque d'erreur")
    print()

    # SEQUENTIAL Naive Bayes Classification
    deb_seq = time.time()
    pred_seq = naif_Bayes2(train, test, labels.Jour, labels.Mois)
    fin_seq = time.time()
    acc_seq = 100 * np.sum(pred_seq == test[:, labels.MixProdElec]) / pred_seq.shape[0]
    print(f"Sequential Précision de la prédiction jour/mois (naïf): {acc_seq}%")
    print(f"Sequential Temps bayes naïf deux variables : {fin_seq - deb_seq} secondes")

    # PARALLEL Naive Bayes Classification
    deb_par = time.time()
    pred_par = parallel_classify(naif_Bayes2, train, test, labels.Jour, labels.Mois)
    fin_par = time.time()
    acc_par = 100 * np.sum(pred_par == test[:, labels.MixProdElec]) / pred_par.shape[0]
    print(f"Parallel Précision de la prédiction jour/mois (naïf): {acc_par}%")
    print(f"Parallel Temps bayes naïf deux variables : {fin_par - deb_par} secondes")

    # Speedup Calculation
    speedup_joint = (fin_seq - deb_seq) / (fin_par - deb_par)
    print(f"Speedup for Naive Bayes two variables : {speedup_joint:.2f}\n")