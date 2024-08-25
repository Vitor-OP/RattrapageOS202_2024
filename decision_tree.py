import numpy  as np
import labels

def gini_impurity(labels : np.ndarray):
    """
    Calcul l'impureté de Gini pour un ensemble de prédictions.

    L'idée de l'impureté de Gini est la probabilité de se tromper dans une prédiction si on la prenait au hasard au sein
    d'un échantillon donné.

    Entrées :
        labels : Un tableau de prédiction

    Sortie:
        La valeur d'impureté de Gini (float)
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts/len(labels)
    return 1.-np.sum(probabilities**2)


def split_data( train_data : np.ndarray, labels : np.ndarray, feature_index : int, threshold : float):
    """
    Partitionne un ensemble d'entraînement (et de réponse) par rapport à une valeur de seuil.

    Entrée:
        train_data    : L'ensemble des données d'entraînement
        labels        : Les réponses correspondantes aux données d'entraînement
        feature_index : L'index du l'étiquette par rapport à laquelle on effectue le partitionnement
        threshold     : Le seuil définissant le partitionnement
    """
    left_mask  = train_data[:, feature_index] < threshold
    right_mask = ~left_mask
    return train_data[left_mask], labels[left_mask], train_data[right_mask], labels[right_mask]


def find_best_split(train_data : np.ndarray, labels : np.ndarray):
    """
    Trouve le meilleur partitionnement pour un ensemble de données d'entrée

    Entrées:
        train_data : Les données d'entrée
        labels     : Les prédictions correspondant à chaque entrée de l'entraînement
    
    Sortie:
        La meilleur étiquette (son index en fait) et la valeur de seuil pour son partitionnement
    """
    best_feature, best_threshold, best_gini = None, None, float('inf')
    for feature_index in range(train_data.shape[1]):
        # Récupère toutes les valeurs mesurées par l'étiquette courante (dans la boucle)
        values = np.unique(train_data[:, feature_index])
        # Recherche du seuil optimal pour cette étiquette :
        for threshold in values:
            _, left_labels, _, right_labels = split_data(train_data, labels, feature_index, threshold)
            if len(left_labels) == 0 or len(right_labels) == 0: continue
            # Calcul de l'indice de gini:
            gini = (len(left_labels)*gini_impurity(left_labels) + len(right_labels)*gini_impurity(right_labels)) / len(labels)
            # Plus l'indice de gini est petit, mieux c'est
            if gini < best_gini:
                best_feature, best_threshold, best_gini = feature_index, threshold, gini
    return best_feature, best_threshold


def build_tree(train_data : np.ndarray, labels : np.ndarray, max_depth : int, min_samples_split : int, depth : int =  0):
    """
    Construit un arbre de décision récursivement

    Entrées :
        train_data        : Les données d'entrée
        labels            : Les prédictions correspondant à chaque entrée de l'entraînement
        max_depth         : La profondeur maximale de l'arbre
        min_samples_split : Le nombre d'échantillons minimal requis pour partitionner un noeud de l'arbre
        depth (optionnel) : La profondeur de l'arbre actuellement traitée (dans la récursion)

    Sortie:
        L'arbre de décision retourné sous la forme d'un dictionnaire
    """
    if depth == max_depth or len(labels) < min_samples_split or gini_impurity(labels) == 0:
        return {'prediction' : np.argmax(np.bincount(labels))}
    feature, threshold = find_best_split(train_data, labels)
    # Si pas trouvé de feature adéquat, on arrête la construction de la branche et
    # on recherche la valeur la plus probable comme réponse (sous forme d'indice) :
    if feature is None: return {"prediction" : np.argmax(np.bincount(labels))}
    left_train_data, left_labels, right_train_data, right_labels = split_data(train_data, labels, feature, threshold)
    return {
        "feature"   : feature,
        "threshold" : threshold,
        "left"      : build_tree(left_train_data, left_labels, max_depth, min_samples_split, depth+1),
        "right"     : build_tree(right_train_data, right_labels, max_depth, min_samples_split, depth+1)
    }

def predict(tree : dict, data_point : np.ndarray):
    """
    Fait une prédiction pour un simple échantillon utilisant l'arbre de décision

    Entrées:
        tree       : L'arbre de décision représentée comme un dictionnaire
        data_point : L'échantillon d'entrée

    Sortie:
        La valeur prédite pour l'étiquette
    """
    if 'prediction' in tree:
        return tree['prediction']
    if data_point[tree['feature']] < tree['threshold']:
        return predict(tree['left'], data_point)
    else:
        return predict(tree['right'], data_point)


def main( file_train : str ):
    import time
    import numpy.random as random
    rng = random.default_rng(seed=2025)
    data = np.loadtxt(file_train)
    np.random.seed(2024)
    msk = np.random.rand(data.shape[0]) < 0.75
    train = data[ msk]
    test  = data[~msk]

    # Affiche le ratio Carbonne,Decarbonne et Mixte :
    _, value_counts = np.unique(train[:,labels.MixProdElec], return_counts=True)
    print(f"Ratio des différents modes de production électrique : Carbonne {100*value_counts[labels.CARBONNE]/len(train)}%\t Decarbonne {100*value_counts[labels.DECARBONNE]/len(train)}%\t Normale {100*value_counts[labels.NORMALE]/len(train)}%")

    n_samples = 4_000
    ind_samples = rng.integers(len(train), size=n_samples)
    x_train = train[ind_samples,2:]
    y_train = train[ind_samples,0 ].astype(np.int32)
    x_test  = test[:,2:]
    y_test  = test[:, 0].astype(np.int32)

    beg = time.time()
    tree = build_tree(x_train, y_train, max_depth=11, min_samples_split=2)
    end = time.time()
    print(f"Temps de construction de l'arbre de décision : {end-beg} secondes")

    n_tests = len(x_test)
    prediction = np.empty(n_tests, dtype=np.int32)
    beg = time.time()
    for i in range(n_tests):
        prediction[i] = predict(tree, x_test[i,:])
    end = time.time()
    print(f"Temps pris pour calculer les predictions : {end-beg} secondes")
    print(f"Précision de l'arbre de précision': {100*np.sum(prediction == test[:,labels.MixProdElec])/prediction.shape[0]}%")


if __name__ == "__main__":
    main("train.npy")