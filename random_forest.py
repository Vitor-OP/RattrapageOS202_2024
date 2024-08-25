import numpy  as np
import decision_tree as decision

n_subsamples = 800
n_trees      = 50
max_depth    = 11
min_samples_split = 2

def bootstrap_sample(data, labels):
    """
    Create a bootstrap sample from the given data and labels.

    Args:
        data (numpy.ndarray): The input data.
        labels (numpy.ndarray): The corresponding labels.

    Returns:
        tuple: A tuple containing the bootstrap sample of data and labels.
    """
    #indices = np.random.choice(len(data), size=len(data), replace=True)
    indices = np.random.choice(len(data), size=n_subsamples, replace=True)
    return data[indices], labels[indices]

def random_forest(data, labels, n_trees, max_depth, min_samples_split):
    """
    Build a random forest classifier.

    Args:
        data (numpy.ndarray): The input data.
        labels (numpy.ndarray): The corresponding labels.
        n_trees (int): The number of trees in the forest.
        max_depth (int): The maximum depth of each tree.
        min_samples_split (int): The minimum number of samples required to split a node.

    Returns:
        list: A list of decision trees representing the random forest.
    """
    forest = []
    for _ in range(n_trees):
        print(f"Construction arbre num. {_}")
        bootstrap_data, bootstrap_labels = bootstrap_sample(data, labels)
        tree = decision.build_tree(bootstrap_data, bootstrap_labels, max_depth, min_samples_split)
        forest.append(tree)
    return forest

def predict_forest(forest, data_point):
    """
    Make a prediction for a single data point using the random forest.

    Args:
        forest (list): The random forest represented as a list of decision trees.
        data_point (numpy.ndarray): The input data point.

    Returns:
        int: The predicted class label.
    """
    predictions = [decision.predict(tree, data_point) for tree in forest]
    return np.argmax(np.bincount(predictions))


def test(file_train : str):
    import time
    import numpy.random as random
    import labels

    rng = random.default_rng(8491)
    data = np.loadtxt(file_train)
    np.random.seed(2024)
    msk = np.random.rand(data.shape[0]) < 0.75
    train = data[ msk]
    test  = data[~msk]

    # Affiche le ratio Carbonne,Decarbonne et Mixte :
    _, value_counts = np.unique(train[:,labels.MixProdElec], return_counts=True)
    print(f"Ratio des différents modes de production électrique : Carbonne {100*value_counts[labels.CARBONNE]/len(train)}%\t Decarbonne {100*value_counts[labels.DECARBONNE]/len(train)}%\t Normale {100*value_counts[labels.NORMALE]/len(train)}%")

    #n_samples = 4_000
    #ind_samples = rng.integers(len(train), size=n_samples)
    x_train = train[:,2:]
    y_train = train[:,0 ].astype(np.int32)
    x_test  = test[:,2:]
    y_test  = test[:, 0].astype(np.int32)

    beg = time.time()
    forest = random_forest(x_train, y_train, n_trees, max_depth, min_samples_split)
    end = time.time()
    print(f"Temps de calcul de la foret d'arbre de decision : {end-beg} secondes")
    n_tests = len(x_test)
    prediction = np.empty(n_tests, dtype=np.int32)
    beg = time.time()
    for i in range(n_tests):
        prediction[i] = predict_forest(forest, x_test[i,:])
    end = time.time()
    print(f"Temps pris pour calculer les predictions : {end-beg} secondes")
    print(f"Précision de l'arbre de précision': {100*np.sum(prediction == test[:,labels.MixProdElec])/prediction.shape[0]}%")


if __name__ == '__main__':
    test("train.npy")
