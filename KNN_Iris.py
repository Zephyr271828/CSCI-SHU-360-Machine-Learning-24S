import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def load_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    classes = iris.target_names

    # check number of elements in each class
    for i in range(len(classes)):
        print(f'# elements in {classes[i]} class: {len([label for label in labels if label == i])}')

    return data, labels

# calculate accuracy for a given dataset and a model
def test(X, y, knn):
    y_pred = knn.predict(X)
    
    M = len(y)
    correct_pred = len([i for i in range(len(y)) if y[i] == y_pred[i]])
    accuracy = correct_pred / M

    return accuracy

def find_optimal_k(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, shuffle = True, random_state = 0)
    train_accs = []
    test_accs = []
    k_range = [i for i in range(1, 50 + 1)]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)

        train_acc = test(X_train, y_train, knn) * 100
        test_acc = test(X_test, y_test, knn) * 100

        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot train and test accuracy against k
    plt.plot(k_range, train_accs, 'o-')
    plt.plot(k_range, test_accs, 'o-')
    plt.legend(['Train', 'Test'])
    plt.ylabel('Accuracy(%)')
    plt.xlabel('k')
    #plt.title("Train and Test Accuracy with different k's")

    plt.show()



if __name__ == '__main__':
    X, y = load_data()

    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X, y)
    accuracy = test(X, y, knn)
    print(f'Accuracy: {accuracy * 100}%')

    find_optimal_k(X, y)

    # predict the class of [5.0, 4.1, 3.8, 1.2]
    knn = KNeighborsClassifier(n_neighbors = 9)
    knn.fit(X, y)
    #print(test(X, y, knn))
    flower = [[5.0, 4.1, 3.8, 1.2]]
    idx = knn.predict(flower)[0]
    print(f'Predicted class id: {idx}')