import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(file_name):

    df = pd.read_csv(file_name, delimiter = ',')
    data = np.array([list(row) for row in df.values])
    return data

def inertia(x):
    x_mean = np.mean(x, axis = 0)
    var = np.sum((x - x_mean) ** 2)
    return var
    

def find_optimal_k(data):
    k_range = [i for i in range(1, 15 + 1)]
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters =  k, n_init = 10)
        # it seems the nstart parameter is replaced by n_init
        kmeans.fit(data)
        labels = kmeans.predict(data)
        sum = 0
        for i in range(k):
            samples = [data[j] for j in range(len(data)) if labels[j] == i]
            sum += inertia(samples)
        #sum /= k
        inertias.append(sum)

    plt.plot(k_range, inertias, 'o-')
    plt.xlabel('k')
    plt.ylabel('Sum{(x_c-mean_c)^2}')
    #plt.title("Within-class inertias with different k's")
    plt.show()

    # the elbow point indicates optimal k = 4

def test_optimal_k(data, k = 4):
    kmeans = KMeans(n_clusters =  k, n_init = 10)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    for i in range(k):
        cnt = len([y for y in labels if y == i])
        print(f'# observation in class {i} is {cnt}')
    sum = 0
    for i in range(k):
        samples = [data[j] for j in range(len(data)) if labels[j] == i]
        sum += inertia(samples)
    #sum /= k
    print(f'inertia when k = {k}: {sum}')
    visualize_data(data, labels, k)

def visualize_data(data, labels, k):
    for i in range(k):
        idx = [j for j in range(len(data)) if labels[j] == i]
        samples = data[idx]
        plt.scatter(samples[:, 0], samples[:, 1])
    plt.legend([f'class {i}' for i in range(k)])
    plt.xlabel('first variable of the data')
    plt.ylabel('second variable of the data')
    plt.show()


if __name__ == '__main__':
    data = load_data(file_name = 'clust_data.csv')
    #print(data)
    find_optimal_k(data)
    test_optimal_k(data, k = 4)

