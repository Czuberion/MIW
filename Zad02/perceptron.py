import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from sklearn.preprocessing import StandardScaler


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            #errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiClassPerceptron:
    def __init__(self, n_classes, eta=0.01, n_iter=10):
        self.n_classes = n_classes
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers = [Perceptron(eta=self.eta, n_iter=self.n_iter) for _ in range(n_classes)]

    def fit(self, X, y):
        for idx, clf in enumerate(self.classifiers):
            # Konwersja etykiet wieloklasowych dla klasyfikacji OvR
            binary_labels = np.where(y == idx, 1, -1)
            clf.fit(X, binary_labels)

    def predict(self, X):
        # Zebranie predykcji z każdego klasyfikatora
        predictions = np.array([clf.net_input(X) for clf in self.classifiers]).T
        # Wybór klasy z najwyższą wartością z wyników klasyfikatora
        return np.argmax(predictions, axis=1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    mcp = MultiClassPerceptron(n_classes=3, eta=0.01, n_iter=50)
    mcp.fit(X_train_std, y_train)

    plot_decision_regions(X_test_std, y_test, classifier=mcp)
    plt.title('MultiClassPerceptron - Iris Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
