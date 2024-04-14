import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions
from sklearn.preprocessing import StandardScaler


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1, C=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            # Regularyzacja
            self.w_[1:] += self.eta * (X.T.dot(errors) - (1 / self.C) * self.w_[1:])
            self.w_[0] += self.eta * errors.sum()

            # cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) + (1 / (2 * self.C)) * np.sum(self.w_[1:] ** 2)
            # self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_classes, eta=0.05, n_iter=100):
        self.n_classes = n_classes
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers = [LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter) for _ in range(n_classes)]

    def fit(self, X, y):
        for idx, clf in enumerate(self.classifiers):
            binary_labels = np.where(y == idx, 1, 0)
            clf.fit(X, binary_labels)

    def predict_prob(self, X):
        # Zebranie wyniku funkcji logistycznej (prawdopodobieństwa) z każdego klasyfikatora
        probs = np.array([clf.predict(X) for clf in self.classifiers]).T
        # Zastosowanie softmax do znormalizowania tych wyników do rozkładu prawdopodobieństwa
        return softmax(probs)

    def predict(self, X):
        # Uzyskanie prawdopodobieństw dla każdej klasy i wybranie klasy o najwyższym prawdopodobieństwie
        prob = self.predict_prob(X)
        return np.argmax(prob, axis=1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [0, 1]]
    # X = iris.data[:, [0, 2]]
    # X = iris.data[:, [0, 3]]
    # X = iris.data[:, [1, 2]]
    # X = iris.data[:, [1, 3]]
    # X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Standaryzacja
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # Trenowanie
    sr = SoftmaxRegression(n_classes=3, eta=0.05, n_iter=1000)
    sr.fit(X_train_std, y_train)

    plot_decision_regions(X_test_std, y_test, classifier=sr)
    plt.title('SoftmaxRegression - Iris Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
