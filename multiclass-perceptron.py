import numpy as np
from sklearn.utils.extmath import softmax


class MultiClassPerceptron:
    def __init__(self, learning_rate=0.01, max_iter=100, fit_intercept=True, verbose=False, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state

        self.W = None
        self.conversion = None

    def fit(self, X_raw, Y_raw):
        X = self._X(X_raw)
        Y = self._Y(Y_raw)

        if self.W is None:
            self.W = self._init_W(Y.shape[1], X.shape[1])

        for iter in range(self.max_iter):
            self.W = self._epoch(X, Y)

            if self.verbose:
                print(f"iter {iter}: {self._loss(X, Y)}")

    def predict(self, X_raw):
        return np.array([self.conversion[idx] for idx in self._proba(self._X(X_raw)).argmax(axis=1)])

    def score(self, X_raw, Y_raw):
        return (self.predict(X_raw) == Y_raw).mean()

    def _epoch(self, X, Y):
        return self.W - self.learning_rate * self._gradient(X, Y)

    def _proba(self, X):
        return softmax(X @ self.W.T)

    def _loss(self, X, Y):
        return -1 / X.shape[0] * (Y * np.log(self._proba(X))).sum()

    def _gradient(self, X, Y):
        return -1 / X.shape[0] * (Y - self._proba(X)).T @ X

    def _init_W(self, K, D):
        np.random.seed(self.random_state)
        return np.random.normal(0, 1, (K, D))

    def _X(self, X_raw):
        if self.fit_intercept:
            return np.hstack((np.ones((X_raw.shape[0], 1)), X_raw))
        else:
            return X_raw

    def _Y(self, Y_raw):
        if self.conversion is None:
            self.conversion = np.unique(Y_raw)

        return np.array([self.conversion == y for y in Y_raw], float)


if __name__ == "__main__":
    from sklearn import datasets, model_selection

    digits = datasets.load_digits()
    X = digits["data"]
    Y = digits["target"]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, random_state=0)

    print(f"train shape: X:{X_train.shape}, Y:{Y_train.shape}")
    print(f"test shape: X:{X_test.shape}, Y:{Y_test.shape}")

    model = MultiClassPerceptron(
        learning_rate=0.05,
        max_iter=200,
        verbose=False,
        random_state=0,
    )
    model.fit(X_train, Y_train)

    print(f"train score: {model.score(X_train, Y_train)}")
    print(f"test score: {model.score(X_test, Y_test)}")
