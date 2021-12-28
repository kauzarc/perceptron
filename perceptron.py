import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=100, fit_intercept=True, verbose=False, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state

        self.W = None
        self.conversion = None

    def fit(self, X_raw, Y_raw):
        """
        Learn the W parameters
        """
        X = self._X(X_raw)
        Y = self._Y(Y_raw)

        if self.W == None:
            self._init_W(X.shape[1])

        for iter in range(self.max_iter):
            self.W = self._epoch(X, Y)

            if self.verbose:
                print(f"iter {iter}: {self._loss(X, Y)}")

    def predict(self, X_raw):
        """
        Try to predict the coresponding Y

        X: N,D matrix
        W: D,1 matrix

        Y_hat = sign(X * W)
        """
        result = np.sign(self._X(X_raw) @ self.W)
        return np.array([self.conversion[0] if x == 1 else self.conversion[1] for x in result])

    def score(self, X_raw, Y_raw):
        """
        Prediction rate between 0 (bad) and 1 (good)
        """
        return (self.predict(X_raw) == Y_raw).mean()

    ## PRIVATE ##

    def _loss(self, X, Y):
        """
        The function we try to minimize

        X: N,D matrix
        W: D,1 matrix
        Y: N,1 matrix

        relu(x) = x if x > 0 else 0
        relu((x1, ..., xd)) = (relu(x1), ..., relu(xd))

        1 / N * sum(relu(-X * W * Y))
        """
        result = -X @ self.W * Y
        return (result * (result > 0)).mean()

    def _epoch(self, X, Y):
        """
        One gradient descent step
        """
        return self.W - self.learning_rate * self._gradient(X, Y)

    def _gradient(self, X, Y):
        """
        loss gradient

        X: N,D matrix
        W: D,1 matrix
        Y: N,1 matrix

        1 / N * sum(i tq -X_i * W * Y_i > 0)(-X_i * Y_i)
        """
        mask = -X @ self.W * Y > 0
        return -1 / X.shape[0] * X[mask].T @ Y[mask]

    def _init_W(self, D):
        np.random.seed(self.random_state)
        self.W = np.random.normal(0, 1, D)

    def _X(self, X_raw):
        """
        x' trics

        x: D vector
        w: D vector
        b: scalar

        x': D+1 vector = (1, x)
        w': D+1 vector = (b, w)

        w * x + b = w' * x'
        """
        if self.fit_intercept:
            return np.hstack((np.ones((X_raw.shape[0], 1)), X_raw))
        else:
            return X_raw

    def _Y(self, Y_raw):
        """
        y must be in {-1, 1}
        """
        if self.conversion == None:
            self.conversion = np.unique(Y_raw)

        return (Y_raw == self.conversion[0]) * 2 - 1


# Test #
if __name__ == "__main__":
    random_state = 0  # For randomness reproducibility

    # import sklearn
    from sklearn import datasets, model_selection

    # Load data
    digits = datasets.load_digits()
    X = digits["data"]
    Y = digits["target"]

    # Split between train and test sets
    mask_0 = Y == 0
    mask_1 = Y == 1
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X[mask_0 | mask_1], Y[mask_0 | mask_1], random_state=random_state)

    # Shape of the data
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # Training
    model = Perceptron(random_state=random_state)
    model.fit(X_train, Y_train)

    # Score
    print(f"train score: {model.score(X_train, Y_train)}")
    print(f"test score: {model.score(X_test, Y_test)}")
