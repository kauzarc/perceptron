import numpy as np


class Perceptron:
    """
    N: number of data sample
    D: number of dimension or feature of the data

    data:
        x1, x2, ..., xN: N vector of R_D
        y1, y2, ..., yN: N value in {-1, 1}

        matrix form:
        X: N,D matrix = xi are the columns
        Y: N,1 matrix = yi are the columns

    model:
        parameters: (b, w) = (b, w1, w2, ..., wD)
                    = w' = (w'1, ..., w'D, w'D+1)

        f: R_D -> R
        f(x) = b + w1 * x1 + w2 * x2 + ... + wD * xD
                    = b + w * x
                    = w' * x' with x' = (1, x)

        x trics:
        we will use D <- D+1, x <- x', w <- w'        

        matrix form:
        W: D,1 matrix

        f:M_N,D -> M_N,1
        f(X) = X * W

    predict:
        y_pred: prediction of the model
        sign(x) = 1 if x > 0 else -1

        y_pred = sign(f(x))
                = sign(w * x)

        matrix form:
        Y_pred: N,1 matrix
        sign((x1, ..., xd)) = (sign(x1), ..., sign(xd))

        Y_pred = sign(X * W)

    loss function: (or error function) The function we try to minimize
        relu(x) = x if x > 0 else 0

        for 1 data sample:
        loss: R_D -> R+
        loss(w) = relu(-f(x) * y)
                    = relu(-w * x * y)

        loss: R_D -> R+
        loss(w) = 1 / N * sum(n)(relu(-f(xn) * yn))
                    = 1 / N * sum(n)(relu(-w * xn * yn))

        numpy matrix form: (@: matrix mul, *: term by term mul)
        loss(W): M_D,1 -> R+
        result = -X @ self.W * Y
        loss(W) = (result * (result > 0)).mean()

    optimisation algorithm:
        gradient descent:
        W' <- W - eta * gradient(loss(W))

        eta: hyper parameter, size of the gradient descent
        eta too small, slow convergence, get locked in local minimum
        eta too big, no convergence

    loss gradient:
        Gloss: R_D -> R_D 
        Gloss(w) = 1 / N * (sum(n tq -wn * w * yn > 0)(-xn * yn) + sum(n tq -xn * w * yn < 0)(0))
                = 1 / N * sum(n tq -xn * w * yn > 0)(-xn * yn)

        numpy matrix form: (@: matrix mul, *: term by term mul)
        mask = -X @ W * Y > 0
        Gloss(W) = -1 / N * X[mask].T @ Y[mask] 
    """

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
        Learn the W parameters by minimising the loss function
        using a gradient descent
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

        sign(x) = 1 if x > 0 else -1
        sign((x1, ..., xd)) = (sign(x1), ..., sign(xd))

        Y_predict = sign(X * W)
        """
        result = np.sign(self._X(X_raw) @ self.W)
        return np.array([self.conversion[0] if x == 1 else self.conversion[1] for x in result])

    def score(self, X_raw, Y_raw):
        """
        Prediction accuracy between 0 (bad) and 1 (good)
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

        1 / N * sum(i)(relu(-X_i * W * Y_i))
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
