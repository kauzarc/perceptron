import numpy as np


class SVM:
    def __init__(self, max_iter=50,  fit_intercept=True, verbose=False):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self.W = None
        self.conversion = None

    def fit(self, X_raw, Y_raw):
        X = self._X(X_raw)
        Y = self._Y(Y_raw)

        Q = -Y @ X @ X.T @ Y
        dual_vars = np.zeros(X.shape[0])

        for iter in range(self.max_iter):
            for k in range(X.shape[0]):
                dual_vars[k] = np.clip(
                    (1 - dual_vars @ Q[k] + dual_vars[k] * Q[k, k]) / Q[k, k],
                    -1,
                    0
                )

            if self.verbose:
                print(
                    f"iter {iter}: {1 / 2 * dual_vars.T @ Q @ dual_vars - dual_vars.sum()}")

        self.W = - X.T @ Y @ dual_vars

    def predict(self, X_raw):
        result = np.sign(self._X(X_raw) @ self.W)
        return np.array([self.conversion[0] if x == 1 else self.conversion[1] for x in result])

    def score(self, X_raw, Y_raw):
        return (self.predict(X_raw) == Y_raw).mean()

    def _X(self, X_raw):
        if self.fit_intercept:
            return np.hstack((np.ones((X_raw.shape[0], 1)), X_raw))
        else:
            return X_raw

    def _Y(self, Y_raw):
        if self.conversion == None:
            self.conversion = np.unique(Y_raw)

        return np.diag((Y_raw == self.conversion[0]) * 2 - 1)


if __name__ == "__main__":
    from sklearn import datasets, model_selection, pipeline

    X, y = datasets.make_classification(
        n_samples=2000,
        n_features=20,
        n_classes=2,
        class_sep=1.0
    )
    print(f"X: {X.shape}, y: {y.shape}")

    scores = model_selection.cross_val_score(
        pipeline.make_pipeline(SVM()),
        X,
        y,
        cv=10,
        n_jobs=-1
    )
    print(f"score: mean={scores.mean()}, std={scores.std()}")
