import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================
        y_pred = np.matmul(X, self.weights_)

        return y_pred


    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution
        #  Use only numpy functions. Don't forget regularization.

        w_opt = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================
        # check what it sais on the notebook

        N = X.shape[0] * 1.0
        bias = np.identity(X.shape[1])
        bias[0][0] = 0
        # the close form solution: calc - inv(X + N * reg_lambda * bias) * X*y
        bias = N * self.reg_lambda * bias
        inv_res = np.linalg.inv(X.T.dot(X) + bias)
        w_opt = inv_res.dot(X.T.dot(y))

        self.weights_ = w_opt
        return self


    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None

        bias = np.ones((len(X), 1), dtype=np.float32)
        xb = np.hstack((bias, X))
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        X[:, 13] = 1.0 / X[:, 13]  # lstat
        X_transformed = PolynomialFeatures(self.degree).fit_transform(X)

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
        """
        Returns the names of features most strongly correlated (correlation is
        close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

        :param df: A pandas dataframe.
        :param target_feature: The name of the target feature.
        :param n: Number of top features to return.
        :return: A tuple of
            - top_n_features: Sequence of the top feature names
            - top_n_corr: Sequence of correlation coefficients of above features
            Both the returned sequences should be sorted so that the best (most
            correlated) feature is first.
        """

        # TODO: Calculate correlations with target and sort features by it

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        top_n = df.corr()[target_feature].drop([target_feature]).abs().nlargest(n)
        top_n_features = list(top_n.index)
        top_n_corr = list(top_n.values)


        return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
        """
        Computes Mean Squared Error.
        :param y: Predictions, shape (N,)
        :param y_pred: Ground truth labels, shape (N,)
        :return: MSE score.
        """

        # TODO: Implement MSE using numpy.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

        temp = y - y_pred
        mse = (np.square(temp)).mean(axis=0)
        return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
        """
        Computes R^2 score,
        :param y: Predictions, shape (N,)
        :param y_pred: Ground truth labels, shape (N,)
        :return: R^2 score.
        """

        # TODO: Implement R^2 using numpy.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
        #ssreg = np.sum((y_pred - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        ssres = np.sum((y - y_pred) ** 2)
        r2 = (ssres / sstot)
        return 1 - r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                            degree_range, lambda_range):
        """
        Cross-validate to find best hyperparameters with k-fold CV.
        :param X: Training data.
        :param y: Training targets.
        :param model: sklearn model.
        :param lambda_range: Range of values for the regularization hyperparam.
        :param degree_range: Range of values for the degree hyperparam.
        :param k_folds: Number of folds for splitting the training data into.
        :return: A dict containing the best model parameters,
            with some of the keys as returned by model.get_params()
        """

        # TODO: Do K-fold cross validation to find the best hyperparameters
        #  Notes:
        #  - You can implement it yourself or use the built in sklearn utilities
        #    (recommended). See the docs for the sklearn.model_selection package
        #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        #  - If your model has more hyperparameters (not just lambda and degree)
        #    you should add them to the search.
        #  - Use get_params() on your model to see what hyperparameters is has
        #    and their names. The parameters dict you return should use the same
        #    names as keys.
        #  - You can use MSE or R^2 as a score.

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        kf = sklearn.model_selection.KFold(k_folds)
        smallest_loss = np.inf
        best_params = {"bostonfeaturestransformer__degree": 1, "linearregressor__reg_lambda": 0.2}
        count = 0


        for lam in lambda_range:
            for deg in degree_range:
                model.set_params(linearregressor__reg_lambda=lam, bostonfeaturestransformer__degree=deg)
                avg_mse = 0.0
                count += 1

                for train_i, test_i in kf.split(X):
                    x_train = X[train_i]
                    y_train = y[train_i]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(X[test_i])
                    avg_mse += np.square(y[test_i] - y_pred).sum() / (2 * X.shape[0])

                avg_mse /= k_folds

                #check if the current params are the best
                if avg_mse <= smallest_loss:
                    smallest_loss = avg_mse
                    best_params = {"linearregressor__reg_lambda": lam, "bostonfeaturestransformer__degree": deg}
                    # ========================
        print(count)
        return best_params
