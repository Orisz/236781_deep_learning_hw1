import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        self.weights = torch.empty(self.n_features, self.n_classes)
        torch.nn.init.normal_(self.weights, mean=0, std=weight_std)
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.
        y_pred, class_scores = None, None
        class_scores = torch.mm(x, self.weights)
        _, y_pred = torch.max(input = class_scores, dim = 1)




        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================
        binary = (y == y_pred)
        sum = binary.sum().item()
        len = y.shape[0]
        acc = sum / len
        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
#             raise NotImplementedError()
            avg_accuracy=0.0
            #train
            for count, train_data in enumerate(dl_train):
                #forward pass
                x_train = train_data[0]
                y_train = train_data[1]
                y_pred, x_scores = self.predict(x_train)
                loss = loss_fn(x_train,y_train,x_scores,y_pred)#note loss is a number per batch 
                gradient = loss_fn.grad()
                regularization_term = weight_decay * self.weights#add regularization term of the derivative
                #backward
                self.weights -= learn_rate * (gradient + regularization_term)
                full_loss_term = loss + 0.5 * weight_decay * (torch.norm(self.weights)**2)
                average_loss += full_loss_term
                avg_accuracy += self.evaluate_accuracy(y_train,y_pred)
#                 print(f'acc: {self.evaluate_accuracy(y_train,y_pred)}')
                
#             print(f'count:{count+1} cum_acc:{avg_accuracy} avg_acc:{avg_accuracy / count+1}')
            train_res.accuracy.append(avg_accuracy / float(count+1))
            train_res.loss.append(average_loss.item() / float(count+1))
            
            average_loss_valid=0.0
            avg_accuracy_valid=0.0
            #validation
            for count_valid, vailid_data in enumerate(dl_valid):
                x_valid = vailid_data[0]
                y_valid = vailid_data[1]
                y_pred_valid, x_scores_valid = self.predict(x_valid)
                loss_valid = loss_fn(x_valid,y_valid,x_scores_valid,y_pred_valid)#note loss is a number per batch
                full_loss_term = loss_valid + 0.5 * weight_decay * (torch.norm(self.weights)**2)
                average_loss_valid += full_loss_term
                avg_accuracy_valid += self.evaluate_accuracy(y_valid,y_pred_valid)
#                 print(f'acc: {self.evaluate_accuracy(y_valid,y_pred_valid)}')
            
#             print(f'count_valid:{float(count_valid+1)} cum_acc:{avg_accuracy_valid} avg_acc:{avg_accuracy_valid / float(count_valid+1)}')
            valid_res.accuracy.append(avg_accuracy_valid / float(count_valid+1))
            valid_res.loss.append(average_loss_valid.item() / float(count_valid+1))
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================
        w = self.weights
        if has_bias:
            w = w[0:len(w) - 1]
        w_images =  w.reshape(self.n_classes, *img_shape)
        return w_images


def hyperparams():
    hp = dict(weight_std=0.005, learn_rate=0.05, weight_decay=0.006)
    return hp


    """weight_std = [i*0.005 for i in range(3)]
    learn_rate = [i*0.01 for i in range(9)]
    weight_decay = [i*0.001 for i in range(7)]


    smallest_loss = None
    #best_params = {"bostonfeaturestransformer__degree": 1, "linearregressor__reg_lambda": 0.2}
    count = 0

    for std in weight_std:
        for lr in learn_rate:
            for dec in weight_decay:
                modle = LinearClassifier(weight_std= std, learn_rate = lr, weight_decay= dec)
                modle.fit(x_train, y_train)
                y_pred = model.predict(X[test_i])
                avg_mse += np.square(y[test_i] - y_pred).sum() / (2 * X.shape[0])
    """
    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
"""

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
"""
