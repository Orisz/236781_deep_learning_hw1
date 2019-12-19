import abc
import torch

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        classes_scores = x_scores[range(x_scores.shape[0]), y]
        classes_scores = classes_scores.reshape(-1, 1)
        my_loss = x_scores + self.delta
        my_loss = my_loss - classes_scores
        my_loss[range(x_scores.shape[0]), y] -= self.delta
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        #save values from forward pass so we can update the weights with them
        self.grad_ctx["curr_x"] = x
        self.grad_ctx["true_y"] = y
        self.grad_ctx["cur_loss_matrix"] = my_loss
        # ========================
        

        my_total_loss = torch.sum(torch.clamp(my_loss, min=0))/len(my_loss)
        return my_total_loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================
#         print(self.grad_ctx["cur_loss_matrix"].shape)
        cur_loss_matrix = self.grad_ctx["cur_loss_matrix"]
        curr_x = self.grad_ctx["curr_x"]
        curr_y = self.grad_ctx["true_y"]
        N = curr_x.shape[0]#num of examples
        G = torch.zeros(cur_loss_matrix.shape)
        #use ther indicator function W.R.T to the elements of 'cur_loss_matrix' i.e. m(i,j)
        idxs = cur_loss_matrix > 0
        #for j != yi (the multipication with xi will come later)
        G[idxs] = 1
        #for j == yi the grad is just minus the sum of all indicator idx for this class
        sum_for_j_equal_yi = torch.sum(G, dim=1)
        G[range(N), curr_y] = -sum_for_j_equal_yi
        # now calc the gradient matrix(of dim (D+1)*(C)) according to the Hint above:
        gradient = torch.mm(curr_x.T, G)
        #now normalize the gradient W.R.T the num of samples N according to the notebook
        gradient = gradient / N

        return gradient
