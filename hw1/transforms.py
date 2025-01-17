import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # TODO: Use Tensor.view() to implement the transform.
        return tensor.view(*self.view_dims)


class InvertColors(object):
    """
    Inverts colors in an image given as a tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) representing an image.
        :return: The image with inverted colors.
        """
        # TODO: Invert the colors of the input image.
        #x.__invert__()
        #if (x.dtype != NoneType):
        #return x.bitwise_not()
        #return x.neg()
        return 1.0 - x
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================


class FlipUpDown(object):
    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) representing an image.
        :return: The image, flipped around the horizontal axis.
        """
        # TODO: Flip the input image so that up is down.
        #if (x != None):

        return torch.flip(x, [1])
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Prepends an element equal to
    1 to each sample in a given tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A pytorch tensor of shape (D,) or (N1,...Nk, D).
        We assume D is the number of features and the N's are extra
        dimensions. E.g. shape (N,D) for N samples of D features;
        shape (D,) or (1, D) for one sample of D features.
        :return: A tensor with D+1 features, where a '1' was prepended to
        each sample's feature dimension.
        """
        assert x.dim() > 0, "Scalars not supported"
 
        orig_data_type = x.dtype
        x_tensor = x.type(torch.FloatTensor)
        # TODO:
        #  Add a 1 at the beginning of the given tensor's feature dimension.
        #  Hint: See torch.cat().
        x_shape = x_tensor.shape
        ones_to_append = torch.ones(x_shape[:-1])#leave out the last dim
        ones_to_append.unsqueeze_(len(x_shape)-1)
        res = torch.cat((ones_to_append, x_tensor), dim=(len(x_shape)-1))
        return res.type(orig_data_type)