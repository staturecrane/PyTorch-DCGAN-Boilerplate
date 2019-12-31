import numpy as np
from scipy.stats import truncnorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1.0, seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(
        np.float32
    )
    return truncation * values
