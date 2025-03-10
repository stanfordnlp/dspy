import numpy as np


def approximate_mode(class_counts, n_draws, rng):
    """Computes approximate mode of multivariate hypergeometric.
    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.
    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.
    Args
    ----------
    class_counts : ndarray of int
        Population per class.
    n_draws : int
        Number of draws (samples to draw) from the overall population.
    rng : random state
        Used to break ties.
    Returns
    -------
    sampled_classes : ndarray of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws

    """
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = n_draws * class_counts / class_counts.sum()
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = np.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = np.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int64)


def stratified_shuffle_split_generate_indices(y, n_train, n_test, rng, n_splits=10):
    """

    Provides train/test indices to split data in train/test sets.
    It's reference is taken from StratifiedShuffleSplit implementation
    of scikit-learn library.

    Args
    ----------

    n_train : int,
        represents the absolute number of train samples.

    n_test : int,
        represents the absolute number of test samples.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError("Minimum class count error")
    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or " "equal to the number of classes = %d" % (n_train, n_classes)
        )
    if n_test < n_classes:
        raise ValueError(
            "The test_size = %d should be greater or " "equal to the number of classes = %d" % (n_test, n_classes)
        )
    class_indices = np.split(np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1])
    for _ in range(n_splits):
        n_i = approximate_mode(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = approximate_mode(class_counts_remaining, n_test, rng)

        train = []
        test = []

        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            train.extend(perm_indices_class_i[: n_i[i]])
            test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])
        train = rng.permutation(train)
        test = rng.permutation(test)

        yield train, test
