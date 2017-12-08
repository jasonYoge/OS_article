import numpy as np
import numpy.random as rng

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, Xtrain, Xval):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes, self.n_examples, self.W, self.h = Xtrain.shape
        self.n_val, self.n_ex_val, _, _ = Xval.shape


    def get_batch(self, n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes, size=(n,), replace=False)
        pairs = [np.zeros((n, self.h, self.w, 3)) for i in range(2)]
        targets = np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0, self.n_examples)
            pairs[0][i, :, :, :] = self.Xtrain[category, idx_1].reshape(self.w, self.h, 3)
            idx_2 = rng.randint(0, self.n_examples)
            category_2 = category if i > n // 2 else (category + rng.randint(1, self.n_classes)) % self.n_classes
            pairs[1][i, :, :, :] = self.Xtrain[category_2, idx_2].reshape(self.w, self.h, 3)
        return pairs, targets


    def make_oneshot_task(self, N):
        """Create pairs of test image, support set for testing N way one-shot"""
