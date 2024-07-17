import numpy as np
from tqdm import tqdm


class HopfieldNN:
    def __init__(self, iterations=100, threshold=0, data_shape=(28, 28)):
        self.iterations = iterations
        self.threshold = threshold
        self.data_shape = data_shape
        self.neurons = 0
        self.w = None

    def fit(self, train_data):
        print("Training weights...")
        len_data = len(train_data)
        self.neurons = train_data[0].shape[0]

        # Initialize weights
        w = np.zeros((self.neurons, self.neurons))
        rho = np.sum([np.sum(t) for t in train_data]) / (len_data * self.neurons)

        # Hebb rule
        for i in tqdm(range(len_data)):
            t = train_data[i] - rho
            w += np.outer(t, t)

        # Make diagonal element of w into 0
        w = w - np.diag(np.diag(w))
        w /= len_data

        self.w = w

    def predict(self, test_data):
        print("Predicting...")

        # Copy to avoid call by reference
        copied_data = np.copy(test_data)

        # Define predict list
        result = []
        for i in tqdm(range(len(test_data))):
            result.append(self._run(copied_data[i]))
        return result

    def _run(self, init_s):
        # Compute initial state energy
        s = init_s
        e = self.energy(s)

        print("Нейроны: ", self.neurons)

        iters = 0
        for i in range(self.iterations):
            iters += 1
            for j in range(self.neurons):
                # Update s
                s[j] = np.sign(self.w[j].T @ s - self.threshold)

            # Compute new state energy
            e_new = self.energy(s)

            # s is converged
            if e == e_new:
                return s
            # Update energy
            e = e_new
        return s

    def energy(self, s):
        return -0.5 * s @ self.w @ s + np.sum(s * self.threshold)
