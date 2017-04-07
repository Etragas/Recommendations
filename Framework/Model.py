class Model:
    def __init__(self):
        self.parameters = None
        self.loss = None
        self.inference = None

    def predict(self, indices):
        """
        Conducts inference over model to generate estimate for point at indices
        :param indices: A tuple (i,j) denoting row and column
        :return: A real number denoting guessed rating
        """

        return self.infer(indices)

    def train(self, data):
        return None

    def infer(self, indices):
        return None
