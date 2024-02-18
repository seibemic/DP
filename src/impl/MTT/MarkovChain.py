import numpy as np

class MarkovChain:
    def __init__(self, initial_distribution):
        self.initial_distribution = initial_distribution
        self.resultMatrix = np.identity(len(self.initial_distribution))
    def transition_matrix(self, pd, pk):
        return np.array([[pd, 1 - pd , 0     ],
                         [pd, pk - pd, pd - pk],
                         [pd, pk - pd, pd - pk]])

    def get_transitionProbs(self, pd, pk):
        self.resultMatrix = self.resultMatrix @ self.transition_matrix(pd, pk)
        return self.initial_distribution @ self.resultMatrix

    def get_visibleProb(self):
        return (self.initial_distribution @ self.resultMatrix)[0]

    def get_hiddenProb(self):
        return (self.initial_distribution @ self.resultMatrix)[1]

    def get_deadProb(self):
        return (self.initial_distribution @ self.resultMatrix)[2]
