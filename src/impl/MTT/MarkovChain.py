import numpy as np


class MarkovChain:
    def __init__(self, initial_distribution, resultMatrix=None):
        self.initial_distribution = initial_distribution
        if resultMatrix is None:
            self.resultMatrix = np.identity(len(self.initial_distribution))
        else:
            self.resultMatrix = resultMatrix
    def transition_matrix(self, pd, pk):
        Th = 0.5#2#3.2
        Td = 2#3
        return np.array([[pd**Td, 1 - pd**Td, 0],
                         [pd, (1-pk**Th) * (1 - pd), pk**Th * (1 - pd)],
                         [pd, (1-pk**Th) * (1 - pd), pk**Th * (1 - pd)]])
        return np.array([[pd, 1 - pd, 0],
                         [pd, pk*(1-pd), (1-pk)*(1-pd)],
                         [pd, pk*(1-pd), (1-pk)*(1-pd)]])


        return np.array([[pd, 1 - pd           , 0     ],
                         [pd, 1 - pk, pk - pd],
                         [pd, 1 - (pk +pd), pk]])

    def get_transitionProbs(self, pd, pk):
        self.resultMatrix = self.resultMatrix @ self.transition_matrix(pd, pk)
        return self.initial_distribution @ self.resultMatrix

    def get_probs(self):
        return self.initial_distribution @ self.resultMatrix
    def get_visibleProb(self):
        return (self.initial_distribution @ self.resultMatrix)[0]

    def get_hiddenProb(self):
        return (self.initial_distribution @ self.resultMatrix)[1]

    def get_deadProb(self):
        return (self.initial_distribution @ self.resultMatrix)[2]
