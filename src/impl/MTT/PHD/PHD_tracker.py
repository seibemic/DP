from src.impl.MTT import TargetTracker
from src.impl.MTT.PHD.PHD import PHD
from scipy.stats import multivariate_normal as mvn
import numpy as np
from src.impl.MTT.ObjectStats import ObjectStats
from copy import deepcopy
from src.impl.MTT.MarkovChain import MarkovChain
class PHDTracker(TargetTracker):
    def __init__(self, F, H, Q, R, ps):
        super().__init__(F, H, Q, R, ps)
        self.T = 10e-5
        self.U = 8 #4
        self.J_max = 40

    def predictBirthTargets(self):
        for sp in self.spawnPoints:
            w = sp.w
            m = sp.m
            m = np.append(m, [0, 0])
            P = sp.cov
            pd = sp.w
            self.trackers.append(PHD(w, m, P, pd))

    def predictExistingTargets(self):
        for target in self.trackers:
            target.predict(self.ps, self.F, self.Q.copy(), self.imageSize)

    def updateComponents(self):
        for target in self.trackers:
            target.updateComponents(self.H, self.R)

    def predict(self):
        self.predictBirthTargets()
        self.predictExistingTargets()

    def update(self, z, xyxy, masks, frame, frame_num,lambd=0.00001):
        Jk = len(self.trackers)
        self.updateComponents()
        measured = np.zeros(shape=(len(self.trackers)))
        for j in range(Jk):
            self.trackers[j].moveMask_and_getPd(frame)
        for l, z in enumerate(z):
            phds_sum = 0
            gatings = 0
            start_index = len(self.trackers)
            for j in range(Jk):
                if self.trackers[j].inGating(z):
                    measured[j] = 1
                    pd = self.trackers[j].pd
                    w = pd * self.trackers[j].w * mvn(self.trackers[j].ny, self.trackers[j].S).pdf(z)
                    m = self.trackers[j].m + self.trackers[j].K @ (z - self.trackers[j].ny)
                    P = self.trackers[j].P
                    phds_sum += w
                    objectstats = ObjectStats(frame, masks[l].copy(), xyxy[l], frame_num)

                    pd = objectstats.get_maskStatsMean(frame, self.trackers[j].mask)
                    markov = deepcopy(self.trackers[j].markovChain)
                    try:
                        hist1=objectstats.maskValues
                        hist2=self.trackers[j].objectStats.maskValues
                        all_vals = []
                        all_vals.append(objectstats.get_cosineSimilarity(hist1,hist2))
                        all_vals = np.array(all_vals)
                        pd = np.mean(all_vals)
                    except:
                        pass
                    self.trackers.append(PHD(w, m, P, pd, xyxy[l], masks[l].copy(), objectstats, markov))
                    gatings += 1
            for j in range(gatings):
                self.trackers[start_index + j].w = self.trackers[start_index + j].w / (lambd + phds_sum)

        for j in range(Jk):
            if measured[j]:
                self.trackers[j].update(frame=frame, frame_num=frame_num)
            else:
                self.trackers[j].update(frame=frame, frame_num=frame_num)


    def pruneByMaxWeight(self, w):
        filters_to_stay = []
        for filter in self.trackers:
            if filter.w > w or (filter.state != 2 and filter.w > w*0.1):
                filters_to_stay.append(filter)
        self.trackers = filters_to_stay

    def argMaxW(self, filters):
        maX = 0
        argmaX = 0
        for i, filter in enumerate(filters):
            if filter.w > maX:
                maX = filter.w
                argmaX = i
        return argmaX



    def mergeTargets(self):
        filters_to_stay = []
        mixed_filters = []
        for filter in self.trackers:
            if filter.w > self.T:
                filters_to_stay.append(filter)

        while len(filters_to_stay) != 0:
            j = self.argMaxW(filters_to_stay)
            L = []  # indexes
            for i in range(len(filters_to_stay)):
                if ((filters_to_stay[i].m - filters_to_stay[j].m).T @
                    np.linalg.inv(filters_to_stay[i].P) @ (filters_to_stay[i].m - filters_to_stay[j].m)) < self.U:
                    L.append(i)
            # print(len(L))
            w_mix = 0
            for t_id in L:
                w_mix += filters_to_stay[t_id].w
            m_mix = np.zeros(4)


            for t_id in L:
                m_mix += filters_to_stay[t_id].w * filters_to_stay[t_id].m
            m_mix /= w_mix
            P_mix = np.zeros_like(filters_to_stay[0].P, dtype="float64")
            for t_id in L:
                P_mix += filters_to_stay[t_id].w * (
                        filters_to_stay[t_id].P + np.outer((m_mix - filters_to_stay[t_id].m),
                                                           (m_mix - filters_to_stay[t_id].m).T))
            P_mix /= w_mix

            """Mask, xyxy"""
            xyxy_mix = np.zeros(shape=(4))
            pd_mix = 0
            max_shape = (0, 0)
            for f in filters_to_stay:
                if f.mask is not None and f.mask.shape[0] > max_shape[0] and f.mask.shape[1] > max_shape[1]:
                    max_shape = f.mask.shape
                    break
            mask_mix = np.zeros(shape=max_shape, dtype=np.float64)
            for t_id in L:
                if filters_to_stay[t_id].xyxy is not None:
                    xyxy_mix += filters_to_stay[t_id].xyxy * filters_to_stay[t_id].w
                pd_mix += filters_to_stay[t_id].pd * filters_to_stay[t_id].w
                if filters_to_stay[t_id].mask is not None:
                    mask_mix += filters_to_stay[t_id].mask.astype(np.float64) * filters_to_stay[t_id].w

            pd_mix /= w_mix
            xyxy_mix /= w_mix

            mask_mix /= w_mix
            mask_mix = np.round(mask_mix)
            mask_mix = np.clip(mask_mix, 0, 1)
            mask_mix = np.array(mask_mix, dtype=np.int8)


            """MARKOV"""
            result_matrix = np.zeros_like(filters_to_stay[0].markovChain.resultMatrix)
            for t_id in L:
                init_distr = filters_to_stay[t_id].markovChain.initial_distribution
                result_matrix += filters_to_stay[t_id].markovChain.resultMatrix * filters_to_stay[t_id].w
            result_matrix /= w_mix
            markov = MarkovChain(init_distr, result_matrix)


            "Object Stats"
            OS_max_timestamp = 0
            OS_frame = None
            OS_mask = None
            OS_w_mix = 0
            OS_xyxy = np.zeros(shape=(4))
            for t_id in L:
                OS_mask = np.zeros_like(filters_to_stay[t_id].objectStats.mask)
                if filters_to_stay[t_id].objectStats is not None and filters_to_stay[t_id].objectStats.timestamp > OS_max_timestamp:
                    OS_max_timestamp = filters_to_stay[t_id].objectStats.timestamp
                    OS_frame = filters_to_stay[t_id].objectStats.frame.copy()
                if filters_to_stay[t_id].objectStats.mask is not None:
                    OS_mask = filters_to_stay[t_id].objectStats.mask.astype(np.float64) * filters_to_stay[t_id].w
                    OS_w_mix += filters_to_stay[t_id].w
                if filters_to_stay[t_id].xyxy is not None:
                    OS_xyxy += filters_to_stay[t_id].objectStats.xyxy.astype(np.float64) * filters_to_stay[t_id].w

            OS_mask /= OS_w_mix
            OS_mask = np.round(OS_mask)
            OS_mask = np.clip(OS_mask, 0, 1)
            OS_mask = np.array(OS_mask, dtype=np.int8)

            OS_xyxy /= OS_w_mix
            OS = ObjectStats(OS_frame, OS_mask, OS_xyxy, OS_max_timestamp)


            mixed_filters.append(PHD(w_mix, m_mix, P_mix, pd_mix, xyxy_mix, mask_mix, OS, markov, fromMerge=True))

            removed = np.delete(filters_to_stay, L)
            filters_to_stay = removed.tolist()

        if len(mixed_filters) > self.J_max:
            self.trackers = mixed_filters
            self.pruneByMaxWeight(0.1)
        else:
            self.trackers = mixed_filters
