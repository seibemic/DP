import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from confidence_ellipse import confidence_ellipse
from PHD import PHD
from MAP2 import Radar


class PHD_map:
    def __init__(self, radar, ps, pd):
        self.radarMap = radar
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.phds = []
        self.phdsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()

        self.J_beta = 1
        self.Q_beta = self.Q
        self.F_beta = self.F
        self.w_beta = 0.2

        self.T = 10e-5
        self.U = 4
        self.J_max = 10
        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def predictionForBirthTargets(self):
        tmp_phds = self.phds.copy()
        for j in range(self.J_beta):
            for l, filter in enumerate(tmp_phds):
                w = filter.w * self.w_beta
                d = mvn([0], 3).rvs(len(filter.m))
                m = d + self.F_beta @ filter.m
                P = self.Q_beta + self.F_beta @ filter.P @ self.F_beta.T
                self.phds.append(PHD(w, m, P))
        for airport in self.radarMap.getAirports():
            w_k = airport.weight
            m_k = np.array(airport.pos)
            P_k = airport.cov
            self.phds.append(PHD(w_k, m_k, P_k))

    def predictionForExistingTargets(self):
        for target in self.phds:
            target.predict(self.ps, self.F, self.Q)
        # for i in range(len(self.phds)):
        #     self.phds[i].predict(self.ps, self.F, self.Q)

    def updateComponents(self, filters):
        for target in filters:
            target.updateComponents(self.H, self.R)
        return filters

    def updateComponents2(self):
        for target in self.phds:
            target.updateComponents(self.H, self.R)

    def update(self):
        for target in self.phds:
            target.update(self.pd)

    def getMeasurements(self, time):
        print(self.measurements[time])
        return self.measurements[time]

    def pruneByMaxWeight(self, w):
        filters_to_stay = []
        # print(" filteres before pruning: ", len(self.phds))
        for filter in self.phds:
            if filter.w > w:
                filters_to_stay.append(filter)
        print("prune len: ", len(filters_to_stay))
        self.phds = filters_to_stay
        # print("survived filteres num: ", len(filters_to_stay))

    def argMax(self, filtres):
        maX = 0
        argmaX = 0
        for i, filter in enumerate(filtres):
            if filter.w > maX:
                maX = filter.w
                argmaX = i
        return argmaX

    def getPHDsToPlot(self):
        self.phdsToPlot=[]
        for filter in self.phds:
            if filter.w > 0.5:
                self.phdsToPlot.append(filter)
        return self.phdsToPlot
    def mergeTargets(self):
        filters_to_stay = []
        mixed_filters = []
        for filter in self.phds:
            if filter.w > self.T:
                filters_to_stay.append(filter)

        while len(filters_to_stay) != 0:
            # print("-------------")
            # print(type(filters_to_stay))
            j = self.argMax(filters_to_stay)
            L = []  # indexes
            # print(len(filters_to_stay))
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
            # print(P_mix + filters_to_stay[0].w * filters_to_stay[0].P + np.outer((m_mix-filters_to_stay[0].m),(m_mix-filters_to_stay[0].m).T))
            #print("mix: ",  np.outer((m_mix - filters_to_stay[0].m),(m_mix - filters_to_stay[0].m).T))
            for t_id in L:
                P_mix += filters_to_stay[t_id].w * (
                            filters_to_stay[t_id].P + np.outer((m_mix - filters_to_stay[t_id].m),
                                                               (m_mix - filters_to_stay[t_id].m).T))
            P_mix /= w_mix
            #print("P mix: ", P_mix)
            mixed_filters.append(PHD(w_mix,m_mix,P_mix))
            removed = np.delete(filters_to_stay, L)
            filters_to_stay = removed.tolist()

        #print("mixed: ", len(mixed_filters))
        if len(mixed_filters) > self.J_max:
            self.phds = mixed_filters
            self.pruneByMaxWeight(0.1)
        else:
            self.phds = mixed_filters

    def updateWithMeasurements(self, time):
        Jk = len(self.phds)
        #print("Jk: ",Jk)
        self.updateComponents2()
        #for l, z in enumerate(zip(self.measurements[time][0], self.measurements[time][1])):
        for l, z in enumerate(np.array(self.measurements[time]).T):
            print("z: ",z)
           # z = np.array(z)
            phds_sum = 0
            for j in range(Jk):
                w = self.pd * self.phds[j].w * mvn(self.phds[j].ny, self.phds[j].S).pdf(z)
                m = self.phds[j].m + self.phds[j].K @ (z - self.phds[j].ny)
                P = self.phds[j].P
                phds_sum += w
                self.phds.append(PHD(w, m, P))
            for j in range(Jk):
                self.phds[(l + 1) * Jk + j].w = self.phds[(l + 1) * Jk + j].w / (self.radarMap.lambd + phds_sum)

        for j in range(Jk):
            self.phds[j].update(self.pd)

    def updateWithMeasurementsTMP(self, time):
        old_phds = self.phds.copy()
        self.update()
        old_phds = self.updateComponents(old_phds)
        self.phds += old_phds
        # for p in self.phds:
        #     print("weight: ", p.w)
        for l, z in enumerate(zip(self.measurements[time][0], self.measurements[time][1])):
            z = np.array(z)
            phds_sum = 0
            new_phds = []
            for j, filter in enumerate(old_phds):
                w = self.pd * filter.w * mvn(filter.ny, filter.S).pdf(z)
                m = filter.m + filter.K @ (z - filter.ny)
                P = filter.P
                phds_sum += w
                new_phds.append(PHD(w, m, P))
            for j, filter in enumerate(new_phds):
                filter.w /= (self.radarMap.lambd + phds_sum)
            self.phds += new_phds
        # for p in self.phds:
        #     print("weight: ", p.w)
        # print("-------------------------")



    def run(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("time: ", t)
            print("     num of phds (before): ", len(self.phds))
            self.predictionForExistingTargets()
            self.predictionForBirthTargets()
            print("     num of phds (after predict): ", len(self.phds))
            # self.updateComponents()
            # self.update()
            # if (t % 2 == 0):
            self.updateWithMeasurements(t)
            # self.pruneByMaxWeight(0.1)
            # self.log()
            #self.getMeasurements(t)
            #self.mergeTargets()

            w=[]
            for r in self.phds:
                w.append(r.w)
                # print(r.P.diagonal())
            print(w)
            self.getPHDsToPlot()
            self.radarMap.animateRadar(t, ax)
            for i, filter in enumerate(self.phdsToPlot):
                ax.plot(filter.m[0], filter.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="PHD")
                confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax,
                                   edgecolor=self.model_colors[i % len(self.model_colors)])
                # print(filter.P.diagonal())
            # print(filter.P)
            # plt.plot(filter)
            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            plt.pause(0.3)


if __name__ == '__main__':
    dt = 1
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.diag([0.1, 0.1, 0.1, 0.1])
    R = np.diag([5, 5]) * 5
    H = np.diag([1, 1])  # 2x4
    H = np.lib.pad(H, ((0, 0), (0, 2)), 'constant', constant_values=(0))

    ndat = 100
    lambd = 0.00001
    Pd = 0.9
    Ps = 0.95

    r = Radar(F, Q, R, H, ndat, lambd)
    r.setRadarPosition([100, 100])
    # r.setRadarRadius(500)
    sx = R[0, 0] * 5
    sy = R[1, 1] * 5
    airportCov = np.array([[sx, 0, sx, 0],
                           [0, sy, 0, sy],
                           [sx, 0, 2 * sx, 0],
                           [0, sy, 0, 2 * sy]])

    r.addNRandomAirports(1, airportCov, 0.15)
    #r.addNBorderAirports(36, airportCov, 0.1)
    # seed = 123
    seed=np.random.randint(1000)
    #r.addSingleTrajectory([-150, 350, 2, -2], seed, ndat, 0, False)
    r.makeRadarMap(full_trajectories=1, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
     # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)

    filter = PHD_map(r, Ps, Pd)
    filter.run()

