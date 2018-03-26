import numpy as np
import pandas as pd
from scipy.spatial import distance


class Gonzalez:
    """docstring for Gonzalez"""
    def __init__(self, k):
        self.dataset = None
        self.distance_matrix = None
        self.k = k
        self.k_centers = []
        self.k_center_idx = []

    def load_and_process_data(self, filepath):
        self.dataset = pd.read_csv(filepath,
                                    header=None,
                                    names=["alpha", "delta", "lon", "lat", "temp"])

        self.dataset["alpha_r"] = self.dataset["alpha"].apply(lambda x: np.deg2rad(x))
        self.dataset["delta_r"] = self.dataset["delta"].apply(lambda x: np.deg2rad(x))

        self.distance_matrix = distance.cdist(self.dataset, self.dataset, self.angular_distance)

        print("Finished processing data")

    def angular_distance(self, point1, point2):
        alpha_1 = point1[2]
        alpha_2 = point2[2]

        delta_1 = point1[3]
        delta_2 = point2[3]

        ca12 = np.cos(alpha_1 - alpha_2)
        cd1 = np.cos(delta_1)
        cd2 = np.cos(delta_2)
        sd1 = np.sin(delta_1)
        sd2 = np.sin(delta_2)
        return np.arccos((sd1*sd2) + (cd1*cd2*ca12))

    def center_classify(self, row):
        return self.k_center_idx[np.argmin(self.distance_matrix[int(row.name), self.k_center_idx])]

    def run(self):
        if self.dataset is None:
            raise Exception("Load data first!")

        init_idx = np.random.randint(0, self.dataset.shape[0])
        self.dataset["center"] = init_idx
        init_center = self.dataset.loc[init_idx]
        self.k_centers.append(init_center)
        self.k_center_idx.append(init_idx)

        for i in range(2, self.k+1):
            multi_idx = np.array([(c.name, c["center"]) for idx, c in self.dataset.iterrows()], dtype=np.int32)
            center_idx = np.argmax(self.distance_matrix[multi_idx[:, 0], multi_idx[:, 1]])
            self.k_center_idx.append(center_idx)
            self.dataset["center"] = self.dataset.apply(self.center_classify, axis=1)
            center = self.dataset.loc[center_idx]
            self.k_centers.append(center)

        assert len(self.k_center_idx) == len(set(self.k_center_idx))