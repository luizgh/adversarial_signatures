from simanneal import Annealer
import numpy as np


class AdversaryAttackProblem(Annealer):
    def __init__(self, x_init, model, threshold, multiplier, norm_weight, std=2, early_stop=False):
        self.model = model
        self.x_init = x_init
        self.threshold = threshold
        self.multiplier = multiplier
        self.norm_weight = norm_weight
        self.early_stop = early_stop
        self.std = std

        super(AdversaryAttackProblem, self).__init__(x_init)

        self.image_list = []
        self.energy_list = []

    def move(self):
        # delta = np.random.uniform(low=-4, high=4, size=self.state.shape)
        delta = np.random.normal(size=self.state.shape) * self.std
        new_img = np.clip(self.state + delta, 0, 255)
        self.state = new_img

    def energy(self):
        score = self.model.predict_score(self.state)
        if self.early_stop and ((self.multiplier == 1 and score <= self.threshold)
                                or (
                                        self.multiplier == -1 and score >= self.threshold)):  # early stop if we pass the threshold
            self.user_exit = True
        return self.multiplier * score + self.norm_weight * np.sqrt(np.mean(np.square(self.x_init - self.state)))

