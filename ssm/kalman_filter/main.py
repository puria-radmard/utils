import torch
from torch import Tensor as T

from purias_utils.ssm.process.base import *
from purias_utils.ssm.process.noise import *


class NonBayesianKalmanFilter:
    """
        This is the Kalman Filter covered in 4F7.
        We start with a given K[X{n}|Y{1:n}], and wlog set n = 0

        We assume we are provided with the dynamics process (fn),
        the output transform (gn), and the ___ noise (qn), which are process types

        TODO: add some size checking here!
    """
    def __init__(
        self,
        initial_estimate: T,
        initial_state_estimate_covariance: T,
        fn_process: ProcessBase,
        gn_process: ProcessBase,
        qn_process: WhiteNoise,
        rn_process: WhiteNoise,
        observation_process: ProcessBase
    ) -> None:
        
        self.state_estimate: T = initial_estimate
        self.state_estimate_covariance: T = initial_state_estimate_covariance

        self.state_prediction: T = None
        self.state_prediction_covariance: T = None

        self.output_prediction: T = None

        self.output_innovation: T = None
        self.innovation_covariance: T = None

        self.kalman_gain: T = None

        self.fn_process = fn_process
        self.gn_process = gn_process
        self.qn_process = qn_process
        self.rn_process = rn_process
        self.observation_process = observation_process

    @property
    def fn(self):
        return self.fn_process.previous_value

    @property
    def gn(self):
        return self.gn_process.previous_value

    @property
    def qn(self):
        return self.qn_process.covariance
    
    @property
    def rn(self):
        return self.rn_process.covariance

    @property
    def latest_observation(self):
        return self.observation_process.previous_value

    def update_state_prediction(self):
        "Prediction step. TODO: include input"
        self.state_prediction = self.fn @ self.state_estimate
        self.state_prediction_covariance = (self.fn @ self.state_estimate_covariance @ self.fn.T) + self.qn

    def update_output_prediction(self):
        "Update prediction of output, innovation, and innovation covariance (the 'denominator' of Kalman gain)"
        # Name mismatch: gn here is actually g{n+1}, which will have been updated
        self.output_prediction = self.gn @ self.state_prediction
        self.output_innovation = self.latest_observation - self.output_prediction
        self.innvation_covariance = (self.gn @ self.state_prediction_covariance @ self.gn.T) + self.rn

    def update_kalman_gain(self):
        "Finish optimal Kalman gain"
        self.kalman_gain = (
            self.state_prediction_covariance @ self.gn.T @ 
            torch.linalg.inv(self.innvation_covariance)
        )

    def update_state_estimate(self):
        """
            This takes the form E[X{n+1} | I{n+1}] = E[X{n+1}] + B I{n+1},
            where I{n+1} = Y{n+1} - K[Y{n+1} | Y{1:n}] are the innovations, 
                and B, as defined in the notes, is the Kalman gain

            The notes apply update the state mean, but the state prediction
            makes more sense

            Also update the state error prediction
        """
        self.state_estimate = self.state_prediction + (self.kalman_gain @ self.output_innovation)

        state_dim = self.state_estimate_covariance.shape[0]

        self.state_estimate_covariance = (
            torch.eye(state_dim) - (self.kalman_gain @ self.gn)
        ) @ self.state_prediction_covariance

    def pre_update_step(self):
        self.update_state_prediction()

    def post_update_step(self):
        self.update_output_prediction()
        self.update_kalman_gain()
        self.update_state_estimate()
