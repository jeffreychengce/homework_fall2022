from torch import nn
import torch
from torch import optim
from cs285.models.base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure import pytorch_util as ptu

import numpy as np

class FFModel(nn.Module, BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        #!!!

        # normalize input data to mean 0, std 1
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std) # TODO(Q1)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std) # TODO(Q1)
        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_network(concatenated_input) # TODO(Q1)
        delta_pred_unnormalized = unnormalize(delta_pred_normalized, delta_mean, delta_std)
        next_obs_pred = obs_unnormalized + delta_pred_unnormalized # TODO(Q1)
        #!!!
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        #!!!
        # TODO(Q1) get the predicted next-states (s_t+1) as a numpy array
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        obs_mean = ptu.from_numpy(data_statistics['obs_mean'])
        obs_std = ptu.from_numpy(data_statistics['obs_std'])
        acs_mean = ptu.from_numpy(data_statistics['acs_mean'])
        acs_std = ptu.from_numpy(data_statistics['acs_std'])
        delta_mean = ptu.from_numpy(data_statistics['delta_mean'])
        delta_std = ptu.from_numpy(data_statistics['delta_std'])


        prediction, _ = self(
            obs, 
            acs, 
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
            )
        prediction = ptu.to_numpy(prediction)
        #!!!
        return prediction

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """
        #!!!
        # TODO(Q1) compute the normalized target for the model.
        # Hint: you should use `data_statistics['delta_mean']` and
        # `data_statistics['delta_std']`, which keep track of the mean
        # and standard deviation of the model.
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        obs_mean = ptu.from_numpy(data_statistics['obs_mean'])
        obs_std = ptu.from_numpy(data_statistics['obs_std'])
        acs_mean = ptu.from_numpy(data_statistics['acs_mean'])
        acs_std = ptu.from_numpy(data_statistics['acs_std'])
        delta_mean = ptu.from_numpy(data_statistics['delta_mean'])
        delta_std = ptu.from_numpy(data_statistics['delta_std'])
        next_observations = ptu.from_numpy(next_observations)

        _, predicted_delta = self(
            observations, 
            actions, 
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
            )

        # target_delta = ptu.from_numpy(next_observations - observations)
        target_delta = next_observations - observations
        target = normalize(target_delta, delta_mean, delta_std)
        
        loss = self.loss(target, predicted_delta) # TODO(Q1) compute the loss
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        #!!!

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
