import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]

            #!!!
            random_sequence = np.random.rand(num_sequences, horizon, self.ac_dim)
            random_action_sequences = (self.high - self.low) * random_sequence + self.low
            #!!!

            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                #!!!
                if i == 0:
                    # uniformly generate random actions
                    sequences = np.random.random(size=[num_sequences, horizon, self.ac_dim])
                    action_sequences = (self.high - self.low) * sequences + self.low
                    # initialize elite means and vars
                    elite_mean = np.mean(action_sequences, axis=0)
                    elite_var = np.var(action_sequences, axis=0)
                else: 
                    # sample candidate sequences
                    action_sequences = np.random.normal(elite_mean, np.sqrt(elite_var), size=(num_sequences, horizon, self.ac_dim))
                    # calculate sequence rewards
                    sequence_rewards = self.evaluate_candidate_sequences(action_sequences, obs)
                    # get elite sequences
                    elite_sequences_indices = np.argsort(sequence_rewards)[-self.cem_num_elites:]
                    elite_sequences = action_sequences[elite_sequences_indices,:,:]
                    # update elite means and vars
                    elite_mean = self.cem_alpha*np.mean(elite_sequences, axis=0) + (1-self.cem_alpha)*elite_mean
                    elite_var = self.cem_alpha*np.var(elite_sequences, axis=0) + (1-self.cem_alpha)*elite_var
                #!!!

            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            #!!!
            # set cem_action to first action of the mean
            cem_action = elite_mean[0,:]
            #!!!

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)

        #!!!
        # initialize rewards matrix
        pred_rewards = []

        for model in self.dyn_models:
            # get model rewards
            model_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            # append to rewards matrix
            pred_rewards.append(model_rewards)
        
        # calculate mean across models
        pred_rewards_avg = np.mean(pred_rewards, axis=0)

        return pred_rewards_avg
        #!!!

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences

        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            #!!!
            best_action_sequence = np.argmax(predicted_rewards)  # TODO (Q2)
            action_to_take = candidate_action_sequences[best_action_sequence][0] # TODO (Q2)
            #!!!
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        #!!!
        # initialize rewards matrix
        rewards = []
        
        # repeat obs N times for batching
        obs_batch = np.repeat(np.expand_dims(obs, axis=0), self.N, axis = 0)

        for i in range(self.horizon):
            # get i-th action of each sequence for batching
            action_batch = candidate_action_sequences[:,i,:]
            # retrieve predicted next observation
            predicted_obs_batch = model.get_prediction(obs_batch, action_batch, self.data_statistics)
            # get rewards from this predicted observation and action
            step_rewards, _ = self.env.get_reward(predicted_obs_batch, action_batch)
            # append to rewards matrix
            rewards.append(step_rewards)
            # make predicted next obsevations current observations
            obs_batch = predicted_obs_batch        
        # sum across horizons
        sum_of_rewards = np.sum(rewards, axis=0)
        #!!!

        return sum_of_rewards
