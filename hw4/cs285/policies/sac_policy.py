from cs285.policies.MLP_policy import MLPPolicy
from cs285.infrastructure.sac_utils import SquashedNormal
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: get this from previous HW
        #!!!
        entropy = self.log_alpha.exp()
        #!!!
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW
        # if not sampling return the mean of the distribution
        
        # #!!!

        # # return the action that the policy prescribes
        # observation = ptu.from_numpy(obs.astype(np.float32))
        # action_distribution = self(observation)
        # if sample:
        #     action = ptu.to_numpy(action_distribution.rsample())
        # else:
        #     action = ptu.to_numpy(action_distribution.mean)
        # #!!!
        # return action

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        dist = self(observation)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return ptu.to_numpy(action)



    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 

        # #!!!
        # if self.discrete:
        #     logits = self.logits_na(observation)
        #     action_distribution = distributions.Categorical(logits=logits)
        #     print("discrete???")
        #     raise NotImplementedError
        # else:
        #     batch_mean = self.mean_net(observation)
        #     batch_dim = batch_mean.shape[0]
        #     #logstd_clipped = TanhTransform.atanh(self.logstd)
        #     logstd_clipped = torch.clamp(self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
            
        #     std = logstd_clipped.exp()
        #     batch_std = std.repeat(batch_dim, 1)

        #     action_distribution = SquashedNormal(
        #         batch_mean,
        #         batch_std,
        #     )
        #     # print(batch_mean.shape)
        #     # print(batch_std.shape)
        # #!!!
        # return action_distribution
        mean = self.mean_net(observation)
        log_std = torch.tanh(self.logstd)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        dist = sac_utils.SquashedNormal(mean, std)
        return dist        

    def update(self, obs, critic):
        # TODO: get this from previous HW
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # #!!!
        # # get actions from observations
        # obs = ptu.from_numpy(obs)
        # action_distribution = self(obs)
        # action = action_distribution.rsample()

        # # get sum of action logprobs
        # logprobs = action_distribution.log_prob(action).sum(dim=1).unsqueeze(1)
        # logprobs_alpha = logprobs.detach().clone()

        # # get q values
        # q1, q2 = critic(obs, action)
        # q = torch.min(q1,q2)

        # assert logprobs.shape == q.shape

        # # actor/policy gradient step
        # actor_loss = torch.mean(self.alpha*logprobs - q)
        # self.optimizer.zero_grad()
        # actor_loss.backward()
        # self.optimizer.step()

        # # alpha gradient step
        # alpha_loss = torch.mean(-self.alpha*logprobs_alpha-self.alpha*self.target_entropy)
        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()
        # #!!!
        # return actor_loss, alpha_loss, self.alpha
        obs = ptu.from_numpy(obs)

        dist = self(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Qs = critic(obs, action)
        actor_Q = torch.min(*actor_Qs)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha.item()