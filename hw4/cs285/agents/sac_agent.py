from collections import OrderedDict

from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure.sac_utils import *
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: get this from previous HW 
        # #!!!
        # # 1. Compute the target Q value. 
        # # HINT: You need to use the entropy term (alpha)
        # # 2. Get current Q estimates and calculate critic loss
        # # 3. Optimize the critic  

        # #!!!
        # # ob_no = ptu.from_numpy(ob_no)
        # # ac_na = ptu.from_numpy(ac_na)
        # # next_ob_no = ptu.from_numpy(next_ob_no)
        # re_n = ptu.from_numpy(re_n)
        # re_n = re_n.unsqueeze(1)
        # terminal_n = ptu.from_numpy(terminal_n)
        # terminal_n = terminal_n.unsqueeze(1)

        # alpha = self.actor.alpha

        # # sample next actions and calculate logprobs
        # next_action = self.actor.get_action(next_ob_no)
        # next_action_distribution = self.actor(ptu.from_numpy(next_ob_no))
        # next_action_logprob = next_action_distribution.log_prob(ptu.from_numpy(next_action)).sum(dim=1).unsqueeze(1)
        
        # # calculate target q values and values
        # q_tp1_1, q_tp1_2 = self.critic_target(ptu.from_numpy(next_ob_no), ptu.from_numpy(next_action))
        # q_tp1 = torch.min(q_tp1_1, q_tp1_2)
        # v_tp1 = q_tp1 - alpha*next_action_logprob

        # # calculate target
        # target = re_n + self.gamma*(1-terminal_n)*v_tp1
        # target = target.detach()

        # # calculate q value
        # q_t_1, q_t_2 = self.critic(ptu.from_numpy(ob_no), ptu.from_numpy(ac_na))
        # q_t = torch.min(q_t_1,q_t_2)

        # assert terminal_n.shape == target.shape
        # assert q_tp1.shape == q_t.shape
        # assert next_action_logprob.shape == target.shape
        # assert q_t.shape == target.shape
        # assert re_n.shape == next_action_logprob.shape
        # assert q_t.shape == v_tp1.shape

        # # update critic
        # critic_loss = 0.5*self.critic.loss(target, q_t)
        # self.critic.optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic.optimizer.step()
        # #!!!
        # return critic_loss

        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(re_n).unsqueeze(1)
        terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)

        with torch.no_grad():
            dist = self.actor(next_ob_no)
            next_action = dist.rsample()
            next_Qs = self.critic_target(next_ob_no, next_action)
            next_Q = torch.min(*next_Qs)
            target_Q = reward_n + ((1-terminal_n) * self.gamma * next_Q)
            next_log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q -= self.gamma * (1-terminal_n) * self.actor.alpha.detach() * next_log_prob

        critic_loss = 0
        # get current Q estimates
        current_Qs = self.critic(ob_no, ac_na)
        for current_Q in current_Qs:
            critic_loss += self.critic.loss(current_Q, target_Q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO: get this from previous HW
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging
        loss = OrderedDict()
        # #!!!
        # for i in range(self.agent_params['num_critic_updates_per_agent_update']):
        #     loss['Critic_Loss'] = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            
        # if (self.training_step % self.critic_target_update_frequency) == 0:
        #     soft_update_params(self.critic, self.critic_target, self.critic_tau)
            
        # if (self.training_step % self.actor_update_frequency) == 0:
        #     for j in range(self.agent_params['num_actor_updates_per_agent_update']):
        #         loss['Actor_Loss'], loss['Alpha_Loss'], loss['Temperature'] = self.actor.update(ob_no, self.critic)
        
        # self.training_step += 1
        # #!!!
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            loss['Critic_Loss'] = critic_loss
            if self.training_step % self.critic_target_update_frequency == 0:
                soft_update_params(self.critic, self.critic_target, self.critic_tau)

        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, temperature = self.actor.update(ob_no, self.critic)
                loss['Actor_Loss'] = actor_loss
                loss['Alpha_Loss'] = alpha_loss
                loss['Temperature'] = temperature

        self.training_step += 1
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
