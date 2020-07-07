from collections import namedtuple
import numpy as np
import torch
import pickle
import random


Transition = namedtuple('Transition', ('state', 'vector', 'action', 'reward', 'nonterminal'))


class Data:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.data = np.array([None] * size)
        self.last_reward_index = 0

    def current_size(self):
        if self.full:
            return self.size
        else:
            return self.index

    def append(self, data):
        assert not self.full
        self.data[self.index] = data
        self.index += 1
        self.full = self.index == self.size

    def get(self, data_index):
        return self.data[data_index % self.size]

    def update_last_reward_index(self):
        assert not self.full
        self.last_reward_index = self.index

    def remove_new_data(self):
        assert not self.full
        removed_ids_list = list(range(self.last_reward_index, self.index))
        removed_amount = len(removed_ids_list)

        self.index = self.last_reward_index

        return removed_amount, removed_ids_list


class Dataset:
    def __init__(self, device, capacity, state_shape, state_vec_shape, state_manager, action_manager,
                 scale_rewards=True):
        self.device = device
        self.capacity = capacity
        self.state_shape = state_shape

        if state_vec_shape is not None:
            self.blank_trans = Transition(torch.zeros(state_shape, dtype=torch.uint8),
                                          torch.zeros(state_vec_shape), None, 0, False)
        else:
            self.blank_trans = Transition(torch.zeros(state_shape, dtype=torch.uint8), None, None, 0, False)

        self.discount = 1.
        self.n = 1

        self.state_manager = state_manager
        self.action_manager = action_manager

        self.transitions = Data(capacity)

        self.scale_rewards = scale_rewards

        self.gatherlog_sample_id_list = []

    def reward_reshaping(self, r):
        if self.scale_rewards:
            if r == 0.:
                return 0.
            else:
                return 1.
        else:
            return r

    def append_sample(self, sample, gatherlog_sample=False, treechop_data=False):

        # Saving ids of samples from the getlog part of data (all data until first reward > 1.)
        if gatherlog_sample and not treechop_data:
            self.gatherlog_sample_id_list.append(self.transitions.index)

        state, action, reward, done = sample[0], sample[1], sample[2], sample[4]

        img, vec = self.state_manager.get_img_vec(state)

        if treechop_data:
            # When dealing with treechop_data that has no inventory information, we insert a random inventory
            # from the other demonstrations:

            random_get_log_id = random.choice(self.gatherlog_sample_id_list)
            torch_vec = self.transitions.data[random_get_log_id].vector.clone()
        else:
            torch_vec = torch.tensor(vec)

        action_id = self.action_manager.get_id(action)

        torch_img = torch.from_numpy(img).permute(2, 0, 1)

        self.transitions.append(Transition(torch_img, torch_vec, action_id, reward, not done))

    def update_last_reward_index(self):
        self.transitions.update_last_reward_index()

    def remove_new_data(self):
        removed_amount, removed_ids_list = self.transitions.remove_new_data()

        for id_ in removed_ids_list:
            if id_ in self.gatherlog_sample_id_list:
                self.gatherlog_sample_id_list.remove(id_)

        return removed_amount

    def save(self, path):
        pickle.dump([self.transitions.index, self.transitions.size, self.transitions.full, self.transitions.data,
                     self.transitions.last_reward_index], open(path, 'wb'))

    def load(self, path):
        self.transitions.index, self.transitions.size, self.transitions.full, self.transitions.data, \
            self.transitions.last_reward_index = pickle.load(open(path, "rb"))

    def _get_transition(self, idx):
        transition = np.array([None] * (self.n + 1))
        transition[0] = self.transitions.get(idx)
        for t in range(1, 1 + self.n):
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx + t)
            else:
                transition[t] = self.blank_trans
        return transition

    def sample_line(self, size, length):
        ids = np.random.randint(0, self.transitions.current_size() - length - self.n, size=size)

        ids = [list(range(i, i + length)) for i in ids]
        ids = [item for sublist in ids for item in sublist]

        states, vecs, next_states, next_vecs, actions, returns, nonterminals = \
            [], [], [], [], [], [], []

        no_vecs = False
        for id_ in ids:
            transition = self._get_transition(id_)

            states.append(transition[0].state.to(device=self.device).to(dtype=torch.float32).div_(255))
            next_states.append(transition[self.n].state.to(device=self.device).to(dtype=torch.float32).div_(255))

            if transition[0].vector is not None:
                vecs.append(transition[0].vector.to(device=self.device).to(dtype=torch.float32))
                next_vecs.append(transition[self.n].vector.to(device=self.device).to(dtype=torch.float32))
            else:
                vecs.append(None)
                next_vecs.append(None)
                no_vecs = True

            actions.append(torch.tensor([transition[0].action], dtype=torch.int64, device=self.device))
            returns.append(
                torch.tensor([sum(self.discount ** n *
                                  self.reward_reshaping(transition[n].reward)
                                  for n in range(self.n))],
                             dtype=torch.float32, device=self.device))
            nonterminals.append(
                torch.tensor([transition[self.n].nonterminal],
                             dtype=torch.float32, device=self.device))

        states, next_states = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)

        if not no_vecs:
            vecs, next_vecs = torch.stack(vecs), torch.stack(next_vecs)

        return states, vecs, actions, returns, next_states, next_vecs, nonterminals
