import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from model import Network, AtariCNN, ImpalaResNetCNN, FixupResNetCNN

from os.path import join as p_join


class Agent:
    def __init__(self, num_actions, image_channels, vec_size, writer,
                 net, batch_size, augment_flip, hidden_size, dueling, learning_rate, adam_eps, device):
        self.num_actions = num_actions
        self.writer = writer
        self.batch_size = batch_size

        self.augment_flip = augment_flip

        self.rev_action_map = None

        if self.augment_flip:
            # flipping the actions horizontally, for the horizontal image flip augmentation:
            self.rev_action_map = [0, 2, 1, 3, 4, 10, 12, 11, 13, 14, 5, 7, 6, 8, 9, 15, 17, 16, 18, 19,
                                   25, 27, 26, 28, 29, 20, 22, 21, 23, 24, 30, 32, 31, 33, 34, 35, 37,
                                   36, 38, 39, 46, 48, 47, 49, 50, 51, 40, 42, 41, 43, 44, 45, 52, 54,
                                   53, 55, 56, 57, 59, 58, 60, 62, 61, 63, 64, 70, 72, 71, 73, 74, 65,
                                   67, 66, 68, 69, 75, 77, 76, 78, 79, 81, 80, 82, 84, 83, 85, 86, 92,
                                   94, 93, 95, 96, 87, 89, 88, 90, 91, 97, 99, 98, 100, 101, 102, 104,
                                   103, 105, 107, 106, 108, 109, 111, 110, 112, 113, 114, 115, 116, 117,
                                   118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
            # this id list can be received by running get_left_right_reversed_mapping() from the ActionManager

        if net == 'normal':
            cnn_module = AtariCNN
        elif net == 'resnet':
            cnn_module = ImpalaResNetCNN
        elif net == 'deep_resnet':
            cnn_module = FixupResNetCNN
        elif net == 'double_deep_resnet':
            cnn_module = lambda x: FixupResNetCNN(x, double_channels=True)
        else:
            raise ValueError("Unknown args.net")

        self.net = Network(num_actions, image_channels, vec_size, cnn_module, hidden_size,
                           dueling=dueling,
                           double_channels=(net == 'double_deep_resnet')).to(device=device)

        self.net.train()

        self.optimiser = optim.Adam(self.net.parameters(), lr=learning_rate, eps=adam_eps, weight_decay=1e-5)

    def act(self, img, vec):
        with torch.no_grad():

            logits = self.net(img, vec)
            probs = F.softmax(logits, 1).detach().cpu().numpy()

            actions = [np.random.choice(len(p), p=p) for p in probs]

            assert len(actions) == 1  # only used with batchsize 1

            return actions[0]

    def learn(self, time_, dataset, write=False):

        states, vecs, actions, returns, next_states, next_vecs, nonterminals = \
            dataset.sample_line(self.batch_size, 1)

        if self.augment_flip:
            if np.random.binomial(n=1, p=0.5):
                states = torch.flip(states, (3,))
                for i in range(actions.shape[0]):
                    actions[i] = self.rev_action_map[actions[i]]

        logits = self.net(states, vecs)

        loss = F.cross_entropy(logits, actions)

        if write:
            if self.writer is not None:
                self.writer.add_scalar('loss/cross_entropy', loss.detach().cpu().numpy(), time_)

        self.net.zero_grad()
        loss.backward()
        self.optimiser.step()

    def save(self, path, id_=None):
        if id_ is None:
            torch.save(self.net.state_dict(), p_join(path, 'model.pth'))
            state = {'optimizer': self.optimiser.state_dict()}
            torch.save(state, p_join(path, 'state.pth'))
        else:
            torch.save(self.net.state_dict(), p_join(path, f'model_{id_}.pth'))
            state = {'optimizer': self.optimiser.state_dict()}
            torch.save(state, p_join(path, f'state_{id_}.pth'))

    def load(self, path, id_=None):
        if id_ is None:
            self.net.load_state_dict(torch.load(p_join(path, 'model.pth')))
            state = torch.load(p_join(path, 'state.pth'))
            self.optimiser.load_state_dict(state['optimizer'])
        else:
            self.net.load_state_dict(torch.load(p_join(path, f'model_{id_}.pth')))
            state = torch.load(p_join(path, f'state_{id_}.pth'))
            self.optimiser.load_state_dict(state['optimizer'])

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()
