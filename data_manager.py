import torch
import numpy as np
from collections import OrderedDict
from itertools import product
import copy


class StateManager:
    """Main minecraft state wrapper, creates image and vector out of the state information. Some inventory items
    (of feasible average amount) are encoded as multi-hot vectors, the rest as normalized float values."""

    def __init__(self, device):
        self.device = device

        # equipped item (1-hot encoded in get_img_vec):
        self.item_list = ['none', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe']

        # dict of avg values in the human data: (converted to normalized float values in get_img_vec)
        self.float_inventory_list = OrderedDict([('dirt', 5.), ('cobblestone', 100.), ('stone', 15.)])

        # dict of maximal values in most of the human data: (converted to multi-hot vectors in get_img_vec)
        self.inventory_list = OrderedDict([('coal', 16), ('crafting_table', 3), ('furnace', 3), ('cobblestone', 16),
                                           ('iron_ingot', 8), ('iron_ore', 8), ('iron_pickaxe', 3), ('log', 32),
                                           ('planks', 64), ('stick', 32),
                                           ('stone_pickaxe', 4), ('torch', 16), ('wooden_pickaxe', 4)])

    def get_img_vec(self, state):
        img = state['pov']
        item_type = state['equipped_items']['mainhand']['type']
        if item_type in self.item_list:
            item_id = self.item_list.index(item_type)
        else:
            item_id = 0
        vec = [0.] * len(self.item_list)
        vec[item_id] = 1.
        for k, v in state['inventory'].items():
            if k in self.float_inventory_list:
                avg = self.float_inventory_list[k]
                vec += [np.clip(float(v) / avg, 0., 5. * avg)]  # norm by avg amount in human data, max: 5 x avg value
            if k in self.inventory_list:
                vec += self._item_vector(v, self.inventory_list[k])

        return img, vec

    def _item_vector(self, amount, total_amount):
        return [1. if i < amount else 0. for i in range(total_amount)]

    def get_torch_img_vec(self, img_list, vec_list):
        img_torch = torch.tensor(img_list, dtype=torch.float32, device=self.device).div_(255).permute(0, 3, 1, 2)
        vec_torch = torch.tensor(vec_list, dtype=torch.float32, device=self.device)
        return img_torch, vec_torch


class ActionManager:
    """Main minecraft action wrapper. Simplifies action space to 130 discrete actions"""

    def __init__(self, device, c_action_magnitude=22.5):
        self.device = device
        self.c_action_magnitude = c_action_magnitude

        self.zero_action = OrderedDict([('attack', 0),
                                        ('back', 0),
                                        ('camera', np.array([0., 0.])),
                                        ('craft', 0),
                                        ('equip', 0),
                                        ('forward', 0),
                                        ('jump', 0),
                                        ('left', 0),
                                        ('nearbyCraft', 0),
                                        ('nearbySmelt', 0),
                                        ('place', 0),
                                        ('right', 0),
                                        ('sneak', 0),
                                        ('sprint', 0)])

        # ['sneak'] is ignored

        # Simplified crafting options:

        self.separate_dict = OrderedDict([
            ('craft', [1, 2, 3, 4]),
            ('equip', [1, 3, 5, 7]),
            ('nearbyCraft', [2, 4, 6, 7]),
            ('nearbySmelt', [1, 2]),
            ('place', [1, 4, 5, 6])
        ])

        self.separate = list(self.separate_dict.keys())
        self.separate_values = list(self.separate_dict.values())

        self.separate_str_lists = OrderedDict([
            ('craft', ["none", "torch", "stick", "planks", "crafting_table"]),
            ('equip', ["none", "air", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe"]),
            ('nearbyCraft', ["none", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe", "furnace"]),
            ('nearbySmelt', ["none", "iron_ingot", "coal"]),
            ('place', ["none", "dirt", "stone", "cobblestone", "crafting_table", "furnace", "torch"])
        ])

        # camera discretization:
        self.camera_dict = OrderedDict([
            ('turn_up', np.array([-c_action_magnitude, 0.])),
            ('turn_down', np.array([c_action_magnitude, 0.])),
            ('turn_left', np.array([0., -c_action_magnitude])),
            ('turn_right', np.array([0., c_action_magnitude]))
        ])

        self.fully_connected_no_camera = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sprint']
        self.camera_actions = ['turn_up', 'turn_down', 'turn_left', 'turn_right']
        self.fully_connected = self.fully_connected_no_camera + self.camera_actions

        # following action combinations are excluded:
        self.exclude = [('forward', 'back'), ('left', 'right'), ('attack', 'jump'),
                        ('turn_up', 'turn_down', 'turn_left', 'turn_right')]

        # sprint only allowed when forward is used:
        self.only_if = [('sprint', 'forward')]

        # Maximal allowed mount of actions within one action:
        self.remove_size = 3

        # if more than 3 actions are present, actions are removed using this list until only 3 actions remain:
        self.remove_first_list = ['sprint', 'left', 'right', 'back',
                                  'turn_up', 'turn_down', 'turn_left', 'turn_right',
                                  'attack', 'jump', 'forward']

        self.fully_connected_list = list(product(range(2), repeat=len(self.fully_connected)))

        remove = []
        for el in self.fully_connected_list:
            for tuple_ in self.exclude:
                if sum([el[self.fully_connected.index(a)] for a in tuple_]) > 1:
                    if el not in remove:
                        remove.append(el)
            for a, b in self.only_if:
                if el[self.fully_connected.index(a)] == 1 and el[self.fully_connected.index(b)] == 0:
                    if el not in remove:
                        remove.append(el)
            if sum(el) > self.remove_size:
                if el not in remove:
                    remove.append(el)

        for r in remove:
            self.fully_connected_list.remove(r)

        self.action_list = []
        for el in self.fully_connected_list:
            new_action = copy.deepcopy(self.zero_action)
            for key, value in zip(self.fully_connected, el):
                if key in self.camera_actions:
                    if value:
                        new_action['camera'] = self.camera_dict[key]
                else:
                    new_action[key] = value
            self.action_list.append(new_action)

        self.separate_id_dict = OrderedDict()
        for i, key in enumerate(self.separate):
            self.separate_id_dict[key] = OrderedDict()
            for id_ in self.separate_values[i]:
                new_action = copy.deepcopy(self.zero_action)
                new_action[key] = id_
                self.separate_id_dict[key][id_] = len(self.action_list)
                self.action_list.append(new_action)

        self.num_action_ids_list = [len(self.action_list)]
        self.act_continuous_size = 0

    def get_action(self, id_):
        a = copy.deepcopy(self.action_list[int(id_)])
        a['camera'] += np.random.normal(0., 0.5, 2)
        return a

    def print_action(self, id_):
        a = copy.deepcopy(self.action_list[int(id_)])
        out = ""
        for k, v in a.items():
            if k != 'camera':
                if v != 0:
                    if k in self.separate_str_lists:
                        out += f'{k} {self.separate_str_lists[k][v]} '
                    else:
                        out += f'{k} '
            else:
                if (v != np.zeros(2)).any():
                    out += k

        print(out)

    def get_id(self, action):

        for key in self.separate:
            if action[key] != 0:
                if action[key] in self.separate_id_dict[key]:
                    action_id = self.separate_id_dict[key][action[key]]
                    return action_id

        action = copy.deepcopy(action)

        # discretize 'camera':
        camera = action['camera']
        camera_action_amount = 0
        if - self.c_action_magnitude / 2. < camera[0] < self.c_action_magnitude / 2.:
            action['camera'][0] = 0.
            if - self.c_action_magnitude / 2. < camera[1] < self.c_action_magnitude / 2.:
                action['camera'][1] = 0.
            else:
                camera_action_amount = 1
                action['camera'][1] = self.c_action_magnitude * np.sign(camera[1])
        else:
            camera_action_amount = 1
            action['camera'][0] = self.c_action_magnitude * np.sign(camera[0])

            action['camera'][1] = 0.

        # simplify action:
        for tuple_ in self.exclude:
            if len(tuple_) == 2:
                a, b = tuple_
                if action[a] and action[b]:
                    action[b] = 0
        for a, b in self.only_if:
            if not action[b]:
                if action[a]:
                    action[a] = 0
        for a in self.remove_first_list:
            if sum([action[key] for key in self.fully_connected_no_camera]) > \
                    (self.remove_size - camera_action_amount):
                if a in self.camera_actions:
                    action['camera'] = np.array([0., 0.])
                    camera_action_amount = 0
                else:
                    action[a] = 0
            else:
                break

        # set one_hot camera keys:
        for key in self.camera_actions:
            action[key] = 0
        for key, val in self.camera_dict.items():
            if (action['camera'] == val).all():
                action[key] = 1
                break

        non_separate_values = tuple(action[key] for key in self.fully_connected)

        return self.fully_connected_list.index(non_separate_values)

    def get_torch_action(self, a_id_batch_list):
        a_id_torch_list = [torch.tensor(a_id_batch, dtype=torch.int64, device=self.device) for a_id_batch in
                           a_id_batch_list]

        return a_id_torch_list

    def get_left_right_reversed_mapping(self):
        action_mapping = []
        for action in self.action_list:
            reversed_action = copy.deepcopy(action)
            if action['left'] == 1:
                reversed_action['left'] = 0
                reversed_action['right'] = 1
                assert action['right'] == 0
            if action['right'] == 1:
                reversed_action['right'] = 0
                reversed_action['left'] = 1
                assert action['left'] == 0
            if (action['camera'] == [0, -22.5]).all():
                reversed_action['camera'][1] = 22.5
            if (action['camera'] == [0, 22.5]).all():
                reversed_action['camera'][1] = -22.5

            rev_action_id = self.get_id(reversed_action)
            action_mapping.append(rev_action_id)

        return action_mapping
