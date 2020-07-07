from gym.core import Wrapper
import minerl.env.spaces as spaces
import numpy as np
from copy import deepcopy


class DummyMinecraft:
    """
    Useful for debugging. (Only very limited gym compatibility is supported.)
    Also used during training, since it doesn't need the actual environment.
    """

    def __init__(self):
        self.state = {
            'equipped_items': {'mainhand': {'damage': 0, 'maxDamage': 0, 'type': 0}},
            'inventory': {'coal': 0,
                          'cobblestone': 0,
                          'crafting_table': 0,
                          'dirt': 0,
                          'furnace': 0,
                          'iron_axe': 0,
                          'iron_ingot': 0,
                          'iron_ore': 0,
                          'iron_pickaxe': 0,
                          'log': 0,
                          'planks': 0,
                          'stick': 0,
                          'stone': 0,
                          'stone_axe': 0,
                          'stone_pickaxe': 0,
                          'torch': 0,
                          'wooden_axe': 0,
                          'wooden_pickaxe': 0},
            'pov': np.full([64, 64, 3], 127, dtype=np.uint8)}

        self.action_space = {
            'attack': spaces.Discrete(2),
            'back': spaces.Discrete(2),
            'camera': spaces.Box(np.full([2], -1), np.full([2], 1)),
            'craft': spaces.Enum('none', 'torch', 'stick', 'planks', 'crafting_table'),
            'equip': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe',
                                 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'),
            'forward': spaces.Discrete(2),
            'jump': spaces.Discrete(2),
            'left': spaces.Discrete(2),
            'nearbyCraft': spaces.Enum('none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe',
                                       'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'),
            'nearbySmelt': spaces.Enum('none', 'iron_ingot', 'coal'),
            'place': spaces.Enum('none', 'dirt', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'),
            'right': spaces.Discrete(2),
            'sneak': spaces.Discrete(2),
            'sprint': spaces.Discrete(2)}

        self.observation_space = {
            'equipped_items': {
                'mainhand': {
                    'damage': spaces.Box(np.full([2], -1), np.full([2], 1), dtype=np.int64),
                    'maxDamage': spaces.Box(np.full([2], -1), np.full([2], 1), dtype=np.int64),
                    'type': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe',
                                        'iron_axe', 'iron_pickaxe', 'other')}},
            'inventory': {
                'coal': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'cobblestone': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'crafting_table': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'dirt': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'furnace': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_axe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_ingot': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_ore': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_pickaxe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'log': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'planks': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'stick': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'stone': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'stone_axe': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'stone_pickaxe': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'torch': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'wooden_axe': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64),
                'wooden_pickaxe': spaces.Box(np.full([1], -1), np.full([1], -1), dtype=np.int64)},
            'pov': spaces.Box(np.full([64, 64, 3], -1), np.full([64, 64, 3], -1), dtype=np.uint8)}

        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': ['rgb_array', 'human']}

        self.t = 0

    def reset(self):
        self.t = 0
        self.state['pov'][:, :, :] = 0
        return deepcopy(self.state)

    def step(self, action):
        self.t += 1
        self.state['pov'][:, :, :] = 0
        if self.t < 1000:
            return deepcopy(self.state), 0.1, False, {}
        else:
            return deepcopy(self.state), 0.1, True, {}

    def close(self):
        pass

    def seed(self, _):
        pass


class Env(Wrapper):
    """Main minecraft wrapper. Wraps the actions and the states with the action and state managers and creates torch
    arrays on the specified device. Also activates always jumping, unless: craft, nearbyCraft, nearbySmelt,
    place or attack"""

    def __init__(self, env, state_manager, action_manager):

        self.state_manager = state_manager
        self.action_manager = action_manager

        self.done = False

        self.last_obs = None  # used for logging

        super().__init__(env)

    def _process_obs(self, obs):
        img, vec = self.state_manager.get_img_vec(obs)

        torch_img, torch_vec = self.state_manager.get_torch_img_vec([img], [vec])

        return torch_img, torch_vec

    def reset(self):
        obs = self.env.reset()

        self.last_obs = obs

        self.done = False
        return self._process_obs(obs)

    def step(self, action):
        assert not self.done

        action = self.action_manager.get_action(action)

        if 'craft' in action:
            if \
                    action['attack'] == 0 and \
                    action['craft'] == 0 and \
                    action['nearbyCraft'] == 0 and \
                    action['nearbySmelt'] == 0 and \
                    action['place'] == 0:
                action['jump'] = 1
        else:
            if action['attack'] == 0:
                action['jump'] = 1

        obs, r, self.done, info = super().step(action)

        self.last_obs = obs

        torch_img, torch_vec = self._process_obs(obs)
        return torch_img, torch_vec, r, self.done


def test_policy(writer, wrapped_env, policy, init_img, init_vec, episodes=100):

    reward_list = []
    steps_list = []

    amount_of_episodes_with_saved_inventory = 30

    first = True
    i = 0
    while i < episodes:

        if first:
            img, vec = init_img, init_vec
            first = False
        else:
            wrapped_env.seed(i)
            img, vec = wrapped_env.reset()

        print("episode {}".format(i))
        reward = 0.
        frame_steps = 0
        steps = 0
        last_obs = wrapped_env.last_obs
        last_meta_id = 0

        if 'inventory' in last_obs:
            last_print_dict = deepcopy(last_obs['inventory'])
            last_print_dict['meta_id'] = last_meta_id
            if i < amount_of_episodes_with_saved_inventory:
                writer.add_scalars(f"episode {i} inventory", last_print_dict, steps)
            
        done = False
        while not done:
            a_id = policy(img, vec)

            img, vec, r, done = wrapped_env.step(a_id)

            reward += r
            steps += 1
            frame_steps += 1

            obs = wrapped_env.last_obs

            if 'inventory' in last_obs:
                print_dict = deepcopy(obs['inventory'])
                if i < amount_of_episodes_with_saved_inventory:
                    if print_dict != last_print_dict:
                        writer.add_scalars(f"episode {i} inventory", print_dict, steps)
                last_print_dict = print_dict

            tmp_p = 6000
            if steps % tmp_p == 0:
                print(f"{frame_steps} / {18000}")

        if steps == 1:
            print("always terminal bug detected")
            raise RuntimeError

        else:
            reward_list.append(reward)
            steps_list.append(steps)
            print(f"episode reward: {reward} , episode terminated after {steps} env steps")
            print(f"avg_reward after {i + 1} (out of {episodes}) episodes: {np.mean(reward_list)}")
            writer.add_scalar("reward", reward, i)
            writer.add_scalar("steps", steps, i)
            writer.flush()

        i += 1

    print("Total avg_reward: {}".format(np.mean(reward_list)))
    writer.add_scalar("avg_reward", np.mean(reward_list), 0)
    writer.add_scalar("avg_steps", np.mean(steps_list), 0)
    writer.flush()
    writer.close()
