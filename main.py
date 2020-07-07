from __future__ import division
import argparse
import os
import sys
import numpy as np
import torch

from agent import Agent
from minecraft import DummyMinecraft, Env, test_policy
from dataset import Dataset, Transition

import pickle
import time
from os.path import join as p_join
from os.path import exists as p_exists

from data_manager import StateManager, ActionManager

from get_dataset import put_data_into_dataset

import minerl
import gym

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description='Rainbow')
    
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--learning_rate', type=float, default=0.0000625)
    parser.add_argument('--adam_eps', type=float, default=1.5e-4)
    parser.add_argument('--enable_cudnn', type=str2bool, default=True)
    
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument("--logdir", default=".", type=str, help="used for logging and to save network snapshots")

    parser.add_argument('--c_action_magnitude', type=float, default=22.5, help="magnitude of discretized camera action")
    parser.add_argument("--net", default='deep_resnet', type=str,
                        choices=['normal', 'resnet', 'deep_resnet', 'double_deep_resnet'])
    parser.add_argument('--hidden_size', type=int, default=1024, help="size of main fully-connected layers")

    parser.add_argument('--dataset_path', type=str, default=None,
                        help="use if dataset is already created")

    parser.add_argument('--trainsteps', type=int, default=3000000)
    parser.add_argument('--augment_flip', type=str2bool, default=True)

    parser.add_argument('--dataset_only_successful', type=str2bool, default=True)
    parser.add_argument('--dataset_use_max_duration_steps', type=str2bool, default=True)
    parser.add_argument('--dataset_continuous_action_stacking', type=int, default=3)
    parser.add_argument('--dataset_max_reward', type=int, default=256)
    parser.add_argument('--minecraft_human_data_dir', type=str, default=None,
                        help="location of MineRL human data")

    parser.add_argument('--save_dataset_path', type=str, default=None)
    parser.add_argument('--quit_after_saving_dataset', type=str2bool, default=False)

    parser.add_argument('--dueling', type=str2bool, default=True)

    parser.add_argument('--scale_rewards', type=str2bool, default=True)

    parser.add_argument('--eval_policy', type=str2bool, default=False)
    parser.add_argument('--eval_policy_path', type=str, default=None)
    parser.add_argument('--eval_policy_model_id', type=str, default="last")
    parser.add_argument('--eval_policy_episodes', type=int, default=100)

    parser.add_argument('--add_treechop_data', type=str2bool, default=True,
                        help="Set to true to create a dataset with additional Treechop trajectories")

    parser.add_argument('--test', type=str2bool, default=False, help="for debugging")

    parser.add_argument('--stop_time', type=int, default=None,
                        help="Maximal training time in hours."
                             "Will save tmp snapshot after the time limit is over."
                             "Starting a training run with identical logdir will "
                             "continue the training from the tmp snapshot.")

    return parser.parse_args(raw_args)


def rl(args):
    """ main function for both training and evaluation. Default is set to train mode.
    Set args.eval_policy = True for evaluation

    :param args: Get default parameters from get_args()
    """

    init_time = time.time()

    if not args.eval_policy:
        if p_exists(p_join(args.logdir, 'model_last.pth')):
            print("Training already finished")
            return

        if p_exists(p_join(args.logdir, "tmp_time.p")):
            print("Detected tmp snapshot, will continue training from there")
            continue_from_tmp = True
        else:
            continue_from_tmp = False

    # Setup

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    assert os.path.exists(args.logdir)

    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))

    assert torch.cuda.is_available()
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
    args.device = torch.device('cuda')

    print(f"Running on {args.device}")

    state_manager = StateManager(args.device)
    action_manager = ActionManager(args.device, args.c_action_magnitude)

    # ########################################### CREATE ENVIRONMENT ###################################################

    if args.eval_policy and not args.test:
        env_ = gym.make('MineRLObtainDiamond-v0')
        env_.seed(0)
    else:
        env_ = DummyMinecraft()
        env_.seed(args.seed)

    env = Env(env_, state_manager, action_manager)

    print("started env")

    img, vec = env.reset()

    print("env reset")

    print("img, vec shapes: ", img.shape, vec.shape)
    
    # ########################################### GET ENV DATA AND WRITER ##############################################

    num_actions = action_manager.num_action_ids_list[0]
    image_channels = img.shape[1]

    vec_size = vec.shape[1]
    vec_shape = vec.shape[1:]

    img_shape = list(img.shape[1:])
    img_shape[0] = int(img_shape[0])

    writer = SummaryWriter(args.logdir)

    with open(p_join(args.logdir, "status.txt"), 'w') as status_file:
        status_file.write('running')

    # extended error exception:
    def handle_exception(exc_type, exc_value, exc_traceback):

        with open(p_join(args.logdir, "status.txt"), 'w') as status_file_:
            status_file_.write('error')

        writer.flush()
        writer.close()
        env.close()
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    # ########################################### GET DATASET ##########################################################

    if not args.eval_policy:
        dataset = Dataset(args.device, 2000000, img_shape, vec_shape,
                          state_manager, action_manager,
                          scale_rewards=args.scale_rewards)

        if args.dataset_path is not None:  # default None
            
            print(f"loading dataset {args.dataset_path}")
            dataset.load(args.dataset_path)
            print(f"loaded dataset")

        else:  # creating dataset:
            
            assert args.minecraft_human_data_dir is not None

            print("creating dataset")

            if args.dataset_use_max_duration_steps:  # default: True
                max_iron_pickaxe_duration = 6000
                max_diamond_duration = 18000
            else:
                max_iron_pickaxe_duration = None
                max_diamond_duration = None

            put_data_into_dataset(
                'MineRLObtainIronPickaxe-v0', action_manager, dataset, args.minecraft_human_data_dir,
                args.dataset_continuous_action_stacking,
                args.dataset_only_successful,
                max_iron_pickaxe_duration,
                args.dataset_max_reward,
                args.test)

            put_data_into_dataset(
                'MineRLObtainDiamond-v0', action_manager, dataset, args.minecraft_human_data_dir,
                args.dataset_continuous_action_stacking,
                args.dataset_only_successful,
                max_diamond_duration,
                args.dataset_max_reward,
                args.test)

            if args.add_treechop_data:
                put_data_into_dataset(
                    'MineRLTreechop-v0', action_manager, dataset, args.minecraft_human_data_dir,
                    args.dataset_continuous_action_stacking,
                    args.dataset_only_successful,
                    None,
                    args.dataset_max_reward,
                    args.test)

            if args.save_dataset_path is not None:
                dataset.save(args.save_dataset_path)
                print(f"saved new dataset{args.save_dataset_path} with {dataset.transitions.index} transitions")
                
                if args.quit_after_saving_dataset:
                    print("stopping after saving the new dataset")
                    return
                else:
                    print("continuing with new dataset")
            else:
                print("continuing with new dataset without saving")

        for j in range(dataset.transitions.index):
            dataset.transitions.data[j] = Transition(
                dataset.transitions.data[j].state.pin_memory(),
                dataset.transitions.data[j].vector.pin_memory(),
                dataset.transitions.data[j].action,
                dataset.transitions.data[j].reward,
                dataset.transitions.data[j].nonterminal
            )

    # ########################################### CREATE NETWORK #######################################################

    agent = Agent(num_actions, image_channels, vec_size, writer,
                  args.net, args.batch_size, args.augment_flip, args.hidden_size, args.dueling,
                  args.learning_rate, args.adam_eps, args.device)

    # ########################################### EVALUATION ###########################################################  

    if args.eval_policy:

        assert args.eval_policy_path is not None

        agent.load(args.eval_policy_path, args.eval_policy_model_id)

        print(f"loaded network {args.eval_policy_path} {args.eval_policy_model_id}")
        
        policy = agent.act

        with open(p_join(args.logdir, "status.txt"), 'w') as status_file:
            status_file.write('running test_policy')

        if args.test:
            args.eval_policy_episodes = 2

        test_policy(writer, env, policy, img, vec, args.eval_policy_episodes)

    # ########################################### TRAINING #############################################################

    else:
        print("starting TRAINING")

        if continue_from_tmp:
            start_int = pickle.load(open(p_join(args.logdir, "tmp_time.p"), "rb"))
            print(f"continuing from {start_int} trainstep")
            agent.load(args.logdir, "tmp")
        else:
            start_int = 0

        agent.train()

        with open(p_join(args.logdir, "status.txt"), 'w') as status_file:
            status_file.write('running training')

        if args.test:
            args.trainsteps = 10
            
        fps_t0 = time.time()

        for i in range(start_int, args.trainsteps):

            agent.learn(i, dataset, write=(i % 1000 == 0))

            if i and i % 100000 == 0:
                agent.save(args.logdir, i // 100000)

            if args.stop_time is not None:
                if ((time.time() - init_time) / 60. / 60.) > args.stop_time:
                    print(f"{(time.time() - init_time) / 60. / 60.} h passed, saving tmp snapshot", flush=True)
                    agent.save(args.logdir, "tmp")
                    pickle.dump(int(i), open(p_join(args.logdir, "tmp_time.p"), 'wb'))
                    writer.close()
                    print('saved')
                    return

            if (i+1) % 5000 == 0:
                fps = float(i - start_int) / (time.time() - fps_t0)
                writer.add_scalar("fps", fps, i)

        agent.save(args.logdir, 'last')

        print("finished TRAINING")

    # ########################################### OUTRO  ###############################################################

    with open(p_join(args.logdir, "status.txt"), 'w') as status_file:
        status_file.write('finished')

    env.close()


if __name__ == '__main__':
    rl(get_args())
