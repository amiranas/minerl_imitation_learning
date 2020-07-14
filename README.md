# Scaling Imitation Learning in Minecraft

**[Accompanying technical report link](https://arxiv.org/abs/2007.02701)**

Code for imitation learning of the [MineRL](https://minerl.io/) `MineRLObtainDiamond-v0` task. 
This implementation is an improved version of our submission to the [Minecraft  competition  on  sampleefficient reinforcement learning at NeurIPS 2019](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition). 
The initial version was able to reach the second place in the competition without using the Minecraft environment during training. A video of one cherry picked episode is available [here](https://youtu.be/ocCJXzNmzHk).

The implementation is partly based on [Kaixhin's Pytorch implementation of Rainbow](https://github.com/Kaixhin/Rainbow), and the `deep_resnet` architecture is based on the network architecture used in [unixpickle's Obstacle Tower Challenge solution](https://github.com/unixpickle/obs-tower2).

## Dependencies:

**Tested with:**
* minerl==0.2.9
* torch==1.2.0

## Training:

Creating a dataset from the MineRL human data:

    python main.py --logdir <PATH_TO_LOG_DATA_FOLDER> --save_dataset_path <DATASET_WILL_BE_SAVED_HERE> --minecraft_human_data_dir <PATH_TO_MineRL_DATA> --quit_after_saving_dataset True

`save_dataset_path` requires a path together with the future dataset name. `minecraft_human_data_dir` requires the path to the human data folder.

If `quit_after_saving_dataset` is set to false, training will start after the dataset creation.

Otherwise training can be started by running:

    python main.py --logdir <PATH_TO_LOG_DATA_FOLDER> --dataset_path <PATH_TO_DATASET>
    
The training will save snapshots into the log folder (every 100,000 train steps). Last snapshot will have an id of 'last'.

## Evaluation:

Evaluating a snapshot:

    xvfb-run -a python main.py --logdir <PATH_TO_LOG_DATA_FOLDER> --eval_policy True --eval_policy_path <PATH_TO_TRAINING_LOG_DATA_FOLDER> --eval_policy_model_id <SNAPSHOT_ID>
    
(The snapshot id is either the according number or 'last')

The minecraft environment will only be started during evaluation.

___

If you use this code, please cite our accompanying tech report:

    @misc{amiranashvili2020scaling,
        title={Scaling Imitation Learning in Minecraft},
        author={Artemij Amiranashvili and Nicolai Dorka and Wolfram Burgard and Vladlen Koltun and Thomas Brox},
        year={2020},
        eprint={2007.02701},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
