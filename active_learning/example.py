import os
import sys

from runners.image_classification_eval import create_argparser


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from guided_diffusion import dist_util
from active_learning.trainer import ActiveLearner
from active_learning.utils import ActiveLearnerConfig, DatasetConfig, TrainConfig


def main():
    dist_util.setup_dist()
    args = create_argparser().parse_args()
    active = ActiveLearner()
    active.learn(args.model)


if __name__ == "__main__":
    main()
