import os
import sys
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from guided_diffusion import dist_util
from active_learning.trainer import ActiveLearner


def main():
    dist_util.setup_dist()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the model")
    parser.add_argument("--width", type=int, required=True, help="Width of the model")
    parser.add_argument("--dropout_rate", type=float, required=True, help="Dropout rate")
    parser.add_argument("--norm", type=str, required=True, help="Normalization layer")
    parser.add_argument("--budget_ratio", type=float, required=True, help="Budget ratio")
    parser.add_argument("--sampling_method", type=str, required=True, help="Sampling method")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--decay_epochs", nargs="+", type=int, required=True, help="Decay epochs")
    parser.add_argument("--decay_rate", type=float, required=True, help="Decay rate")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument("--eval_freq", type=int, required=True, help="Evaluation frequency")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")

    args = parser.parse_args()

    active = ActiveLearner(
        budget_ratio=args.budget_ratio,
        sampling_method=args.sampling_method,
        args=args,
    )
    active.learn(args)


if __name__ == "__main__":
    main()
