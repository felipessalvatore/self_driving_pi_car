import tensorflow as tf
import os
import argparse
from Config import Config
from DataHolder import DataHolder
from util import int2command

def data_search(channels, 
                data_path, 
                label_path, 
                mode,
                flip=False,
                augmentation=False,
                gray=False,
                green=False,
                binary=False):

    config = Config(channels=channels)
    data = DataHolder(config,
                      data_path=data_path,
                      label_path=label_path,
                      record_path=mode,
                      flip=flip,
                      augmentation=augmentation,
                      gray=gray,
                      green=green,
                      binary=binary,
                      records=None)

    data.create_records() 

def main():
    """
    Main script to perform data search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="pure",
                        help="mode for data: pure, flip, aug, bin, gray, green (default=pure)")  # noqa

    parser.add_argument("-f",
                        "--flip",
                        action="store_true",
                        default=False,
                        help="flag to flip x-axis (default=False)")  # noqa

    parser.add_argument("-a",
                        "--augmentation",
                        action="store_true",
                        default=False,
                        help="flag to augment dataset (default=False)")  # noqa

    parser.add_argument("-gy",
                        "--gray",
                        action="store_true",
                        default=False,
                        help="flag to transform dataset in grayscale (default=False)")  # noqa

    parser.add_argument("-gr",
                        "--green",
                        action="store_true",
                        default=False,
                        help="flag to keep only the green channel (default=False)")  # noqa

    parser.add_argument("-b",
                        "--binary",
                        action="store_true",
                        default=False,
                        help="flag to binarize dataset (default=False)")  # noqa
 
    parser.add_argument("-d",
                        "--train_data",
                        type=str,
                        default=None,
                        help="train data path (default=None)")
 
    parser.add_argument("-l",
                        "--train_label",
                        type=str,
                        default=None,
                        help="label data path (default=None)")

    args = parser.parse_args()
    
    if args.mode == "bin" or args.mode == "gray" or args.mode == "green":
        channels = 1
    else:
        channels = 3
    
    #cond1 = type(args.train_data) == str
    #cond2 = type(args.train_label) == str
    #have_records = not (cond1 and cond2)
    data_search(channels, 
                args.train_data,
                args.train_label, 
                args.mode,
                args.flip,
                args.augmentation,
                args.gray,
                args.green,
                args.binary)


if __name__ == "__main__":
    main()

