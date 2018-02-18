'''
Dataset histogram
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(labels, path):
    """
    Plot dataset histogram

    :param label_path: array of labels
    :type label_path: np.array
    :param path: name to save histogram
    :type path: np.str
    """

    data_hist = plt.hist(labels, bins=np.arange(4) - 0.5, edgecolor='black')
    axes = plt.gca()  # Get Current Axes
    axes.set_ylim([0, len(labels)])

    plt.title("Histogram of {} images".format(len(labels)))
    plt.xticks(np.arange(4), ['up', 'left', 'right'])
    plt.xlabel("Label")
    plt.ylabel("Frequency")

    for i in range(3):
        plt.text(data_hist[1][i] + 0.25,
                 data_hist[0][i] + (data_hist[0][i] * 0.01),
                 str(int(data_hist[0][i])))

    plt.savefig(path)


def main():
    """
    Plot label's histogram
    """
    description = "plot label's histogram"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('labels_path',
                        type=str, help='path to labels')
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="histogram",
                        help="name to save histogram plot (default=histogram)")  # noqa
    args = parser.parse_args()
    labels = np.load(args.labels_path)
    plot_histogram(labels, args.name)


if __name__ == '__main__':
    main()
