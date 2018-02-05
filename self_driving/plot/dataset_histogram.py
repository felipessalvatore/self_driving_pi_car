'''
Dataset histogram
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(labels):
    """
    Plot dataset histogram

    :param label_path: array of labels
    :type label_path: np.array
    """

    data_hist = plt.hist(labels, bins=np.arange(5) - 0.5, edgecolor='black')
    axes = plt.gca()  # Get Current Axes
    axes.set_ylim([0, len(labels)])

    plt.title("Label histogram of {} image dataset".format(len(labels)))
    plt.xticks(np.arange(5), ['up', 'down', 'left', 'right'])
    plt.xlabel("Label")
    plt.ylabel("Frequency")

    for i in range(4):
        plt.text(data_hist[1][i] + 0.25,
                 data_hist[0][i] + (data_hist[0][i] * 0.01),
                 str(int(data_hist[0][i])))

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_path',
                        type=str, help='path to labels')
    user_args = parser.parse_args()
    labels = np.load(user_args.labels_path)
    plot_histogram(labels)


if __name__ == '__main__':
    main()
