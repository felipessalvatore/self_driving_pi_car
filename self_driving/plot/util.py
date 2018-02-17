import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from pandas_ml import ConfusionMatrix  # noqa
import itertools # noqa


def plot_confusion_matrix(cm,
                          classes,
                          title,
                          normalize=False,
                          cmap=plt.cm.Oranges,
                          path="confusion_matrix.png"):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    'cmap' controls the color plot. colors:

    https://matplotlib.org/1.3.1/examples/color/colormaps_reference.html

    :param truth: true labels
    :type truth: np array
    :param predictions: model predictions
    :type predictions: np array
    :param path: path to save image
    :type path: str
    :param label_dict: dict to transform int to str
    :type label_dict: dict
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.savefig(path)


def plotconfusion(truth, predictions, path, label_dict, classes):
    """
    This function plots the confusion matrix and
    also prints useful statistics.

    :param truth: true labels
    :type truth: np array
    :param predictions: model predictions
    :type predictions: np array
    :param path: path to save image
    :type path: str
    :param label_dict: dict to transform int to str
    :type label_dict: dict
    """
    acc = np.array(truth) == np.array(predictions)
    size = float(acc.shape[0])
    acc = np.sum(acc.astype("int32")) / size
    truth = [label_dict[i] for i in truth]
    predictions = [label_dict[i] for i in predictions]
    cm = ConfusionMatrix(truth, predictions)
    cm_array = cm.to_array()
    cm_diag = np.diag(cm_array)
    sizes_per_cat = []
    for n in range(cm_array.shape[0]):
        sizes_per_cat.append(np.sum(cm_array[n]))
    sizes_per_cat = np.array(sizes_per_cat)
    sizes_per_cat = sizes_per_cat.astype(np.float32) ** -1
    recall = np.multiply(cm_diag, sizes_per_cat)
    print("\nRecall:{}".format(recall))
    print("\nRecall stats: mean = {0:.6f}, std = {1:.6f}\n".format(np.mean(recall), # noqa
                                                                    np.std(recall))) # noqa
    title = "Confusion matrix of {0} examples\n accuracy = {1:.6f}".format(int(size), # noqa
                                                                           acc)
    plot_confusion_matrix(cm_array, classes, title=title, path=path)
    cm.print_stats()
