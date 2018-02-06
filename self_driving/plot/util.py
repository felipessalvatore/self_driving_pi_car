import matplotlib.pyplot as plt
import numpy as np
from pandas_ml import ConfusionMatrix


def plotconfusion(truth, predictions, path, label_dict):
    """
    Function to plot the confusion fuction between the
    truth and predictions array.
    :type truth: np array
    :type predictions: np array
    :type path: str
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
    tp = np.multiply(cm_diag, sizes_per_cat)
    print("\nTrue positiv {0}:\nmean = {1:.6f}, std = {2:.6f}\n".format(tp,
                                                                        np.mean(tp), # noqa
                                                                        np.std(tp))) # noqa
    plt.figure(figsize=(9, 9))
    cm.plot(backend='seaborn', normalized=True)
    title = "Confusion matrix of {0} examples\n accuracy = {1:.6f}".format(int(size), # noqa
                                                                           acc)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.savefig(path)
    cm.print_stats()
