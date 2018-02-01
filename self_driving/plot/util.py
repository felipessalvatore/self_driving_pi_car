import matplotlib.pyplot as plt
import numpy as np
from pandas_ml import ConfusionMatrix

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]: i[0] for i in command2int.items()}


def plotconfusion(truth, predictions):
    """
    Function to plot the confusion fuction between the
    truth and predictions array.
    :type truth: np array
    :type predictions: np array
    """
    acc = np.array(truth) == np.array(predictions)
    size = acc.shape[0]
    acc = np.sum(acc.astype("int32")) / size
    truth = [int2command[i] for i in truth]
    predictions = [int2command[i] for i in predictions]
    cm = ConfusionMatrix(truth, predictions)
    plt.figure(figsize=(9, 9))
    cm.plot(backend='seaborn')
    title = "Confusion matrix of {} examples\n accuracy = {}".format(size, acc) 
    plt.title(title, fontsize=24, fontweight='bold')
    plt.show()


def accuracy_per_category(pred, label, categories, cat_dict=None):
    pred, label = list(pred), list(label)
    results = []
    result_str = ''
    for cat in range(categories):
        f = lambda x: 1 if x == cat else 0 # noqa
        vfunc = np.vectorize(f)
        mapped_pred = vfunc(pred)
        mapped_labels = vfunc(label)
        right = float(np.dot(mapped_pred, mapped_labels))
        total = np.sum(mapped_labels)
        if total == 0:
            results.append(0.0)
        else:
            results.append((right / total))

    for i, result in enumerate(results):
        if cat_dict is None:
            result_str += str(i) + " : {0:.3f}\n".format(result)
        else:
            result_str += cat_dict[i] + " : {0:.3f}\n".format(result)

    return result_str
