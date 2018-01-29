import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix


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