# import argparse
# import inspect
# import pickle
import os
from scipy import misc
import cPickle as pickle
import numpy as np

command_dict = {"up": 0, "down": 1, "left": 2, "right": 3}


def command2int(command):
    """
    Command to int representation

    :param command: command label
    :type command: string
    :rtype: int
    """
    return command_dict[command]


def folder2array(folder_path,
                 pickle_path,
                 image_width=160,
                 image_height=90,
                 image_channels=3):
    """
    Function to transform all images from the folder folder_name
    into a tuple of np arrays.

    :type folder_path: str
    :type
    :rtype: (np.array,np.array
    """
    all_images = []
    all_labels = []
    image_width = 160
    image_height = 90
    image_channels = 3
    flat_shape = image_width * image_height * image_channels
    with open(pickle_path, "rb") as f:
        label_dict = pickle.load(f)
    print("Trying to convert images from {} \n".format(folder_path))
    for filename in os.listdir(folder_path):
        key = filename[:- 4]
        label = command2int(label_dict[key])
        image_path = os.path.join(folder_path, filename)
        image = feature_normalization(misc.imread(image_path))
        image = image.reshape(flat_shape)
        all_images.append(image)
        all_labels.append(label)
    all_labels = np.array(all_labels)
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    all_images = np.array(all_images)
    return all_images, all_labels


def feature_normalization(image):
    """
    Feature normalization for otimization
    https://en.wikipedia.org/wiki/Feature_scaling

    :type image: np array
    :rtype: np array
    """
    image = image.astype('float32')
    return image / 255


def create_data_set_as_pickle(folder_path, pickle_name="data"):
    """
    Giving one path to a folder of folders of images,
    this function do all the preprocessing
    and stores the data as a pickle.

    :type folder_name: str
    :type pickle_name: str
    :type verbose: boolean
    """
    assert os.path.exists(folder_path)
    all_images = []
    all_labels = []
    for folder in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder)
        if os.path.isdir(folder):
            print(folder)
            pickle_path = folder + "_pickle"
            images, labels = folder2array(folder, pickle_path)
            all_images.append(images)
            all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_images.shape, all_labels.shape)
    pickle_name += ".p"
    dict_file = {'images': all_images,
                 'labels': all_labels}
    # p = open(pickle_name, "wb")
    # pickle.dump(dict_file, p, protocol=pickle.HIGHEST_PROTOCOL)
    # p.close()
    with open(pickle_name, 'wb') as f:
        pickle.dump(dict_file, f)
    print("\npickle can be found in {}".format(pickle_name))


if __name__ == '__main__':
    folder_path = "/home/felsal/Desktop/pista1"
    create_data_set_as_pickle(folder_path)
