import os
import argparse
import numpy as np
import image_manipulation
# from PIL import Image

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]:i[0] for i in command2int.items()}


def get_image_and_command(data_index, 
                         label_index):
    """
    Get and reshape image with parameters: 90(height), 160(width), 3(channels) from data_index
    and get it's label in label_index (e.g. 'right') 

    :type data_index: numpy.ndarray
    :type label_index: numpy.ndarray
    :rtype: numpy.ndarray, str
    """
    img_array = data_index.reshape((90, 160,3))
    command = int2command[label_index[0]]
    return img_array, command

def get_image(data_index):
    """
    Get and reshape image with parameters: 90(height), 160(width), 3(channels) from data_index

    :type data_index: numpy.ndarray
    :rtype: numpy.ndarray
    """
    return data_index.reshape((90, 160,3))

def get_flat_shape(image):
    """
    Multiply each shape component of image (tuple of array dimensions)

    :type image: numpy.ndarray
    :rtype: int
    """
    flat = 1
    for i in range(len(image.shape)):
        flat *= image.shape[i] 
    return flat

def shape2filename(data_shape):
    """
    Get each shape component and return a string formatted to 'height_width_channels_'

    :type data_shape: numpy.ndarray
    :rtype: str
    """
    name = ""
    for i in data_shape:
        name += "{}_".format(i)
    return name

def load_dataset(data_path, 
                 labels_path):
    """
    Load and return dataset arrays from data_path and label arrays from labels_path

    :type data_path: str (.npy file)
    :type labels_path: str (.npy file)
    :rtype: numpy.ndarray, numpy.ndarray
    """
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def save_dataset(data, 
                 labels, 
                 folder_path, 
                 data_shape, 
                 name):
    """
    Save data and labels in a directory as a numpy array binary file (NPY)

    :type data: numpy.ndarray
    :type labels: numpy.ndarray
    :type folder_path: str 
    :type data_shape: tuple of numpy array dimensions 
    :type name: str 
    :rtype: 
    """
    shape = shape2filename(data_shape)
    data_name = "{}_{}data.npy".format(name, shape)
    label_name = "{}_{}labels.npy".format(name, shape)
    data_path = os.path.join(folder_path,data_name)
    labels_path = os.path.join(folder_path,label_name)
    np.save(data_path, data)
    np.save(labels_path, labels)

def extend_dataset_flip_axis(data, 
                             labels):
    """
    Balance and extend dataset by generating new images flipping the horizontal axis (only applicable to images labeled 'left' or 'right')

    :type data: numpy.ndarray
    :type labels: numpy.ndarray
    :rtype: numpy.ndarray, numpy.ndarray
    """
    all_images = []
    all_labels = []
    flat_shape = data.shape[1]
    for i in range(data.shape[0]):
        orig_label = labels[i]
        if (orig_label / 2 < 1):
            continue
        frame, cmd = get_image_and_command(data[i], labels[i])
        if (orig_label % 2 == 0):
            flip_cmd = 3
        else:
            flip_cmd = 2
        flip = np.flip(frame, axis=1)
        flip = np.array(flip.reshape(flat_shape))
        all_images.append(flip)
        all_labels.append(flip_cmd)       
    all_labels = np.array(all_labels).astype('uint8')
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    extended_images = np.concatenate((data, all_images), axis=0)
    extended_labels = np.concatenate((labels, all_labels), axis=0)
    return extended_images, extended_labels

def dataset_augmentation(data):
    """
    Create new datasets by applying in data image transformations, available at image_manipulation. 
    Returns a new dataset and the original shape of its contents

    :type data: numpy.ndarray
    :rtype: numpy.ndarray, tuple
    """
    new_dataset = []
    for i in range(data.shape[0]):
        image = get_image(data[i])
        new_image = image_manipulation.binarize_image(image)

        original_shape = new_image.shape
        new_image = new_image.reshape(get_flat_shape(new_image))
        new_dataset.append(new_image)
    new_dataset = np.array(new_dataset).astype('uint8')
    return new_dataset, original_shape


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',
                        type=str, help='path to current data')
    parser.add_argument('labels_path',
                        type=str, help='path to current labels')
    parser.add_argument('new_data_folder_path',
                        type=str, help='path to data and labels to be saved')    
    parser.add_argument('dataset_name',
                        nargs='?', default='dataset', type=str, help='name for dataset. (Default) dataset')

    user_args = parser.parse_args()

    data, labels = load_dataset(user_args.data_path, 
                                user_args.labels_path)
    data, labels = extend_dataset_flip_axis(data, 
                                            labels)
    new_dataset, data_shape = dataset_augmentation(data)
    save_dataset(new_dataset, 
                 labels, 
                 user_args.new_data_folder_path,
                 data_shape,
                 user_args.dataset_name)


if __name__ == '__main__':
    main()