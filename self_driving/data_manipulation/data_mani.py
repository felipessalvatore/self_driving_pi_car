import argparse
import pickle
import os
from scipy import misc
import numpy as np

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]: i[0] for i in command2int.items()}


def folder2array(folder_path,
                 pickle_path,
                 image_width,
                 image_height,
                 image_channels):
    """
    Function to transform all images from the folder folder_name
    into a tuple of np arrays.

    :param
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
        label = command2int[label_dict[key]]
        image_path = os.path.join(folder_path, filename)
        image = change_type_to_uint8(misc.imread(image_path))
        image = image.reshape(flat_shape)
        all_images.append(image)
        all_labels.append(label)
    all_labels = change_type_to_uint8(np.array(all_labels))
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    all_images = np.array(all_images)
    return all_images, all_labels


def change_type_to_uint8(image):
    """
    Change type to uint8 Unsigned integer (0 to 255)

    :type image: np array
    :param
    :rtype: np array
    """
    image = image.astype('uint8')
    return image


def create_data_set_as_np_array(folder_path,
                                data_name="data",
                                label_name="labels",
                                image_width=160,
                                image_height=90,
                                image_channels=3):
    """
    Giving one path to a folder of folders of images,
    this function transform all images in two arrays
    one 'data_name' with all the flatted images
    and other 'label_name' with all the respective labels

    :type folder_name: str
    :param
    :type data_name: str
    :type data_name: str
    """
    assert os.path.exists(folder_path)
    all_images = []
    all_labels = []
    for folder in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder)
        if os.path.isdir(folder):
            print(folder)
            pickle_path = folder + "_pickle"
            images, labels = folder2array(folder,
                                          pickle_path,
                                          image_width,
                                          image_height,
                                          image_channels)
            all_images.append(images)
            all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_images.shape, all_labels.shape)
    np.save(data_name, all_images)
    np.save(label_name, all_labels)


def main():
    """
    to do
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('img_folder_path',
                        type=str, help='path to image folder')
    parser.add_argument('data_path',
                        type=str, help='path to data to be saved')
    parser.add_argument('labels_path',
                        type=str, help='path to labels to be saved')
    parser.add_argument("-w",
                        "--image_width",
                        type=int,
                        default=160,
                        help="width number (default=160)")
    parser.add_argument("-H",
                        "--image_height",
                        type=int,
                        default=90,
                        help="height number (default=90)")
    parser.add_argument("-c",
                        "--image_channels",
                        type=int,
                        default=3,
                        help="number of channels (default=3)")
    user_args = parser.parse_args()
    create_data_set_as_np_array(user_args.img_folder_path,
                                user_args.data_path,
                                user_args.labels_path,
                                user_args.image_width,
                                user_args.image_height,
                                user_args.image_channels)


if __name__ == '__main__':
    main()
