import argparse
import numpy as np
from PIL import Image, ImageDraw

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]:i[0] for i in command2int.items()}

data = np.load("data.npy")
labels = np.load("labels.npy")

print("data shape = {}".format(data.shape))
print("data type = {}".format(data.dtype))
print("labels shape = {}".format(labels.shape))
print("labels type = {}".format(labels.dtype))

def get_image_array_and_command(index):
    img_array = data[index].reshape((90, 160,3))
    command = int2command[labels[index][0]]
    return img_array, command

def flip_horizontal_axis(data_name, label_name):
    all_images = []
    all_labels = []
    image_width = 160
    image_height = 90
    image_channels = 3
    flat_shape = image_width * image_height * image_channels
    for i in range(data.shape[0]):
        orig_label = labels[i]
        if (orig_label / 2 < 1):
            continue
        frame, cmd = get_image_array_and_command(i)
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
   
    ext_images = np.concatenate((data, all_images), axis=0)
    ext_labels = np.concatenate((labels, all_labels), axis=0)
    print(ext_images.shape, ext_labels.shape)
    np.save(data_name, ext_images)
    np.save(label_name, ext_labels)


def binarize_dataset():
    # to do
    return 

def green_dataset():
    # to do
    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',
                        type=str, help='path to data to be saved')
    parser.add_argument('labels_path',
                        type=str, help='path to labels to be saved')
    user_args = parser.parse_args()
    flip_horizontal_axis(user_args.data_path,
                         user_args.labels_path)

if __name__ == '__main__':
    main()