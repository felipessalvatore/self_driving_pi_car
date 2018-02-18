# Self Driving Raspberry-NXT car 

[![Build Status](https://travis-ci.org/felipessalvatore/self_driving_project.svg?branch=master)](https://travis-ci.org/felipessalvatore/self_driving_project)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/self_driving_project/blob/master/LICENSE)

## Introduction
short info about the project and team members (table of contents with hyperlinks maybe?)

<p align = 'center'>
<img src = 'images/track.png' height = '270px'>
</p>
<p align = 'center'>
Robot driving on a track
</p>


## Getting Started

### Install

The first thing you need to do is to install all the libraries for the Raspberry Pi. To do so, open a terminal in Raspberry Pi and run

```
$ cd raspi_utils/
$ bash install.sh
```

In the computer that you will perform the training -- from now on referred as the "training computer" (protip: don't train the model in the Raspberry pi!) -- install all the requirements by runnig

```
$ pip install -r requirements.txt
```

## Usage

**Attention**
In the Master branch all python code is written for Python 2. If you would like to run this project in Python3, please switch to the Python3 branch of this repository.

"Don't use Windows..."

### Collecting data

Before doing any kind of training you need to collect the track data. So in the Raspberry Pi -- with the assembled robot -- run the data collection script:
```
  $ cd self_driving/data_collection/ 
  $ python DataCollector.py -n <images_folder_name>
```

Inside the folder `<images_folder_name>` there will be subdirectories organized by timestamps similar to `2018-02-17-23-27-02` with the collected `*.png` images. All the associated labels are saved in a pickle file `2018-02-17-23-27-02_pickle` in `<images_folder_name>`.

Compress `<images_folder_name>` directory and export it from Raspberry Pi to other computer (using scp command, cloud, email, etc).
```
  $ tar cvf <images_folder_name>.tar <images_folder_name>
```

**Attention**
Please continue following the instructions in the computer that will be use for training.

### Generating npy and tfrecords

Before generating tfrecords, you need to transform the untar `<images_folder_name>` containing all folders of images and pickles into a tuple of np.arrays. Running the following script will result in the creation of `<npy_files_name>_90_160_3_data.npy` and `<npy_files_name>_90_160_3_labels.npy` files:
```
  $ cd self_driving/data_manipulation/
  $ python img2array.py <images_folder_path> <npy_folder_path> <npy_files_name>
```

To generate tfrecords from `*.npy` and augment or manipulate (e.g. binarize) the data, run:
 ```
  $ cd ../ml_training/ 
  $ python generate_tfrecords.py <npy_data_path> <npy_labels_path> -n <name_tfrecords> 
```

Resulting in `<name_tfrecords>_train.tfrecords`, `<name_tfrecords>_test.tfrecords` and `<name_tfrecords>_valid.tfrecords` files.

### Hyperparameters optimization

**Attention**
All code in this section can be runned on both Python 2 and 3 with TensorFlow 1.2.1 (and above) and with GPU support, if possible.

Now it's time to test different architectures, learning rates and optimizers, in the hopes of improving accuracy. 

#### Best architecture search

Running the following script will creat `architecture_results.txt` file with the results for a given configuration passed through optional arguments.
 ```
  $ python best_architecture.py -n <name_tfrecords>
```

#### Best learning rate search

Running the following script will creat `learning_rate_results.txt` file with the results for a given configuration passed through optional arguments.
 ```
  $ python best_learning_rate.py -n <name_tfrecords>
```

#### Best optimizer search

Running the following script will creat `optimizer_results.txt` file with the results for a given configuration passed through optional arguments.
 ```
  $ python best_optimizer.py -n <name_tfrecords>
```

### Training the model (FINALLY)

**Attention**
Back to Python 2

After searching for an appropriate combination of hyperparameters, you must train the model running this script with additional arguments relative to the model:

```
  $ python train.py -n <name_tfrecords> -v
```

The result will be a `checkpoints` directory with all files needed to deploy the model.

Having a checkpoints directory .....

#### Accuracy test

#### Simulation

### Self driving


### Running the tests

There is two kind of tests: the ones from the Raspberry Pi and the ones for the training computer.
In the Raspberry Pi run

```
$ python setup.py test 
```
These tests serve to check if the conection with the NXT robot is working.

And in the training computer
```
  $ bash test_script.sh 
```
These last tests check if the image manipulation functions and the tensorflow model are doing what they suppose to be doing.



## Built With

* [Tensorflow](https://www.tensorflow.org/)
* [NXT-Python](https://github.com/Eelviny/nxt-python)

### Demo
![alt text](images/run_readme.gif)

### Citation
```
  @misc{self_driving_project2018,
    author = {Paula Moraes, Felipe Salvatore},
    title = {Self Driving Project},
    year = {2018},
    howpublished = {\url{https://github.com/felipessalvatore/self_driving_project}},
    note = {commit xxxxxxx}
  }
```
### Related Work
- Project M (https://medium.com/@project_m)
