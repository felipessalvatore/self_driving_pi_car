# Self Driving Raspberry-NXT car 

[![Build Status](https://travis-ci.org/felipessalvatore/self_driving_project.svg?branch=master)](https://travis-ci.org/felipessalvatore/self_driving_project)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/self_driving_project/blob/master/LICENSE)

## Introduction
short info about the project and team members (table of contents maybe?)

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
  $ tar cvf <filename>.tar <images_folder_name>
```

**Attention**
Please continue following the instructions in the computer that will be used for training.

### Generating npy and tfrecords

Before generating tfrecords, you need to transform the `<images_folder_name>` containing all folders of images and pickles into a tuple of np.arrays. Running the following script will result in the creation of `data.npy` and `labels.npy`:
```
  $ python img2array.py <images_folder_path> <npy_folder_path> <npy_files_name>
```

### Running the tests

There is two kind of tests: the ones from the Raspberry pi and the ones for the training computer.
In the Raspberry pi run

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
