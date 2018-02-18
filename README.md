# self_driving_project

[![Build Status](https://travis-ci.org/felipessalvatore/self_driving_project.svg?branch=master)](https://travis-ci.org/felipessalvatore/self_driving_project)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/self_driving_project/blob/master/LICENSE)


<p align = 'center'>
<img src = 'images/track.png' height = '2783px'>
</p>
<p align = 'center'>
Robot driving on a track
</p>


## Getting Started

### Install

The first thing you need to do is to install all the libraries for the Raspberry pi. To do so, open a terminal in Raspberry pi and run

```
$ cd raspi_utils/
$ bash install.sh
```

In the computer that you will perform the training -- from now on referred as the "training computer" (protip: don't train the model in the Raspberry pi!) -- install all the requirements by runnig

```
$ pip install -r requirements.txt
```

## Running the tests

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

## Usage

Before doing any kind of training you need to collect the track data. So in the Raspberry pi -- with the assembled robot -- run the data collection script:
```
  $ cd self_driving/data_collection/ 
  $ python DataCollector.py -n <images_folder_name>
```



## Built With

* [Tensorflow](https://www.tensorflow.org/)
* [NXT-python](https://github.com/Eelviny/nxt-python)

### Demo
![Alt Text](https://media.giphy.com/media/1j8Qf5yPZXev2zfDZY/giphy-downsized-large.gif)

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