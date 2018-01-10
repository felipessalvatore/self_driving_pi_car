# Raspberry config
sudo apt-get remove --purge libreoffice*
sudo apt-get purge minecraft-pi
sudo apt-get purge wolfram-engine

# Installing tensorflow 1.2.1 on python2

sudo apt-get update
sudo apt-get upgrade
sudo reboot
sudo apt-get install python-pip python-dev python-numpy python-wheel
wget https://github.com/DeftWork/rpi-tensorflow/raw/master/tensorflow-1.2.1-cp27-none-linux_armv7l.whl
sudo pip install tensorflow-1.2.1-cp27-none-linux_armv7l.whl
pip uninstall mock
pip install mock


# Installing opencv-3.3.0 on python2
sudo apt-get install build-essential git cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
unzip opencv_contrib.zip
cd ~/opencv-3.3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
    -D BUILD_EXAMPLES=ON ..
make -j2
sudo make install
sudo ldconfig

# Installing libraries for nxt robot
sudo pip install nxt-python
sudo pip install keyboard
sudo pip install pybluez
sudo apt-get install bluetooth libbluetooth-dev

# Getting our repo for self-driving car!
git clone git@github.com:felipessalvatore/self_driving_project.git

