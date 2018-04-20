
![Smart Dollhouse with Gesture Recognition](./images/myhouse.myhouse.JPG)
===========

This repository contains code necessary to use a Raspberry Pi as a smart home controller using gesture recognition while utilizing Machine Learning at the edge. This code can work on any Raspberry Pi, including Raspberry Pi Zero and does not require internet connection for training or inference. 

Currently, the software reads accelerometer/gyro data from a sensor and enables you to use gestures to control the following:

- Fan (on, off, normal speed, double speed)
- Lights (M, Y letters are separately activated)
- Shutters (open or close)
- TV (on or off)


The code in this repo works with PlayStation Move controller but can be made to work with any other accelerometer without any data preprocessing. For example using smart phone, smart watch, small joystick with sensors are all valid input methods. 

The program is efficient enough to be ran even on the Pi Zero and should only take about 1-2% CPU with inference process taking about 20-30 milliseconds. 

Please read more about MYHouse and its story on my blog: https://maxoffsky.com/research-progress/project-myhouse-a-smart-dollhouse-with-gesture-recognition/

Core Features of MYhouse
------------------------

- Gesture recognition via accelerometer / gyroscope sensors
- Uses simple neural network that is fast and efficient in recognizing gestures
- Can be made to work with your own gestures
- Takes very little CPU power, making it a good candidate for low power device
- Can be made to run on Arduino with some work (by implementing MLP algorithm) and training on computer first



Pin connections
---------------

Currently the pins of the Raspberry Pi are connected in the following way:

- 6 = Fan servo 
- 4 = Left shutter servo
- 22 = Right shutter servo
- 18 = Neopixel strip that consists of 29 LEDs that are split in two groups for letter M and letter Y

Prerequisites:
------------------------

In order to run the training and inference programs, you need to have the following installed:

- Python 3
- MXNet https://github.com/apache/incubator-mxnet
- PSMoveAPI https://github.com/thp/psmoveapi

Note: The `collect.py` and `myhouse-gesture-recognition.py` scripts need SUDO permissions in order to use Bluetooth stack of the Raspberry Pi

Machine Learning
----------------

The training and inferring process information will be updated soon

![Data Structure](./images/ml-data-structure.jpg)
![Network Structure](./images/ml-network-structure.jpg)

More Information
----------------

 * License: MIT License
 * Maintainer: [@msurguy](https://twitter.com/msurguy) on Twitter
 * Website: https://maxoffsky.com/research-progress/project-myhouse-a-smart-dollhouse-with-gesture-recognition/
