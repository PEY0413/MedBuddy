# MedBuddy
MedBuddy is a health buddy robot system based on ROS (Robot Operating System). It is able to predict your disease based on described symptoms (Google Speech Recognition) and provide prescription and medical advice (Google Text-to-Speech). It uses Natural Language Processing techniques to understand your input (symptoms) and match against the answer in the system (disease, prescription and medical advice). This project was developed by six of us: Beh Sin Yee, Christina Ku Pei San, Lee Keat En, Phuah En Yi, Soo Jin Xue, and Tee Yee Taung. Below are the codes used for environment setup and installation.

## ROS Configuration
1. Installation of ROS Melodic on Ubuntu 18.04
2. Create a ROS workspace
```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```
3. Create a ROS package
```
$ cd ~/catkin_ws/src/
$ catkin_create_pkg med_buddy roscpp rospy std_msgs
```
4. Build catkin workspace
```
$ cd ..
$ catkin_make
```
5. Source setup file
```
$ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```
## User Manual
1. Clone the project repository from GitHub
```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/PEY0413/med_buddy.git
```
2. Install dependencies
```
$ pip install -r med_buddy/requirements.txt
```
3. Build catkin workspace
```
$ cd ..
$ catkin_make
```
4. To launch the robot application, open 2 terminals and run the following commands:
- Terminal 1:
```
$ roscore
```
- Terminal 2:
```
$ roslaunch med_buddy med_buddy.launch
```
