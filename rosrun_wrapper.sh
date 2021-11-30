#!/bin/bash

source $HOME/catkin_ws/devel/setup.bash;
rosrun lab_automation go_to_position $1 $2 $3

