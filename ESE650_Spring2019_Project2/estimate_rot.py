#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter



# Import the Python Libraries
import numpy as np
from scipy.io import loadmat
import scipy.stats
import os
from matplotlib import pyplot as plt
import pdb


# Import The Quaternion and 
import Quaternion as Qt


def estimate_rot(data_num=1):
	#your code goes here
	return roll,pitch,yaw
