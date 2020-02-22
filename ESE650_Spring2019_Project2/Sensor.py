import numpy as np
class Sensor:
    def __init__(self):
        pass
    # Matrix to convert IMU data
    def acc_vec_conv(self, imu_acc_vector):
        rot_matrix=np.array([[-1,0,0],[0,-1,0],[0,0,1]],np.int16)
        print(imu_acc_vector.shape) 
       # imu_acc_conv=np.matmul(rot_matrix,imu_acc_vector)
        return imu_acc_conv

    def gyro_vec_conv(self, imu_gyro_vector):
        rot_matrix=np.array([[0,1,0],[0,0,1],[1,0,0]],np.int16) 
        imu_gyro_conv=np.matmul(rot_matrix,imu_gyro_vector)
        return imu_gyro_conv