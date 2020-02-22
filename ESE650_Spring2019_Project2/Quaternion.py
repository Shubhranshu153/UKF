import numpy as np
from scipy.io import loadmat
import scipy.stats
import os
from matplotlib import pyplot as plt
import pdb
import math as mat
from scipy.spatial.transform import Rotation as R

class Sensor:
    def __init__(self):
        pass
    # Matrix to convert IMU data
    def acc_vec_conv(self, imu_acc_vector):
        rot_matrix=np.array([[-1,0,0],[0,-1,0],[0,0,1]]) 
        imu_acc_conv=np.matmul(rot_matrix,imu_acc_vector)
        return imu_acc_conv

    def gyro_vec_conv(self, imu_gyro_vector):
        rot_matrix=np.array([[0,1,0],[0,0,1],[1,0,0]]) 
        imu_gyro_conv=np.matmul(rot_matrix,imu_gyro_vector)
        return imu_gyro_conv
    
    def acc_bias(self,imu_acc):
        acc_bias=505
        acc_s = 32.3
        acc_sf = 3300/(1023*acc_s)
        imu_unbiased_acc=(imu_acc-acc_bias)*acc_sf
        return imu_unbiased_acc
    
    def gyr_bias(self,imu_gyr):
        gyr_bias_roll=369.8
        gyr_bias_pitch=373.5
        gyr_bias_yaw=375.4
        gyr_s = 3.3
        gyr_sf = 3300*np.pi/(1023*180*gyr_s)
        imu_unbiased_gyr=np.zeros(imu_gyr.shape)
        imu_unbiased_gyr[1,:]=(imu_gyr[0,:]-gyr_bias_roll)*gyr_sf
        imu_unbiased_gyr[2,:]=(imu_gyr[1,:]-gyr_bias_pitch)*gyr_sf
        imu_unbiased_gyr[0,:]=(imu_gyr[2,:]-gyr_bias_yaw)*gyr_sf
        return imu_unbiased_gyr

      
    def acc_to_euler(self,imu_acc_vect):
        accelX=imu_acc_vect[0,:]
        accelY=imu_acc_vect[1,:]
        accelZ=imu_acc_vect[2,:]
        pitch =  np.arctan2(accelX, np.sqrt(accelY*accelY + accelZ*accelZ))
        roll =  -np.arctan2(accelY, np.sqrt(accelX*accelX + accelZ*accelZ))
        return roll,pitch
    
    

class DataHandling: 
    def __init__(self):
        pass
    
    def import_data(self):
        filename = os.path.join(os.path.dirname(__file__), "vicon/viconRot1.mat")
        
        vicon_data = loadmat(filename)
        vicon_data_read=np.array(vicon_data['rots'],dtype=float)
        vicon_data_time=np.array(vicon_data['ts'],dtype=float)

        imu_filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw1.mat")
        imu_data = loadmat(imu_filename)
        imu_data_read=np.array(imu_data['vals'],dtype=float)
        imu_data_time=np.array(imu_data['ts'],dtype=float)
        return vicon_data_read,vicon_data_time,imu_data_read,imu_data_time
    
    def vicon_to_euler(self,rot):
        theta = -np.arcsin(rot[2,0,:])
        psi = np.arctan2(rot[2,1,:]/np.cos(theta), rot[2,2,:]/np.cos(theta))
        phi = np.arctan2(rot[1,0]/np.cos(theta), rot[0,0]/np.cos(theta))
        return theta, psi, phi
    
    
    

    def plot_ang_acc(self,roll_imu,roll_vicon,pitch_imu,pitch_vicon,yaw_vicon,imu_time,vicon_time):
        dim=roll_imu.shape
        print(dim)
        # Unit Test For Euler to Quat and Back to Euler
        # yaw=np.zeros(dim[0])
        # euler=(np.vstack((roll_imu,pitch_imu,yaw))).reshape(5645,3)
        # q_r=R.from_euler('xyz',euler)
        # quat=q_r.as_quat()
        # q_quat=R.from_quat(quat)
        # Q_euler=q_quat.as_euler('xyz')
       
        plt.figure(1)
        plt.plot(imu_time.T,roll_imu,'r',label='Roll_imu')
        plt.plot(vicon_time.T,roll_vicon,'r',linestyle='--',label='roll_vicon')
        plt.plot(imu_time.T,pitch_imu,'g',label='Pitch_imu')
        plt.plot(vicon_time.T,pitch_vicon,'g',linestyle='--',label='pitch_vicon')
        plt.plot(vicon_time.T,yaw_vicon,'b',linestyle='--',label='yaw_vicon')
        plt.legend()
       
    
    def plot_ang_gyr(self,omega,imu_time):
        plt.figure(2)
        plt.plot(imu_time.T,omega[1],'r',label='Roll') #in X
        plt.plot(imu_time.T,omega[2],'g',label='pitch') #in Y
        plt.plot(imu_time.T,omega[0],'b',label='yaw') #in Z
        plt.legend()
       
        



class Quaternion:
    def __init__(self):
        pass

    # Checked
    def normalize(self, q):
        norm = np.linalg.norm(q, axis=(0))
        return q/norm
    
    def rot_mat_to_quat(self,rot_mat):
       r=R.from_matrix(rot_mat)
       q=r.as_quat()
       return q
    
   
    
    def rot_mat_to_quat(self,rot_mat):
        r=R.from_matrix(rot_mat)
        q=r.as_quat()
        return q
    
    def rot_vec_to_quat(self,rot_vec):
        r=R.from_rotvec(rot_vec)
        q=r.as_quat()
        return q
    
    def euler_to_quat(self, roll,pitch,yaw):
        r=R.from_euler('xyz',[roll,pitch,yaw])
        q=r.as_quat()
        return q
    
   
    def quat_mult(self, q1, q2):
        dim_q1=q1.shape
        dim_q2=q2.shape
        if len(dim_q1)==1 and len(dim_q2)==1:
            r1=R.from_quat(q1)
            r2=R.from_quat(q2)
            quat=r1*r2
       
       
        elif dim_q1[1]==1:
            r1=R.from_quat(q1.reshape(4))
            r2=R.from_quat(q2)
            quat=r1*r2
        
        elif dim_q2[1]==1:
            r1=R.from_quat(q1)
            r2=R.from_quat(q2.reshape(4))
            quat=r1*r2

        return quat.as_quat().T
    
    def quat_mean(self,q):
        r=R.from_quat(q)
        return r.mean().as_quat()

    def ang_mean(self,ang):
        dim=ang.shape

        sum=np.sum(ang,axis=1)
        return sum/dim[1]

    def err_vec(self,y_i,y_mean):
        r=R.from_quat(y_i)
        r_mean=R.from_quat(y_mean)
        r_mean_inv=r_mean.inv()
        err=r*r_mean_inv
        err_rot_vec=err.as_rotvec()
        return err_rot_vec
    
    def err_ang(self,y_i,y_mean):
        return y_i-y_mean

    def generate_quat(self, angle, axis):

        quat = np.zeros((4,1))
        quat[0] = np.cos(angle/2)

        if isinstance(axis, np.ndarray):
            quat[1:,:] = axis*np.sin(angle/2)
        else:
            vec = np.array(axis).reshape((3,1))
            quat[1:,:] = vec*np.sin(angle/2)

        return self.normalize(quat)   

class UKF:
    def __init__(self):
        #Reducing the Value of Noise
        self.P = 0.0001*np.eye(6)
        self.Q = 0.0001*np.eye(6)
        self.R = 0.0001*np.eye(6)

        self.thresh = 0.0001
        self.gamma=1.5

    def generate_sigma_points(self,x_hat,P_hat):
        Q=Quaternion()
        n = self.P.shape[0]
        
        #We generate the Variance of the Points
        self.W = np.linalg.cholesky(P_hat + self.Q)
        sigma_pts = np.hstack((np.sqrt(self.gamma)*self.W, -np.sqrt(self.gamma)*self.W))
        sigma_pts_transpose=sigma_pts.T
       
        #Top 3X3 Matrix is Rotation Vector. Need to Convert to Quaternion
        sigma_pts_quat=Q.rot_vec_to_quat(sigma_pts_transpose[:,0:3])
        delta_sigma_mat = np.vstack((sigma_pts_quat.T, sigma_pts[3:,:]))
        delta_sigma_mat_trans=delta_sigma_mat.T
       #The angle part go into quaternion multiplication 
        Xi=np.zeros((x_hat.shape[0], sigma_pts.shape[1]))
        
        Xi[0:4,:] = Q.quat_mult(x_hat[0:4], delta_sigma_mat_trans[:,0:4])   
        
        #We can sum the angular velocities
        Xi[4:,:] = x_hat[4:,:] #+ delta_sigma_mat[4:,:]
        
        return Xi
       
    def process_model_test(self,omega,imu_time):
        Q=Quaternion()
        # Delta angle space
        dim=omega.shape
        alpha=np.zeros(dim[0]).reshape(3,1)
        for i in range(dim[1]-1):
            dt = imu_time[0,i+1] - imu_time[0,i] 
            dt.reshape(1,1)
            temp=np.vstack((omega[0,i]*dt,omega[1,i]*dt,omega[2,i]*dt))
            alpha=np.hstack((alpha,temp))
        #ang=np.cumsum(alpha,axis=1)
        r_ang=R.from_euler('xyz',alpha.T)
        r_quat=r_ang.as_quat()
        Euler_angs=np.zeros((dim[1],3))
        init_quat=r_quat[0,:]
    
        print(r_quat.shape)
        for i in range(dim[1]-1):
            temp_quat=Q.quat_mult(init_quat.reshape(4),r_quat[i+1,:].reshape(4))
            temp_r=R.from_quat(temp_quat)
            init_quat=temp_quat
            eul=temp_r.as_euler('xyz')
            Euler_angs[i,:]=eul
      
        plt.figure(4)
        plt.plot(imu_time.T,Euler_angs[:,1],'r',label='Roll') #in X
        plt.plot(imu_time.T,Euler_angs[:,2],'g',label='pitch') #in Y
        plt.plot(imu_time.T,Euler_angs[:,0],'b',label='yaw') #in Z
        plt.legend()
       
    def process_model(self,X_i,omega,dt):
        Q=Quaternion()
        
        Yi=np.zeros(X_i.shape,dtype=float)
    # Omega are simply transformed 1-1 as we dont know acceleration
        Yi[4:,:] = X_i[4:,:]
        dim=X_i.shape
      
        
    # Convert angle to quaternion and do quaternion Multiplication
        alpha=[]
        dt.reshape(1,1)
        temp=np.asarray([omega[0]*dt,omega[1]*dt,omega[2]*dt])
      
        #ang=np.cumsum(alpha,axis=1)
        r_ang=R.from_euler('xyz',temp)
        r_quat=r_ang.as_quat()
        Quat_angs=np.zeros((dim[1],4))
       
    
        for i in range(dim[1]):
            temp_quat=Q.quat_mult(X_i[0:4,i].reshape(4), r_quat.reshape(4))
            Yi[0:4,i] = temp_quat 
        
    
        return Yi
    
    def measurement_model_actual(self,Z_act):
        Zi = np.zeros((Z_act.shape))
        quat_g=[0,0,0,9.8]
        R_q=R.from_rotvec(Z_act[0:3].reshape(3))
        R_g=R.from_quat(quat_g)

        
        R_q_inv=R_q.inv()
        acc_est_q=R_q_inv*R_g*R_q
        acc_rot_vec=acc_est_q.as_rotvec()
        Zi[0:3]=acc_rot_vec.reshape(3,1)  
        Zi[3:] = Z_act[3:].reshape(3,1)
        
        return Zi
        
    def measurement_model(self,Yi):
        Zi = np.zeros((Yi.shape[0]-1, Yi.shape[1]))
        quat_g=[0,0,0,9.8]
        Yi_trans=Yi.T
        R_q=R.from_quat(Yi_trans[:,0:4])
        R_g=R.from_quat(quat_g)
     
        acc_vect=[]

        for i in range(Yi.shape[1]):
            R_q_inv=R_q[i].inv()
            acc_est_q=R_q_inv*R_g*R_q[i]
            acc_rot_vec=acc_est_q.as_rotvec()
            acc_vect=np.hstack((acc_vect,acc_rot_vec))
        Zi[0:3,:]=acc_vect.reshape(3,Yi.shape[1])
        
        Yi_trans=Yi.T
        Rot_mat=(R.from_quat(Yi_trans[:,0:4])).as_matrix()


        for i in range(Yi.shape[1]):
            #Zi[3:,i] = np.matmul(np.transpose(Rot_mat[i]),Yi[4:,i])
            Zi[3:,i] = Yi[4:,i]
        
        return Zi
  

    def measurement_update(self,Zi,Z_actual,W_err,x_hat_old):

        #Z mean and variance
        z_mean = (np.sum(Zi, axis=1)/Zi.shape[1]).reshape((Zi.shape[0], 1))
        Pzz = np.dot((Zi - z_mean), np.transpose(Zi - z_mean))/Zi.shape[1]
        
        Z_act_conv=self.measurement_model_actual(Z_actual)
        #Innovation
        v_inn = Z_act_conv-z_mean
       
        

        Pvv = Pzz + self.R
        Pxz = np.dot(W_err.T, np.transpose(Zi - Z_act_conv))/W_err.shape[0]
        kalman = np.dot(Pxz, np.linalg.inv(Pvv))
      
     
        v_inn_kal=np.matmul(kalman,v_inn)
     
       
       
        r_v_inn_kal=R.from_rotvec(v_inn_kal[0:3].reshape(3))
        r_v_inn_quat=r_v_inn_kal.as_quat()    
        
        
         
        x_hat_new=np.zeros(x_hat_old.shape)
        
        temp_quat=Q.quat_mult(x_hat_old[0:4].reshape(4), r_v_inn_quat.reshape(4))
        x_hat_new[0:4]=temp_quat.reshape(4,1)
        x_hat_new[4:7]=x_hat_old[4:7]+v_inn_kal[3:6]

        P_hat_old = np.zeros(self.P.shape)
        P_hat_old = np.dot(W_err.T,W_err)/Zi.shape[1]
        
        P_hat_new=P_hat_old- np.matmul(kalman,np.matmul(Pvv,kalman.T))
        
        return x_hat_new,P_hat_new
  
    
    def filter(self,acc,omega,imu_time,roll_imu,pitch_imu):
        Q_quat=Quaternion()
        q_init=Q_quat.euler_to_quat(roll_imu,pitch_imu,0)
        q_init.reshape(4,1)
        x_hat_init=np.array(np.concatenate((q_init,omega[:,0]),axis=None).reshape(7,1),dtype=float)
        
       
        dim=omega.shape
        roll = np.zeros((1,dim[1]),dtype=float)
        pitch = np.zeros((1,dim[1]),dtype=float)
        yaw = np.zeros((1,dim[1]),dtype=float)
       
        # Generate Sigma Points 12
        # Send the Sigma Points through the process Model and get Yi
        # A priori Estimate mean and Covariance is computed from the Yi
        # Transformation into 6 dimensional state Check once more
        # Measurement Model
        # measurement update
        roll[0,0]=x_hat_init[0,0]
        pitch[0,0]=x_hat_init[1,0]
        x_hat=x_hat_init
        y_mean=np.zeros(x_hat.shape,dtype=float)
        W_err=np.zeros((12,6)) #set some shape parameters
        Zi = np.zeros((x_hat.shape[0]-1, x_hat.shape[1]))
        Z_measured=np.vstack((acc,omega))
        P_hat=self.P
        temp_rpy=np.zeros((3,1),dtype=float)
        print(Z_measured.shape)

       
        #for i in range(5):
        for i in range(dim[1]-1):
            dt = imu_time[0,i+1] - imu_time[0,i]
            x_i=self.generate_sigma_points(x_hat,P_hat)
        
        # Apply Process Model
            y_i=self.process_model(x_i,omega[:,i],dt)  
            y_i_trans=y_i.T
            
        # y mean and W space err which is 6 dimensional
            y_mean[0:4,0]=Q_quat.quat_mean(y_i_trans[:,0:4])
            y_mean[4:7,0]=Q_quat.ang_mean(y_i[4:7,:])
            W_err[:,0:3]=Q_quat.err_vec(y_i_trans[:,0:4],y_mean[0:4,0])
            W_err[:,3:6]=Q_quat.err_ang(y_i[4:7,0],y_mean[4:7,0])
            
            #measurement model
            Zi=self.measurement_model(y_i)
            x_hat,P_hat=self.measurement_update(Zi,(Z_measured[:,i+1]).reshape(6,1),W_err,y_mean)
            r_temp=R.from_quat(x_hat[0:4].reshape(4))
            temp_rpy=r_temp.as_euler('xyz')
            #x_hat[0:4]=y_mean[0:4]
            #x_hat[4:]=Z_measured[3:,i+1].reshape(3,1)
            roll[0,i]=temp_rpy[0]
            pitch[0,i]=temp_rpy[1]
            yaw[0,i]=temp_rpy[2]
            

            print('Iteration {} '.format(i))
        
        plt.figure(3)
        plt.plot(roll[0,:],'r',label='Roll') #in X
        plt.plot(pitch[0,:],'g',label='pitch') #in Y
        plt.plot(yaw[0,:],'b',label='yaw') #in Z
        plt.legend()
  


if __name__ == "__main__":

    #DataHandling
    data=DataHandling()
    vicon_data,vicon_data_time,imu_data,imu_data_time=data.import_data()
    
    #Vicon Data rotation Matrix to Euler Angles
    pitch_vicon,roll_vicon,yaw_vicon=data.vicon_to_euler(vicon_data)
    
      
    
    #Sensor Data Transformation                                                           
    s=Sensor()
    imu_unbiased_acc=np.array(s.acc_bias(imu_data[0:3,:]))
    imu_unbiased_omega=np.array(s.gyr_bias(imu_data[3:6,:]))
    imu_acc=s.acc_vec_conv(imu_unbiased_acc)
    imu_gyr=s.gyro_vec_conv(imu_unbiased_omega)
    data.plot_ang_gyr(imu_unbiased_omega,imu_data_time)
    roll_imu,pitch_imu=s.acc_to_euler(imu_unbiased_acc)
    data.plot_ang_acc(roll_imu,roll_vicon,pitch_imu,pitch_vicon,yaw_vicon,imu_data_time,vicon_data_time) 
    
    #TODO: TAKE CARE OF YAW
    
    #quaternion Transforms
    Q=Quaternion() 

    #Checked This Coversion is correct
    q=Q.rot_mat_to_quat(vicon_data.T)
    
    
    UKF=UKF()
    UKF.process_model_test(imu_gyr,imu_data_time)
    UKF.filter(imu_acc,imu_gyr,imu_data_time,roll_imu[0],pitch_imu[0])
    
    plt.show()


  
   
    

    
 



