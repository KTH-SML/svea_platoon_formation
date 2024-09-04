#! /usr/bin/env python3

import numpy as np
import time
from scipy.signal import savgol_filter

## import path generator function
from path_generator import *

## import rospy and useful message type
import rospy
from svea.states import VehicleState
from std_msgs.msg import Float32MultiArray as FloatArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Path
from svea.simulators.viz_utils import publish_path
from geometry_msgs.msg import Pose, PoseStamped, PoseArray


## import module for simulation
from svea.models.bicycle import SimpleBicycleModel
from svea.simulators.sim_SVEA import SimSVEA

## import svea interfaces
from svea.interfaces import LocalizationInterface
from svea.controllers.pure_pursuit import PurePursuitController
from svea.svea_managers.svea_archetypes import SVEAManager
from svea.data import TrajDataHandler, RVIZPathHandler

## import computer vision packages
import cv2
from cv2 import aruco
from cv_bridge import CvBridge
import message_filters as mf
from sensor_msgs.msg import Image, CameraInfo

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def publish_initialpose(state, n=10):

    p = PoseWithCovarianceStamped()
    p.header.frame_id = 'map'
    p.pose.pose.position.x = state.x
    p.pose.pose.position.y = state.y

    q = quaternion_from_euler(0, 0, state.yaw)
    p.pose.pose.orientation.z = q[2]
    p.pose.pose.orientation.w = q[3]

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rate = rospy.Rate(10)

    for _ in range(n):
        pub.publish(p)
        rate.sleep()

def lists_to_pose_stampeds(x_list, y_list, yaw_list=None, t_list=None):
    poses = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]

        curr_pose = PoseStamped()
        curr_pose.header.frame_id = 'map'
        curr_pose.pose.position.x = x
        curr_pose.pose.position.y = y

        if not yaw_list is None:
            yaw = yaw_list[i]
            quat = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
            curr_pose.pose.orientation.x = quat[0]
            curr_pose.pose.orientation.y = quat[1]
            curr_pose.pose.orientation.z = quat[2]
            curr_pose.pose.orientation.w = quat[3]

        if not t_list is None:
            t = t_list[i]
            curr_pose.header.stamp = rospy.Time(secs = t)
        else:
            curr_pose.header.stamp = rospy.Time.now()

        poses.append(curr_pose)
    return poses

class svea_platoon:
    def __init__(self):
        ## Initialize ROS Node
        rospy.init_node("svea_platoon")

        ## Generate reference path
        wp_x = [-8, 10.5]
        wp_y = [-16, 11.7]

        wp_x = [0.0, 0.9, 1.1, 0.8, 0.0, -0.8, -1.0, -0.8, 0.0]
        wp_y = [-1.7, -1.4, 0.0, 0.8, 1.1, 0.8, 0.0, -1.4, -1.7]


        #wp_x = [1.1, 0.8, 0.4, -0.3, -0.9]
        #wp_y = [-1.7, 0,0.85,1.05,1.1]

        self.ref_path = CubicSpline2D(wp_x, wp_y)   
        wp_x, wp_y,_,_,_ = calc_spline_course(wp_x, wp_y, ds=0.2)
        
        width = 0.5
        lx, ly, rx, ry = calc_spline_edge(wp_x, wp_y, width, ds=0.2)

        path_topic = "left_edge"
        self.l_pub = \
            rospy.Publisher(path_topic, Path, queue_size=1, latch=True)
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'map'
        path.poses = lists_to_pose_stampeds(lx, ly, None, None)
        self.l_pub.publish(path)

        path_topic = "right_edge"
        self.r_pub = \
            rospy.Publisher(path_topic, Path, queue_size=1, latch=True)
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'map'
        path.poses = lists_to_pose_stampeds(rx, ry, None, None)
        self.r_pub.publish(path)


        self.Vpub = rospy.Publisher('/Vehiclepose', PoseWithCovarianceStamped, queue_size = 1, tcp_nodelay = True) 
        self.Ppub = rospy.Publisher('/Pathpose', PoseWithCovarianceStamped, queue_size = 1, tcp_nodelay = True) 


        ## Parameters for SVEA
        self.USE_RVIZ = load_param("~use_rviz", False)
        self.STATE = load_param("~state", [0, 0, 0, 0])
        self.IS_SIM = load_param("~is_sim",False)
        
        name_dict = {"svea0":0, "svea1":1, "svea2":2, "svea3":3, "svea4":4, "svea5":5}
        index_dict = {0:"svea0", 1:"svea1", 2:"svea2", 3:"svea3", 4:"svea4", 5:"svea5"}
        self.marker_dict = {"svea0":0, "svea1":1, "svea2":2, "svea3":3, "svea4":4, "svea5":5}
        self.vehicle_name = load_param("~name")

        state = VehicleState(*self.STATE)
        rospy.loginfo(state)
        publish_initialpose(state)
        
        # is the mapping correct here?
        self.x = state.x
        self.y = state.y
        self.yaw = state.yaw
        self.v = state.v

        self.op_estimate = -0.4
        self.log_ds_estimate = 0
        
        self.s_path, self.x_p , self.y_p = self.ref_path.calc_closest_point_on_path(self.x, self.y)
        yaw_temp = self.ref_path.calc_yaw(self.s_path)        
        self.yaw_path = self.yaw - yaw_temp
        self.y_path =  (self.x - self.x_p)*np.cos(yaw_temp+np.pi/2) + (self.y- self.y_p)*np.sin(yaw_temp+np.pi/2)
        self.kappa = self.ref_path.calc_curvature(self.s_path)
        self.dkappa = self.ref_path.calc_curvature_change(self.s_path)
        self.v_path = self.v*np.cos(self.yaw_path)/(1-self.kappa*self.y_path)

        self.DELTA_TIME = 0.01
        if self.IS_SIM:
            self.sim_model = SimpleBicycleModel(state)
            self.simulator = SimSVEA(self.sim_model,
                                     dt = self.DELTA_TIME,
                                     run_lidar = True,
                                     start_paused=True).start()
            

        self.svea = SVEAManager(LocalizationInterface, PurePursuitController, data_handler = RVIZPathHandler if self.USE_RVIZ else TrajDataHandler)
        self.svea.start(wait = True)

        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()


        self.svea.data_handler.update_traj(wp_x,wp_y)
        self.control_dt = 0.1


        self.road_width_left = 0.5
        self.road_width_right = 0.5
        self.road_width = 1
        self.r_eta = 0.15        
        self.dw_sample_left = []
        self.dw_sample_right = []
        self.dw_t_sample = []
        self.opflow_wall_left = 0
        self.opflow_wall_right = 0

        self.r_g = 0.2
        self.ds_sample = []
        self.t_sample = []
        self.opflow_leader = 0
        self.desired_velocity = 0.4  #set the desired platoon velocity
        self.desired_dist = 0.6      #set the desired platoon distance
        self.k1 = 3
        self.k2 = 3
        self.k3 = 0
        self.k4 = 5
        self.k5 = 5
        self.k6 = 0

        self.L = 0.324               #wheel base length
        self.error_sum = 0.0
        self.max_error_sum = 2
        self.K_p = 1.0
        self.K_i = 0.8

        self.target_velocity = self.desired_velocity
        self.target_steering = 0 #  kappa = tan(delta)/L
            
        self.leader_vehicle = {"vehicle_name": index_dict[name_dict[self.vehicle_name]-1], "x":0.0, "y":0.0, "yaw":0.0, "v":0.0, "s_path":self.s_path+0.7, "yaw_path":0.0, "y_path":0.0, "kappa":0.0, "dkappa":0.0, "v_path":self.desired_velocity, "acc":0.0}

        self.control_timer = rospy.get_time()
        self.state_timer = rospy.get_time()
        self.create_control_publisher()
        self.create_state_publisher()
        self.create_debug_publisher()
        self.create_debug_op_publisher()

        self.pid_timer = rospy.get_time()
        
    def run(self):
        self.svea.wait_for_state()
        self.r = rospy.Rate(1/self.control_dt)
        while self.keep_alive():
            self.spin()
            self.r.sleep()

    def keep_alive(self):
        return not rospy.is_shutdown()


    def spin(self):
        self.get_control()
        # comment to use either control direct or through pid
        # self.svea.send_control(self.target_steering, self.target_velocity)
        self.svea.visualize_data()

    def get_control(self):
        self.x = self.svea.state.x
        self.y = self.svea.state.y
        self.yaw = self.svea.state.yaw
        self.v = self.svea.state.v

        yaw_raw = self.yaw

        self.s_path, self.x_p , self.y_p = self.ref_path.calc_closest_point_on_path(self.x, self.y)

        yaw_temp = self.ref_path.calc_yaw(self.s_path)

        if np.sign(self.yaw) != np.sign(yaw_temp) and (abs(self.yaw)>3 or abs(yaw_temp)>3):
            if self.yaw < 0:
                self.yaw = 2*np.pi + self.yaw
            elif yaw_temp < 0:
                yaw_temp = 2*np.pi + yaw_temp
                    
        self.yaw_path = self.yaw - yaw_temp
        self.y_path =  (self.x - self.x_p)*np.cos(yaw_temp+np.pi/2) + (self.y- self.y_p)*np.sin(yaw_temp+np.pi/2)
        self.kappa = self.ref_path.calc_curvature(self.s_path)
        self.dkappa = self.ref_path.calc_curvature_change(self.s_path)
        self.v_path = self.v*np.cos(self.yaw_path)/(1-self.kappa*self.y_path)

        tild_y = self.y_path
        tild_theta = self.yaw_path
        v = self.v
        kappa = self.kappa
        dkappa = self.kappa
        tild_v = v - self.desired_velocity


        p = PoseWithCovarianceStamped()
        p.header.frame_id = 'map'
        p.pose.pose.position.x = self.x
        p.pose.pose.position.y = self.y

        q = quaternion_from_euler(0, 0, self.yaw)
        p.pose.pose.orientation.z = q[2]
        p.pose.pose.orientation.w = q[3]
        self.Vpub.publish(p)


        p = PoseWithCovarianceStamped()
        p.header.frame_id = 'map'
        p.pose.pose.position.x = self.x_p
        p.pose.pose.position.y = self.y_p

        q = quaternion_from_euler(0, 0, yaw_temp)
        p.pose.pose.orientation.z = q[2]
        p.pose.pose.orientation.w = q[3]
        self.Ppub.publish(p)

        #add case for vehicle 1!!
        if self.leader_vehicle["vehicle_name"] == "svea0":   
            current_time = rospy.get_time()
            delta_time = current_time - self.state_timer
            self.state_timer = current_time

            self.leader_vehicle["v"] = self.desired_velocity
            self.leader_vehicle["v_path"] = self.desired_velocity
            self.leader_vehicle["acc"] = 0.0
            leader_s = self.leader_vehicle["s_path"]
            self.leader_vehicle["s_path"] = self.leader_vehicle["s_path"] + self.leader_vehicle["v_path"] * delta_time
            if self.leader_vehicle["s_path"] - self.s_path>= self.ref_path.sx.x[-1]:
                self.leader_vehicle["s_path"] = self.leader_vehicle["s_path"] - self.ref_path.sx.x[-1]
            
            
            if leader_s - self.s_path>= self.ref_path.sx.x[-1]:
                leader_s = leader_s - self.ref_path.sx.x[-1]

            ds = leader_s - self.s_path - self.r_g        
            alpha1 = 1
            alpha2 = 1
            epsilon = 0.1
            measurement = np.log(ds)
            self.log_ds_estimate = self.log_ds_estimate + delta_time*(self.op_estimate + (alpha1/epsilon)*(measurement - self.log_ds_estimate))
            self.op_estimate = self.op_estimate + delta_time*((alpha2/epsilon)*(measurement - self.log_ds_estimate))

            self.op_real = (self.leader_vehicle["v_path"] - self.v_path)/ds
            debug_op = [ds, measurement, self.log_ds_estimate, self.op_estimate, self.op_real]  #wrap the local state into a FloatArray
            debug_msg_op = FloatArray(data=debug_op)
            self.debug_op_publisher.publish(debug_msg_op)


        tild_s = self.leader_vehicle["s_path"] - self.s_path
        tild_nu = self.leader_vehicle["v_path"] - self.v_path

        if tild_y >= self.road_width_left - self.road_width/2:
            opflow_wall = self.opflow_wall_left
        else:
            opflow_wall = self.opflow_wall_right
        
        if tild_theta == 0:
            tild_theta = 1e-9
                
        tild_e = tild_s - self.desired_dist

        vehicle_kappa = -self.k1*tild_y*np.sin(tild_theta)/tild_theta - self.k2*np.sign(v)*tild_theta - self.k3*np.sign(v)*opflow_wall + kappa*np.cos(tild_theta)/(1-kappa*tild_y)

        #self.opflow_leader = self.op_estimate
        acc = self.leader_vehicle["acc"] + self.k4*tild_e + self.k5*tild_nu + self.k6*self.opflow_leader
        
        dtild_theta = v*(vehicle_kappa - kappa*np.cos(tild_theta)/(1-kappa*tild_y))
        vehicle_acc = 1/np.cos(tild_theta)*(acc*(1-kappa*tild_y))+v*np.sin(tild_theta)*dtild_theta-self.v_path*(dkappa*self.v_path*tild_y+kappa*v*np.sin(tild_theta))

        self.target_velocity = v + self.control_dt * vehicle_acc 
        self.target_steering = np.arctan2(vehicle_kappa*self.L,1.0) #  kappa = tan(delta)/L
        
        current_time = rospy.get_time()
        delta_time = current_time - self.pid_timer
        self.pid_timer = current_time

        error = self.target_velocity - v
        self.error_sum += error * delta_time
        P = error * self.K_p

        if self.error_sum > self.max_error_sum:
            self.error_sum = self.max_error_sum
        elif self.error_sum < - self.max_error_sum:
            self.error_sum = - self.max_error_sum        

        I = self.error_sum * self.K_i
        correction = P + I
        speed_val = self.target_velocity + correction
        if speed_val <= 0:
            speed_val = 0

        #send control to low level
        self.svea.send_control(self.target_steering, speed_val)

        control_input = [acc, vehicle_kappa]  #wrap the local state into a FloatArray
        control_input_msg = FloatArray(data=control_input)
        self.control_publisher.publish(control_input_msg)
        
        #debug = [tild_e, tild_nu, self.v_path, vehicle_acc, self.target_velocity, v, self.leader_vehicle["s_path"] - self.s_path, tild_y*np.sin(tild_theta)/tild_theta, self.target_steering, vehicle_kappa, np.sign(v)*tild_theta]  #wrap the local state into a FloatArray
        debug = [v, self.v_path, self.k5*tild_nu, tild_s, self.k4*tild_e, self.leader_vehicle["s_path"], self.s_path, acc, vehicle_acc, self.target_velocity, tild_theta, tild_y, yaw_temp, self.target_steering, self.yaw, vehicle_kappa, kappa, kappa*np.cos(tild_theta)/(1-kappa*tild_y), -self.k1*tild_y*np.sin(tild_theta)/tild_theta, -self.k2*tild_theta]
        debug_msg = FloatArray(data=debug)
        self.debug_publisher.publish(debug_msg)  

    def create_control_publisher(self):
        topic_name = "/" + self.vehicle_name + "/control"
        self.control_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True) 

    def create_debug_publisher(self):
        topic_name = "/" + self.vehicle_name + "/debug"
        self.debug_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True) 
    
    def create_debug_op_publisher(self):
        topic_name = "/" + self.vehicle_name + "/debug_op"
        self.debug_op_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True) 

    def create_state_publisher(self):
        topic_name = "/" + self.vehicle_name + "/local_state"
        self.state_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True)

if __name__ == '__main__':
    ## Start node ##
    svea = svea_platoon()
    svea.run()
