#! /usr/bin/env python3

import numpy as np
from scipy.signal import savgol_filter

## import path generator function
from path_generator import *

## import rospy and useful message type
import rospy
from std_msgs.msg import Float32MultiArray as FloatArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path
from svea.simulators.viz_utils import publish_path
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

## import svea interfaces
from svea.states import VehicleState
from svea.interfaces import LocalizationInterface
from svea.controllers.pure_pursuit import PurePursuitController
from svea.svea_managers.svea_archetypes import SVEAManager
from svea.data import TrajDataHandler, RVIZPathHandler

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new):
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith("~")
    is_global = new.startswith("/")
    assert not (is_private or is_global)
    ns, _ = split_last(old.split("/"))
    ns += new.split("/")
    return "/".join(ns)

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
        curr_pose.header.frame_id = 'mocap'
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
        wp_x = [0.0, 0.3376, 0.448, 0.75, 1.024, 1.1755, 1.1, 0.8, 0.0, -0.8, -1.0, -0.982, -0.827, -0.666, -0.408, -0.215, 0.0]
        wp_y = [-1.71, -1.6977, -1.6745, -1.572, -1.3568, -0.7071, 0.0, 0.8, 1.1, 0.8, 0.0, -1.136, -1.4409, -1.563, -1.66, -1.6997, -1.71]

        width = 0.2
        lx, ly, x_slack, y_slack = calc_spline_edge(wp_x, wp_y, width, ds=0.2)
        width = 1.0
        x_slack, y_slack, rx, ry = calc_spline_edge(wp_x, wp_y, width, ds=0.2)

        ## Parameters for SVEA
        self.USE_RVIS = load_param("~use_rvis", False)
        self.STATE = load_param("~state", [0, 0, 0, 0])
        state = VehicleState(*self.STATE)
        publish_initialpose(state)      
        
        self.vehicle_name = load_param("~name")
        self.svea = SVEAManager(LocalizationInterface, PurePursuitController, data_handler = RVIZPathHandler if self.USE_RVIS else TrajDataHandler)
        self.svea.start(wait = True)

        self.switch_flag = 0
        width = 0.7
        lxr, lyr, rxr, ryr = calc_spline_edge(wp_x, wp_y, width, ds=0.2)
        lxr.append(lxr[0]); lyr.append(lyr[0]); rxr.append(rxr[0]);ryr.append(ryr[0])    
        self.ref_path_old = CubicSpline2D(rxr, ryr)   
        refO_x, refO_y,_,_,_ = calc_spline_course(rxr, ryr, ds=0.2)
        self.publish_line(refO_x, refO_y, 'path_plan_old', 'mocap')

        width = 0.1
        lxr, lyr, rxr, ryr = calc_spline_edge(wp_x, wp_y, width, ds=0.2)
        lxr.append(lxr[0]); lyr.append(lyr[0]); rxr.append(rxr[0]);ryr.append(ryr[0])    
        self.ref_path_new = CubicSpline2D(rxr, ryr)   
        refN_x, refN_y,_,_,_ = calc_spline_course(rxr, ryr, ds=0.2) 
        self.publish_line(refN_x, refN_y, 'path_plan_new', 'mocap')        

        #start on the old path
        self.ref_path = self.ref_path_old       

        self.publish_line(lx, ly, 'left_edge', 'mocap')
        self.publish_line(rx, ry, 'right_edge', 'mocap')

        self.v_sample = []
        self.to_switch = 1
        self.L = 0.324          
        self.control_dt = 0.1
        self.epsilon_w = 0.15        
        self.epsilon = 0.6
        self.desired_velocity = 1.2  #set the desired platoon velocity
        self.desired_dist = 1.5      #set the desired platoon distance
        self.k1 = 5; self.k2 = 2; self.k3 = 0
        self.k4 = 0.3; self.k5 = 2.5; self.k6 = 0.            
        self.error_sum = 0.0; self.max_error_sum = 2; self.K_p = 2.5; self.K_i = 1.5

        
        self.control_counter = 0  #check if control loop is initiated
        self.state_counter = 0    #check if state is received from mocap

        self.msg_timer = rospy.get_time()
        self.pid_timer = rospy.get_time()
        self.state_timer = rospy.get_time()

        self.create_control_publisher()
        self.create_state_publisher()
        self.create_debug_publisher()
        self.create_subscription_to_state()

        self.leader_vehicle_old = {"vehicle_name": "svea0", "x":0.0, "y":0.0, "yaw":0.0, "v":0.0, "s_path":0.0, "yaw_path":0.0, "y_path":0.0, "kappa":0.0, "dkappa":0.0, "v_path":self.desired_velocity, "acc":0.0}
        self.leader_vehicle_new = {"vehicle_name": "svea1", "x":0.0, "y":0.0, "yaw":0.0, "v":0.0, "s_path":0.0, "yaw_path":0.0, "y_path":0.0, "kappa":0.0, "dkappa":0.0, "v_path":self.desired_velocity, "acc":0.0}

        self.create_subscription_to_leader_new()
        self.create_subscription_to_leader_input_new()


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
        if self.state_counter == 1: #check if state is received from mocap
            x = self.x
            y = self.y
            yaw = self.yaw 
            v = self.v
            s_path = self.s_path
            tild_theta = self.yaw_path
            tild_y = self.y_path
            kappa = self.kappa
            dkappa = self.dkappa 
            v_path = self.v_path  

            x_p = self.x_p
            y_p = self.y_p
            path_yaw = self.path_yaw

            local_state = [x, y, yaw, v, s_path, tild_theta, tild_y, kappa, dkappa, v_path, self.switch_flag]  #wrap the local state into a FloatArray
            local_state_msg = FloatArray(data=local_state)
            self.state_publisher.publish(local_state_msg)

            if self.control_counter == 0: #check if control loop is initiated            
                self.leader_vehicle_old["v"] = self.desired_velocity
                self.leader_vehicle_old["v_path"] = self.desired_velocity
                self.leader_vehicle_old["acc"] = 0.0
                self.leader_vehicle_old["s_path"] = s_path + 1.0
                self.leader_vehicle_old["x"], self.leader_vehicle_old["y"] = self.ref_path.calc_position(self.leader_vehicle_old["s_path"])
                #self.leader_vehicle_old["yaw"] = self.ref_path.calc_yaw(self.leader_vehicle_old["s_path"])

                op_old = 0
                
                self.control_counter = 1
            else:
                current_time = rospy.get_time()
                delta_time = current_time - self.state_timer
                self.state_timer = current_time
                self.leader_vehicle_old["v"] = self.desired_velocity
                self.leader_vehicle_old["v_path"] = self.desired_velocity
                self.leader_vehicle_old["acc"] = 0.0

                #obtain the optical flow
                leader_s = self.leader_vehicle_old["s_path"]
                if leader_s - s_path>= self.ref_path.sx.x[-1]:
                    leader_s = leader_s - self.ref_path.sx.x[-1]
                ds = leader_s - s_path - self.epsilon        
                op_old = (self.leader_vehicle_old["v_path"] - v_path)/ds
                
                #rospy.loginfo('-----------------')
                #rospy.loginfo(self.leader_vehicle_old["v_path"] - v_path)
                #rospy.loginfo(ds)
                #rospy.loginfo(op_old)

                #update leader s_path
                self.leader_vehicle_old["s_path"] = self.leader_vehicle_old["s_path"] + self.leader_vehicle_old["v_path"] * delta_time
                if self.leader_vehicle_old["s_path"] - s_path>= self.ref_path.sx.x[-1]:
                    self.leader_vehicle_old["s_path"] = self.leader_vehicle_old["s_path"] - self.ref_path.sx.x[-1]

                if self.leader_vehicle_old["s_path"] >  self.ref_path.sx.x[-1]:
                    leader_s = self.leader_vehicle_old["s_path"] - self.ref_path.sx.x[-1]  
                else:
                    leader_s = self.leader_vehicle_old["s_path"]
                self.leader_vehicle_old["x"], self.leader_vehicle_old["y"] = self.ref_path.calc_position(leader_s)
                #self.leader_vehicle_old["yaw"] = self.ref_path.calc_yaw(self.leader_vehicle_old["s_path"])

            
            if self.leader_vehicle_new["s_path"] - s_path < 0:
                leader_s =  self.ref_path.sx.x[-1] + self.leader_vehicle_new["s_path"] 
            else:
                leader_s =  self.leader_vehicle_new["s_path"]           
            leader_v = self.leader_vehicle_new["v_path"] 
            ds = leader_s - s_path - self.epsilon
            op_new = (leader_v - v_path)/ds

            if self.switch_flag == 1:
                leader = self.leader_vehicle_new
                opflow_leader = op_new
            else:
                leader = self.leader_vehicle_old
                opflow_leader = op_old
            #check if leader is circled by svea        
            if leader["s_path"] - s_path < 0:
                leader_s =  self.ref_path.sx.x[-1] + leader["s_path"]
            else:
                leader_s =  leader["s_path"]  
            
            #rospy.loginfo(opflow_leader)
            if self.switch_flag == 1:
                target_v = 0.7
            else:
                target_v = 1.2
            tild_v = v-target_v

            self.publish_pose(x, y, yaw, "vehicle_pose", "mocap")
            self.publish_pose(x_p, y_p, path_yaw, "path_pose", "mocap")
            self.publish_pose(leader["x"], leader["y"], leader["yaw"], "leader_pose", "mocap")

            tild_s = leader_s - s_path
            tild_e = tild_s - self.desired_dist
            tild_nu = leader["v_path"] - v_path            
            if tild_theta == 0:
                tild_theta = 1e-9

            if self.switch_flag == 1:
                road_width_left = 0.3
                road_width_right = 0.9
            else:
                road_width_left = 0.9
                road_width_right = 0.3
            if tild_y >= road_width_left - 0.6:
                alpha = -1
                dperp = road_width_left + alpha*tild_y - self.epsilon_w
                A = 'left'
            else:
                alpha = 1
                dperp = road_width_right + alpha*tild_y - self.epsilon_w 
                A = 'right'  
            dperp_dot = alpha*np.sin(tild_theta)     
            opflow_wall = dperp_dot/dperp

            vehicle_kappa = -self.k1*tild_y*np.sin(tild_theta)/tild_theta - self.k2*np.sign(v)*tild_theta - self.k3*np.sign(v)*alpha*opflow_wall + kappa*np.cos(tild_theta)/(1-kappa*tild_y)
            acc = leader["acc"] + self.k4*tild_e + self.k5*tild_nu + self.k6*opflow_leader
            dtild_theta = v*(vehicle_kappa - kappa*np.cos(tild_theta)/(1-kappa*tild_y))
            vehicle_acc = 1/np.cos(tild_theta)*(acc*(1-kappa*tild_y))+v*np.sin(tild_theta)*dtild_theta-v_path*(dkappa*v_path*tild_y+kappa*v*np.sin(tild_theta))
            
            self.target_velocity = v + self.control_dt * vehicle_acc 
            self.target_steering = np.arctan2(vehicle_kappa*self.L, 1.0) + 0.1 #  kappa = tan(delta)/L
            measure_steering = np.arctan2(self.yaw_rate*self.L/v, 1.0)

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

            #publish control signal to follower
            control_input = [acc, vehicle_kappa]
            control_input_msg = FloatArray(data=control_input)
            self.control_publisher.publish(control_input_msg)

            #publish debug data
            debug = [tild_e, tild_s, leader["s_path"], s_path, tild_nu, leader["v_path"] , v_path, v, self.target_velocity, tild_theta, tild_y, self.target_steering-0.1, measure_steering, opflow_leader, dperp, opflow_wall, acc, vehicle_kappa, tild_v]
            debug_msg = FloatArray(data=debug)
            self.debug_publisher.publish(debug_msg)  

    def create_control_publisher(self):
        topic_name = "/" + self.vehicle_name + "/control"
        self.control_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True) 

    def create_state_publisher(self):
        topic_name = "/" + self.vehicle_name + "/local_state"
        self.state_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True)

    def create_debug_publisher(self):
        topic_name = "/" + self.vehicle_name + "/debug"
        self.debug_publisher = rospy.Publisher(topic_name, FloatArray, queue_size = 1, tcp_nodelay = True) 

    def create_subscription_to_state(self):
        topic_name = "/qualisys/" + self.vehicle_name + "/odom"
        rospy.Subscriber(topic_name, Odometry, self.update_vehicle_state_mocap, tcp_nodelay = True, queue_size = 1)

    def update_vehicle_state_mocap(self, state):
        self.state_counter = 1
        self.x = state.pose.pose.position.x
        self.y = state.pose.pose.position.y
        quaternion = (state.pose.pose.orientation.x, state.pose.pose.orientation.y, state.pose.pose.orientation.z, state.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.yaw = euler[2]
        self.v = np.sqrt(state.twist.twist.linear.x**2 + state.twist.twist.linear.y**2) 
        self.yaw_rate = state.twist.twist.angular.z

        if 1==self.to_switch:
            if self.switch_flag == 0:
                s_path_check, xp_check, yp_check= self.ref_path_new.calc_closest_point_on_path(self.x, self.y)
                if self.leader_vehicle_new["s_path"] - s_path_check < 0:
                    leader_s =  self.ref_path_new.sx.x[-1] + self.leader_vehicle_new["s_path"]
                else:
                    leader_s =  self.leader_vehicle_new["s_path"]
                tild_s = leader_s - s_path_check
                check_L = 0.4; check_U = 0.8
                if check_L<= tild_s <=check_U:
                    self.switch_flag = 1 
                    self.ref_path = self.ref_path_new
                    rospy.loginfo('------switch lane------')
                    rospy.loginfo(tild_s)

        self.s_path, self.x_p , self.y_p = self.ref_path.calc_closest_point_on_path(self.x, self.y)
        yaw_temp = self.ref_path.calc_yaw(self.s_path)
        #to handle actan discontinuity
        if np.sign(self.yaw) != np.sign(yaw_temp) and (abs(self.yaw)>3 or abs(yaw_temp)>3):
            if self.yaw < 0:
                self.yaw = 2*np.pi + self.yaw
            elif yaw_temp < 0:
                yaw_temp = 2*np.pi + yaw_temp

        self.path_yaw = yaw_temp
        self.yaw_path = self.yaw - yaw_temp
        self.y_path =  (self.x - self.x_p)*np.cos(yaw_temp+np.pi/2) + (self.y- self.y_p)*np.sin(yaw_temp+np.pi/2)
        self.kappa = self.ref_path.calc_curvature(self.s_path)
        self.dkappa = self.ref_path.calc_curvature_change(self.s_path)
        self.v_path = self.v*np.cos(self.yaw_path)/(1-self.kappa*self.y_path)

    def create_subscription_to_leader_new(self):
        topic_name = "/" + self.leader_vehicle_new["vehicle_name"] + "/local_state"
        rospy.Subscriber(topic_name, FloatArray, self.update_leader_state_new, tcp_nodelay = True, queue_size = 1)

    def create_subscription_to_leader_input_new(self):
        topic_name = "/" + self.leader_vehicle_new["vehicle_name"] + "/control"
        rospy.Subscriber(topic_name, FloatArray, self.update_leader_input_new, tcp_nodelay = True, queue_size = 1)        

    def update_leader_input_new(self, msg):
        self.leader_vehicle_new["acc"] = msg.data[0]

    def update_leader_state_new(self, msg):
        self.leader_vehicle_new["x"] = msg.data[0]
        self.leader_vehicle_new["y"] = msg.data[1]
        self.leader_vehicle_new["yaw"] = msg.data[2]
        self.leader_vehicle_new["v"] = msg.data[3]

        self.leader_vehicle_new["s_path"] = msg.data[4]
        self.leader_vehicle_new["yaw_path"] = msg.data[5]
        self.leader_vehicle_new["y_path"] = msg.data[6]
        self.leader_vehicle_new["kappa"] = msg.data[7]
        self.leader_vehicle_new["dkappa"] = msg.data[8]
        self.leader_vehicle_new["v_path"] = msg.data[9]
        if self.leader_vehicle_new["vehicle_name"] == "svea2":
            self.switch_flag = msg.data[10]

    def publish_line(self, x_list, y_list, topic_name,frame):
        edge_pub = \
            rospy.Publisher(topic_name, Path, queue_size=1, latch=True)
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = frame
        path.poses = lists_to_pose_stampeds(x_list, y_list, None, None)
        edge_pub.publish(path) 

    def publish_pose(self, x, y, yaw, topic_name, frame):
        pose_pub = \
            rospy.Publisher(topic_name, PoseWithCovarianceStamped, queue_size = 1, tcp_nodelay = True)                    
        p = PoseWithCovarianceStamped()
        p.header.frame_id = frame
        p.pose.pose.position.x = x
        p.pose.pose.position.y = y

        q = quaternion_from_euler(0, 0, yaw)
        p.pose.pose.orientation.z = q[2]
        p.pose.pose.orientation.w = q[3]
        pose_pub.publish(p)


if __name__ == '__main__':
    ## Start node ##
    svea = svea_platoon()
    svea.run()