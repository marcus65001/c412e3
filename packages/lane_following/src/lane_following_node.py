#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from typing import cast
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String


class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Subscriber
        self.sub_comp_img = rospy.Subscriber('~cam', CompressedImage, self.cb_img)  # camera image topic
        
        # Publisher
        self.pub_wheel_command = rospy.Publisher("~wheel_control", Twist2DStamped, queue_size=1)
        self.pub = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="The stream of JPEG compressed images from the modified camera feed",
        )

        #PID Variables for the Controller
        # American P Gain
        self.proportionalGain = 7.5

        # English P Gain
        # self.proportionalGain = 7.5

        #Optional
        # self.derivativeGain = 0.0
        # self.integralGain = 0.0
        # self.valueLast = 0.0
        # self.errorLast = 0.0
        # self.derivativeInitialized = True
        # self.time_interval = rospy.Time.now().to_sec()
        
        self.curr_dist = 0
        # American Driver Target
        self.target = 205
 
        # English Driver
        # self.target = 190
        self.velocity  = 0.25
        self.omega = 0

        # Camera Information
        self.cam_width = 640
        self.cam_height = 480

        # Additional parameters and internal objects
        self.image = None
        self._bridge = CvBridge()
        #American driver
        self.yellow_lower = np.array([25, 100, 140])
        self.yellow_higher = np.array([35, 255, 255])
        # English driver
        # self.yellow_lower = np.array([18, 45, 155])
        # self.yellow_higher = np.array([28, 135, 255])
        
        rospy.loginfo("Initialized Lane Following Node")

    # Camera calback
    def cb_img(self, msg):
        # image callback
        if self._bridge:
            self.image = self.read_image(msg)
            # self.image = self.image[220:self.cam_height, 300:600]

    # Reads the camera feed image and sets it as a cv2 compatible image
    def read_image(self, msg):
        try:
            img=self._bridge.compressed_imgmsg_to_cv2(msg)
            if (img is not None) and (self.image is None):
                self.log("got first msg")
            return img
        except Exception as e:
            self.log(e)
            return np.array([])

    #Calculates the midpoint of the contoured object
    def midpoint (self, x, y, w, h):
        mid_x = int(x + (((x+w) - x)/2))
        mid_y = int(y + (((y+h) - y)/2))
        return (mid_x, mid_y)

    #Detects the yellow dash lanes of our duckietown
    def lane_detection(self):
        
        half_width = int(self.cam_width*0.5)
        
        #Converts color info from bgr to hsv
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #Creates a middle line for the frame
        cv2.line (self.image, (half_width,0), (half_width,self.cam_height), (255,0,0), 2)

        # This to detect the color yellow on the duckietown track and places the contour
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_higher)
        contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0: 
            # Select the largest contour to be dectected           
            max_contour = max(contours, key=cv2.contourArea)
            # Generates the size and the cordinantes of the bounding box and draw
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(self.image,(x,y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(self.image, self.midpoint(x,y,w,h), 2, (63, 127, 0), -1)
            # Calculate the pixel distance from the middle of the frame
            pixel_distance = math.sqrt(math.pow((half_width - self.midpoint(x,y,w,h)[0]),2))
            cv2.line (self.image, self.midpoint(x,y,w,h), (half_width,self.midpoint(x,y,w,h)[1]), (255,0,0), 2)
            self.curr_dist = pixel_distance
            # print("Pixel Distance: " + str(pixel_distance))
        

    def P_controller (self):

        error = self.curr_dist - self.target
        # print("ERROR: " + str(error))


        #Calculate P term
        p = self.proportionalGain * error 
        # print("P value: " + str(p))

        # American driver
        self.omega = (p/180)

        # English driver
        # self.omega = - (p/180)

        ### TESTING D #### DOES NOT WORK
        # #Calculating both D terms
        # errorRateofChange = (error - self.errorLast) / time_interval
        # self.errorLast = error

        # valueRateofChange = (curr_dist - self.valueLast) / time_interval
        # self.valueLast = curr_dist
        # velocity = valueRateofChange

        # deriveMeasure = 0.0

    def publish_robot_cmd (self):
        
        msg = Twist2DStamped()
        msg.v = self.velocity
        msg.omega = self.omega
        self.pub_wheel_command.publish(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.lane_detection()
                self.P_controller()
                # print("OMEGA: " + str(self.omega))
                self.publish_robot_cmd()
                # Publish image
                image_msg = self._bridge.cv2_to_compressed_imgmsg(self.image, dst_format="jpeg")
                self.pub.publish(image_msg)
                rate.sleep()

if __name__ == '__main__':
    # create the node
    node = LaneFollowNode(node_name='lane_following_node')
    rospy.loginfo_once("Lane Following process is running...")
    # keep spinning
    node.run()
    rospy.spin()