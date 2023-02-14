#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage, CameraInfo
from typing import cast
import numpy as np
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY
from cv_bridge import CvBridge
import yaml
from rectification import Rectify
from dt_apriltags import Detector


class TagDetectorNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(TagDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # subscriber
        self.sub_comp_img = rospy.Subscriber('~cam', CompressedImage, self.cb_img)
        self.sub_cam_info = rospy.Subscriber('~cam_info', CameraInfo, self.cb_cam_info)

        # publisher
        self.pub = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="The stream of JPEG compressed images from the modified camera feed",
        )
        self.image = None
        self.cam_info = None
        self._bridge = CvBridge()
        self._rect = None
        self._at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        self._at_detector_cam_para=None

    def read_image(self, msg):
        try:
            img=self._bridge.compressed_imgmsg_to_cv2(msg)
            if img and not self.image:
                self.log("got first msg")
            return img
        except Exception as e:
            self.log(e)
            return []

    def init_rect(self, cam_inf):
        self.log("init rectification")
        self._rect=Rectify(cam_inf)

    def cb_cam_info(self,msg):
        if not self.cam_info:
            self.cam_info = msg
            self.log('read camera info')
            self.log(self.cam_info)
            # init rectification
            self.init_rect(self.cam_info)
            # init tag detector
            camera_matrix=np.array(self.cam_info["K"]).reshape((3,3))
            self._at_detector_cam_para=(camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])


    def cb_img(self, msg):
        if self._bridge and self._rect:
            rec_img=self._rect.rectify(self.read_image(msg))
            self.image=rec_img
            tags = self._at_detector.detect(self.image, True, self._at_detector_cam_para, ?)
            print(tags)

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.
            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def run(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.image is not None:
                image_msg = self._bridge.cv2_to_compressed_imgmsg(self.image, dst_format="jpeg")
                self.pub.publish(image_msg)
                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = TagDetectorNode(node_name='apriltag_node')
    # keep spinning
    node.run()
    rospy.spin()