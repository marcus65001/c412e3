#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, Quaternion, Point, TransformStamped, Transform, Vector3, PoseStamped
from typing import cast
import numpy as np
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY
from cv_bridge import CvBridge
import yaml
from dt_apriltags import Detector
from duckietown_msgs.srv import ChangePattern, ChangePatternResponse
from std_msgs.msg import String
from tf import transformations as tr
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


class TagDetectorNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(TagDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.veh = rospy.get_param("~veh")

        # subscriber
        self.sub_comp_img = rospy.Subscriber('~cam', CompressedImage, self.cb_img)  # camera image topic
        self.sub_cam_info = rospy.Subscriber('~cam_info', CameraInfo, self.cb_cam_info)  # camera info topic

        # publisher
        self.pub = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="The stream of JPEG compressed images from the modified camera feed",
        )
        self.pub_pose = rospy.Publisher(
            "~pose",
            Pose,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
        )

        # services
        self.srvp_led_emitter = rospy.ServiceProxy(
            "~set_pattern", ChangePattern
        )

        # parameters and internal objects
        self.image = None
        self.cam_info = None
        self._bridge = CvBridge()
        self._at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        self._at_detector_cam_para=None

        self.ci_cam_matrix = None
        self.ci_cam_dist = None

        # color mappings
        self.tag_cat_id={
            "ua":[93,94,200,201],
            "t":[58,62,133,153],
            "stop":[162,169],
            # "other":[227]
        }
        self.tag_all_id=np.concatenate([self.tag_cat_id[i] for i in self.tag_cat_id])
        self.tag_color={
            None: "WHITE",
            "ua": "GREEN",
            "stop": "RED",
            "t": "BLUE",
            "other": "LIGHT_OFF"
        }
        self.led_color = "white"
        self.init_dr=True
        self.tag_det=None
        self.tag_det_dist=1.4

        # tranform
        self._tf_broadcaster = TransformBroadcaster()
        self._tf_buffer = Buffer(rospy.Duration(100.0))  # tf buffer length
        self._tf_listener = TransformListener(self._tf_buffer)


    def read_image(self, msg):
        try:
            img=self._bridge.compressed_imgmsg_to_cv2(msg)
            if (img is not None) and (self.image is None):
                self.log("got first msg")
            return img
        except Exception as e:
            self.log(e)
            return np.array([])


    def cb_cam_info(self, msg):
        if not self.cam_info:
            self.cam_info = msg
            self.log('read camera info')
            self.log(self.cam_info)
            # init camera info matrices
            self.ci_cam_matrix=np.array(self.cam_info.K).reshape((3,3))
            self.ci_cam_dist=np.array(self.cam_info.D).reshape((1,5))

            # init tag detector parameters
            camera_matrix=np.array(self.cam_info.K).reshape((3,3))
            self._at_detector_cam_para=(camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])

    def undistort(self, u_img):
        h, w = u_img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.ci_cam_matrix, self.ci_cam_dist, (w, h), 1, (w, h))
        dst = cv2.undistort(u_img, self.ci_cam_matrix, self.ci_cam_dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def tag_id_to_color(self, id):
        cat=None
        for k,v in self.tag_cat_id.items():
            if id in v:
                cat = k
        return self.tag_color[cat] if cat else self.tag_color['other']

    def draw_segment(self, image, pt_A, pt_B, color):
        defined_colors = {
            'RED': ['rgb', [1, 0, 0]],
            'GREEN': ['rgb', [0, 1, 0]],
            'BLUE': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'purple': ['rgb', [1, 0, 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'WHITE': ['rgb', [1, 1, 1]],
            'LIGHT_OFF': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, pt_A, pt_B, (b * 255, g * 255, r * 255), 5)
        return image

    def set_led(self, color):
        if color==self.led_color:
            return
        self.log("Change LED: {}".format(color))
        msg = String()
        msg.data = color
        try:
            self.srvp_led_emitter(msg)
            self.led_color=color
        except Exception as e:
            self.log("Set LED error: {}".format(e))


    def tag_detect(self, img):
        tags = self._at_detector.detect(img, True, self._at_detector_cam_para, 0.051)
        # print(tags)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dist=np.inf
        rcand=None
        for r in tags:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box
            color=self.tag_id_to_color(r.tag_id)
            self.draw_segment(img, ptA, ptB, color)
            self.draw_segment(img, ptB, ptC, color)
            self.draw_segment(img, ptC, ptD, color)
            self.draw_segment(img, ptD, ptA, color)

            # draw the center
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

            # dist
            if (tdist:=np.linalg.norm(r.pose_t))<dist:
                dist=tdist
                rcand=r
        print(rcand)
        if rcand:
            if dist > self.tag_det_dist:
                self.log("tag {} too far ({}), ignored".format(rcand.tag_id, dist))
            else:
                self.tag_det=rcand

        # led
        self.set_led(self.tag_id_to_color(rcand.tag_id) if rcand else "WHITE")
        return img


    def cb_tag_pose_update(self, timer):
        if self.tag_det is None:
            return
        rcand = self.tag_det
        self.tag_det=None
        t = np.zeros((4, 4))
        t[:3, :3] = np.array(rcand.pose_R)
        t[3, 3] = 1.
        rot = tr.euler_from_matrix(t)
        # 1roll->pitch, 2pitch->yaw, 3yaw->roll
        # rotq = Quaternion(*tr.quaternion_from_euler(rot[2]-1.5708, -rot[0], -1.5708-rot[1]))
        rotq = Quaternion(*tr.quaternion_from_euler(*rot))
        tmsg = TransformStamped(
            child_frame_id="{}/at_det".format(self.veh),
            transform=Transform(
                translation=Vector3(*rcand.pose_t), rotation=rotq
            ),
        )
        tmsg.header.stamp = rospy.Time.now()
        tmsg.header.frame_id = "{}/camera_optical_frame".format(self.veh)
        self._tf_broadcaster.sendTransform(
            tmsg
        )
        if rcand.tag_id in self.tag_all_id:
            try:
                t_at_base = self._tf_buffer.lookup_transform("{}/at_det".format(self.veh),
                                                             "{}/footprint".format(self.veh)
                                                             , rospy.Time(0))
                t_at_base.child_frame_id = "{}/v_pred".format(self.veh)
                t_at_base.header.frame_id = "at_{}_static".format(rcand.tag_id)
                self.log("t_at_base: {}".format(t_at_base))
                self._tf_broadcaster.sendTransform(
                    t_at_base
                )

                # project
                t_base_w = self._tf_buffer.lookup_transform(
                    "world",
                    "{}/v_pred".format(self.veh),
                    rospy.Time(0))

                pose_new = Pose(Point(t_base_w.transform.translation.x,
                                      t_base_w.transform.translation.y,
                                      0),
                                t_base_w.transform.rotation)
                self.pub_pose.publish(pose_new)
            except Exception as e:
                self.log(e)

    def cb_img(self, msg):
        # image callback
        if self._bridge and (self.ci_cam_matrix is not None):
            # rectify
            self.image=self.read_image(msg)


    def init_msg(self):
        if self.pub_pose.get_num_connections()>0:
            self.init_dr=False
            self.log("init pos published")
            q = tr.quaternion_from_euler(0,0,1.5708)
            pm = Pose(Point(0.32, 0.32, 0), Quaternion(*q))
            self.pub_pose.publish(pm)
            return pm

    def run(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.init_dr:
                self.init_msg()
                timer_tag_pose=rospy.Timer(rospy.Duration(0.3),self.cb_tag_pose_update)
            if self.image is not None:
                # publish image
                if not self.image.size:
                    return
                ud_image = self.undistort(self.image)
                # grayscale
                g_image = cv2.cvtColor(ud_image, cv2.COLOR_BGR2GRAY)
                # tag detection
                td_image = self.tag_detect(g_image)
                # publish
                image_msg = self._bridge.cv2_to_compressed_imgmsg(td_image, dst_format="jpeg")
                self.pub.publish(image_msg)
                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = TagDetectorNode(node_name='apriltag_node')
    # keep spinning
    node.run()
    rospy.spin()