from styx_msgs.msg import TrafficLight
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from sensor_msgs.msg import Image as ImageMsg
import time
from scipy.stats import norm
import cv2
import tensorflow as tf
import rospy
from cv_bridge import CvBridge, CvBridgeError

GRAPH_FILE="ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"

class TLClassifier(object):
    def __init__(self):
        rospy.init_node('tl_classifier', log_level=rospy.DEBUG)

        self.traffic_light_list = []
        self.traffic_light_scores = []
        self.graph_file = GRAPH_FILE
        
        cmap = ImageColor.colormap
        #print("Number of colors =", len(cmap))
        self.COLOR_LIST = sorted([c for c in cmap.keys()])

        #TODO load classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.frame_count = 0

    
    def detect_traffic_lights(self, img, confidence_level=0.2, detect_class_id=[10]):
        #rospy.loginfo("Detection start: %s", time.time())
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        width, height = image.size
        rospy.logdebug("detect_traffic_lights, img.shape=%s", img.shape)

        with tf.Session(graph=self.detection_graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, 
                                                    self.detection_scores, 
                                                    self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_level, boxes, scores, classes, detect_class_id)
        #rospy.logdebug("scores.shape=%s, classes.shape=%s, boxes.shape=%s", scores.shape, classes.shape, boxes.shape)
        #rospy.logdebug("scores=%s, classes=%s", scores, classes)
        #rospy.logdebug(" boxes=%s", boxes)

        box_coords = self.to_image_coords(boxes, height, width)
        #rospy.logdebug(" box_coords.shape=%s, box_coords=%s", box_coords.shape, box_coords)
        #rospy.logdebug(" box_coords=%s", box_coords)
        self.traffic_light_list = []
        for box in box_coords:
            top,left,bottom,right = box
            traffic_light = image.crop((left,top,
                                        right, bottom ) )
            traffic_light = traffic_light.copy()
            traffic_light = traffic_light.resize((32, 32))
            
            self.traffic_light_list.append(traffic_light)
            
        self.traffic_light_scores = scores

        #Debug code start
        '''
        self.draw_boxes(image, box_coords, classes)
        cv2_output_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  

        """
        try:
            image_message = self.bridge.cv2_to_imgmsg(cv2_output_img, encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.detection_traffic_light_pub.publish(image_message)
        """
        cv2.imshow("Image window", cv2_output_img)
        cv2.waitKey(3)
        '''
        #Debug code end


        #pic_filename = "./result/%08d.png"%self.frame_count
        #image.save(pic_filename)
        self.frame_count += 1
        
        if len(scores) < 1:
            rospy.logdebug("No traffic light!")
        else:
            #rospy.logdebug("%d Traffic light(s) detected!, .shape=%s",len(scores), scores.shape)
            rospy.logdebug("%d traffic_light_scores=%s", len(scores), scores)

        return len(scores)
        
    def get_classification(self):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        #return TrafficLight.UNKNOWN
        color_scores = [0.0, 0.0, 0.0]
        for i in range(min(3,len(self.traffic_light_scores))):
            result = self.red_green_yellow(self.traffic_light_list[i])
            rospy.logdebug("Top prob #%d: %.4f %s", i, self.traffic_light_scores[i], result)
            color_scores[result] += self.traffic_light_scores[i]

        rospy.logdebug("color_scores.len=%s, color_scores=%s", len(color_scores), color_scores)
        
        if color_scores[0] > 2 * (color_scores[1] + color_scores[2]):
            return TrafficLight.RED
        elif color_scores[1] > 2 * (color_scores[0] + color_scores[2]):
            return TrafficLight.YELLOW
        elif color_scores[2] > 2 * (color_scores[0] + color_scores[1]):
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN
        

    
    def filter_boxes(self, min_score, boxes, scores, classes, filter_classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] in filter_classes and scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = self.COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    def findNoneZero(self, rgb_image):
        rows,cols,_ = rgb_image.shape
        counter = 0
        for row in range(rows):
            for col in range(cols):
                pixels = rgb_image[row,col]
                if sum(pixels)!=0:
                    counter = counter+1
        return counter

    def red_green_yellow(self, rgb_image, display=False):
        '''
        Determines the red , green and yellow content in each image using HSV and experimentally
        determined thresholds. Returns a Classification based on the values
        '''

        #image = cv2.imread(img_file)
        #image = cv2.resize(image,(32,32))

        #rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        #rgb_image = Image.open(img_file)  
        #rospy.loginfo("Detection Result:%f", rgb_image[1])
        
        #rgb_image.save('./test.jpg')
        rgb_image = np.asarray(rgb_image)

        hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)

        sum_saturation = np.sum(hsv[:,:,1])# Sum the brightness values
        area = 32*32
        avg_saturation = sum_saturation / area #find average
        
        sat_low = int(avg_saturation*1) 
        sat_low_red = int(avg_saturation*1) 
        sat_low_yellow = int(avg_saturation) 
        val_low = 140
        
        #print(sat_low, val_low)


        #Green
        lower_green = np.array([40,sat_low,val_low])
        upper_green = np.array([100,255,255])
        green_mask = cv2.inRange(hsv,lower_green,upper_green)
        green_result = cv2.bitwise_and(rgb_image,rgb_image,mask = green_mask)
        
        #Yellow
        lower_yellow = np.array([10,sat_low_yellow,val_low])
        upper_yellow = np.array([60,255,255])
        yellow_mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
        yellow_result = cv2.bitwise_and(rgb_image,rgb_image,mask=yellow_mask)
        
        # Red 
        lower_red = np.array([150,sat_low_red,val_low])
        upper_red = np.array([180,255,255])
        red_mask1 = cv2.inRange(hsv,lower_red,upper_red)

        lower_red2 = np.array([0,sat_low_red,val_low])
        upper_red2 = np.array([30,255,255])
        red_mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

        red_mask = np.bitwise_or(red_mask1, red_mask2)


        red_result = cv2.bitwise_and(rgb_image,rgb_image,mask = red_mask)
        '''
        if display==True:
            _,ax = plt.subplots(1,5,figsize=(20,10))
            ax[0].set_title('rgb image')
            ax[0].imshow(rgb_image)
            ax[1].set_title('red result')
            ax[1].imshow(red_result)
            ax[2].set_title('yellow result')
            ax[2].imshow(yellow_result)
            ax[3].set_title('green result')
            ax[3].imshow(green_result)
            ax[4].set_title('hsv image')
            ax[4].imshow(hsv)
            plt.show()
        '''
        sum_green = self.findNoneZero(green_result)
        sum_red = self.findNoneZero(red_result)
        sum_yellow = self.findNoneZero(yellow_result)
        #print(sum_red, sum_yellow, sum_green)
        
        if sum_red >= 1.5*sum_yellow and sum_red>=sum_green:
            return 0
        if sum_yellow>=sum_green:
            return 1
        return 2