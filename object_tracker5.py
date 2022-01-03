# General imports
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow imports
# Comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Set memory for GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Image imports
from PIL import Image
import cv2


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('top_color', 'green', 'color of the object to be tracked')
flags.DEFINE_string('bottom_color', 'green', 'color of the object to be tracked')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to trackermageFrame,
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  

        # detect colors into the bboxes of detected people
        # cast into integer
        boxs2 = boxs.astype(int)

        # if there is no detection skip boxes analysis
        if boxs2.size is not 0:
            #print(type(boxs[0][0]))
            # analysis each box
            for [x,y,w,h] in boxs2:
                # upper frame
                pixels_box_upper = frame[y:int(y+h/2), x:x+w]
                # lower frame
                pixels_box_lower = frame[int(y+h/2):y+h, x:x+w]
                # master box
                master_box = frame[y:y+h, x:x+w]

                # Color definitions
                red_lower = np.array([136, 87, 111], np.uint8)
                red_upper = np.array([180, 255, 255], np.uint8)

                black_lower = np.array([0, 0, 0], np.uint8)
                black_upper = np.array([180, 255, 30], np.uint8)

                green_lower = np.array([39, 138, 255], np.uint8)
                green_upper = np.array([255, 43, 255], np.uint8)

                blue_lower = np.array([0, 131, 9], np.uint8)
                blue_upper = np.array([164, 253, 113], np.uint8)


                # detect color red
                hsvFrame_lower = cv2.cvtColor(pixels_box_lower, cv2.COLOR_RGB2HSV) 
                hsvFrame_upper = cv2.cvtColor(pixels_box_upper, cv2.COLOR_RGB2HSV) 
                kernal = np.ones((5, 5), "uint8")

                # Bottom frame analysis
                if FLAGS.bottom_color == "red":
                    mask_lower = cv2.inRange(hsvFrame_lower, red_lower, red_upper)
                    color_string_lower = "Red Color"
                    color_box_lower = (255, 0, 0)
                elif FLAGS.bottom_color == "black":
                    mask_lower = cv2.inRange(hsvFrame_lower, black_lower, black_upper)
                    color_string_lower = "Black Color"
                    color_box_lower = (0, 0, 0)
                elif FLAGS.bottom_color == "blue":
                    mask_lower = cv2.inRange(hsvFrame_lower, blue_lower, blue_upper)
                    color_string_lower = "Blue Color"
                    color_box_lower = (0, 0, 255)
                else:
                    mask_lower = cv2.inRange(hsvFrame_lower, green_lower, green_upper)
                    color_string_lower = "Green Color"
                    color_box_lower = (0, 255, 0)

                # Top frame analysis
                if FLAGS.top_color == "red":
                    mask_upper = cv2.inRange(hsvFrame_upper, red_lower, red_upper)
                    color_string_upper = "Red Color"
                    color_box_upper = (255, 0, 0)
                elif FLAGS.top_color == "black":
                    mask_upper = cv2.inRange(hsvFrame_upper, black_lower, black_upper)
                    color_string_upper = "Black Color"
                    color_box_upper = (0, 0, 0)
                elif FLAGS.top_color == "blue":
                    mask_upper = cv2.inRange(hsvFrame_upper, blue_lower, blue_upper)
                    color_string_upper = "Blue Color"
                    color_box_upper = (0, 0, 255)
                else:
                    mask_upper = cv2.inRange(hsvFrame_upper, green_lower, green_upper)
                    color_string_upper = "Green Color"
                    color_box_upper = (0, 255, 0)

                
                mask_lower = cv2.dilate(mask_lower, kernal)
                res_mask_lower = cv2.bitwise_and(pixels_box_lower, pixels_box_lower , mask = mask_lower)

                mask_upper = cv2.dilate(mask_upper, kernal)
                res_mask_upper = cv2.bitwise_and(pixels_box_upper, pixels_box_upper , mask = mask_upper) 

                contours_lower, hierarchy_lower = cv2.findContours(mask_lower, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_upper, hierarchy_upper = cv2.findContours(mask_lower, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
                
                match_top = False
                match_low = False

                for pic, contour_lower in enumerate(contours_lower): 
                    area = cv2.contourArea(contour_lower) 
                    if(area > 5000): 
                        x, y, w, h = cv2.boundingRect(contour_lower) 
                        pixels = cv2.rectangle(pixels_box_lower, (x, y), (x + w, y + h), color_box_lower, 2)    
                        cv2.putText((pixels), color_string_lower, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_box_lower)
                        match_low = True

                for pic, contour_upper in enumerate(contours_upper): 
                    area = cv2.contourArea(contour_upper) 
                    if(area > 5000): 
                        x1, y1, w1, h1 = cv2.boundingRect(contour_upper) 
                        pixels1 = cv2.rectangle(pixels_box_upper, (x1, y1), (x1 + w1, y1 + h1), color_box_upper, 2)    
                        cv2.putText((pixels1), color_string_upper, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_box_upper)
                        match_top = True 

                if match_low and match_top:
                    hsvFrame_master = cv2.cvtColor(master_box, cv2.COLOR_RGB2HSV)



        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if boxs2.size is not 0:
            lower = cv2.cvtColor(hsvFrame_lower, cv2.COLOR_HSV2BGR)
            upper = cv2.cvtColor(hsvFrame_upper, cv2.COLOR_HSV2BGR)
            if match_low and match_top:
                master = cv2.cvtColor(hsvFrame_master, cv2.COLOR_HSV2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
            if boxs2.size is not 0:
                cv2.imshow("Lower frame analysis", lower)
                cv2.imshow("Upperframe analysis", upper)
                if match_low and match_top:
                    cv2.imshow("master", master)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
