"""
A demo for object detection using a TFLite model, from the model zoo.
Is non performant or production ready. The purpose was only to demonstrate
the difference between running a TF full model vs TFLite with a code
structured in the same way for helping comparison.
See: https://github.com/carlosbravoa/tensorflow-recipes/blob/master/my_TF_object_detection.py

This runs with the starter coco quantized model in the following URL:
https://www.tensorflow.org/lite/models/object_detection/overview
"""
import argparse
import numpy as np
import time
from collections import deque
import tensorflow as tf

#For webcam capture and drawing boxes
import cv2


# Parameters for visualizing the labels and boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.6
FONT_THICKNESS = 1
LINE_WEIGHT = 1
SHOW_CONFIDENCE_IN_LABEL = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)
    parser.add_argument(
        '--label', help='Path of the labels file.')
#    parser.add_argument(
#        '--mode', help='Mode for de detection: OBJECT_DETECTION or IMAGE_CLASSIFICATION',
#        required=True)
    parser.add_argument(
        '--camera', help='Camera source (if multiple available)', type=int, required=False)

    args = parser.parse_args()

    # Initialize the camera
    camera = args.camera if args.camera else 0
    cam = cv2.VideoCapture(camera)
    ##cam = cv2.VideoCapture('videoplayback2.mp4') ## If you want to capture from a video

    # Initialize model. args.model should point to the tflite file
    MODEL_NAME = args.model

    #read labels (pbtxt from model zoo)
    PATH_TO_LABELS = args.label

    print("Starting interpreter")
    interpreter = tf.lite.Interpreter(model_path=MODEL_NAME)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if PATH_TO_LABELS:
        labels = read_label_file(PATH_TO_LABELS)
    else:
        labels = {}

    # Initialize the timer for fps
    start_time = time.time()
    frame_times = deque(maxlen=40)

    # Start capturing
    print("Starting the camera")
    #while True:
    while True:
        ret, cv2_im = cam.read()

        start_inference_time = time.time() * 1000

        #This is where inference happens
        output_dict = identify_with_npimage(cv2_im,
                                            interpreter,
                                            input_details,
                                            output_details,
                                            threshold=0.5)

        last_inference_time = time.time() * 1000 - start_inference_time

        if SHOW_CONFIDENCE_IN_LABEL:
            confidence = output_dict['detection_scores']
        else:
            confidence = {}

        real_num_detection = output_dict['num_detections']
        draw_rectangles(cv2_im,
                        real_num_detection,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        confidence,
                        labels=labels)

        frame_times.append(time.time())
        fps = len(frame_times)/float(frame_times[-1] - frame_times[0] + 0.001)
        draw_text(cv2_im, "{:.1f}fps / {:.2f}ms".format(fps, last_inference_time) + " Detections: "+ str(real_num_detection))
        

        # flipping the image:
        #cv2.flip(cv2_im, 1)
        #resizing the image
        #cv2_im = cv2.resize(cv2_im, (800, 600))
        cv2.imshow('object detection', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    #end
    exit()

def identify_with_npimage(cv2_im, interpreter, input_details, output_details, threshold=0.5):
    # Resize and normalize image for network input
    input_shape = input_details[0]['shape']
    frame = cv2.resize(cv2_im, ((input_shape[2], input_shape[1]))).reshape(input_shape)

    # run model
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()

    # get results
    output_dict = {}
    output_dict['detection_boxes'] = interpreter.get_tensor(output_details[0]['index'])[0]
    output_dict['detection_classes'] = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int64)
    output_dict['detection_scores'] = interpreter.get_tensor(output_details[2]['index'])[0]
    output_dict['num_detections'] = int(interpreter.get_tensor(output_details[3]['index'])[0])

    return output_dict

def read_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def draw_rectangles(image_np, num_detections, boxes, classes, scores, labels={}):
    i = 0
    im_height, im_width, channels = image_np.shape
    for box in boxes:
        if scores[i] > 0.5:
            ymin = box[0] * im_height
            xmin = box[1] * im_width
            ymax = box[2] * im_height
            xmax = box[3] * im_width

            rectangle = [[xmin, ymin], [xmax, ymax]]
            if len(scores) > 0:
                score = "({0:.2f})".format(scores[i])
            else:
                score = ""

            if len(labels) > 0:
                draw_single_rectangle(rectangle, image_np, labels[classes[i]]  + score)
            else:
                draw_single_rectangle(rectangle, image_np, str(classes[i]) + score)
            i += 1
            if i >= num_detections:
                break

def draw_single_rectangle(rectangle, image_np, label=None):
    BOXCOLOR = (255, 0, 0)
    p1 = (int(rectangle[0][0]), int(rectangle[0][1]))
    p2 = (int(rectangle[1][0]), int(rectangle[1][1]))
    cv2.rectangle(image_np, p1, p2, color=BOXCOLOR, thickness=LINE_WEIGHT)
    if label:
        size = cv2.getTextSize(label, FONT, FONT_SIZE, FONT_THICKNESS)
        center = p1[0] + 5, p1[1] + 5 + size[0][1]
        pt2 = p1[0] + 10 + size[0][0], p1[1] + 10 + size[0][1]
        cv2.rectangle(image_np, p1, pt2, color=(255, 0, 0), thickness=-1)

        cv2.putText(image_np, label, center, FONT, FONT_SIZE, (255, 255, 255), 
                    FONT_THICKNESS, cv2.LINE_AA)
    #imgname = str(time.time())
    #cv2.imwrite('/home/pi/development/Coral-TPU/imgs/' + imgname + '.jpg', image_np)

def draw_text(image_np, label, pos=0):
    p1 = (0, pos*30+20)
    #cv2.rectangle(image_np, (p1[0], p1[1]-20), (800, p1[1]+10), color=(0, 255, 0), thickness=-1)
    cv2.putText(image_np, label, p1, FONT, FONT_SIZE, (0, 255, 0), 1, cv2.LINE_AA)

if __name__ == '__main__':
    main()
