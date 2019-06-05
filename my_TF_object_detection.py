"""A demo for object detection using TF models from the model zoo.
The intention was to have a single script able to run a model without
the need of any other utility or dependency than cv2 on top of the regular ones.
There is a lot of good code ready to use from the TF repo.
Reference for this code:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

The model zoo: 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Download one of the models. The script will look for the frozen file.
Add the labels in txt format and ready to go


TODO: Adjustable threshold. Currently it displays all
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
SHOW_CONFIDENCE_IN_LABEL = False

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

    # Initialize model.
    MODEL_NAME = args.model
    # Path to frozen detection graph. This is the model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    #read labels (pbtxt from model zoo)
    PATH_TO_LABELS = args.label

    print("Starting session")
    detection_graph = tf.Graph()

    sess = tf.Session(graph=detection_graph)

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        print("Loading frozen graph")
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # Get handles to input and output tensors
            print("Getting both ends of the model")
            input_tensor = get_input_tensor() #The image
            output_tensor = get_output_tensor() #The evaluation results

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
            while cam.isOpened():
                ret, cv2_im = cam.read()
                start_inference_time = time.time() * 1000

                #This is where inference happens
                output_dict = identify_with_npimage(cv2_im, 
                                                    sess, 
                                                    output_tensor, 
                                                    input_tensor, 
                                                    threshold=0.05, 
                                                    top_k=10)

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
                                labels=labels,
                                scores=confidence)

                frame_times.append(time.time())
                fps = len(frame_times)/float(frame_times[-1] - frame_times[0] + 0.001)
                draw_text(cv2_im, "{:.1f}fps / {:.2f}ms".format(fps, last_inference_time) + " Detections: "+ str(real_num_detection))
                #draw_text(cv2_im, "{:.1f}".format(fps))

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

def get_input_tensor():
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return image_tensor

def get_output_tensor():
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image
        # coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        #detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        #    detection_masks, detection_boxes, image.shape[0], image.shape[1])
        #detection_masks_reframed = tf.cast(
        #    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        #tensor_dict['detection_masks'] = tf.expand_dims(
        #    detection_masks_reframed, 0)
    return tensor_dict

def identify_with_npimage(image_np, sess, tensor_dict, image_tensor, threshold=0.5, top_k=10):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

#Not needed for this particular example
def load_pil_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def read_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def draw_rectangles(image_np, num_detections, boxes, classes, labels={}, scores={}):
    i = 0
    im_height, im_width, channels = image_np.shape
    for box in boxes:
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
            draw_single_rectangle(rectangle, image_np, labels[classes[i]-1]  + score)
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
