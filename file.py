import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import necessary modules from the extracted project
import sys
# Temporarily add the project directory to the system path to import project modules
if output_directory not in sys.path:
    sys.path.append(output_directory)

from config import YOLO_CONFIG, VIDEO_CONFIG, FRAME_SIZE, TRACK_MAX_AGE
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from tracking import detect_human # Assuming detect_human is reusable

# Define paths from the config (adjusted for the current output_directory)
WEIGHTS_PATH = os.path.join(output_directory, YOLO_CONFIG["WEIGHTS_PATH"])
CONFIG_PATH = os.path.join(output_directory, YOLO_CONFIG["CONFIG_PATH"])
DEEPSORT_MODEL_PATH = os.path.join(output_directory, 'model_data/mars-small128.pb')

# Load the YOLOv3-tiny pre-trained COCO dataset
# Ensure the batch and subdivisions are set to 1 in the .cfg file (already done in previous steps)
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
ln = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]

# Tracker parameters
max_cosine_distance = 0.7
nn_budget = None
encoder = gdet.create_box_encoder(DEEPSORT_MODEL_PATH, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=TRACK_MAX_AGE)

# Load the image provided by the user
image_path = '/content/pexels-jopwell-2422290.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Resize image for consistent processing if FRAME_SIZE is set and image is larger
    (h, w) = image.shape[:2]
    if w > FRAME_SIZE:
        image = cv2.resize(image, (FRAME_SIZE, int(h * FRAME_SIZE / w)))
    elif h > FRAME_SIZE:
        image = cv2.resize(image, (int(w * FRAME_SIZE / h), FRAME_SIZE))

    # The detect_human function expects a frame, let's call it with a dummy record_time
    # Note: detect_human modifies the frame in place, so make a copy if original is needed
    processed_image = image.copy()
    record_time = 0 # Dummy value as it's not a video stream

    # detect_human function returns a list of humans_detected and expired tracks. We are only interested in detections
    # The function also draws on the image if SHOW_PROCESSING_OUTPUT was True, but we turned it off.
    # We will manually draw bounding boxes here.

    # To get raw detections (boxes, scores, classIDs), we might need to modify tracking.py or extract the logic.
    # For simplicity, let's try to adapt the output of detect_human or its internal components.

    # Re-reading tracking.py and video_process.py, detect_human primarily works with DeepSORT
    # and doesn't directly return bounding boxes for drawing. It updates the tracker.
    # It would be better to extract YOLO detection logic directly from video_process.py or main.py
    # This part requires a more direct implementation of YOLO detection on an image.

    # Let's extract the YOLO detection part for a single image.

    # Get image dimensions
    (H, W) = processed_image.shape[:2]
    # Construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(processed_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    for output in layer_outputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions and ensure it's a 'person' (COCO dataset class ID for person is 0)
            if confidence > 0.5 and class_id == 0: # Assuming 0 is the class ID for 'person'
                # Scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) # confidence_threshold, NMS_threshold

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y, w, h) = boxes[i]

            # Draw a bounding box rectangle and label on the image
            color = (0, 255, 0) # Green color for bounding box
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), color, 2)
            text = f"Person: {confidences[i]:.2f}"
            cv2.putText(processed_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert image from BGR to RGB for matplotlib display
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_image_rgb)
    plt.title(f"Crowd Analysis: Detections on {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
