import cv2
import numpy as np

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

# camera
cap = cv2.VideoCapture('video.mp4')
min_width_rect = 80 # Minimum width rectangle
min_height_rect = 80 # Minimum height rectangle
count_line_position = 550

# Initialize Substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
#algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h): # Point in the center of the detected object
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6  # Allowable error between pixels
counter = 0
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame1 = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame1,confThreshold=0.55)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    # Applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3) # Line posistion

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (w >= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2) # Object detection 
        cv2.putText(frame1, "Object" + str(counter), (x, y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 244, 0), 1)
        

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        if (len(ClassIndex)!=0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                if (ClassInd<=80):
                    cv2.rectangle(frame1,boxes,(255, 0, 0), 2)
                    cv2.putText(frame1,classLabels[ClassInd-1] + str(counter),(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0), thickness=3)

        for (x, y) in detect: 
            if y < (count_line_position + offset) and y > (count_line_position - offset):   # Cross the line by objects
                counter += 1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove((x, y))

            print("Objects counter: " + str(counter))

    cv2.putText(frame1, "OBJECTS COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('Detector', dilatada)
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()