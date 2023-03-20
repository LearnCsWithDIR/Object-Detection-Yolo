# this way yolo use to image objects detection

from ultralytics import YOLO
import cv2

# only one type Yolo file can be run in one time

model1 = YOLO('yolov8n.pt')  # n is nano (detection minimum details)| v8 is version 8
#model2 = YOLO('yolov8m.pt')  # m is medium (detection medium details) | v8 is version 8
#model3 = YOLO('yolov8l.pt')  # l is large (detection large details) | v8 is version 8

results1 = model1("Images/animal_img.jpg",show = True)
#results2 = model2("Images/person_img.jpg",show = True)
#results3 = model3("Images/road_img.jpg",show = True)

cv2.waitKey(0) # Display to output detection GUI (If this part is not include then, GUI Appear & close one time )



