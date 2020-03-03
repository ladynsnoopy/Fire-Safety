from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import csv
from time import *
import threading

# countdown thread
def countdown():
    my_timer = 30
    global stop_threads
    global alr_running
    for x in range(30):
        if (stop_threads and alr_running):
            break
        my_timer -= 1
        print("time left:"+ str(my_timer))
        sleep(1)
        if(my_timer == 0):
            print("Send Warning")



# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolo-obj_final.weights'
class_path='config/obj.name.txt'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = 'images/Object_Cooking_IR_02.avi'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]


vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()
fps = vid.get(cv2.CAP_PROP_FPS)
cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".avi", "-det.avi"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()
row_list = []
row_list.append(["Time Stamp","x1","y1","Height","Width","Class Name","Class Probability"])

# start the  countdown thread
stop_threads = False
alr_running = False
countdown_thread = threading.Thread(target=countdown)
countdown_thread.start()
while(True):
    ret, frame = vid.read()
    old_cls = "no person"
    if not ret:
        # write to csv
        with open('results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        numpyArr = detections.cpu().detach().numpy()
        classProb = numpyArr[0][4]
        tracked_objects = mot_tracker.update(detections.cpu())
        current = frames / fps
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            originalX = x1
            originalY = y1
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            arr = [current,originalX,originalY,box_h,box_w,cls,classProb]
            if(cls == "person"):
                # kill the countdown thread
                stop_threads = True
                alr_running = True
                if countdown_thread is not None:
                    countdown_thread.join()
                old_cls = "person"
            row_list.append(arr)
            print(arr)

    if(old_cls != "person"):
        stop_threads = False
        # start a new countdown thread
        if(alr_running == True):
            countdown_thread = threading.Thread(target=countdown)
            countdown_thread.start()
            alr_running = False

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()