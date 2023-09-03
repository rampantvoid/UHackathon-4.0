import yolov5
import matplotlib.pyplot as plt
# import torch
# import cv2
# import numpy as np
# import time

# pass these when calling the function
areaArr = [[(430,67),(111,488),(962,478),(610,61)],[(626,246),(163,490),(731,487),(718,254)],[(362,145),(242,485),(927,485),(618,135)],[(608,209),(365,490),(775,495),(715,189)]]
videofiles = ['intersection1.mp4','intersection2.mp4','intersection3.mp4','intersection4empty.mp4']


# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)

import torch
import cv2
import numpy as np

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

cap=cv2.VideoCapture('intersection1.mp4')

# model = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained=True)
model = yolov5.load('yolov5x.pt')
model.conf = 0.01
model.iou = 0.45  
model.agnostic = False 
model.multi_label = False  
model.max_det = 1000 
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        cx= int(x1+x2)//2
        cy= int(y1+y2)//2
        result = cv2.pointPolygonTest(np.array([(430,67),(111,488),(962,478),(610,61)],np.int32),((cx,cy)),False)
        d=(row['name'])
        print(d)
        if result >=0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
            cv2.circle(frame,(cx,cy),3,(255,0,0),-1)
    cv2.polylines(frame,[np.array([(430,67),(111,488),(962,478),(610,61)],np.int32)],True,(0,0,255),2)

    cv2.imshow("ROI",frame)
    cv2.imwrite('result.jpg',frame)
    # results.save(save_dir='results')
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()         


