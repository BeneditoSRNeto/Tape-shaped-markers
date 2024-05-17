import numpy as np
import cv2
from time import process_time

contador=0
texto=[]
fim=0
cont_frame=0
inicio=process_time()
reloading=0.0000   
url1="C:\\tape-shaped-marker\\videos\\conditionA\\aruco.mp4"    
 

camera = cv2.VideoCapture(url1)

while True:
    
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
   

    img=frame.copy()
    img2=frame.copy()
    cv2.imshow("cam", img)
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    
    
    Aruco_Dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    Aruco_Params = cv2.aruco.DetectorParameters()
    (marker_corners, marker_id, rejected_markers) = cv2.aruco.detectMarkers(gray, Aruco_Dict,parameters=Aruco_Params)

    if len(marker_corners) > 0:
            
            marker_id = marker_id.flatten()

            for (markerCorner, markerID) in zip(marker_corners, marker_id):
                
                

                    marker_corners = markerCorner.reshape((4, 2))
                    (top_Left, top_Right, bottom_Right, bottom_Left) = marker_corners
                    top_Right = (int(top_Right[0]), int(top_Right[1]))
                    bottom_Right = (int(bottom_Right[0]), int(bottom_Right[1]))
                    bottom_Left = (int(bottom_Left[0]), int(bottom_Left[1]))
                    top_Left = (int(top_Left[0]), int(top_Left[1]))

                    cv2.line(img, top_Left, top_Right, (0, 255, 0,255), 2)
                    cv2.line(img, top_Right, bottom_Right, (0, 255, 0,255), 2)
                    cv2.line(img, bottom_Right, bottom_Left, (0, 255, 0,255), 2)
                    cv2.line(img, bottom_Left, top_Left, (0, 255, 0,255), 2)
                    cX = int((top_Left[0] + bottom_Right[0]) / 2.0)
                    cY = int((top_Left[1] + bottom_Right[1]) / 2.0)
                    cv2.circle(img, (cX, cY), 4, (0, 255, 0,255), -1)
                    cv2.putText(img, str(markerID),(top_Left[0], top_Left[1]-5), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 0,255), 2)
                    cv2.imshow("cam",img)
                    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()

cv2.waitKey()           
cv2.destroyAllWindows()

