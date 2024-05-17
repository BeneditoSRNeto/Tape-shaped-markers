import numpy as np
import cv2
from time import process_time
import stag

contador=0
texto=[]
fim=0
cont_frame=0
inicio=process_time()
reloading=0.0000   
url1="C:\\tape-shaped-marker\\videos\\conditionA\\stag.mp4"    
  

camera = cv2.VideoCapture(url1)

while True:
    
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
    
    tmp=True
    cont_frame=cont_frame+1
    
  
    img=frame.copy()
    img2=frame.copy()
    cv2.imshow("cam", img)
    # detect markers
    (corners, ids, rejected_corners) = stag.detectMarkers(img, 11)
    if len(corners) > 0:

            for (markerCorner, markerID) in zip(corners, ids):
                
                if tmp:
                    contador=contador+1
                    
                
                    tmp=False

                            # draw detected markers with ids
                    stag.drawDetectedMarkers(img, corners, ids)

                   
                    cv2.imshow("cam",img)
                   



    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()

cv2.waitKey()           
cv2.destroyAllWindows()

