import numpy as np
import cv2
from time import process_time
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
contador=0
texto=[]
reloading=0.000000
inicio=process_time()
cont_frame=0

url1="C:\\tape-shaped-marker\\videos\\conditionA\\qrcode.mp4"    
  
camera = cv2.VideoCapture(url1)    
while True:
    
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
   
    img=frame.copy()
    img2=frame.copy()
    cv2.imshow("cam", img)
   
    tmp=True
    cont_frame=cont_frame+1
    barcodes = pyzbar.decode(img, symbols=[ZBarSymbol.QRCODE])
    found = []
    for barcode in barcodes:
            text = barcode.data.decode('utf-8')
            if len(text)>0:
               
                    x, y, w, h = barcode.rect
                
                    tex=str(text)
                    cor=(255, 255, 0,255)
                    fonte = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, tex, (x,y-5), fonte, 0.5, cor, 0, cv2.LINE_AA)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0,255), 2)
                    cv2.imshow("cam",img)
                   


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()



cv2.waitKey()           
cv2.destroyAllWindows()


