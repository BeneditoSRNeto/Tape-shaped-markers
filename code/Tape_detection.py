
from operator import is_not
import numpy as np
import cv2
import numpy as np
from numpy import array
from hashlib import blake2b

import math
import matplotlib.pyplot as plt
import json

from time import process_time

texto=[]


def loadJson():
          json_path='metadados.json' 
          
          f = open (json_path, "r")
          data = json.loads(f.read())
          f.close()
          return data



def isCM(identificador):
    
    isCM=False
    data=loadJson()
    for i in data:
        if i['idcm']== identificador:
            isCM=True
           
    return isCM

def isTM(identificador):
    
    isTM=False
    data=loadJson()
    for i in data:
        if i['idtm']== identificador:
            isTM=True
           
    return isTM

def lerCMToDecimal(identificador):
    decimal=0
    y=0
    isCM=0
    pontos=[]
    data=loadJson()
    for i in data:
        if i['idcm']== str(identificador):
            isCM=1
            
            
            decimal=i['idcm_decimal']
            
    return decimal

def lerCMToChecsum(identificador):
    checksum=0
    y=0
    isCM=0
    pontos=[]
    data=loadJson()
    for i in data:
        if i['idcm']== str(identificador):
            isCM=1
           
            
            checksum=i['checksum']
            
    return checksum



def escreve_local(img, texto, posicao, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (int(posicao[0]),int(posicao[1]+20)), fonte, 0.3, cor, 0, cv2.LINE_AA)



def getLine(img):
    Array_linhas=[]
    
    h, w = img.shape[:2]
    rot=np.zeros((h,w))
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.equalizeHist(gray)
    
    dst = cv2.Canny(gray, 50, 150, apertureSize =3)

    lines_1 = cv2.HoughLines(dst, 1, np.pi / 180, 200)
    
    
    Mt=[]
    
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)  
    
    if lines_1 is not None:
        
        for l in lines_1:
            lines_list=[]
           
            rho = l[0][0]
           
            theta = l[0][1]
            
            a = math.cos(theta)
            b = math.sin(theta) 
            x0 = a * rho
            y0 = b * rho       
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                
            M = cv2.getRotationMatrix2D((cX, cY), math.degrees(theta) -90, 1.0)
            
            lines_list.append([pt1[0],pt2[0] ]) #X
            lines_list.append([pt1[1],pt2[1] ] ) #Y
            lines_list.append([1,1 ] )
            
            Array_linhas.append({'M':M,'lines_list':lines_list})

        return Array_linhas
    else:
        return None
   


def bit_to_bars(stri):
    string=""
    string=stri
    bars = []
      
    current_length = 1
      
    lenght=0
    
    if len(string)>0:
       
            
       
        for i in range(len(string)-1):
            if string[i] == string[i+1]:
                current_length = current_length + 1
            else:
                bars.append(current_length * str(string[i]))
                current_length = 1
        
    return bars



def remove_quieteZone(b):
    barra_q=[]
    valida=0
    inicio=0
    fim=0
    
    cont=0
    barra_q=b
    barra=[]
   
    for i in range(0,len(barra_q)-6):
        if abs(len(barra_q[i])-len(barra_q[i+1]))< 2:    
            if   ( abs(len(barra_q[i])-len(barra_q[i+2]))< 2  and abs(len(barra_q[i])-len(barra_q[i+3]))< 2   and abs(len(barra_q[i+1])-len(barra_q[i+3]))< 2 and  abs(len(barra_q[i+5])-len(barra_q[i+6]))< 2 and abs(len(barra_q[i+4])-len(barra_q[i+6]))< 2   ):
               
                inicio=i
                break
                    
    
    
    if inicio!= 0 :
        
        barra=barra_q
        
    else: 
        barra=[]
    
    return barra


def remove_barras(barra):
    new_bar=barra  
    if len(new_bar) >= 7:
        for j in range(7):
            new_bar.pop(0)         
        ##inversa
        
        for j in range(7):
            if len(new_bar) > 6:
                new_bar.pop(len(new_bar)-1)                
        #remove marcador
        
   # 
    return new_bar

def remove_quadrado(barra):
    bar=barra
    new=[]
    new2=[]
    maior=0
    
    if len(bar) > 0:
        for k in range(len(bar)-1):
            if len(bar[k]) > 10:
                new.append(bar[k])
                                   
  
    return new

def encode_bars(array_bar):
    
    array=array_bar
    s = ""
    for value in array:
        if value[0]=="1":
            s = s + "0"
        if value[0]=="0":
            s = s + "1"
      
    return s


def array_to_string(array1):
    array=array1
    s = ""
    
   
    for value in array:
        
        s = s + str(value)
        
    

    return s


def valida_detect(array):
    #
    bi_t=""
    saida=""
    tp=0
    if len(array) >1:
        for k in range(0,len(array)):
            contador=0
            binario=array[k]        
            for i in range(0,len(array)):
                if array[i]==binario:
                    contador=contador+1
                               
                if tp<contador:
                    tp=contador
                    bi_t=binario
        if abs(tp/len(array))>=0.7:
              saida=bi_t
    
    return saida 

def escreve(img, texto, cor=(0,0,220)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (250,250), fonte, 1.5, cor, 0, cv2.LINE_AA)


def FiltragemInicial(img):
    obj3=cv2.medianBlur(img,3)
    B,G,R=cv2.split(obj3)
 
    
    v1= np.absolute(R.astype(float)-B.astype(float)) < 30
    
    v2= np.absolute(G.astype(float)-B.astype(float)) < 30
    
    v3= np.absolute(R.astype(float)-G.astype(float)) < 30
    
    obj2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    v6= obj2 < 60
    v7= obj2 > 155 
    
    v4=np.logical_and(v1,v2)
    v5=np.logical_and(v3,v4)
    
    
    v8=np.logical_and(v5,v6)
    
    v9=np.logical_and(v5,v7)
       
    obj2[v8]=[0]
    obj2[v9]=[255]
   
    return obj2

  
def Detectect(img):
    V=[]
    sucesso2=False  
    img = cv2.equalizeHist(img)
    obj= cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 4)
   
    
    contours, hierarchy = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    frame2=frame.copy()
   
    cont=0
   
    for i in range(0,len(contours) ):
        rect = cv2.minAreaRect(contours[i])
        epsilon = 0.1*  cv2.arcLength(contours[i],True)
        approx = cv2.approxPolyDP(contours[i],epsilon,True)
        if rect[1][1] > 0:
            aspect=rect[1][0]/rect[1][1]
            

            if aspect >= 0.6 and aspect <= 1.0 and approx.shape[0]==4 and cv2.arcLength(contours[i],True) :#and hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] == -1:
                j=hierarchy[0][i][2]
                if hierarchy[0][j][2] >= 0:
                   k=hierarchy[0][j][2]
                   if hierarchy[0][k][3] >= 0:
                        
                        cnt=contours[i]
                        rect = cv2.minAreaRect(cnt)
                        pontos = cv2.boxPoints(rect)
                        box = np.int0(pontos)
                                             
                        retangulo = cv2.minAreaRect(contours[i])
                                                
                        box = cv2.boxPoints(retangulo)
                                                            
                        V.append(box)
                              
    
    return V


    
def Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame):
    saida2=""
    B=[]
    tag=False
    warped_img=Warp(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
   
    temp=FiltragemInicial(warped_img.copy())
    temp = cv2.equalizeHist(temp)
   
    suave = cv2.GaussianBlur(temp, (13, 13), 0) # aplica blur
    ret,th = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    imagem_cinza=temp
   
    A=np.zeros((7,16))
    size=832/32
   
    for a in range (0,16 ):
        for b in range (0,7 ):
            col=(a+8)*size      
            lin=b*size
            cropped_image = temp[int(lin):int(lin+size), int(col): int(col +size)]
            
            
            if cropped_image[9,9] >= 90 : # pixel maiores que 127 viram 1
                A[b,a]=1
                
    B = A
    
    bits=""
    Str=""
    saida=""
    checksum=""
    saida_check=""
    for l in B:
        Str=Str+str(l).replace("[", "").replace("]", "").replace(".", "").replace(" ", "")
    
    if len(Str) >100:
        checksum=checksum+str(Str[18] )
        checksum=checksum+str(Str[21] )
        checksum=checksum+str(Str[24] )
        checksum=checksum+str(Str[29] )
        checksum=checksum+str(Str[66] )
        checksum=checksum+str(Str[69] )
        checksum=checksum+str(Str[72] )
        checksum=checksum+str(Str[77] )
       
        for i in range(0, len(Str) ):
            if i==0: #bit orienta��o
                continue
            elif i==14: #bit orienta��o
                continue
            elif i==15: #bit orienta��o
                continue
            elif i==16:
                continue
            elif i==18:
                continue
            elif i==21:
                continue    
            elif i==24:
                continue
            elif i==29:
                continue    
            elif i==30:
                continue
            elif i==31:
                continue
            elif i==32:
                continue
            elif i==48:
                continue
            elif i==64:
                continue
            elif i==66:
                continue
            elif i==69:
                continue
            elif i==72:
                continue
            elif i==77:
                continue    
            elif i==80:
                continue
            elif i==96:
                continue
            elif i==97:
                continue
            elif i==98:
                continue
            elif i==99:
                continue
            elif i==100:
                continue
            elif i==101:
                continue
            elif i==102:
                continue
            elif i==103:
                continue
            elif i==104:
                continue
            elif i==105:
                continue
            elif i==106:
                continue
            elif i==107:
                continue
            elif i==108:
                continue
            elif i==109:
                continue
            elif i==110:
                continue
            elif i==111:
                continue

            else:
                bits=bits+str(Str[i] )
    
   
    for i in range(0, len(bits)):
        if bits[i]=="1":
            saida+="0"
                
        elif bits[i]=="0":
            saida+="1"

    for i in range(0, len(checksum)):
        if checksum[i]=="1":
            saida_check+="0"
                
        elif checksum[i]=="0":
            saida_check+="1"
    
   
                  
    if lerCMToChecsum(saida)== saida_check:
        
        return saida
    else:
        return ""

def draw_boundBox(final_top_left,final_bottom_right,copy):
    
    image_out = cv2.rectangle(copy, (int(final_bottom_right[0]),(final_bottom_right[1])),(int(final_top_left[0]),int(final_top_left[1])) ,(255, 0, 0), 4)#azul
    
    return image_out


def Warp(final_top_left,final_top_right,final_bottom_left,final_bottom_right,img):
    
    input_pts = np.float32([[final_top_left],[final_top_right],[final_bottom_left],[final_bottom_right] ])

    output_pts = np.float32([[0,0],[832,0],[0,182],[832,182]]) # matriz homografica
   
   
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
    warped_img = cv2.warpPerspective(img,M,(832, 182))
   
    
    return warped_img 



def escreve_id(img, texto, posicao, cor=(0,0,220)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (int(posicao[0]),int(posicao[1])-5), fonte, 1.5, cor, 0, cv2.LINE_AA)

def rotate_detect90(VT,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        for i in range(0,len(VT)-2 ):
            if len(VT[i])==4:
                                     
                p1,p2,p3,p4=VT[i]
                
                p5,p6,p7,p8=VT[i+1]

                final_top_right=(p3[0],p3[1])#azul0
                final_top_left=(p6[0],p6[1])#vermelho
                final_bottom_right=(p4[0],p4[1])#rosa0
                final_bottom_left=(p5[0],p5[1])#verde
                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose1=(int(p1[0]),int(p1[1]))
                pose2=(int(p7[0]),int(p7[1]))
                     
                if isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                    
                else:
                    result=[]
                    
        return result

def rotate_detect45(VT,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        dados=[]
        for i in range(0,len(VT)-2 ):
            if len(VT[i])==4:
                p1,p2,p3,p4=VT[i]
                p5,p6,p7,p8=VT[i+1]

                final_top_left=(p1[0],p1[1])#azul
                final_bottom_left=(p4[0],p4[1])#verme
                final_top_right=(p6[0],p6[1])#rosa
                final_bottom_right=(p7[0],p7[1])#verde
                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose2=(int(p3[0]),int(p3[1]))
                pose1=(int(p5[0]),int(p5[1]))
                    
                    
                if isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                    
                else:
                    result=[]
                    
        return result

def rotate_detect_0(VT,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        dados=[]
        final_top_left=(0,0)#azul
        final_bottom_left=(0,0)#verme
        final_top_right=(0,0)#rosa
        final_bottom_right=(0,0)#verde
        
        if len(VT)==3:
            for i in range(0,len(VT)-2 ):
            
                p1,p2,p3,p4=VT[i+1]
                p5,p6,p7,p8=VT[i]
               

                final_top_left=(p1[0],p1[1])#verme
                final_bottom_left=(p4[0],p4[1])#azul
                final_top_right=(p6[0],p6[1])#rosa
                final_bottom_right=(p7[0],p7[1])#verde
                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                pose1=(int(p3[0]),int(p3[1]))
                pose2=(int(p5[0]),int(p5[1]))
                    
                    
                if isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
        elif len(VT)>3:
            for i in range(0,len(VT)-1 ):
            
                p1,p2,p3,p4=VT[i+1]
                p5,p6,p7,p8=VT[i]
               

                final_top_left=(p1[0],p1[1])#verme
                final_bottom_left=(p4[0],p4[1])#azul
                final_top_right=(p6[0],p6[1])#rosa
                final_bottom_right=(p7[0],p7[1])#verde
                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                pose1=(int(p3[0]),int(p3[1]))
                pose2=(int(p5[0]),int(p5[1]))
                    
                    
                if isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                     
        else:
                    result=[]
                    
        return result

def rotate_detect180(VT,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        for i in range(0,len(VT)-2 ):
            if len(VT[i])==4:
                p1,p2,p3,p4=VT[i]
                
                p5,p6,p7,p8=VT[i+1]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde
                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose1=(int(p3[0]),int(p3[1]))
                pose2=(int(p5[0]),int(p5[1]))
                    
                if isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                    
                else:
                    result=[]
                    
        return result                                

def detect_lowScale(img_detect,copy):
        sucesso=False
        id1=None
        pontos=[]
        V=[]
        V_info=[]
        p1=0
        p2=0
        D=[]
       
        img2=FiltragemInicial(img_detect)
        V1=Detectect(img2)
        
        V_info=rotate_detect45(V1,copy)
       
        if not V_info:
           
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                 
                id1=lerCMToDecimal(str(bits) )
                
                
                D.append(p1)
                D.append(p2)
                D.append(id1)
               
                sucesso=True
        
        V_info=rotate_detect90(V1,copy)
        
        if not V_info:
            
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                 
                id1=lerCMToDecimal(str(bits) )
              
                D.append(p1)
                D.append(p2)
                D.append(id1)
               
    
        V_info=rotate_detect180(V1,copy)
       
        if not V_info:
           
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                 
                id1=lerCMToDecimal(str(bits) )
                
                D.append(p1)
                D.append(p2)
                D.append(id1)
                
        V_info=rotate_detect_0(V1,copy)
        
        if not V_info:
           
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
               
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                   
                id1=lerCMToDecimal(str(bits) )
                
                D.append(p1)
                D.append(p2)
                D.append(id1)               
                
                       
        return D


contador=0

fim=0
cont_frame=0
inicio=process_time()
reloading=0.0000


url="C:\\tape-shaped-marker\\videos\\conditionA\\tape.mp4"    

cont=0
camera = cv2.VideoCapture(url)
while True:
    
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
   
    img=frame.copy()
    img2=frame.copy()
    cv2.imshow("cam", img2)
   
    tmp=True
    cont_frame=cont_frame+1
    D=detect_lowScale(img,img2)
    if D:
        for i in range(0,len(D),3):
            P1=D[i]
            P2=D[i+1]
            ID=D[i+2]
            
            cv2.rectangle(img2, (int(P2[0]),(P2[1])),(int(P1[0]),int(P1[1])) ,(255,0,0,255), 4)#azul
            escreve_id(img2, str(ID),P1 )
           
            cv2.imshow("cam", img2)
          
    else:
        txt=""
        binario=""
        sucesso=False
        linha2=[]
        cod_anterior=""
        tamanho=3
       
        Arr=getLine(img)
        
        (h, w) = img.shape[:2]
        rot = np.zeros((h, w, 4), dtype='uint8')
        linha=[]
        validos=[]
        p1=(0,0)
        p2=(0,0)
      
        if not Arr is None:
                    for lin in Arr: 
                       
                        linha=np.dot(lin['M'],lin['lines_list']).astype(np.int32)
                
                        rot= cv2.warpAffine(img, lin['M'], (w, h))

                       
                        encode=[]
                        x=0
                        if len(linha) > 0:
                            if rot.size>0:
                                x1=linha[0][0]-w
                                y1=linha[1][0]
                                x2=linha[0][1]+w
                                y2=linha[1][1]
                                cv2.line(rot,(linha[0][0],linha[1][0]),(x2,y2),(0,0,255,255),5)
                              

                                crop_img=rot[y1-20:y2+10,x1: x2]
                                
                                h1, w1 = crop_img.shape[:2]
                      
                                if h1 > 0 and w1 > 0:
                        
                                
                                    temp = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                               
                                    thresh = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 201, 5)
                                    thresh = cv2.bitwise_not(thresh)
                                   
                                    for l in range(0, h1-1):
                                
                                        lines= thresh[l]
                                
                                        for i in range(len(lines)):
                                            if lines[i] == 255:
                                                lines[i] = 1
                                
                                        string=array_to_string(lines)
                                        
                                        bar=bit_to_bars(string)
                                        
                                        data_string=""
                                        linha_limpa=remove_quieteZone(bar)
                                        sem_barras=  remove_barras(linha_limpa)                                  
                                        squadrado=""
                                        
                                        squadrado=remove_quadrado(sem_barras)
                                        bits=""
                                        bits=encode_bars(squadrado)
                                        if isTM(bits):
                                            if  len(bits)>tamanho: 
                                                tamanho=len(bits)
                                            if len(bits)==tamanho :
                                                encode.append(bits)
                                    
                            binario=valida_detect(encode)
                           
                        
                        if(len(binario) > 2):
                                
                                    txt=binario
                                    posicao=(w/2,h/2)
                                    cor=(0,0,220,255)
                                    fonte = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(rot, txt, (int(posicao[0]),int(posicao[1])-5), fonte, 3, cor, 0, cv2.LINE_AA)
                                    
                                  
                                   
                                    cv2.imshow("cam",rot)
                                   
                                    break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
       

cv2.waitKey()
cv2.destroyAllWindows()
