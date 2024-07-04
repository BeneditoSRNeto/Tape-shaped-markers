from asyncio.windows_events import NULL
from operator import is_not
import numpy as np
import cv2
import numpy as np
from numpy import array
from hashlib import blake2b

import math
import matplotlib.pyplot as plt
import json
import os.path
from time import process_time





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
        if i['idcm'] == identificador:
            isCM=True
           
    return isCM

def isTM(identificador):
    
    isTM=False
    data=loadJson()
    for i in data:
        if i['idtm']== str(identificador):
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


 #fun��o que converte a cadeia de bits para modulos de bits

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


#fun��o que codificas as cadeias de bits em valores unitarios 111111100000001111111 =101(simplifica��o)
def encode_bars(array_bar):
    
    array=array_bar
    s = ""
    for value in array:
        if value[0]=="1":
            s = s + "0"
        if value[0]=="0":
            s = s + "1"
      
    return s

#fun��o que converte as linhas da imahgem em array de bits 0 e 1 
def array_to_string(array1):
    array=array1
    s = ""
    
  
    for value in array:
        
        s = s + str(value)
        
  
    return s

#fun��o que valida as os codigos que tem maior incidencia nas linhas
def valida_detect(array):
   
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
                    #print("contador de repeticao : ",contador)            
                if tp<contador:
                    tp=contador
                    bi_t=binario
        if abs(tp/len(array))>=0.7:
              saida=bi_t
   
    return saida 
#escreve tento na imagemde detec��o maior
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
    #cv2.imshow("warp temp 1", obj2)
    return obj2





#fun��o que faz a detec��o dos contornos quadrados que delimitam a regi�o de encoder    
def Detectect(img):
    V=[]
    sucesso2=False 
    img_origem=img 
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
            

            if aspect >= 0.6 and  approx.shape[0]==4 :#and cv2.arcLength(contours[i],True) :#and hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] == -1:
                j=hierarchy[0][i][2]
                if hierarchy[0][j][2] >= 0:
                   k=hierarchy[0][j][2]
                   if hierarchy[0][k][3] >= 0:
                        cnt=contours[i]
                        rect = cv2.minAreaRect(cnt)
                        pontos = cv2.boxPoints(rect)
                        box = np.intp(pontos)
                    
                       
                                                
                        retangulo = cv2.minAreaRect(contours[i])
                                                
                        box = cv2.boxPoints(retangulo)
                                                           
                        V.append(box)
                          
                
                                                     
    
    return V

# fun��o que realiza a extra��o da informa��o da regi�o do encoder em escala menor 
    
def Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame):
    saida2=""
    B=[]
    tag=False
    warped_img=Warp(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
    #######
    #warped_img=Warp(final_bottom_left,final_bottom_right,final_top_left,final_top_right,frame)

    temp=FiltragemInicial(warped_img.copy())
    temp = cv2.equalizeHist(temp)
 
    suave = cv2.GaussianBlur(temp, (33, 33), 0) # aplica blur

    #suave = cv2.equalizeHist(suave)
    ret,th = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #obj_bin= cv2.adaptiveThreshold(temp, 200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imagem_cinza=temp
    
    A=np.zeros((7,16))
    size=832/32

    for a in range (0,16 ):
        for b in range (0,7 ):
            col=(a+8)*size      
            lin=b*size
            cropped_image = th[int(lin):int(lin+size), int(col): int(col +size)]
            
            
            if cropped_image[9,9] >= 180 : # pixel maiores que 127 viram 1
                A[b,a]=1
                cv2.rectangle(temp, (int(col),int(lin)),(  int(col+ size),int(lin+size)) ,(0, 0, 0), 1)                                        

   
   
    B = A
    #print(B)
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
        #print("check B ",checksum)
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



# fun��o que faz a transforma��o da perspectiva da imagem da regi�o de encoder ficar plana para extra��o das caracteristicas     

def Warp(final_top_left,final_top_right,final_bottom_left,final_bottom_right,img):
    
    input_pts = np.float32([[final_top_left],[final_top_right],[final_bottom_left],[final_bottom_right] ])
    

    output_pts = np.float32([[0,0],[832,0],[0,182],[832,182]]) # matriz homografica
   
   
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
    warped_img = cv2.warpPerspective(img,M,(832, 182))

    
    return warped_img 

def WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,img):
    
    input_pts = np.float32([[final_top_left],[final_top_right],[final_bottom_left],[final_bottom_right] ])
    
    output_pts = np.float32([[0,0],[1496,0],[0,60],[1496,60]]) # matriz homografica
   
   
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
    # Apply the perspective transformation to the image
    #warped_img = cv2.warpPerspective(img,M,(1000, 219))
    warped_img = cv2.warpPerspective(img,M,(1496, 70))

    
    return warped_img 

#escreve tento na imagem de detec��o menor
def escreve_id(img, texto, posicao, cor=(0,0,255)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (int(posicao[0]),int(posicao[1])-5), fonte, 2.5, cor, 0, cv2.LINE_AA)

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
                    p4,p1,p2,p3=VT[i]
                
                    p8,p5,p6,p7=VT[i+2]
                    
                    final_bottom_left=(p3[0],p3[1])#azul0
                    final_bottom_right=(p6[0],p6[1])#vermelho
                    final_top_left=(p4[0],p4[1])#rosa0
                    final_top_right=(p5[0],p5[1])#verde


                    bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    pose1=(int(p4[0]),int(p4[1]))
                    pose2=(int(p7[0]),int(p7[1]))
                    
                        
                    if isCM(str(bits)):
                        result.append(pose1)
                        result.append(pose2)
                        result.append(bits)
                        
                    
                    else:
                        p4,p1,p2,p3=VT[i+2]
                
                        p8,p5,p6,p7=VT[i]
                     


                        final_bottom_right=(p3[0],p3[1])#azul0q
                        final_bottom_left=(p7[0],p7[1])#vermelho
                        final_top_right=(p2[0],p2[1])#rosa0q
                        final_top_left=(p8[0],p8[1])#verdeq
                   

                        bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                        pose1=(int(p8[0]),int(p8[1]))
                        pose2=(int(p3[0]),int(p3[1]))
                           
                        if isCM(str(bits)):
                            result.append(pose1)
                            result.append(pose2)
                            result.append(bits)
                           

                        else: #novo caso
                            p1,p2,p3,p4=VT[i+1]
                
                            p5,p6,p7,p8=VT[i]
                           

                            final_top_left=(p3[0],p3[1])#azul0
                            final_top_right=(p7[0],p7[1])#vermelho
                            final_bottom_left=(p4[0],p4[1])#rosa0
                            final_bottom_right=(p8[0],p8[1])#verde
                            
                            bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                            pose1=(int(p3[0]),int(p3[1]))
                            pose2=(int(p5[0]),int(p5[1]))
                                
                            if isCM(str(bits)):
                                result.append(pose1)
                                result.append(pose2)
                                result.append(bits)
                               
                            else:
                                p1,p2,p3,p4=VT[i]
                    
                                p5,p6,p7,p8=VT[i+2]
                            

                                final_top_left=(p2[0],p2[1])#azul0
                                final_top_right=(p7[0],p7[1])#vermelho
                                final_bottom_left=(p1[0],p1[1])#rosa0
                                final_bottom_right=(p8[0],p8[1])#verde
                                
                                bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                                pose1=(int(p3[0]),int(p3[1]))
                                pose2=(int(p5[0]),int(p5[1]))
                                
                                if isCM(str(bits)):
                                    result.append(pose1)
                                    result.append(pose2)
                                    result.append(bits)
                                   
                                else:
                                    p1,p2,p3,p4=VT[i]
                
                                    p5,p6,p7,p8=VT[i+1]

                                    final_top_right=(p3[0],p3[1])#azul0
                                    final_top_left=(p5[0],p5[1])#vermelho
                                    final_bottom_right=(p4[0],p4[1])#rosa0
                                    final_bottom_left=(p8[0],p8[1])#verde
                                 
                                    bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                                    pose1=(int(p1[0]),int(p1[1]))
                                    pose2=(int(p6[0]),int(p6[1]))
                                       
                                    if isCM(str(bits)):
                                        result.append(pose1)
                                        result.append(pose2)
                                        result.append(bits)
                                      
                                    else:
                                        p1,p2,p3,p4=VT[i]
                
                                        p5,p6,p7,p8=VT[i+1]
                                       
                                        final_bottom_left=(p3[0],p3[1])#azul0
                                        final_bottom_right=(p6[0],p6[1])#vermelho
                                        final_top_left=(p4[0],p4[1])#rosa0
                                        final_top_right=(p5[0],p5[1])#verde
                                        bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                                        pose1=(int(p1[0]),int(p1[1]))
                                        pose2=(int(p7[0]),int(p7[1]))
                                           
                                        if isCM(str(bits)):
                                            result.append(pose1)
                                            result.append(pose2)
                                            result.append(bits)
                                           
                                
                    
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
                   
                    p1,p2,p3,p4=VT[i]
                    p5,p6,p7,p8=VT[i+1]

                    final_top_left=(p1[0],p1[1])#azul
                    final_bottom_left=(p4[0],p4[1])#verme
                    final_top_right=(p5[0],p5[1])#rosa
                    final_bottom_right=(p6[0],p6[1])#verde
                    bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    
                  
                    pose2=(int(p3[0]),int(p3[1]))
                    pose1=(int(p5[0]),int(p5[1]))
                        
                        
                    if isCM(str(bits)):
                        result.append(pose1)
                        result.append(pose2)
                        result.append(bits)
                      
                    else:
                      
                        
                        bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    
                        pose2=(int(p1[0]),int(p1[1]))
                        pose1=(int(p7[0]),int(p7[1]))
                            
                            
                        if isCM(str(bits)):
                            result.append(pose1)
                            result.append(pose2)
                            result.append(bits)
                          
                        else:
                         
                            p1,p2,p3,p4=VT[i+1]
                
                            p5,p6,p7,p8=VT[i+2]
                            final_bottom_left=(p3[0],p3[1])#rosa0
                            final_top_left=(p4[0],p4[1])#azul0
                            final_top_right=(p5[0],p5[1])#vermelho
                            final_bottom_right=(p6[0],p6[1])#verde
                            
                            
                            bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                            bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                            
                       
                            pose2=(int(p1[0]),int(p1[1]))
                            pose1=(int(p7[0]),int(p7[1]))
                                
                                
                            if isCM(str(bits)):
                                result.append(pose1)
                                result.append(pose2)
                                result.append(bits)
                              
                        
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
                   
                else:
                    p1,p2,p3,p4=VT[i+1]
                    p5,p6,p7,p8=VT[i+2]
                    final_top_right=(p2[0],p2[1])#verme
                    final_bottom_right=(p3[0],p3[1])#azul
                    final_top_left=(p5[0],p5[1])#rosa
                    final_bottom_left=(p8[0],p8[1])#verde
                    bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    if isCM(str(bits)):
                        pose1=(int(p1[0]),int(p1[1]))
                        pose2=(int(p7[0]),int(p7[1]))
                        result.append(pose1)
                        result.append(pose2)
                        result.append(bits)
                        
                    else:
                        p1,p2,p3,p4=VT[i]
                        p5,p6,p7,p8=VT[i+2]
                        final_top_right=(p2[0],p2[1])#verme
                        final_bottom_right=(p3[0],p3[1])#azul
                        final_top_left=(p5[0],p5[1])#rosa
                        final_bottom_left=(p8[0],p8[1])#verde
                        bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                        if isCM(str(bits)):
                            pose1=(int(p1[0]),int(p1[1]))
                            pose2=(int(p7[0]),int(p7[1]))
                            result.append(pose1)
                            result.append(pose2)
                            result.append(bits)
                          
                            
        elif len(VT)>3:
            
            for i in range(0,len(VT)-1 ):
                    bits=NULL
            
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
 
                    
                        
        elif len(VT)==2:
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
                    p1,p2,p3,p4=VT[i+1]
                    p5,p6,p7,p8=VT[i]
                    final_top_right=(p2[0],p2[1])#verme
                    final_bottom_right=(p3[0],p3[1])#azul
                    final_top_left=(p5[0],p5[1])#rosa
                    final_bottom_left=(p8[0],p8[1])#verde
                    bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    if isCM(str(bits)):
                        pose1=(int(p1[0]),int(p1[1]))
                        pose2=(int(p7[0]),int(p7[1]))
                        result.append(pose1)
                        result.append(pose2)
                        result.append(bits)
                      
                    else:
                      
                        p1,p2,p3,p4=VT[i]
                        p5,p6,p7,p8=VT[i+1]
                    
                   

                        final_bottom_left=(p3[0],p3[1])#verme
                        final_top_left=(p4[0],p4[1])#azul
                        final_bottom_right=(p6[0],p6[1])#rosa
                        final_top_right=(p5[0],p5[1])#verde
                        bits=Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                       
                        pose1=(int(p1[0]),int(p1[1]))
                        pose2=(int(p7[0]),int(p7[1]))
                            
                            
                        if isCM(str(bits)):
                            result.append(pose1)
                            result.append(pose2)
                            result.append(bits)
                            
                            
                    
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
#fun��o de detec��o em escala menor
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


def Detectect_TM(img):
    V=[]
    maior=0
    temp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.equalizeHist(temp)
    obj= cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 4)
       
        
    contours, hierarchy = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    frame2=frame.copy()
    for i in range(0,len(contours) ):
            rect = cv2.minAreaRect(contours[i])
            epsilon = 0.1*  cv2.arcLength(contours[i],True)
            approx = cv2.approxPolyDP(contours[i],epsilon,True)
            if rect[1][1] > 0:
                aspect=rect[1][0]/rect[1][1]
                

               
                if  hierarchy[0][i][2] == -1 and hierarchy[0][i][3] == -1 and cv2.arcLength(contours[i],True)>= 20:
                    if (aspect>=0.14 and aspect < 0.18) or aspect==5.5 or aspect==6.666666666666667 or aspect==6.333333333333333 or aspect==6.75 or aspect==6.5 or aspect==6.75 or aspect==7.0 or aspect==7.25 or aspect==7.5 or aspect==6.0 or aspect==7.33 or aspect==8.5 or aspect==7.666666666666667 or aspect==9.0 or aspect==10.0 or aspect==11.0 or aspect==12.0 or aspect==13.0 or aspect==14.0  or aspect==15.0  or aspect==9.5  or aspect==8.75  :
                   
                            cnt=contours[i]
                            rect = cv2.minAreaRect(cnt)
                            pontos = cv2.boxPoints(rect)
                            box = np.intp(pontos)
                            
                            retangulo = cv2.minAreaRect(contours[i])
                                                
                            box = cv2.boxPoints(retangulo)
                            
                            V.append(box)
                             
   
    return V



def LerTM(barras):
    code=[]
    if barras is NULL:
        return []
    else:
         for i in range(0,len(barras)-12,1): 
            if abs(len(barras[i])-len(barras[i+4]))< 5 and abs(len(barras[i+1])-len(barras[i+3]))< 5 and abs(len(barras[i+2])-len(barras[i]))> 10 and len(barras[i+7]) > len(barras[i+6]) and len(barras[i+7])> 80 and  abs(len(barras[i+8])-len(barras[i+12]))< 5 and abs(len(barras[i+9])-len(barras[i+11]))< 5  and len(barras[i+10]) > len(barras[i+8]) : #bit 0
                
                code.append(1)
            elif abs(len(barras[i])-len(barras[i+4]))< 5 and abs(len(barras[i+1])-len(barras[i+3]))< 5 and abs(len(barras[i+2])-len(barras[i]))> 10 and len(barras[i+7]) < len(barras[i+6]) and len(barras[i+6])> 80 and abs(len(barras[i+8])-len(barras[i+12]))< 5 and abs(len(barras[i+9])-len(barras[i+11]))< 5  and len(barras[i+10]) > len(barras[i+8]) : #bit 1
               
                code.append(0)
    return code





def Extract_TM(img):
    saida2=""
    B=[]
    binario=NULL
    tag=False
    
    temp=FiltragemInicial(img.copy())
    temp = cv2.equalizeHist(temp)
    suave = cv2.GaussianBlur(temp, (11, 11), 0) # aplica blur
    
    ret,th = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(th)
  
    h1, w1 = thresh.shape[:2]
  
    encode=[]
    for l in range(0, h1-1):
                                
        lines= thresh[l]
        for i in range(len(lines)):
            if lines[i] == 255:
                lines[i] = 1
        string=array_to_string(lines)
        
        bar=bit_to_bars(string)
        
        bits=LerTM(bar)
        caracter=""
        for x in range(0,len(bits)):
            caracter+=str(bits[x])
            
        if isTM(caracter) and len(caracter)>2:
             encode.append(caracter)
   
    binario=valida_detect(encode)
   
    return binario


def Decode_TM(V):
    codigo=""
    pontos=[]
    print("qtd :",len(V))
    for i in range(0,len(V)-2 ):
            if len(V)==12:
                p3,p4,p1,p2=V[i]
                
                p7,p8,p5,p6=V[len(V)-4]

                final_bottom_right=(p1[0]+5,p1[1])#azulq
                final_top_right=(p4[0]+5,p4[1])#verme
                final_bottom_left=(p6[0]-5,p6[1])#rosa
                final_top_left =(p7[0]-5,p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
            
                codigo=Extract_TM(warped_img)
                
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2=V[len(V)-2]
                
                    p5,p6,p7,p8=V[i]

                    final_bottom_left=(p1[0]+5,p1[1])#azulq
                    final_top_left=(p4[0]+5,p4[1])#verme
                    final_bottom_right=(p5[0]-5,p5[1])#rosa
                    final_top_right =(p7[0]-5,p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                  
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                    
                        pontos.append(final_top_right)
                        pontos.append(final_bottom_left)
                        pontos.append(codigo)
                        break
                    else:
                        p3,p4,p1,p2=V[i+2]
                
                        p5,p6,p7,p8=V[len(V)-2]

                        final_bottom_left=(p1[0]+5,p1[1])#azulq
                        final_top_left=(p4[0]+5,p4[1])#verme
                        final_bottom_right=(p5[0]-5,p5[1])#rosa
                        final_top_right =(p7[0]-5,p7[1])#verde

                        warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                      
                        codigo=Extract_TM(warped_img)
                        if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break
                    
            
            elif len(V)==7:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-3]

                final_bottom_right=(p1[0]+5,p1[1])#azul
                final_top_right=(p4[0]+5,p4[1])#verme
                final_bottom_left=(p6[0]-5,p6[1])#rosa
                final_top_left =(p7[0]-5,p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
              
                codigo =Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                if codigo == "":
                    p3,p4,p1,p2,=V[i+1]
                
                    p7,p8,p5,p6=V[len(V)-2]

                    final_bottom_right=(p1[0],p1[1])#azul
                    final_top_right=(p4[0],p4[1])#verme
                    final_bottom_left=(p6[0],p6[1])#rosa
                    final_top_left =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                  
                    codigo=Extract_TM(warped_img) 
                    if codigo!= "":
                    
                        pontos.append(final_top_right)
                        pontos.append(final_bottom_left)
                        pontos.append(codigo)
                        break 
              
            elif len(V)==8:
                p3,p4,p1,p2=V[i]
                
                p7,p8,p5,p6=V[len(V)-1]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img)
              
                
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
            
            elif len(V)==15:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-3]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img)

                
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2=V[i]
                
                    p5,p6,p7,p8=V[len(V)-5]

                    final_bottom_left=(p1[0],p1[1])#azulq
                    final_top_left=(p4[0],p4[1])#verme
                    final_bottom_right=(p5[0],p5[1])#rosa
                    final_top_right =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                                           
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break
               
            elif len(V)==6:
                p3,p4,p1,p2=V[len(V)-4]
                
                p7,p8,p5,p6=V[i+2]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                codigo=Extract_TM(warped_img)
                
                codigo =Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2,=V[i]
                
                    p7,p8,p5,p6=V[len(V)-4]

                    final_bottom_right=(p1[0],p1[1])#azul
                    final_top_right=(p4[0],p4[1])#verme
                    final_bottom_left=(p6[0],p6[1])#rosa
                    final_top_left =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                   
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                    
                        pontos.append(final_top_right)
                        pontos.append(final_bottom_left)
                        pontos.append(codigo)
                        break 
                    else:
                        p3,p4,p1,p2=V[i]
                    
                        p5,p6,p7,p8=V[len(V)-2]

                        final_bottom_left=(p1[0],p1[1])#azulq
                        final_top_left=(p4[0],p4[1])#verme
                        final_bottom_right=(p5[0],p5[1])#rosa
                        final_top_right =(p7[0],p7[1])#verde

                        warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                      
                            
                        codigo=Extract_TM(warped_img)
                        if codigo!= "":
                            
                                pontos.append(final_top_right)
                                pontos.append(final_bottom_left)
                                pontos.append(codigo)
                                break
               
            elif len(V)==9:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-4]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
              
                codigo=Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                        p1,p2,p3,p4=V[i+2]
                    
                        p5,p6,p7,p8=V[len(V)-2]

                        final_bottom_right=(p1[0],p1[1])#azulq
                        final_top_right=(p2[0],p2[1])#verme
                        final_bottom_left=(p5[0],p5[1])#rosa
                        final_top_left =(p7[0],p7[1])#verde

                        warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                      
                        codigo=Extract_TM(warped_img)
                        if codigo!= "":
                            
                                pontos.append(final_top_right)
                                pontos.append(final_bottom_left)
                                pontos.append(codigo)
                                break    
            elif len(V)==11:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-4]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break    
                else: 
                    p3,p4,p1,p2=V[len(V)-8]
                
                    p7,p8,p5,p6=V[i+1]

                    final_bottom_right=(p1[0],p1[1])#azul
                    final_top_right=(p4[0],p4[1])#verme
                    final_bottom_left=(p6[0],p6[1])#rosa
                    final_top_left =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    
                    codigo=Extract_TM(warped_img) 
                    if codigo!= "":
                    
                        pontos.append(final_top_right)
                        pontos.append(final_bottom_left)
                        pontos.append(codigo)
                        break 
                    else:
                        p3,p4,p1,p2=V[i+2]
                
                        p5,p6,p7,p8=V[len(V)-2]

                        final_bottom_left=(p1[0]+5,p1[1])#azulq
                        final_top_left=(p4[0]+5,p4[1])#verme
                        final_bottom_right=(p5[0]-5,p5[1])#rosa
                        final_top_right =(p7[0]-5,p7[1])#verde

                        warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                      
                        
                        codigo=Extract_TM(warped_img)
                        if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break
            elif len(V)==10:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-4]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img)  
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2=V[i+2]
                
                    p5,p6,p7,p8=V[len(V)-2]

                    final_bottom_left=(p1[0]+5,p1[1])#azulq
                    final_top_left=(p4[0]+5,p4[1])#verme
                    final_bottom_right=(p5[0]-5,p5[1])#rosa
                    final_top_right =(p7[0]-5,p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                    
                    
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                    
                        pontos.append(final_top_right)
                        pontos.append(final_bottom_left)
                        pontos.append(codigo)
                        break

            elif len(V)==16:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-7]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
               
                codigo=Extract_TM(warped_img) 
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2=V[i]
                
                    p5,p6,p7,p8=V[len(V)-6]

                    final_bottom_left=(p1[0],p1[1])#azulq
                    final_top_left=(p4[0],p4[1])#verme
                    final_bottom_right=(p5[0],p5[1])#rosa
                    final_top_right =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                   
                        
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break 
            elif len(V)==14:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-7]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img) 
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
                else:
                    p3,p4,p1,p2=V[i]
                
                    p5,p6,p7,p8=V[len(V)-5]

                    final_bottom_left=(p1[0],p1[1])#azulq
                    final_top_left=(p4[0],p4[1])#verme
                    final_bottom_right=(p5[0],p5[1])#rosa
                    final_top_right =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                  
                        
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break
            elif len(V)==13:
                p3,p4,p1,p2=V[i+1]
                
                p7,p8,p5,p6=V[len(V)-4]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#vermeq
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
              
                codigo=Extract_TM(warped_img) 
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
               
            elif len(V)==5:
                p3,p4,p1,p2=V[i]
                
                p7,p8,p5,p6=V[len(V)-1]

                final_bottom_right=(p1[0]+5,p1[1])#azul
                final_top_right=(p4[0]+5,p4[1])#verme
                final_bottom_left=(p6[0]-5,p6[1])#rosa
                final_top_left =(p7[0]-5,p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
              
                codigo=Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
            
            elif len(V)==3:
                p3,p4,p1,p2=V[i]
                
                p7,p8,p5,p6=V[len(V)-1]

                final_bottom_right=(p1[0]+5,p1[1])#azul
                final_top_right=(p4[0]+5,p4[1])#verme
                final_bottom_left=(p6[0]-5,p6[1])#rosa
                final_top_left =(p7[0]-5,p7[1])#verde

                warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                
                codigo=Extract_TM(warped_img)
                if codigo!= "":
                    
                    pontos.append(final_top_right)
                    pontos.append(final_bottom_left)
                    pontos.append(codigo)
                    break
            elif len(V)==25:
                    p3,p4,p1,p2=V[i]
                
                    p5,p6,p7,p8=V[len(V)-5]

                    final_bottom_left=(p1[0],p1[1])#azulq
                    final_top_left=(p4[0],p4[1])#verme
                    final_bottom_right=(p5[0],p5[1])#rosa
                    final_top_right =(p7[0],p7[1])#verde

                    warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                   
                        
                    codigo=Extract_TM(warped_img)
                    if codigo!= "":
                        
                            pontos.append(final_top_right)
                            pontos.append(final_bottom_left)
                            pontos.append(codigo)
                            break
            elif len(V)==35 or len(V)==21 or len(V)==33:
                   
                            p1,p2,p3,p4=V[i+2]
                        
                            p5,p6,p7,p8=V[len(V)-2]

                            final_top_left=(p2[0],p2[1])#azulq
                            final_bottom_left=(p3[0],p3[1])#verme
                            final_top_right=(p7[0],p7[1])#rosa
                            final_bottom_right =(p8[0],p8[1])#verde

                            warped_img=WarpTM(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                         
                                
                            codigo=Extract_TM(warped_img)
                            if codigo!= "":
                                
                                    pontos.append(final_top_right)
                                    pontos.append(final_bottom_left)
                                    pontos.append(codigo)
                                    break
           
    return pontos

url="C:\\tape-shaped-marker\\videos\\conditionA\\tape.mp4"

 
camera = cv2.VideoCapture(url)
while True:
    
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
    
    
    img=frame.copy()
    img2=frame.copy()
    cv2.imshow("cam", img2)

   
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
        V=Detectect_TM(img)
        
        TM=Decode_TM(V)
        if TM:
            
            P1=TM[0]
            P2=TM[1]
            ID=TM[2]
        
            cv2.rectangle(img2, (int(P2[0]),int(P2[1])),(int(P1[0]),int(P1[1])) ,(255,0,0,255), 4)#azul
            escreve_id(img2, str(ID),P2 )
                
            cv2.imshow("cam", img2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
        
cv2.destroyAllWindows()
