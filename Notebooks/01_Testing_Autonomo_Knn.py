#Version Total 1
import os
from tkinter import filedialog
import cv2
import numpy as np
import serial,time
from threading import Thread
import math
import joblib

# Carga el modelo del archivo modelo_knn.joblib
modelo = joblib.load('./modelos/modelo_autonomo_knn.joblib')

"""Pesos de la Red Neuronal"""
#pesos = np.genfromtxt('Recursos/pesos_ANN.csv', delimiter=',') #Cambiar ruta a la de la ubicación donde se encuentran los pesos

kernel_dit= np.ones((10,10),np.uint8)
kernel_ero=np.ones((15,15),np.uint8)


'''Intervalos para correción y mantenerse en el carril
'''
'''Centrado en Carril'''
centro_izq=90-5
centro_der=90+5

rigth_inicio =centro_der+10
left_inicio = 69

camara = cv2.VideoCapture(0)
frame_index = 0

"""SE COMENTAN LINEAS """
#def angulo_2_arduino(pdi):
#    cadena1=str(pdi)
#    cadena2='\n'
#    comando=cadena1+cadena2
#    salida=comando.encode('utf-8')
#    #arduino.write(salida)
#    return arduino.write(salida)
#
#
#def angulo_3_arduino(pdi):
#    pdi_final=round(pdi)
#    cadena1=str(pdi_final)
#    cadena2='\*'
#    cadena3='\n'
#    comando=cadena1+cadena2+cadena3
#    salida=comando.encode('utf-8')
#    arduino.write(salida)

def ajustarROI(im):
    '''Calculamos el porcentaje de reducción de acuerdo a la resolución deseada'''
    roi = [[0,20],[0, 130],[640, 20],[640,130]]

    pts1 = np.float32(roi)
    pts2 = np.float32([[0,0], [0,180], [180,0], [180,180]])

    '''Se aplican transformadas de perspectiva'''
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_paj = cv2.warpPerspective(im, M, (180,180))
    return img_paj

def deteccionContorno(img):


    img_paj = ajustarROI(img)
    #cv2.imshow('Vista pajaro', img_paj)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''Ajustamos las coordenadas para la construcción del área para vista de pájaro'''
    # Our operations on the frame come here
    gray = cv2.cvtColor(img_paj, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray,(3,3),0) 
    
    t, dst = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)  
    conts, cont = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # la variable _
    
    # Si tenemos algún contorno
    
    if len(conts) > 0:
        # Buscamos el que tenga más área
        c = max(conts, key=cv2.contourArea) 
        M = cv2.moments(c)
        cx=0
        cy=0
        #en el if  antes estaba si M['m00'] es diferente de  0, lo acomode a si M['m00'] es igual 0
        # al valor de M['m00'] le asignamos 1, pero daba valores cercanos a 100, entonces dividí el valor resultante entre 10
        if M['m00'] != 0:
           
            #cambiar la condición hizo que nos diera mejores resultados al obtener el PDI
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dif = (cx-(cy*-1)-100)
           #cv2.circle(gray, (cx, cy), 1, (0, 0, 255), -1)
        left_most = tuple(c[c[:, :, 0].argmin()][0])
        #cv2.circle(gray, left_most,3 , (0, 0, 255), -1)
        right_most = tuple(c[c[:, :, 0].argmax()][0])
        #cv2.circle(gray, right_most, 3, (0, 0, 255), -1)
        #cv2.drawContours(gray, [c], -1, (0, 0, 255), 2, cv2.LINE_AA)  
        #d_izq = -1 * math.sqrt(pow((left_most[0]-cx),2)+pow((left_most[1]-cy),2))
        #d_der = math.sqrt(pow((right_most[0]-cx),2)+pow((right_most[1]-cy),2))
        #d_total = (d_izq + d_der)
        

        
        

        '''Definimos los intervalos de distancias para calcular los grados de giro'''

        
        return [gray,dif] 

def preprocesamientoErosion(frame):
    cv2.imshow('frame1',frame)
    crop_img = np.array(frame[20:200, 0:640])
    # Our operations on the frame come here
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    imagem = (255-gray)
    #cv2.imshow('Invertida',imagem)
    ret,thresh1 = cv2.threshold(imagem,150,255,cv2.THRESH_BINARY)
    #cv2.imshow('Binarizada',thresh1)
    edges = cv2.Canny(thresh1,200,245)
    #cv2.imshow('Bordes',edges)

    erosion = cv2.dilate(edges,kernel_dit,iterations = 5)
    imagen_erocion = cv2.erode(erosion, kernel_ero, iterations=1)
    #cv2.imshow("imagen entrada NN",imagen_erocion)
    #np.savetxt("prueba_1.txt",imagen_erocion, delimiter = ',')
    renglon = np.array(np.arange(0,180, 10), dtype=np.intp)
    columna = np.array(np.arange(0,640, 10), dtype=np.intp)
   
    pixeles = np.asarray(imagen_erocion)
    pixeles = pixeles[renglon[:, np.newaxis], columna]
   # cv2.imshow("Inter", pixeles)
    pixeles = pixeles.flatten()
    return pixeles
    
def redANN():
    dirname = filedialog.askdirectory(title="Abrir archivo",
                                        initialdir="C:Documents/")
    inicio = time.time()
   
    for nombre_directorio,dirs, ficheros in os.walk(dirname):
        posicion=os.path.realpath(nombre_directorio).split('\\')[-1]
        #print(posicion)
        directorio=os.path.realpath(nombre_directorio)
        aux_dir = directorio.find(posicion)
        directorio = directorio[:aux_dir-1]
        #print(directorio)
        for nom_fichero in ficheros:
            #Funciones aplicadas a imagenes
            #Leer Directorio
            com_dir = os.path.relpath(nombre_directorio+'/'+nom_fichero)
            frame = cv2.imread(os.path.realpath(com_dir)) #Se lee la imagen
            k = cv2.waitKey(1) & 0xFF
                
            crop_img = np.array(frame[20:200, 0:640])
            pixeles = np.array([preprocesamientoErosion(frame)])
            prediccion = modelo.predict(pixeles)[0]
            #print(prediccion)
            #print("LEFT,STRI,RIGHT")
            #print('{:.2f},{:.2f},{:.2f}'.format(LEFT,STRI,RIGHT))
            crop_img_contorns=deteccionContorno(frame)
            #print('{:.2f}'.format(crop_img_contorns[1]))
            pdi=(crop_img_contorns[1])
            if pdi >= -90:
                pdi = 180-pdi
            elif pdi < 90:
                pdi = pdi+90
            #type(pdi)

            if prediccion == 'RIGHT':
                print("DERECHA")
                #/*PDI>170 AND PDI<=200*/ 
                #print("Imagen Clasificada como RIGHT\n")
                #if pdi > 170:
                #    print("DER GIRO A DERECHA: \n")
                #    angulo_3_arduino(180)
                #
                #if pdi <170 and pdi >centro_izq:
                #    print("DER SE MANTIENE EN DER PDI>{:.2f},<{:.2F},\n".format(centro_izq,pdi))
                #    angulo_3_arduino(80.0)
                #
                #else:
                #    print("DER SE MANTIENE EN DER PDI<{:.2f},<{:.2F},\n".format(centro_izq,pdi))
                #    angulo_3_arduino(80.0


            if prediccion == 'LEFT':
                print("IZQUIERDA")
                #if pdi<=20:
                #    print("LEFT GIRO A IZQUIERDA <20, PDI= {:.2f}, \n".format(centro_der,pdi))
                #    #print("LEFT GIRO A IZQUIERDA")
                #    angulo_3_arduino(10)
                #if pdi >20 and pdi < centro_der:
                #    print("LEFT SE MANTIENE EN IZQ PDI>{:.2f},<{:.2F},\n".format(pdi,centro_izq))
                #    angulo_3_arduino(120.0)
                #else:
                #    print("LEFT SE MANTIENE EN IZQ PDI>{:.2f},<{:.2F},\n".format(pdi,centro_izq))
                #    angulo_3_arduino(120.0
            if prediccion == 'STR':
                print("CENTRO")
                #if pdi>=centro_izq and pdi<=centro_der:
                #
                #    print("CENTRO: El PDI está centrado en {:.2f}-{:.2f} PDI= {:.2f}, \n".format(centro_izq,centro_der,pdi))
                #    angulo_3_arduino(pdi)
                #if pdi>centro_der and pdi<=170:
                #    print("CENTRO: El PDI está más a la derecha, PDI > {:.2f}, \n".format(centro_der))
                #    angulo_3_arduino(pdi)
                #
                #if pdi<centro_izq and pdi >=21:
                #    print("CENTRO: El PDI está más a la izquierda, PDI < {:.2f}, PDI= {:.2f}, \n".format(centro_izq,pdi))
                        #    angulo_3_arduino(pdi)
#
                #def capturadora(): 
                #
                #    # El parametro img, debe de ser una imagen 
                #    frame_name = "Frame{}.png".format(frame_index)
                #    cv2.imwrite(frame_name, imagen)



if __name__ == '__main__':
    inicio = time.time() #Incio de tiempo ejecución
    #capturar = Thread(target=capturadora, args=()) #Se pasa como parametros n y m, que definen la resolución en la que se realizará el preprocesamiento de las imágenes
    redA = Thread(target=redANN, args=())  
    #capturar.start()
    #capturar.join()
    redA.start()
    redA.join()
    fin = time.time()
    #arduino = serial.Serial("/dev/ttyACM0", 9600, timeout=None)
    #arduino = serial.Serial("/dev/ttyUSB0", 115200, timeout=None)
    #score = 95
    #
    #auxDistancia = 90
    #while (camara.isOpened()):
    #    ret, imagen = camara.read()
    #    if ret == True:
    #        inicio = time.time() #Incio de tiempo ejecución
    #        capturar = Thread(target=capturadora, args=()) #Se pasa como parametros n y m, que definen la resolución en la que se realizará el preprocesamiento de las imágenes
    #        redA = Thread(target=redANN, args=())  
    #        capturar.start()
    #        capturar.join()
    #        redA.start()
    #        redA.join()
    #        fin = time.time() #Fin de tiempo de ejecución
    #        #print(fin-inicio) #Imprimimos tiempo total
    #        if cv2.waitKey(1) & 0xFF == ord('ñ'):
    #            break
    #    else: break
    #
    #
    #camara.release()       
    print("Camara cerrada")    
    cv2.destroyAllWindows()