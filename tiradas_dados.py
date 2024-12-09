import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)
def segmentar(imagen, margen_inferior:np.array, margen_superior:np.array):
    """
    Recibe una imagen y dos margenes y devuelve una mascara en escala de grises 
    poniedo a todos los pixeles que tienen valores entre esos margenes un valor  de 255
    """
    mask = cv2.inRange(imagen, margen_inferior, margen_superior)
    return mask
def leer_video(ruta_video):
    limite_inferior_rojo = np.array([75, 0, 0])
    limite_superior_rojo = np.array([255, 65, 75])
    limite_inferior_verde = np.array([0, 0, 0])
    limite_superior_verde = np.array([165, 215,180])
    limite_inferior_verde_hsv = np.array([65, 180, 60])
    limite_superior_verde_hsv = np.array([95, 255,215])
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imag = cv2.VideoCapture(ruta_video)
    frames = []
    frames_gr = []
    frames_hsv =[]
    i=0
    while True:
        ret,frame = imag.read()
        if not ret:
            break
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        if i == 0:
            mascara = segmentar(frame_hsv,limite_inferior_verde_hsv,limite_superior_verde_hsv)
            i += 1
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # plt.imshow(frame)
        # plt.show()
        masked_frame = cv2.bitwise_and(frame, frame, mask = mascara)
        frames.append(masked_frame)
        frame_gr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        masked_frame_gr = cv2.bitwise_and(frame_gr, frame_gr, mask = mascara)
        frames_gr.append(masked_frame_gr)
        frames_hsv.append(frame_hsv)
        # plt.imshow(masked_frame)
        # plt.show()
        # cv2.imshow('frame',masked_frame)
        # cv2.imshow('movimiento', frame_hsv)
    #     if cv2.waitKey(1) == ord('q'):
    #         break 
    # imag.release()
    # cv2.destroyAllWindows()
    return frames_hsv,frames,frames_gr
##############ESRE ES EL DE AHORA
limite_inferior_rojo = np.array([75, 0, 0])
limite_superior_rojo = np.array([255, 65, 75])
limite_inferior_verde = np.array([0, 0, 0])
limite_superior_verde = np.array([165, 215,180])
limite_inferior_verde_hsv = np.array([65, 180, 60])
limite_superior_verde_hsv = np.array([95, 255,215])
for tirada in range(1,5):
    frameshsv,frames,frames_gr = leer_video(f"videos_dados/tirada_{tirada}.mp4")
    frame_min = 40
    freno = 0
    band = 0
    band2 = 0
    movimiento = len(frames_gr) - 1 
    for i in range(len(frames_gr)-2):
        frame1 = segmentar(frames[i],limite_inferior_rojo,limite_superior_rojo)
        frame2 = segmentar(frames[i+1],limite_inferior_rojo,limite_superior_rojo)
        frame3 = segmentar(frames[i+2],limite_inferior_rojo,limite_superior_rojo)
        # plt.imshow(frame1)
        # plt.show()
        # frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # frame1=frames_gr[i]
        # frame2=frames_gr[i+1]
        diff = cv2.absdiff(frame1,frame2)
        # diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # plt.imshow(diff_thresh,cmap = 'gray')
        # plt.show()
        movement = np.sum(diff_thresh > 0)
        diff2 = cv2.absdiff(frame2,frame3)
        _, diff_thresh2 = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)
        movement2 = np.sum(diff_thresh2 > 0)
        if movement < 2000 and i > frame_min and band == 0 and movement2 <2000:
            # print("ACA 2",i)
            band=1
            freno = i
        elif movement > 2000 and movement2 > 2000 and band==1 and band2==0:
            # print("ACA",i)
            band2 = 1
            movimiento = i
            break
        # print(freno)
        # print(movimiento)
        # print(i, ":" ,movement)
    #Freno: frame en el que deja de haber movimiento
    #Movimiento: Frame en el que vuelve a haber movimiento o ultimo frame del video
    salida = []
    for i in range(len(frames)):
        if i < freno or i > movimiento:
            salida.append(frames[i])
        else:
            im = frames[i]
            mascara = segmentar(im, limite_inferior_rojo,limite_superior_rojo)
            # plt.imshow(mascara,cmap='gray')
            # plt.show()
            result = cv2.bitwise_and(im, im, mask = mascara)
            # plt.imshow(mascara,cmap='gray')
            # plt.show()
            contours, jerarquia = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # for j,cnt in enumerate(contours):
            #     print(cv2.contourArea(cnt))
            filtered_contours = [(i,cnt) for i,cnt in enumerate(contours) if cv2.contourArea(cnt) > 100]
            dados = []
            indices = []
            for i,cnt in filtered_contours:
                # Crear un rectángulo alrededor del contorno
                x, y, w, h = cv2.boundingRect(cnt)
                indices.append(i)
                # Recortar el objeto en la imagen original en color
                objeto = result[y:y+h, x:x+w]
                # Mostrar el objeto
                dados.append(objeto)
                # plt.imshow(objeto)
                # plt.title("Contornos encontrados")
                # plt.show()
            dict_contornos = {}
            padres = 0
            hijos = 0
            conteo_dado = 0
            copia = im.copy()
            for indice in indices:
                _, _, first_child, parent = jerarquia[0][indice]
                # print(indice,jerarquia[0][indice])
                # print(first_child,parent)
                if parent == -1:
                    #Es padre 
                    padres +=1 
                    color = (0, 0, 255)  # Azul para contornos padres
                    dict_contornos[indice] = []  # Inicializo la lista de hijos para este padre
                else:  
                    # Contorno hijo
                    hijos += 1
                    color = (0, 0, 255)  # Azul para contornos hijos
                    # Agregar este contorno hijo a la lista del contorno padre
                    if parent in dict_contornos:
                        dict_contornos[parent].append(indice)
                conteo_dado = hijos
                cv2.drawContours(copia, contours, indice, color, 1)
            conteo_dados = hijos
            for parent_index in dict_contornos:
                # Obtenemos las coordenadas del rectángulo que envuelve el dado
                if len(dict_contornos[parent_index])==0:
                    continue
                x, y, w, h = cv2.boundingRect(contours[parent_index])
                # Asignamos el número de puntos detectados en el dado (hijos)
                numero = len(dict_contornos[parent_index])
                # Escribimos el número sobre el dado
                cv2.putText(copia, str(numero), (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            # salida += [copia for _ in range(movimiento - freno)]# Si no se mueve pego directamente 20 frames de estos falla algo con los frames agregados
            salida.append(copia)
            # i += (movimiento-freno)
    output_video = f"videos_dados/salida_tirada_{tirada}.mp4"
    frame_height, frame_width, _ = salida[0].shape
    # Creo el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (frame_width, frame_height))
    # Agregar cada frame al video
    for frame in salida:
        out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    # Libero el objeto VideoWriter
    out.release()
