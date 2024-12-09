import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
#Patentes 29.4 x 12.9
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
def image_paths():
    paths = []
    for i in range(1,13):
        if i<10:
            num = '0' + str(i)
        else:
            num = str(i)
        img_path = f"imagenes/img{num}.png"
        paths.append(img_path)
    return paths
def get_images(img_paths):
    """
    Recibe una lista con los path de las imagenes a leer y retorna una lista de las imagenes leidas en bgr
    img_paths: Lista con las direcciones de las imagenes
    imagenes: Lista de las imagenes leidas
    """
    imagenes = []
    for path in img_paths:
        imagenes.append(cv2.imread(path))
    return imagenes
def images_gris(imgs_bgr):
    imagenes = []
    for img in imgs_bgr:
        imagenes.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    return imagenes
def umbralice(imagen_gris, min = 125,max =255, tipo = "THRESH_BINARY"):
    """
    Recibe una imagen en escala de grises y devuelve la imagen umbralizada
    """
    if tipo == "THRESH_BINARY":
        _, imagen_auto_gris = cv2.threshold(imagen_gris, min, max,cv2.THRESH_BINARY)
    else:
        _, imagen_auto_gris = cv2.threshold(imagen_gris, min, max,cv2.THRESH_BINARY_INV)
    return imagen_auto_gris
#
paths = get_images(image_paths())
grises = images_gris(paths)
gri = grises[0]
gri = umbralice(gri)
diccionario_posibles_patentes = {i: [] for i in range(len(grises))}
aspect_ratio = 294/129
for i in range(len(grises)):
    # imagen_umbralizada = umbralice(grises[i],120,255)
    im  = grises[i]
    imagen_blurreada = cv2.GaussianBlur(im, (5, 5), 0)
    imagen_umbralizada = cv2.adaptiveThreshold(imagen_blurreada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    # Operación morfológica para mejorar limpieza
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(imagen_umbralizada, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cleaned = 255 - cleaned
    # edges = cv2.Canny(imagen_blurreada, 200, 350)
    ###Muestra de Imagen umbralizada y original
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(grises[i],cmap = 'gray'), plt.title('Imagen en escala de grises')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(imagen_umbralizada, cmap="gray"), plt.title('Imagen Umbralizada')
    # plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(cleaned, cmap="gray"), plt.title('Cleaned')
    # plt.show(block=False)
    num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    binary_mask = np.zeros_like(labels, dtype=np.uint8)
    # Recorremos cada etiqueta y dibujamos las regiones en la máscara binaria
    for label in range(1, num_labels):  # Empezar desde 1 para ignorar el fondo
        binary_mask[labels == label] = 255
    # plt.imshow(binary_mask,cmap = 'gray')
    # plt.show()
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(labels,cmap = 'gray'), plt.title('Componentes de umbralizada')
    # plt.show(block=False)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours and bounding boxes to visualize
    output_image = cv2.cvtColor(grises[i], cv2.COLOR_GRAY2BGR)
    # Filtro contornos en base a area, altura, ancho y aspect ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 100 <= cv2.contourArea(contour) <=  2500 and w < 200 and 10 < h < 45 and np.isclose(w/h,aspect_ratio,atol = 0.7): #and 0 < w < 75 and 0 < h < 35:#30 ,  and 1 < w < 30 and 1 < h < 6 
            cv2.rectangle(output_image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
            diccionario_posibles_patentes[i].append(grises[i][y : y + h, x: x + w])
    # Show the result
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(output_image,cmap = 'gray'), plt.title('Contornos')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(binary_mask, cmap="gray"), plt.title('Mask')
    # plt.show(block=False)

valid_components = {i: [] for i in range(len(grises))}
for keys,value in diccionario_posibles_patentes.items():
    for region in range(len(value)):
        # _, binary_plate = cv2.threshold(value[region], 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # bin = binary_plate.copy()
        # binary_plate = skimage.segmentation.clear_border(binary_plate)
        _ , thimg = cv2.threshold(value[region],121,255,cv2.THRESH_BINARY)#122
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        thimg = cv2.morphologyEx(thimg, cv2.MORPH_CLOSE, kernel,iterations=1)
        # thimg = cv2.dilate(thimg, kernel, iterations = 1)
        # thimg = skimage.segmentation.clear_border(thimg)
        # plt.figure()
        # plt.subplot(121), plt.imshow(value[region],cmap='gray')
        # plt.title('imagen original')
        # plt.subplot(122), plt.imshow(thimg, cmap='gray')
        # plt.title('Th image')
        # plt.show()
        #####
        contours, _ = cv2.findContours(thimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = cv2.cvtColor(value[region], cv2.COLOR_GRAY2BGR)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10 <= cv2.contourArea(contour) and h > 7 and w < 19:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                valid_components[keys].append(value[region][y : y + h, x : x + w])
        #####
        # num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(binary_plate, connectivity=8)
        # output_image = cv2.cvtColor(value[region], cv2.COLOR_GRAY2BGR)
        # # Iterar por todas las componentes conectadas menos el fonde (label 0)
        # for i in range(1, num_labels):
        #     x, y, w, h, area = stats[i]
            
        #     # Filter components based on area, height, width, and aspect ratio
        #     if 100 <= area <= 2500 and 10 < h < 45 and 0.5 < w / h < 3 or True:
        #         # Draw a rectangle around each valid component
        #         cv2.rectangle(output_image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        # plt.figure()
        # plt.subplot(121), plt.imshow(output_image)
        # plt.title('Connected Components')
        # plt.subplot(122), plt.imshow(binary_plate, cmap='gray')
        # plt.title('Binary Plate Mask')
        # plt.show()
        # plt.figure()
        # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(output_image,cmap = 'gray'), plt.title('Componentes de umbralizada')
        # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(thimg, cmap="gray"), plt.title('Mask')
        # plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(value[region], cmap="gray"), plt.title('Normal')
        # # plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(bin, cmap="gray"), plt.title('Bin')
        # plt.show(block=False)
no_detectadas=[]
for i in range(len(valid_components)):
    # print(f"en el indice {i}:{len(valid_components[i])}")
    if len(valid_components[i]) != 6:
        valid_components[i]=[]
        no_detectadas.append(i)
# print(no_detectadas)


diccionario_posibles_patentes_2 =  {i: [] for i in no_detectadas}
for i in no_detectadas:
    imagen = grises[i]
    _ , thimg = cv2.threshold(grises[i],120,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Use a larger kernel to connect nearby characters
    thimg = cv2.morphologyEx(thimg, cv2.MORPH_OPEN, kernel,iterations=1)
    # thimg = 255 - thimg
    # plt.imshow(thimg,cmap='gray')
    # plt.show()
    # contours, _ = cv2.findContours(thimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # output_image2 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if 10 <= cv2.contourArea(contour) and h > 5 and w < 30:
    #         cv2.rectangle(output_image2, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(output_image2,cmap = 'gray'), plt.title('Componentes de umbralizada')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(thimg, cmap="gray"), plt.title('Mask')
    # plt.show(block=False)
    num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(thimg, connectivity=8)
    # num_labels2, labels2, stats2, centroids2 =  cv2.connectedComponentsWithStats(edges, connectivity=8)
    # Recorremos cada etiqueta y dibujamos las regiones en la máscara binaria
    binary_mask = np.zeros_like(labels, dtype=np.uint8) # Empezar desde 1 para ignorar el fondo
    for label in range(1, num_labels):
        if (stats[label][4] > 10 and stats[label][4]<70 and np.isclose(stats[label][2]/stats[label][3] ,1.5/3, atol=0.7)): 
            binary_mask[labels == label] = 255
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(labels,cmap = 'gray'), plt.title('Componentes de umbralizada')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(thimg, cmap="gray"), plt.title('Componentes canny')
    # plt.show(block=False)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image2 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 2 <= cv2.contourArea(contour) and 3 < h < 17 and 5 < w < 10 :
            cv2.rectangle(output_image2, (x, y), (x + w, y + h), (255, 0, 0), 2)
            diccionario_posibles_patentes_2[i].append((x, y, w, h))
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(output_image2,cmap = 'gray'), plt.title('Componentes de umbralizada')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(thimg, cmap="gray"), plt.title('Mask')
    # plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(binary_mask, cmap="gray"), plt.title('binary mask')
    # plt.subplot(244,sharex=ax1,sharey=ax1), plt.imshow(labels, cmap = 'gray'), plt.title('binary mask')
    # plt.show(block=False)


# for key,value in diccionario_posibles_patentes_2.items():
#     print(f"{key}:{len(value)}")
# len(diccionario_posibles_patentes_2)
# prueba = diccionario_posibles_patentes_2[8]
# prueba = sorted(diccionario_posibles_patentes_2[8], key=lambda x: x[0])

contornos_caracteres = {i: [] for i in no_detectadas}
for clave in no_detectadas:
    possible_contours = sorted(diccionario_posibles_patentes_2[clave], key=lambda x: (x[0], x[1]))  # Ordeno por x y despues y
    if len(possible_contours)>=6:
        selected_contours = []
        conteo=[]
        found = False  # Flag to break both loops
        for i in range(len(possible_contours)):
            if found:
                break  # Exit the outer loop if we've found 6 contours
            current_contour = possible_contours[i]
            current_x, current_y, current_w, current_h = current_contour
            # conteo = [current_contour]  # Start a new group with the current contour
            conteo = [current_contour]
            # Try to add the 5 closest contours (next 5 that are close in x and y)
            for j in range(i + 1, len(possible_contours)):
                next_contour = possible_contours[j]
                next_x, next_y = next_contour[0], next_contour[1]
                if next_x - 100 < current_x < next_x + 100 and next_y - 30 < current_y < next_y + 30:
                    conteo.append(next_contour)
                if len(conteo) == 6:
                    found = True  # Cambio bandera a True
                    break  # Salgo del primer bucle
            if len(conteo) == 6:
                # print("Breakeo 2")
                selected_contours = conteo  # GUardo el contorno
                contornos_caracteres[clave].append(conteo)
                break  # Salgo del segundo Bucle
for clave,value in contornos_caracteres.items():
    # print("clave",clave)
    for i in range(len(value)):
        # print("i",i)
        for j in range(len(value[i])):
            x,y,w,h = value[i][j]
            valid_components[clave].append(grises[clave][y:y+h,x:x+w])
            # print(f"clave:{clave}\ni:{i}\nj:{j}")
            # valid = grises[clave][y:y+h,x:x+w]
            # plt.imshow(valid,cmap='gray')
            # plt.show()
    
for key,value in valid_components.items():
    if len(value)!=6:
        print(f"Auto nro {key}: Patente No encontrada")
    else:
        # print(value)
        plt.figure()
        plt.title(f"Patente Nro: {key}")
        ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(value[0],cmap = 'gray')
        plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(value[1], cmap="gray")
        plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(value[2], cmap="gray")
        plt.subplot(244,sharex=ax1,sharey=ax1), plt.imshow(value[3], cmap = 'gray')
        plt.subplot(245,sharex=ax1,sharey=ax1), plt.imshow(value[4], cmap = 'gray')
        plt.subplot(246,sharex=ax1,sharey=ax1), plt.imshow(value[5], cmap = 'gray')
        plt.show(block=False)