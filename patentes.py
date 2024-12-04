import numpy as np
import cv2
import matplotlib.pyplot as plt

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

paths = get_images(image_paths())
grises = images_gris(paths)
coso = paths[0].copy()
for i in range(len(grises)):
    th_im = umbralice(grises[i], 70, 250)
    blurred = cv2.GaussianBlur(th_im, (5, 5), 0)
    edges = cv2.Canny(blurred, 160, 320)
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(grises[i],cmap = 'gray'), plt.title('Imagen en escala de grises')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(th_im, cmap="gray"), plt.title('Threshold')
    # plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(edges, cmap="gray"), plt.title('Canny')
    # plt.show(block=False)
    # thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    # character_candidates = []
    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # Draw all contours on a copy of the original image
    # output_image = grises[i].copy()
    # output_image2 = grises[i].copy()
    # cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    # plt.imshow(output_image, cmap = 'gray')
    # plt.show()
    # if i == 5:
    #     for cnt in contours:
    #         if cv2.contourArea(cnt)>40:
    #             print(cv2.contourArea(cnt))
    #             cv2.drawContours(output_image2, cnt, -1, (0, 255, 0), 2)
    #     # plt.imshow(output_image2, cmap = 'gray')
    #     # plt.show()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    rectangular_contours = []
    for cnt in contours:
    # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Check if the contour has four vertices (is a rectangle)
        if len(approx) == 4:
            # Calculate the bounding box and aspect ratio
            print("LEN 4 ")
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            # Aspect ratio for a license plate (~2.07) and size constraints
            if  42 < w < 105 and 11 < h < 45:  # Adjust thresholds as needed
                "ENTRO AL ISCLOSE"
                rectangular_contours.append(cnt)
    output_image = grises[i].copy()
    cv2.drawContours(output_image, rectangular_contours, -1, (0, 255, 0), 2)
    # Display the result
    plt.imshow(output_image,cmap = 'gray')
    plt.show()




################## otra cosa
paths = get_images(image_paths())
grises = images_gris(paths)
for i in range(len(grises)):
    w, h = 4,9
    th_im = umbralice(grises[i], 62, 250)
    blurred = cv2.GaussianBlur(th_im, (5, 5), 0)
    edges = cv2.Canny(blurred, 200, 350)
    s = cv2.getStructuringElement(cv2.MORPH_RECT,(h,w))
    imagen_close = cv2.morphologyEx(edges.copy(),cv2.MORPH_CLOSE,s,iterations=2)
    plt.figure()
    ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(grises[i],cmap = 'gray'), plt.title('Imagen en escala de grises')
    plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(edges, cmap="gray"), plt.title('Canny')
    plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(imagen_close, cmap="gray"), plt.title('Close')
    plt.show(block=False)
    num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(imagen_close, connectivity=8)
    filtered_imagen = np.zeros_like(imagen_close, dtype = np.uint8)
    for j in range(1, num_labels):
        if stats[j][4] >= 1500:
            filtered_imagen[labels == j] = 255
    plt.figure()
    plt.imshow(filtered_imagen,cmap='gray'),plt.title("Close filtro area")
    plt.show()
    imagen_open = cv2.morphologyEx(filtered_imagen.copy(), cv2.MORPH_OPEN, s, iterations=2)
    plt.figure()
    plt.imshow(imagen_open,cmap='gray'),plt.title("COpenlose filtro area")
    plt.show()
