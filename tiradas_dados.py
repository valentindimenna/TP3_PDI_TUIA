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
    Recibe una imaggen en BGR y dos margenes y devuelve una mascara de todos los pixeles que tienen valores entre los margenes
    """
    mask = cv2.inRange(imagen, margen_inferior, margen_superior)
    
    return mask
def detectar_dados_movimientos():
    pass
img_path = "videos_dados/tirada_1.mp4"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
imag = cv2.VideoCapture(img_path)
frames=[]
for i in range(100):
    ret,img = imag.read()
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    frames.append(img)
    # plt.imshow(img)
    # plt.show()
while imag.isOpened():
    ret,img = imag.read()
    if ret == False:
        break
    frames.append(img)
    cv2.imshow('Iamgen',img)
    
plt.imshow(frames[75])
plt.show()
prueba = frames[75]
limite_inferior = np.array([80, 0, 0])  # Limite Inferior de cada Canal
limite_superior = np.array([255, 50, 70])# Limite Superior de cada Canal
# mask = cv2.inRange(imghsv, limite_inferior, limite_superior)
# result = cv2.bitwise_and(img,img,mask=mask)
mask = segmentar(prueba,limite_inferior,limite_superior)
result = cv2.bitwise_and(prueba, prueba, mask=mask)
imshow(mask)