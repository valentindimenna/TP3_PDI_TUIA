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


img_path = "videos_dados/tirada_4.mp4"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
imag = cv2.VideoCapture(img_path)
frames=[]

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
plt.imshow(mask,cmap='gray')
plt.show()
###Leo y guardo todos los frames del video en una lista

img_path = "videos_dados/tirada_1.mp4"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
imag = cv2.VideoCapture(img_path)
frames = []
frames_gr = []
i=0
while True:
    ret,frame = imag.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_gr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames_gr.append(frame_gr)
    frames.append(frame)
    # if i>50:
        # plt.imshow(frame_gr,cmap='gray')
        # plt.show()
        # print(i)
    i+=1

limite_inferior = np.array([80, 0, 0])  # Limite Inferior de cada Canal
limite_superior = np.array([255, 50, 70])# Limite Superior de cada Canal
frames_consecutivos = 0
# for i in range(len(frames)):


frames_consecutivos = 0
i=0
while True:
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(frames[i]), plt.title('primero')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(frames[i+1], cmap="gray"), plt.title('Segundo')
    # plt.show(block=False)
    diff = cv2.absdiff(frames_gr[67], frames_gr[80])
    _, diff_thresh = cv2.threshold(diff, 90, 255, cv2.THRESH_BINARY)
    plt.figure()
    ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(frames_gr[67],cmap='gray'), plt.title('primero')
    plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(frames_gr[77], cmap="gray"), plt.title('Segundo')
    plt.subplot(243,sharex=ax1,sharey=ax1), plt.imshow(diff, cmap="gray"), plt.title('Segundo')
    plt.subplot(244,sharex=ax1,sharey=ax1), plt.imshow(diff_thresh, cmap="gray"), plt.title('Segundo')
    plt.show(block=False)
    # plt.imshow(diff,cmap='gray')
    # plt.show()
    print(diff)
    # _, diff_thresh = cv2.threshold(diff, 90, 255, cv2.THRESH_BINARY)
    # plt.imshow(diff_thresh,cmap='gray')
    # plt.show()
    motion_intensity = np.sum(diff_thresh)
    print(motion_intensity)
    break
    mascara1 = segmentar(frames[i+5],limite_inferior,limite_superior)
    if i < (len(frames)-1):
        mascara2 = segmentar(frames[i+20],limite_inferior,limite_superior)
    else:
        i+=1
        continue
    # plt.figure()
    # ax1 = plt.subplot(241); plt.xticks([]), plt.yticks([]), plt.imshow(mascara1,cmap = 'gray'), plt.title('Mascara1')
    # plt.subplot(242,sharex=ax1,sharey=ax1), plt.imshow(mascara2, cmap="gray"), plt.title('Segundo')
    # plt.show(block=False)
    similarity = np.sum(mascara1 == mascara2) / mascara1.size
    if similarity > 0.99 and not (mascara1 == np.zeros(mascara1.shape)).all():
        print("Máscaras similares")
        frames_consecutivos += 1
    else:
        print("Reseteo")
        frames_consecutivos = 0
    # if (mascara1 == mascara2).all() and not (mascara1 == np.zeros(mascara1.shape)).all():
    #     print("entro")
    #     frames_consecutivos += 1
    # else:
    #     print("Reseteo")
    #     frames_consecutivos = 0
    # print("el i ")
    i += 1
    # print("se incrementa")
# Frame en el que se frenan = i 
img_path = "videos_dados/tirada_1.mp4"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
imag = cv2.VideoCapture(img_path)
frames = []
frames_gr = []
i=0
while True:
    ret,frame = imag.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_gr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames_gr.append(frame_gr)
    frames.append(frame)



for i in range(len(frames_gr)):
    print(i)
    frame1 = frames_gr[i]
    frame2 = frames_gr[i+13]
    diff = cv2.absdiff(frame1, frame2)
    _, diff_thresh = cv2.threshold(diff, 90, 255, cv2.THRESH_BINARY)
    motion_intensity = np.sum(diff_thresh)
    if motion_intensity < 45000:
        break
frenos.append(i)