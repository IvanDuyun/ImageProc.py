import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_float, data, feature,color, io
from skimage.filters import gaussian, median, sobel
from scipy import signal
from math import*
from numpy import*

import os
cd = os.curdir

def myrgb2hsv(img):
    # В сером
    img = img/255
    Cmin = img.min(-1)
    out = np.empty_like(img)

    # V канал
    Cmax = img.max(-1)
    v = Cmax

    # S канал
    delta = Cmax-Cmin
    s = delta / Cmax
    s[Cmax == 0] = 0  # Если Сmax = 0 , то s = 0

    # H канал
    # Максимальный красный
    i = (img[..., 0] == v)
    out[i, 0] = (img[i, 1] - img[i, 2]) / delta[i]

    # Максимальный зеленый
    i = (img[..., 1] == v)
    out[i, 0] = 2 + (img[i, 2] - img[i, 0]) / delta[i]

    # Максимальный синий
    i = (img[..., 2] == v)
    out[i, 0] = 4 + (img[i, 0] - img[i, 1]) / delta[i]

    h = (out[..., 0] / 6.) % 1.
    h[delta == 0.] = 0.

    out[..., 0] = h
    out[..., 1] = s
    out[..., 2] = v
    return out

def myrgb2gray(img):
    img = 0.2125*img[..., 0]+0.7154*img[..., 1]+0.0721*img[..., 2]
    return img

def myrgb2yuv(img):
    YUV_K_R = 0.299
    YUV_K_G = 0.587
    YUV_K_B = 0.114
    out = np.empty_like(img)
    y_out = YUV_K_R * img[:, :, 0] + YUV_K_G * img[:, :, 1] + YUV_K_B * img[:, :, 2]
    u_out = 0.493*(img[:, :, 2] - y_out)
    v_out = 0.877*(img[:, :, 0] - y_out)
    out[:, :, 0] = y_out
    out[:, :, 1] = u_out
    out[:, :, 2] = v_out
    return out

def transformations():
    rgb_image = Image.open(cd+'/1.jpg')
    rgb_image = img_as_float(rgb_image)

    hsv_image = myrgb2hsv(rgb_image)
    gray_image = myrgb2gray(rgb_image)
    yuv_image = myrgb2yuv(rgb_image)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 2))
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title("RGB original")
    axs[0, 1].imshow(hsv_image)
    axs[0, 1].set_title("HSV")
    axs[1, 0].imshow(gray_image)
    axs[1, 0].set_title("Grayscale")
    axs[1, 1].imshow(yuv_image)
    axs[1, 1].set_title("YUV")

    plt.show()

def find_face():
    # Задание диапазона телесных цветов
    h_min = 0.05
    h_max = 0.09
    s_min = 0.2
    s_max = 0.45

    rgb_img = data.astronaut()
    hsv_img = myrgb2hsv(rgb_img)

    h, s = hsv_img[..., 0], hsv_img[..., 1]
    # Отбор значений, входящий в диапазон
    h_mask = np.logical_and(h > h_min, h < h_max)
    s_mask = np.logical_and(s > s_min, s < s_max)
    # Маска изображения только с искомым диапазоном цветов
    mask = np.logical_not(np.logical_and(h_mask, s_mask))
    # Наложение маски на исходное RGB изображение
    filter_img = rgb_img.copy()
    filter_img[mask] = 255

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].imshow(rgb_img)
    ax[0].set_title("RGB")
    ax[1].imshow(filter_img)
    ax[1].set_title("RGB")
    fig.tight_layout()
    plt.show()

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) /sd, 2) / 2)

# Определение функции свертки гаусса
def convolution(image, kernel, average=False):
    image = color.rgb2gray(image)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col +
    (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row +
            kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=sqrt(kernel_size))
    return convolution(image, kernel)

def median(img, r):
    rows, columns = img.shape

    result = copy(img)

    sectorL = 2 * r + 1

    curentSector = zeros(sectorL * sectorL)

    for i in range(r, rows - r):
        for j in range(r, columns - r):
            for l in range(sectorL):
                for k in range(sectorL):
                    curentSector[l * sectorL + k] = img[i - r + l, j - r + k]

            curentSector = sort(curentSector)

            result[i, j] = curentSector[sectorL + sectorL // 2 + 1]

    return result[r:rows - r, r:columns - r]

def decline_noise():
    original_image = io.imread('chel.jpg')
    original_image = color.rgb2gray(original_image)

    gauss_filter_image = gaussian_blur(original_image, 5)
    median_filter_image = median(original_image,1)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
    ax0.imshow(original_image, cmap='gray')
    ax0.set_title("Original")
    ax1.imshow(gauss_filter_image, cmap='gray')
    ax1.set_title("Gauss")
    ax2.imshow(median_filter_image, cmap='gray')
    ax2.set_title("Median filter")

    plt.show()

def mysobel(imeg):
    img = img_as_float(imeg)
    # Создаем маски, используемые оператором Собеля
    kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
    kv = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)
    # Сворачиваем в двумерный массив. Gx, Gy содержат приближенные производные по x и y
    gx = signal.convolve2d(img, kh, mode='same', boundary='symm')
    gy = signal.convolve2d(img, kv, mode='same', boundary='symm')
    # Приближенное значение градиента
    g = np.sqrt(gx * gx + gy * gy)
    # Нормализация
    g *= 255.0 / np.max(g)
    return g

def mylaplas(img):
    img_out = np.zeros(img.shape, dtype='float64')
    M = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    for row in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            value = M * img[(row - 1):(row + 2), (col - 1):(col + 2)]
            img_out[row, col] = min(100, max(0, value.sum()))
    return img_out

#Проверки точки на принадлежность изображению
def isCorrectIndex(img,x,y):

    if x<0 or x>img.shape[1]-1 or y<0 or y>img.shape[0]-1:
        return 0
    else: return 1

#Сравнение значения:
def Check(img,x,y,v):
    if isCorrectIndex(img,x,y) == 0:
        return 0
    elif img[int(x),int(y)] < v:
        return  1
    else: return 0

def errCode(x,y):
    alpha = int(round(atan2(x,y) * 180 /pi,0))
    tr = [0,45,90,135,180,225,270]
    if alpha == tr:
        return True
def mycanny(img):
    x: int
    y: int
    rows, columns = img.shape

    g = zeros((rows-1, columns-1))
    ang = zeros((rows-1, columns-1))
    r = zeros((rows, columns))

    for y in range(0, rows-1):
        for x in range(0, columns-1):
            g[x,y] = img[x,y]
    #Сглаживание
    img = gaussian_blur(img,5)
    #Поиск градиентов
    img = mysobel(img)
    #Подавление не-максимумов
    for y in range(1,rows-1):
        for x in range(1,columns-1):
            if errCode(x,y):
                continue
            dx = np.sign(cos(img[x,y]))
            dy = -np.sign(sin(img[x,y]))
            if Check(g,x-dx,y-dy,g[x,y])==1:
                r[x-int(dx),y-int(dy)] = 0
            if Check(g,x+dx,y+dy,g[x,y])==1:
                r[x+int(dx),y+int(dy)] = 0
            r[x,y] = g[x,y]
    return  r

def finf_contour():
    original_image = data.camera()

    #canny_image = feature.canny(original_image)
    canny_image = mycanny(original_image)
    sobel_image = mysobel(original_image)
    laplas_image = mylaplas(original_image)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original image")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(canny_image, cmap='gray')
    axs[0, 1].set_title("Canny filter")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(sobel_image, cmap='gray')
    axs[1, 0].set_title("Sobel filter")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(laplas_image, cmap='gray')
    axs[1, 1].set_title("Laplace filter")
    axs[1, 1].axis('off')
    plt.show()

transformations()
find_face()
decline_noise()
finf_contour()

