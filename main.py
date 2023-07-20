import cv2 as cv
import numpy as np
import scipy as sc
img = cv.imread('pics/jocelyn-morales-urzLXNmiJi8-unsplash.jpg')


def resizing(frame):
    width = (int)(frame.shape[1])
    #height = (int)(frame.shape[0] * s)

    dimensions = (513, 513)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img2 = resizing(img)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow('img_orgnl', img2)
img = np.asarray(img2)

kernel = np.zeros((5,5))
kernel[0:5,0:5] = 1

print(kernel)

def convolution(img, kernel):
    M = img.shape[1]
    N = img.shape[0]
    n = kernel.shape[0]
    m = kernel.shape[1]

    arr = np.pad(img, ((m-1,m-1), (n-1,n-1)))

    s = np.zeros((M+m-1, N+n-1, 3), 'uint8')

    for x in range(M+m-1):
        for y in range(N+n-1):
            for i in range(m):
                for j in range(n):
                    s[x, y] = s[x, y] + (kernel[i, j] * arr[(x - i)%M + m-1, (y - j)%N + n-1])/((kernel.shape[0])*(kernel.shape[1]))

    return s


def faster_convolution(img, kernel):
    M = img.shape[1]
    N = img.shape[0]
    n = kernel.shape[0]
    m = kernel.shape[1]
    K1 = np.pad(kernel, pad_width = (img.shape[1] - kernel.shape[1])//2)

    F = np.fft.fft2(img)
    K = np.fft.fft2(K1)


    G = F * K/((kernel.shape[0])*(kernel.shape[1]))

    a = np.fft.ifft2(G)
    a = np.fft.fftshift(a)
    #a = img - a
   # a = a - np.min(a)

    #a = 255 * (a / np.max(a))

    a = a.astype('uint8')

    return a

#s = convolution(img , kernel)
a = faster_convolution(img, kernel)
cv.imshow('img', a)

#grd = grad - np.min(grad)
#grad = 255 * (grad / np.max(grad))

cv.waitKey(0)