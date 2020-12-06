import numpy as np
cimport numpy as np


cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b


def createIntImg(np.ndarray[np.int32_t, ndim=2] img):
    cdef int height = img.shape[0], width = img.shape[1]

    for i in range(1, height):
        img[i, 0] = img[i, 0] + img[i-1, 0]
    for j in range(1, width):
        img[0, j] = img[0, j] + img[0, j-1]

    for i in range(1, height):
        for j in range(1, width):
            img[i, j] = img[i, j] + img[i-1, j] + img[i, j-1] - img[i-1, j-1]

    return img


def compiou(np.ndarray[np.float32_t, ndim=2] dt, np.ndarray[np.float32_t, ndim=2] gt, ig):
    cdef int m = dt.shape[0], n = gt.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] ious = np.zeros([m, n], dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] de = dt[:, [0, 1]] + dt[:, [2, 3]]  # x2, y2
    cdef np.ndarray[np.float32_t, ndim=1] da = dt[:, 2] * dt[:, 3]  # area
    cdef np.ndarray[np.float32_t, ndim=2] ge = gt[:, [0, 1]] + gt[:, [2, 3]]  # x2, y2
    cdef np.ndarray[np.float32_t, ndim=1] ga = gt[:, 2] * gt[:, 3]  # area
    cdef np.ndarray[np.uint8_t, ndim=1] igr = np.array(ig, dtype=np.uint8)

    cdef int i, j

    for i in range(m):      # loop through det boxes
        for j in range(n):  # loop through gt boxes
            w = min(de[i, 0], ge[j, 0]) - max(dt[i, 0], gt[j, 0])
            h = min(de[i, 1], ge[j, 1]) - max(dt[i, 1], gt[j, 1])
            if w <= 0 or h <= 0:
                continue
            t = w * h
            if igr[j]:
                u = da[i]
            else:
                u = da[i] + ga[j] - t
            ious[i, j] = t / u
    return ious
