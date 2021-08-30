import cv2
import numpy as np

def draw_chessboard(row, col, size):
    img = np.zeros([(row+1)*size, (col+1)*size])
    colors = [0, 255]
    for i in range(row+1):
        for j in range(col+1):
            img[i*size:(i+1)*size, j*size:(j+1)*size] = colors[j % 2]
        colors = colors[::-1]

    img = np.pad(img, ((120, 120), (150, 150)), constant_values=255)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    draw_chessboard(5, 11, 140)