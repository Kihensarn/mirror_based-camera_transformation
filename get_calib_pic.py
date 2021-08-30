import cv2
import numpy as np
import os
from pathlib import Path

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


def get_camera():
    npzfile = np.load('calibrate.npz')
    camera_matrix = npzfile['mtx']

    file = Path(file_to_store)
    if not file.is_dir():
        file.mkdir(parents=True)

    camera_store_path = file / 'camera.txt'
    np.savetxt(str(camera_store_path), camera_matrix, fmt='%f', delimiter=' ')

def set_params(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080);#宽度

    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960);#高度

    capture.set(cv2.CAP_PROP_FPS, 30);#帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, -100);#亮度 

    capture.set(cv2.CAP_PROP_CONTRAST,10);#对比度 40

    capture.set(cv2.CAP_PROP_SATURATION, 50);#饱和度 50

    capture.set(cv2.CAP_PROP_HUE, 50)#色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, 10);#曝光 50 获取摄像头参数

def reduce_highlights(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy  = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8) 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) 
        img_zero[y:y+h, x:x+w] = 255 
        mask = img_zero 

    print("Highlight part: ")
    # show_img(mask)

    # alpha，beta 共同決定高光消除後的模糊程度
    # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
    # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
    result = cv2.illuminationChange(img, mask, alpha=0.2, beta=0.4) 
    # show_img(result)

    return result

def get_calib_pic(size):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # set_params(cap)
    count = 1  # count 用来标志成功检测到的棋盘格画面数量
    NumberofCalibrationImages = 10

    Nx_cor = size[0]
    Ny_cor = size[1]

    # W = 640
    # H = 480 #360
    # print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, W))
    # print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H))

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (1920, 1080))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
            # If found, add object points, image points
            if ret == True:
                file_path =file_to_store + '/' + 'input{}.jpg'.format(count)
                cv2.imwrite(file_path, gray)
                print('Num of imgs {}/{}'.format(count, NumberofCalibrationImages))
                count += 1

                if count > NumberofCalibrationImages:
                    break
            else:
                print('not find chessboard')
                print(type(corners))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def find_chessboard(img_path, size):
    assert os.path.exists(img_path)
    img = cv2.imread(str(img_path))
    # img = cv2.resize(img, (1920, 1080))
    # img = reduce_highlights(img)
    ok, corners = cv2.findChessboardCorners(img, size, None)
 
    # show the detected corners
    if ok:
        for pt in corners:
            point = pt[0]
            cv2.circle(img, center=(int(point[0]), int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print ('cannot find chessboard points')

    # select 
    select_input_index = [0, 50, 54]
    select_model_index = [50, 0, 4]

    # sort the results at the beginning of right bottom
    corners = corners.reshape(corners.shape[0], corners.shape[2])
    if corners[0].sum() < corners[size[0]*size[1]-1].sum():
        corners = corners[::-1, :]

    corners_select = corners[select_input_index, :]
    img_path = Path(img_path)
    file_name = img_path.stem
    file_path = file_to_store + '/' + file_name + '.txt'
    file_select_path = file_to_store + '/' + file_name + '_3p.txt'

    if file_name == 'model':
        corners = np.pad(corners, ((0, 0), (0, 1)), constant_values=0.0)
        corners = corners * 0.2745
        corners_select = corners[select_model_index, :]

    np.savetxt(file_path, corners, fmt='%f', delimiter=' ')
    np.savetxt(file_select_path, corners_select, fmt='%f', delimiter=' ')

def show_results(file_dir):    
    mat_list = []
    for file in file_dir.iterdir():
        if file.stem[0:3] == 'mat':
            mat = np.loadtxt(file)
            mat = mat.reshape(-1)[9:13].tolist() # show the transformation matrix
            mat.append(float(file.stem[3:])) # average error
            if mat[3] < 0.7:
                mat_list.append(mat)
    mat = np.array(mat_list)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    print(mat)


if __name__ == '__main__':
    file_to_store = './data4'
    size = (5, 11)
    file_dir = Path(file_to_store)

    # get intrinsic parameters from .npz file
    get_camera()

    # get calibrate pictures
    get_calib_pic(size)
    for file in file_dir.iterdir():
        if file.suffix == '.jpg':
            find_chessboard(file, size)

    # show the results of the combined pictures 
    # show_results(file_dir)





