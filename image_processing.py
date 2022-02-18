import math
import cv2
import glob
import os
import re
import pandas as pd

def inverse_color(image):  #1. 灰度处理
    height, width = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j])
    return img2         # 反色处理

path = "C:\\Users\\Min's laptop\\Desktop\\Documents\\ISSP\\AuNR_Photothermal\\raw data\\image process\\AuNR1\\*.jpg"

df = pd.DataFrame(columns=['diameter (nm)', 'AR'])

for file in glob.glob(path):
    file_name = os.path.splitext(file)[0] #读取文件名
    file_number = ''.join(re.findall('(\d+)', file_name))
    img1 = cv2.imread(file)           #原图
    img2 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)     #灰度处理
    cv2.imwrite(file, img1)  # 保存原图
    img = inverse_color(img2)
    height, width = img.shape
    scale_bar = 1613/width    # nanometer per pixel
    print(file_name)
    #print(height,width)
    var = 1
    thr = input('Please enter new threshold value (0 - 255): = ')
    while var == 1:  # 调节灰度
        _, threshold = cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)  # 单色
        binary, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓
        cv2.imshow("threshold", threshold)
        cv2.imshow("original", img1)
        cv2.waitKey(500)
        thre = input('Set threshold = ' + str(thr) + ' , Accept? Yes(y) or enter new threshold value (0 - 255): ')
        if not thre == 'y':
            thr = thre
        else:
            break

    for contour in contours:        #2. 区域识别
        approx = cv2.approxPolyDP(contour, 0.001*cv2.arcLength(contour, True), True)       #多边形拟合
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        area = cv2.contourArea(contour)
        if area <= 100 or area > 300000:  # 忽略过小和过大对象
            continue
        if len(approx) >= 15:
            fit = (x, y), (MA, ma), angle = cv2.fitEllipse(approx)
            x1, y1, w1, h1 = cv2.boundingRect(approx)
            img3 = cv2.imread(file)  # 原图显示形状拟合
            cv2.ellipse(img3, fit, (0, 0, 255), 5)
            cv2.imshow("shape fit", img3)
            cv2.waitKey(500)
            user_input = input('cluster? Yes(y) or No(n)： ')
            if user_input == 'y':
                cv2.ellipse(img1, fit, (0,0,255), 5)
                diameter = math.sqrt(MA*ma)
                diameter_1 = round(float(diameter*scale_bar), 2)
                cv2.imshow("cluster sum", img1)
                aspectRatio = max(float(MA / ma), float(ma / MA))
                print("D = " + str(diameter_1) + " nm")
                print("AR = " + str(aspectRatio))
                df = df.append({'diameter (nm)': diameter_1, 'AR': aspectRatio}, ignore_index=True)
        cv2.destroyAllWindows()
        cv2.imwrite("C:\\Users\\Min's laptop\\Desktop\\Documents\\ISSP\\AuNR_Photothermal\\raw data\\image process\\" + file_number + "_processed.jpg", img1) # 保存

df.to_csv("C:\\Users\\Min's laptop\\Desktop\\Documents\\ISSP\\AuNR_Photothermal\\raw data\\image process\\AuNR1\\results.csv", index=False,sep=',')