import math

import cv2 as cv
import sys
import numpy as np

# 为了judge_rect_exist_pix 函数写的小型修补函数，考虑粘连
def get_freq(x,y,l,img,type):
    if type == 'h':
        count_1 = 0
        count_2 = 0
        for i in range(l-1):
            if img[y,x+i] == 0 and img[y,x+i+1]== 255:
                count_1 += 1
            if img[y,x+i] == 255 and img[y,x+i+1] == 0:
                count_2 += 1
        count = max(count_1,count_2)
    return count

# 判断矩形框上是否有像素点并修改矩形框
def judge_rect_exist_pix(x,y,width,height,img):
    h,w = img.shape
    y1 = y
    x1 = x
    x2 = x + width-1
    y2 = y + height-1
    while True:
        flag = False
        for i in range(height):
            if img[y+i,x1] == 255:
                flag = True
                break
        if x1 == 0 or flag == False:
            break
        if flag == True:
            x1 = x1 -1
    while True:
        flag = False
        for i in range(height):
            if img[y+i,x2] == 255:
                flag = True
                break
        if x2 == w-1 or flag == False:
            break
        if flag == True:
            x2 = x2 + 1

    while True:
        flag = False
        freq = get_freq(x1,y1,x2-x1,img,type='h')
        for i in range(width):
            if img[y1,x+i] == 255:
                flag = True
                break
        if y1 == 0 or flag == False or freq <= 1:
            break
        if flag == True:
            y1 = y1-1
    while True:
        flag = False
        freq = get_freq(x1, y2, x2-x1, img, type='h')
        for i in range(width):
            if img[y2,x+i] == 255:
                flag = True
                break
        if y2 == h or flag == False or freq <= 1:
            break
        if flag == True:
            y2 = y2 + 1


    new_width = x2 - x1
    new_height = y2 - y1
    return  x1,y1,new_width,new_height

# 检测图片中某切片中白像素个数
def CountImgWhitePix(x,y,width,height,img):
    number = 0
    pic = img[y:y+height,x:x+width]
    for i in range(height):
        for j in range(width):
            if pic[i,j] == 255:
                number += 1
    return number

# 处理遗留层（主要是为了特殊的字符，如“一”）
def deal_leave_img(x,y,width,height,img):
    h,w = img.shape
    left = int(width / 2) if x- int(width/2) >=0 else x
    right = int(width / 2) if x+width+int(width/2) <= w else w-x-width
    number_1 = CountImgWhitePix(x-left,y,left, height,img)
    number_2 = CountImgWhitePix(x+width,y,right,height,img)
    if number_1 != 0 and number_2!=0:
        newx = x - left
        newwidth = width+left+right
    if number_1 == 0 and number_2 !=0:
        newx = x
        newwidth = width+right
    if number_1!=0 and number_2 == 0:
        newx = x - left
        newwidth = width+left
    if number_1 == 0 and number_2 == 0:
        newx = x
        newwidth = width
    return newx,y,newwidth,height


# 雨刷后处理操作
def YS_after(contours,img):
    for contour in contours:
        x, y, Width, Height = cv.boundingRect(contour)
        for i in range(y+1,y+Height):
            for j in range(x+1,x+Width):
                if img[i,j] == 0 :
                    left = j - x
                    right = x+Width - j
                    top = i - y
                    bottom = y+Height - i
                    flag1 = False
                    flag2 = False
                    flag3 = False
                    flag4 = False
                    for k in range(left):
                        if img[i,j-k] == 255:
                            flag1 = True
                            break
                    for k in range(right):
                        if img[i,j+k] == 255:
                            flag2 = True
                            break
                    for k in range(top):
                        if img[i-k,j] == 255:
                            flag3 = True
                            break
                    for k in range(bottom):
                        if img[i+k,j] == 255:
                            flag4 = True
                            break
                    if flag1 and flag2 and flag3 and flag4:
                        img[i,j] = 255
    return img

# 雨刷操作
def YS(img,ys_len):
    new_img_1 = np.zeros(shape=img.shape)
    new_img_2 = np.zeros(shape=img.shape)
    new_img_3 = np.zeros(shape=img.shape)
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i,j] == 255:
                if j + ys_len <= w:
                    new_img_1[i,j:j+ys_len] = 255
                elif j + ys_len > w:
                    new_img_1[i,j:w] = 255
                if j - ys_len >= 0:
                    new_img_2[i,j - ys_len:j] = 255
                else:
                    new_img_2[i,0:j] = 255
    for i in range(h):
        for j in range(w):
            if new_img_1[i,j] == 255 and new_img_2[i,j] == 255:
                new_img_3[i,j] = 255
    new_img_3 = np.array(new_img_3,dtype='uint8')
    contours, hierarchy = cv.findContours(new_img_3, mode=0, method=1)
    for contour in contours:
        x,y,Width,Height = cv.boundingRect(contour)
        number = CountImgWhitePix(x,y,Width,Height,img)
        if number == 0 :
            new_img_3[y:y+Height,x:x+Width] = 0
    contours, hierarchy = cv.findContours(new_img_3, mode=0, method=1)
    new_img_3 = YS_after(contours,new_img_3)

    # contours, hierarchy = cv.findContours(new_img_3, mode=1, method=1)
    # print(len(contours))
    # for  contour in contours:
    #     x, y, Width, Height = cv.boundingRect(contour)
    #     if new_img_3[y-1,x-1] == 255 and new_img_3[y+Height+1,x-1] == 255 and new_img_3[y-1,x+Width+1] == 255 and new_img_3[y+Height+1,x+Width+1] ==255:
    #         print("yes")
    #         new_img_3[y:y+Height,x:x+Width] = 255

    return new_img_3

# 去噪点
def delete_ZD(img):
    h, w = img.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            if img[i,j] == 255:
                if img[i,j-1] == 0 and img[i,j+1] ==0 and img[i+1,j] == 0 and img[i-1,j] == 0 and img[i-1,j-1] ==0 and img[i-1,j+1] ==0\
                        and img[i+1,j-1] == 0 and img[i+1,j+1] == 0:
                    img[i,j] = 0
                if img[i,j+1] != 0:
                    if j+1 !=w:
                        if img[i,j-1] == 0 and img[i+1,j] ==0 and img[i-1,j] ==0 and img[i-1,j+1] ==0 and img[i+1,j+1]==0 and img[i,j+2] ==0\
                                and img[i-1,j-1]== 0 and img[i+1,j-1]==0 and img[i-1,j+2]==0 and img[i+1,j+2] == 0:
                            img[i,j] = 0
                            img[i,j+1] =0
                    if j+1 == w:
                        if img[i, j - 1] == 0 and img[i + 1, j] == 0 and img[i - 1, j] == 0 and img[
                            i - 1, j + 1] == 0 and img[i + 1, j + 1] == 0 \
                                and img[i - 1, j - 1] == 0 and img[i + 1, j - 1] == 0 :
                            img[i, j] = 0
                            img[i, j + 1] = 0
                if img[i+1,j] !=0:
                    if i+1 !=h:
                        if img[i-1,j-1]==0 and img[i-1,j]==0 and img[i-1,j+1]==0 and img[i,j-1]==0 and img[i,j+1]==0 and img[i+1,j-1]==0\
                            and img[i+1,j+1]==0 and img[i+2,j-1]==0 and img[i+2,j] ==0 and img[i+2,j+1]==0:
                            img[i,j] = 0
                            img[i+1,j] = 0
                    if i+1 == h:
                        if img[i-1,j-1]==0 and img[i-1,j]==0 and img[i-1,j+1]==0 and img[i,j-1]==0 and img[i,j+1]==0 and img[i+1,j-1]==0\
                            and img[i+1,j+1]==0:
                            img[i,j] = 0
                            img[i+1,j] = 0
    return img

# 检测连通域边界上像素在原图是否连续
def detectContourPix(contour, img):
    h,w = img.shape
    flag = False
    rect = cv.boundingRect(contour)
    left_top_x = rect[0]
    left_top_y = rect[1]
    Width = rect[2]
    Height = rect[3]
    # print("x:",left_top_x,"y:",left_top_y,"width:",Width,"Height:",Height)
    white = np.ones(shape=img.shape) * 255
    number = 0
    pixlen = len(contour)
    for i in range(pixlen):
        x = contour[i][0][0]
        y = contour[i][0][1]
        if x != 0 and x!= w-1 and y!=0 and y!= h-1:
            if x == left_top_x:
                if img[y-1,x-1] == 0:
                    flag = True
                if img[y,x-1] == 0:
                    flag = True
                if img[y+1,x-1] == 0:
                    flag = True
            if x == left_top_x + Width -1:
                if img[y-1,x+1] == 0:
                    flag = True
                if img[y,x+1] == 0:
                    flag = True
                if img[y+1,x+1]==0:
                    flag = True
            if y == left_top_y:
                if img[y-1,x-1] == 0:
                    flag = True
                if img[y-1,x] == 0:
                    flag = True
                if img[y-1,x+1] == 0:
                    flag = True
            if y == left_top_y + Height -1:
                if img[y+1,x-1] == 0:
                    flag = True
                if img[y+1,x] == 0:
                    flag = True
                if img[y+1,x+1] == 0:
                    flag = True
    return flag

# 检测外接矩形边上黑像素的点是否连续
def detectBlackPix(contour,img):
    flag = False

    rect = cv.boundingRect(contour)
    left_top_x = rect[0]
    left_top_y = rect[1]
    Width = rect[2]
    Height = rect[3]
#     上边界
    for i in range(Width):
        if img[left_top_y ,left_top_x+i] == 0:
            if img[left_top_y-1, left_top_x+i - 1] == 0:
                flag = True
            if img[left_top_y - 1, left_top_x + i ] == 0:
                flag = True
            if img[left_top_y - 1, left_top_x + i +1] == 0:
                flag = True
    # 下边界
    for i in range(Width):
        if img[left_top_y+Height -1 ,left_top_x+i] == 0:
            if img[left_top_y +Height , left_top_x + i - 1] == 0:
                flag = True
            if img[left_top_y +Height, left_top_x + i] == 0:
                flag = True
            if img[left_top_y +Height, left_top_x + i + 1] == 0:
                flag = True
    # 左边界
    for i in range(Height):
        if img[left_top_y+i, left_top_x] == 0:
            if img[left_top_y +i - 1, left_top_x  - 1] == 0:
                flag = True
            if img[left_top_y +i, left_top_x  - 1] == 0:
                flag = True
            if img[left_top_y +i +1, left_top_x  - 1] == 0:
                flag = True
    # 右边界
    for i in range(Height):
        if img[left_top_y+i, left_top_x+Width-1] == 0:
            if img[left_top_y + i - 1, left_top_x +Width + 1] == 0:
                flag = True
            if img[left_top_y + i, left_top_x +Width + 1] == 0:
                flag = True
            if img[left_top_y + i + 1, left_top_x +Width + 1] == 0:
                flag = True
    return flag

# 判断该连通域是否是残缺的
def judgeContourComplete(contour, originImg):
    flag = False
    pixLen = len(contour)
    print(pixLen)
    rect = cv.boundingRect(contour)
    left_top_x = rect[0]
    left_top_y = rect[1]
    Width = rect[2]
    Height = rect[3]
    x = left_top_x - 1
    y = left_top_y - 1
    new_Width = Width + 2
    new_Height = Height + 2
    number_1 = CountImgBlackPix(left_top_x,left_top_y,Width,Height,originImg)
    number_2 = CountImgBlackPix(x,y,new_Width,new_Height,originImg)
    if number_1 == number_2:
        flag = True

    return flag

# 获取最小外接矩形的高宽比
def getHeightWidthRatio_(contour):
    rect = cv.minAreaRect(contour)
    Height,Width  = rect[1]
    if Width == 0 :
        Width = float("1e-8")
    ratio = Height / Width
    if ratio < 1:
        ratio = 1 / ratio
    return ratio

# 获取连通域外接矩形的高宽比
def getHeightWIdthRatio(contour):
    rect = cv.boundingRect(contour)
    Width = rect[2]
    Height = rect[3]
    ratio = Height / Width
    return ratio

# 获取连通域外界矩形的总像素数量、白色像素数量、黑色像素数量
def getPixRatio(img, contour):
    rect = cv.boundingRect(contour)
    left_top_x = rect[0]
    left_top_y = rect[1]
    Width = rect[2]
    Height = rect[3]

    rect_min = cv.minAreaRect(contour)
    h,w = rect_min[1]
    if w == 0:
        w =float("1e-8")
    area = h *w

    all_pix_number = Width * Height
    black_pix_number = 0
    white_pix_number = 0
    img_part = img[left_top_y : left_top_y + Height,left_top_x : left_top_x + Width]
    # cv.imwrite("img_part.jpg",img_part)
    for i in range(Height):
        for j in range(Width):
            if img_part[i][j] == 0:
                black_pix_number += 1
            else:
                white_pix_number += 1
    ratio = black_pix_number / all_pix_number

    return ratio,black_pix_number
# 获取所有连通域平均高度
def getAveCCsLen(contours):
    n = len(contours)
    L = 0
    for i in range(n):
        rect = cv.boundingRect(contours[i])
        W = rect[2]
        H = rect[3]
        if W > H:
            L += W
        else:
            L += H
    AveLen = L / n
    return AveLen

# 获取连通域长度分布
def getCCsLenDistribution(contours):
    dict = {}
    n = len(contours)
    for i in range(n):
        L = 0
        rect = cv.boundingRect(contours[i])
        W = rect[2]
        H = rect[3]
        if W > H:
            L = W
        else:
            L = H
        if L in dict:
            dict[L] += 1
        else:
            dict[L] = 1
    return sorted(dict.items(),key=lambda x:x[1], reverse=True)

# 连通域附近是否有其他连通域？
def judge_single_cc(contour,img):
    # new_contours = []
    h, w = img.shape
    left_top_x = cv.boundingRect(contour)[0]
    left_top_y = cv.boundingRect(contour)[1]
    Width = cv.boundingRect(contour)[2]
    Height = cv.boundingRect(contour)[3]
    L = Width if Width >= Height else Height
    number = CountImgWhitePix(left_top_x, left_top_y,Width,Height,img)
    number_ = CountImgWhitePix(left_top_x-L if left_top_x >= L else 0, left_top_y,3 * L if left_top_x + 2* L <= w else L + w - left_top_x, Height, img)
    flag = True if number == number_ else False


    return flag

# 获取线条层，黑底二值化图，白底二值化图
def get_line_img(img_file):
    img = cv.imread(img_file)
    h = img.shape[0]
    w = img.shape[1]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 获取二值化图像
    ret, binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    ret, binary_ = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 获取竖线
    min_len = int(h / 30) if h >= w else int(w / 30)
    structure1 = cv.getStructuringElement(0, (1, min_len))
    eroded = cv.erode(binary, structure1, iterations=1)
    dilatedrow = cv.dilate(eroded, structure1, iterations=1)
    # cv.imwrite('vertical_line.jpg', dilatedrow)
    # 获取横线
    structure2 = cv.getStructuringElement(0, (min_len, 1))
    eroded = cv.erode(binary, structure2, iterations=1)
    dilatedcol = cv.dilate(eroded, structure2, iterations=1)
    # cv.imwrite('horizontal_line.jpg', dilatedcol)
    line_img = 255 - (dilatedcol + dilatedrow)
    return line_img, binary, binary_
# 获取无长线条图、无线条图连接轮廓、连接轮廓的长度分布、连接轮廓的平均长度
def get(line_img, binary, binary_):
    no_line_img = line_img - binary_
    no_line_img_white = 255 + binary_ - line_img
    # 去噪处理
    no_line_img = delete_ZD(no_line_img)
    contours, hierarchy = cv.findContours(no_line_img, mode=0, method=1)

    AverageLen = getAveCCsLen(contours)
    CCLens = getCCsLenDistribution(contours)
    return no_line_img,contours,CCLens,AverageLen

# 初步过滤非文字部分
def filter_1(no_line_img,contours, CCLens, Avelen, binary, binary_):
    img = np.zeros(shape=binary.shape)
    leave = np.zeros(shape=binary.shape)
    for contour in contours:
        left_top_x = cv.boundingRect(contour)[0]
        left_top_y = cv.boundingRect(contour)[1]
        Width = cv.boundingRect(contour)[2]
        Height = cv.boundingRect(contour)[3]
        number = CountImgWhitePix(left_top_x,left_top_y,Width,Height,no_line_img)
        L = Width if Width>=Height else Height
        H, W = cv.minAreaRect(contour)[1]
        L2 = min(round(H),round(W))
        ratio = number / (H*W) if L2 !=0 else 1
        if L < Avelen * 1.5 and number != 0 and number != Width * Height and float(number / (Width * Height))<0.7 :
            if number != 0  and number != Width * Height and ratio < 0.9:
                img[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = no_line_img[
                                                                                 left_top_y: left_top_y + Height,
                                                                                  left_top_x: left_top_x + Width]
            else:
                leave[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = no_line_img[
                                                                                 left_top_y: left_top_y + Height,
                                                                                  left_top_x: left_top_x + Width]
        # else:
            # leave[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = no_line_img[
            #                                                                          left_top_y: left_top_y + Height,
            #                                                                          left_top_x: left_top_x + Width]

    img_1 = np.array(img,dtype='uint8')
    contours, hierarchy = cv.findContours(img_1, mode=0, method=1)
    for contour in contours:
        left_top_x = cv.boundingRect(contour)[0]
        left_top_y = cv.boundingRect(contour)[1]
        Width = cv.boundingRect(contour)[2]
        Height = cv.boundingRect(contour)[3]
        flag = judge_single_cc(contour,img_1)
        if flag == True:
            img[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = 0
    return img, leave

# 二次过滤非文字部分
def filter_2(img):
    contours, hierarchy = cv.findContours(img, mode=0, method=1)
    for contour in contours:
        H, W = cv.minAreaRect(contour)[1]
        x,y,Width,Height = cv.boundingRect(contour)
        number = CountImgWhitePix(x,y,Width,Height,img)
        ratio1 = number / (Width * Height)
        L2 = min(round(H), round(W))
        ratio2 = number / (H*W) if L2 !=0 else 1
        L = Width if Width >= Height else Height
        if L<6 or number <6 or ratio1>0.8 or ratio2>1:
            img[y:y+Height,x:x+Width] = 0
    img_1 = np.array(img, dtype='uint8')
    contours, hierarchy = cv.findContours(img_1, mode=0, method=1)
    for contour in contours:
        left_top_x = cv.boundingRect(contour)[0]
        left_top_y = cv.boundingRect(contour)[1]
        Width = cv.boundingRect(contour)[2]
        Height = cv.boundingRect(contour)[3]
        flag = judge_single_cc(contour, img_1)
        if flag == True:
            img[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = 0
    img_2 = np.array(img, dtype='uint8')
    cv.imwrite('t.jpg',img_2)


    contours, hierarchy = cv.findContours(img_2, mode=0, method=1)
    for contour in contours:
        left_top_x = cv.boundingRect(contour)[0]
        left_top_y = cv.boundingRect(contour)[1]
        Width = cv.boundingRect(contour)[2]
        Height = cv.boundingRect(contour)[3]
        flag = judge_single_cc(contour, img_2)
        if flag == True:
            img[left_top_y: left_top_y + Height, left_top_x: left_top_x + Width] = 0
    return img

def get_finally_text_img(img1, leave, no_line_img):
    h,w = no_line_img.shape
    text = np.zeros(shape=no_line_img.shape,dtype='uint8')
    ys = YS(img1,16)
    contours, hierarchy = cv.findContours(ys, mode=0, method=1)
    for contour in contours:
        x,y,Width,Height = cv.boundingRect(contour)
        newx,newy,newwidth,newheight = judge_rect_exist_pix(x,y,Width,Height,no_line_img)
        newx,newy,newwidth,newheight = deal_leave_img(newx,newy,newwidth,newheight,leave)
        newx, newy, newwidth, newheight = deal_leave_img(newx, newy, newwidth, newheight, leave2)
        # newx,newy,newwidth,newheight = judge_rect_exist_pix(newx,newy,newwidth,newheight,no_line_img)
        text[newy:newy+newheight,newx:newx+newwidth] = no_line_img[newy:newy+newheight,newx:newx+newwidth]
        # list_rect = [0,0,0,0]
        # list_rect[0] = newx
        # list_rect[1] = newy
        # list_rect[2] = newwidth
        # list_rect[3] = newheight
        # max_rect = tuple(list_rect)
        # image = cv.rectangle(no_line_img, max_rect, (255, 255), 1, 1, 0)
    # cv.imshow('0',text)
    # cv.waitKey(0)
    return text

def drawCCs(contours,CCsDict, AveLen,binary,binary_):
    # cv.imshow('0',binary_)
    white = np.ones(shape=binary.shape) * 255
    L1 = math.ceil(CCsDict[0][0] / 4)
    L2 = AveLen * 2
    print(L1, L2)
    n = len(contours)
    for i in range(n):
        rect = cv.boundingRect(contours[i])
        left_top_x = rect[0]
        left_top_y = rect[1]
        Width = rect[2]
        Height = rect[3]
        L_ = Width if Width > Height else Height
        # 长度限制
        if L_< L2 and L_>L1:
            # print(L_)
            ratio_1 = getHeightWIdthRatio(contours[i])
            ratio_2 = getHeightWidthRatio_(contours[i])
            ratio_3,black_pix_number = getPixRatio(binary_, contours[i])
            if (ratio_3 < 0.6) or (0.8<ratio_3 < 1):
                flag = detectContourPix(contours[i],binary_)
                # print("flag=", flag)
                if flag == True:
                    white[left_top_y : left_top_y + Height,left_top_x : left_top_x + Width] = binary_[left_top_y : left_top_y + Height,left_top_x : left_top_x + Width]
    cv.imshow('1',white)
    cv.waitKey(0)
    return white

def get_text_img(img_file):


    img = cv.imread(img_file)
    h = img.shape[0]
    w = img.shape[1]
    print(h,w)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 获取二值化图像
    ret, binary = cv.threshold(img_gray , 0, 255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    ret, binary_ = cv.threshold(img_gray , 0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)


    # cv.imwrite("2_white.jpg",binary_)
    # cv.imwrite("2_black.jpg",binary)


    # 获取竖线
    structure1 = cv.getStructuringElement(0,(1,100))
    eroded = cv.erode(binary,structure1,iterations=1)
    dilatedrow = cv.dilate(eroded,structure1,iterations=1)
    cv.imwrite('vertical_line.jpg',dilatedrow)

    # 获取横线
    structure2 = cv.getStructuringElement(0,(100,1))
    eroded = cv.erode(binary,structure2, iterations=1)
    dilatedcol = cv.dilate(eroded, structure2,iterations=1)
    cv.imwrite('horizontal_line.jpg',dilatedcol)

    # 识别交点
    # bitwiseAnd = cv.bitwise_and(dilatedcol,dilatedrow)
    # cv.imshow("bitwiseAnd Image",bitwiseAnd)
    # cv.waitKey(0)



    line_img = 255-(dilatedcol + dilatedrow)
    # 去除横线与竖线
    new_img = 255 + binary_ - line_img
    new_img_black = 255 - new_img
    new_img_black = delete_ZD(new_img_black)


# 腐蚀膨胀cc
    structure3 = cv.getStructuringElement(1, (3,3))
    cc = cv.dilate(new_img_black,structure3,iterations=1)





    contours, hierarchy = cv.findContours(cc,mode=0,method=1)
    print(len(contours))
    AverageLen = getAveCCsLen(contours)
    CCLens = getCCsLenDistribution(contours)
    print("AveLen:",AverageLen)
    print('CClens:',CCLens)

    white_img = drawCCs(contours, CCLens, AverageLen,binary, binary_)
    # cv.imwrite('train_text.jpg',white_img)
    return white_img

# def get_text_data(img):

# 最后一步：相邻高度相同

if __name__ == '__main__':
    # 输入你想分解的图片
    line_img, binary, binary_ = get_line_img('.//img/test_50.jpg')
    h,w = line_img.shape

    no_line_img,contours,CCLens,AverageLen = get(line_img, binary, binary_)
    cv.imwrite('no_lines.jpg',no_line_img)

    img,leave = filter_1(no_line_img,contours,CCLens,AverageLen,binary,binary_)
    # cv.imwrite('compare.jpg',img)
    cv.imwrite('leave.jpg',leave)

    newimg = YS(img,24)
    cv.imwrite('compare_YS.jpg',newimg)
    #开处理操作
    kernel = cv.getStructuringElement(0,(5,5))
    open = cv.morphologyEx(newimg, cv.MORPH_OPEN,kernel=kernel)
    cv.imwrite('compare_YS_open.jpg',open)
    leave2 =  newimg - open
    cv.imwrite('leave2.jpg',leave2)
    #
    text_img = np.zeros(shape=open.shape,dtype='uint8')
    contours, hierarchy = cv.findContours(open,mode=0,method=1)
    Avelen = getAveCCsLen(contours)
    cclen = getCCsLenDistribution(contours)
    # print(Avelen)
    # print(cclen)
    for contour in contours:
        x,y,Width,Height = cv.boundingRect(contour)
        L = Width if Width >= Height else Height
        if L>(Avelen / 3 ):
            top = y if y >=0 else 0
            bottom = y+Height if y+Height <= h else h
            left = x if x >=0 else 0
            right = x+Width if x+Width <= w else w
            text_img[top:bottom,left:right] = no_line_img[top:bottom,left:right]
    cv.imwrite('text_img.jpg',text_img)
    text_img_2 = filter_2(text_img)
    cv.imwrite("text_img_2.jpg",text_img_2)

    text = get_finally_text_img(text_img_2, leave, no_line_img)
    # cv.imwrite('text_img_2.jpg',text_img_2)
    text_img_f = 255 - text
    cv.imwrite('text_img_final.jpg',text_img_f)

    element_img = 255 - (no_line_img - text)
    cv.imwrite('element.jpg',element_img)

    no_text_img = binary



