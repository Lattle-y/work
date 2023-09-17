import random
import os
import cv2 as cv
import numpy as np
import copy
from PIL import ImageFont,ImageDraw,Image
# 自己存放图形符号的位置
path = 'C:/Users/92090/Desktop/image_preHandle/img/element'
byq_path = path + '/byq'
byq_list = os.listdir(byq_path)
byq=[]
for i,name in enumerate(byq_list):
    if name == '3_148x145.jpg':
        file_name = byq_path + '/' + name
        byq_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        byq.append((byq_element.shape[0],byq_element.shape[1],byq_element))

dr_path = path + '/dr'
dr_list = os.listdir(dr_path)
dr=[]
for i,name in enumerate(dr_list):
    if name == '64x104.jpg':
        file_name = dr_path + '/' + name
        dr_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        dr.append((dr_element.shape[0],dr_element.shape[1],dr_element))

dg_path = path + '/dg'
dg_list = os.listdir(dg_path)
dg=[]
for i,name in enumerate(dg_list):
    if name == '54x110.jpg':
        file_name = dg_path + '/' + name
        dg_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        dg.append((dg_element.shape[0],dg_element.shape[1],dg_element))


fdj_path = path + '/fdj'
fdj_list = os.listdir(fdj_path)
fdj=[]
for i,name in enumerate(fdj_list):
    if name == '130x143.jpg':
        file_name = fdj_path + '/' + name
        fdj_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        fdj.append((fdj_element.shape[0],fdj_element.shape[1],fdj_element))

jd_path = path + '/jd'
jd_list = os.listdir(jd_path)
jd=[]
for i,name in enumerate(jd_list):
    if name == '131x63.jpg':
        file_name = jd_path + '/' + name
        jd_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        jd.append((jd_element.shape[0],jd_element.shape[1],jd_element))

Disconnector_path = path + '/Disconnector'
Disconnector_list = os.listdir(Disconnector_path)
Disconnector=[]
for i,name in enumerate(Disconnector_list):
    if name == '51x113.jpg':
        file_name = Disconnector_path + '/' + name
        Disconnector_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        Disconnector.append((Disconnector_element.shape[0],Disconnector_element.shape[1],Disconnector_element))

jdkg_path = path + '/jdkg'
jdkg_list = os.listdir(jdkg_path)
jdkg=[]
for i,name in enumerate(jdkg_list):
    if name == '48x120.jpg':
        file_name = jdkg_path + '/' + name
        jdkg_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        jdkg.append((jdkg_element.shape[0],jdkg_element.shape[1],jdkg_element))

glkg_path = path + '/glkg'
glkg_list = os.listdir(glkg_path)
glkg=[]
for i,name in enumerate(glkg_list):
    if name == '75x167.jpg':
        file_name = glkg_path + '/' + name
        glkg_element = cv.imread(file_name,cv.IMREAD_GRAYSCALE)
        glkg.append((glkg_element.shape[0],glkg_element.shape[1],glkg_element))
#
# glkg1 = cv.imread('./element/glkg/58x94.jpg', cv.IMREAD_GRAYSCALE)
# glkg2 = cv.imread('./element/glkg/2.jpg', cv.IMREAD_GRAYSCALE)
# glkg = [(glkg1.shape[0], glkg1.shape[1],glkg1), (glkg2.shape[0], glkg2.shape[1],glkg2)]
#
# Disconnector1 = cv.imread('./element/Disconnector/64x104.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector2 = cv.imread('./element/Disconnector/2.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector3 = cv.imread('./element/Disconnector/3.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector4 = cv.imread('element/Disconnector/47x91.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector5 = cv.imread('element/Disconnector/44x89.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector6 = cv.imread('./element/Disconnector/6.jpg', cv.IMREAD_GRAYSCALE)
# Disconnector = [(Disconnector1.shape[0], Disconnector1.shape[1],Disconnector1), (Disconnector2.shape[0], Disconnector2.shape[1],Disconnector2),
#                 (Disconnector3.shape[0], Disconnector3.shape[1], Disconnector3), (Disconnector4.shape[0], Disconnector4.shape[1],Disconnector4),
#                 (Disconnector5.shape[0], Disconnector5.shape[1],Disconnector5), (Disconnector6.shape[0], Disconnector6.shape[1],Disconnector6)]



# img = np.ones(shape=(1200,1200),dtype='uint8') * 255

count = 0
# y = np.random.randint(0,1200)
# x = np.random.randint(0,1200)
def gama_transfer(img,power1):
    if len(img.shape) == 3:
        img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = 255*np.power(img/255,power1)
        img = np.around(img)
        img[img>255] = 255
        out_img = img.astype(np.uint8)
    else:
        img = 255 * np.power(img / 255, power1)
        img = np.around(img)
        img[img > 255] = 255
        out_img = img.astype(np.uint8)
    return out_img
def detect_white(y, x, h, w, img):
    flag = False
    if y + h  >=img.shape[0] or x + w >= img.shape[1] or y <0 or x <0:
        flag = True
    else:
        for i in range(h):
            for j in range(w):
                if img[y + i, x + j]  != 255:
                    flag = True
                    break
            if flag == True:
                break
    return flag
def deal_element(element,r):
    if np.random.randint(0,9)<5:
        element = cv.flip(element,np.random.randint(0,3) - 1)
    if np. random.randint(0,9)<5:
        if random.randint(0,3) == 0:
            element = cv.rotate(element,cv.ROTATE_90_CLOCKWISE)
        if random.randint(0,3) == 1:
            element = cv.rotate(element,cv.ROTATE_90_COUNTERCLOCKWISE)
        if random.randint(0, 3) == 2:
            element = cv.rotate(element,cv.ROTATE_180)
    if np. random.randint(0,9)<5:
        for i in range(np.random.randint(0,4)):
            element = cv.transpose(element)
    r = np.random.randint(90,110) / 100
    element = cv.resize(element,(0,0),fx=1,fy=1,interpolation=cv.INTER_AREA)
    element = gama_transfer(element,np.random.randint(180,220)/100)
    return element , element.shape[0], element.shape[1]
def draw_line_down(orientation, y, x, h, w, img):
    line_len = np. random.randint(25, 35)
    if orientation == 'v':
        flag = detect_white(y + h,x , line_len, w, img)
        if flag == False:
            for i in range(w):
                if img[y + h - 1 , x + i] <=  100:
                    line_c = random.randint(0, 100)
                    for j in range(line_len):
                        img[y + h - 1  + j, x + i] = line_c
    if orientation == 'h':
        flag = detect_white(y , x +w, h, line_len, img)
        if flag == False:
            for i in range(h):
                if img[y + i, x + w -1] <= 100 :
                    line_c = random.randint(0, 100)
                    for j in range(line_len):
                        img[y + i, x + w - 1 + j] = line_c
    return img,line_len, flag
def draw_line_up(orientation, y, x, h, w, img):
    line_len = np.random.randint(25, 35)
    if orientation == 'v':
        flag = detect_white(y - line_len, x, line_len, w, img)
        if flag == False:
            for i in range(w):
                if img[y , x + i] <= 100:
                    line_c = random.randint(0, 100)
                    for j in range(line_len):
                        img[y - 1 - j, x + i] = line_c
    if orientation == 'h':
        flag = detect_white(y , x - line_len, h, line_len, img)
        if flag == False:
            for i in range(h):
                if img[y + i, x   ] <= 100:
                    line_c = random.randint(0, 100)
                    for j in range(line_len):
                        img[y + i, x - 1 - j] = line_c
    return img, line_len, flag
def draw_disconnector(y, x, img,r):
    record = []
    element_info = Disconnector[np.random.randint(0,len(Disconnector))]
    h , w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element,dtype='uint8')
    orientation = 'h' if h <= w else 'v'
    flag = detect_white(y, x, h, w, img)
    if flag == False :
        img[y : y + h, x: x + w] = element
        record = [(x,y,w,h),'disconnector']
    return img,orientation,h,w,flag,record
def draw_dr(orientation,y, x, img,r):
    record = []
    element_info = dr[np.random.randint(0,len(dr))]
    h , w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element,dtype='uint8')
    dr_orientation = 'h' if h <= w else 'v'
    if orientation == dr_orientation and dr_orientation == 'v':
        flag = detect_white(y, x - int(w / 2), h , w, img)
        if flag == False:
            img[y : y + h, x - int(w / 2) : x - int(w / 2) + w] = element
            record = [(x - int(w/2), y, w, h),'dr']
    elif orientation == dr_orientation and dr_orientation == 'h':
        flag = detect_white(y - int(h / 2), x, h, w, img)
        if flag == False:
            img[y - int(h / 2): y - int(h / 2) + h, x : x  + w] = element
            record = [(x, y-int(h/2),w,h),'dr']
    else:
        flag = True
    return img,flag,record
def draw_dg(orientation,y,x,img,r):
    record = []
    element_info = dg[np.random.randint(0,len(dg))]
    h , w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element,dtype='uint8')
    dg_orientation = 'h' if h <= w else 'v'
    if orientation == dg_orientation and dg_orientation == 'v':
        flag = detect_white(y, x - int(w / 2), h , w, img)
        if flag == False:
            img[y : y + h, x - int(w / 2) : x - int(w / 2) + w] = element
            record = [(x - int(w / 2),y,w,h),'dg']
    elif orientation == dg_orientation and dg_orientation == 'h':
        flag = detect_white(y - int(h / 2), x, h, w, img)
        if flag == False:
            img[y - int(h / 2): y - int(h / 2) + h, x : x  + w] = element
            record = [(x,y-int(h/2),w,h),'dg']
    else:
        flag = True
    return img,flag,record
def draw_glkg_down(orientation,y,x,img,r):
    record = []
    element_info = glkg[np.random.randint(0,len(glkg))]
    h , w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element,dtype='uint8')
    glkg_orientation = 'h' if h <= w else 'v'
    if orientation == glkg_orientation and glkg_orientation == 'v':
        flag = detect_white(y, x - int(w / 2), h , w, img)
        if flag == False:
            img[y : y + h, x - int(w / 2) : x - int(w / 2) + w] = element
            record = [(x - int(w / 2),y,w,h),'glkg']
    elif orientation == glkg_orientation and glkg_orientation == 'h':
        flag = detect_white(y - int(h / 2), x, h, w, img)
        if flag == False:
            img[y - int(h / 2): y - int(h / 2) + h, x : x  + w] = element
            record = [(x,y - int(h / 2),w,h),'glkg']
    else:
        flag = True
    return img,flag,record
def draw_glkg_up(orientation,y,x,img,r):
    record = []
    element_info = glkg[np.random.randint(0, len(glkg))]
    h, w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element, dtype='uint8')
    glkg_orientation = 'h' if h <= w else 'v'
    if orientation == glkg_orientation and glkg_orientation == 'v':
        flag = detect_white(y - h, x - int(w / 2), h, w, img)
        if flag == False:
            img[y - h : y , x - int(w / 2) : x - int(w / 2) + w] = element
            record = [(x - int(w / 2),y-h,w,h),'glkg']
    elif orientation == glkg_orientation and glkg_orientation == 'h':
        flag = detect_white(y - int(h / 2), x - w, h, w, img)
        if flag == False:
            img[y - int(h / 2): y - int(h/2) + h, x - w: x ] = element
            record = [(x-w,y - int(h / 2),w,h),'glkg']
    else:
        flag = True
    return img,flag,record
def draw_jd(y, x, img,r):
    record = []
    element_info = jd[np.random.randint(0, len(jd))]
    h, w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element, dtype='uint8')
    orientation = 'h' if h <= w else 'v'
    flag = detect_white(y, x, h, w, img)
    if flag == False:
        img[y: y + h, x: x + w] = element
        record = [(x,y,w,h),'jd']
    return img, orientation, h, w, flag,record
def draw_jdkg(y, x, img,r):
    record = []
    element_info = jdkg[np.random.randint(0, len(jdkg))]
    h, w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element, dtype='uint8')
    orientation = 'h' if h <= w else 'v'
    flag = detect_white(y, x, h, w, img)
    if flag == False:
        img[y: y + h, x: x + w] = element
        record = [(x,y,w,h),'jdkg']
    return img, orientation, h, w, flag,record
def draw_byq(y,x,img,r):
    record = []
    element_info = byq[np.random.randint(0, len(byq))]
    h, w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element, dtype='uint8')
    orientation = 'h' if h <= w else 'v'
    flag = detect_white(y, x, h, w, img)
    if flag == False:
        img[y: y + h, x: x + w] = element
        record = [(x,y,w,h),'byq']
    return img, orientation, h, w, flag,record
def draw_fdj(y,x,img,r):
    record = []
    element_info = fdj[np.random.randint(0, len(fdj))]
    h, w, element = element_info
    element, h, w = deal_element(element,r)
    element = np.array(element, dtype='uint8')
    orientation = 'h' if h <= w else 'v'
    flag = detect_white(y, x, h, w, img)
    if flag == False:
        img[y: y + h, x: x + w] = element
        record = [(x,y, w,h),'fdj']
    return img, orientation, h, w, flag,record
def draw_struct_1(before_img,r, *args):
    if not args:
        flag = True
        while flag == True:
            elements_list = []
            y = np.random.randint(0, 1200)
            x = np.random.randint(0, 1200)
            img = copy.deepcopy(before_img)
            img, orientation, h, w, flag,record = draw_disconnector(y, x, img, r)
            if flag == True:
                continue
            else:
                elements_list.append(record)
            img,line_len, flag = draw_line_down(orientation, y, x, h, w, img)
            if flag == True:
                continue
            if orientation == 'v':
                if np.random.randint(0,9) < 3:
                    img,flag,record = draw_dr(orientation ,y + h - 1 + line_len, x + int(w / 2), img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                elif np.random.randint(0,9) > 5:
                    img,flag,record = draw_dg(orientation ,y + h - 1 + line_len, x + int(w / 2), img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                else:
                    img,flag,record = draw_glkg_down(orientation ,y + h - 1 + line_len, x + int(w / 2), img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                img,line_len, flag = draw_line_up(orientation, y, x, h, w, img)
                if flag == True:
                    continue
                img,flag,record = draw_glkg_up(orientation, y - line_len , x + int(w / 2), img ,r)
                if flag == True:
                    continue
                else:
                    elements_list.append(record)
            if orientation == 'h':
                if np.random.randint(0, 9) < 3:
                    img, flag, record = draw_dr(orientation, y + int(h / 2), x + w - 1 + line_len, img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                elif np.random.randint(0, 9) > 5:
                    img, flag,record = draw_dg(orientation, y + int(h / 2), x + w - 1 + line_len, img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                else:
                    img, flag,record = draw_glkg_down(orientation, y + + int(h / 2), x + w - 1 + line_len, img,r)
                    if flag == True:
                        continue
                    else:
                        elements_list.append(record)
                img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
                if flag == True:
                    continue
                img, flag, record = draw_glkg_up(orientation, y + int(h / 2), x - line_len, img,r)
                if flag == True:
                    continue
                else:
                    elements_list.append(record)
    if args:
        elements_list = []
        y = args[0]
        x = args[1]
        img = copy.deepcopy(before_img)
        img, orientation, h, w, flag, record = draw_disconnector(y, x, img,r)
        if flag == False:
            elements_list.append(record)
        img, line_len, flag = draw_line_down(orientation, y, x, h, w, img)
        if orientation == 'v':
            if np.random.randint(0, 9) < 3:
                img, flag, record = draw_dr(orientation, y + h - 1 + line_len, x + int(w / 2), img,r)
                if flag == False:
                    elements_list.append(record)
            elif np.random.randint(0, 9) > 5:
                img, flag, record = draw_dg(orientation, y + h - 1 + line_len, x + int(w / 2), img,r)
                if flag == False:
                    elements_list.append(record)
            else:
                img, flag, record = draw_glkg_down(orientation, y + h - 1 + line_len, x + int(w / 2), img,r)
                if flag == False:
                    elements_list.append(record)
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, flag, record = draw_glkg_up(orientation, y - line_len, x + int(w / 2), img,r)
            if flag == False:
                elements_list.append(record)
        if orientation == 'h':
            if np.random.randint(0, 9) < 3:
                img, flag, record = draw_dr(orientation, y + int(h / 2) , x + w - 1 + line_len, img,r)
                if flag == False:
                    elements_list.append(record)
            elif np.random.randint(0, 9) > 5:
                img, flag, record = draw_dg(orientation, y + int(h / 2) , x + w - 1 + line_len, img,r)
                if flag == False:
                    elements_list.append(record)
            else:
                img, flag, record = draw_glkg_down(orientation, y + + int(h / 2) , x + w - 1 + line_len, img,r)
                if flag == False:
                    elements_list.append(record)
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, flag, record = draw_glkg_up(orientation, y + int(h / 2), x - line_len, img,r)
            if flag == False:
                elements_list.append(record)
    return img,y,x,orientation,elements_list
def draw_struct_2(before_img,r):
    flag = True
    while flag == True:
        element_list = []
        y = np.random.randint(0, 1200)
        x = np.random.randint(0, 1200)
        img = copy.deepcopy(before_img)
        img, orientation, h, w, flag, record = draw_jd(y, x, img,r)
        if flag == True:
            continue
        else:
            element_list.append(record)
        if random.randint(0,100)<50:
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, line_len, flag = draw_line_down(orientation, y, x, h, w, img)
    return img,element_list
def draw_struct_3(before_img,r):
    flag = True
    while flag == True:
        elements_list = []
        y = np.random.randint(0, 1200)
        x = np.random.randint(0, 1200)
        img = copy.deepcopy(before_img)
        img, orientation, h, w, flag, record = draw_byq(y, x, img,r)
        if flag == True:
            continue
        else:
            elements_list.append(record)
        if random.randint(0,100)<50:
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, line_len, flag = draw_line_down(orientation, y, x, h, w, img)
    return img,elements_list
def draw_struct_4(before_img,r):
    flag = True
    while flag == True:
        elements_list = []
        y = np.random.randint(0, 1200)
        x = np.random.randint(0, 1200)
        img = copy.deepcopy(before_img)
        img, orientation, h, w, flag, record = draw_fdj(y, x, img,r)
        if flag == True:
            continue
        else:
            elements_list.append(record)
        if random.randint(0,100)<50:
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, line_len, flag = draw_line_down(orientation, y, x, h, w, img)
    return img,elements_list
def draw_struct_5(before_img,r):
    flag = True
    while flag == True:
        elements_list = []
        y = np.random.randint(0, 1200)
        x = np.random.randint(0, 1200)
        img = copy.deepcopy(before_img)
        img, orientation, h, w, flag, record = draw_jdkg(y, x, img,r)
        if flag == True:
            continue
        else:
            elements_list.append(record)
        if random.randint(0,100)<50:
            img, line_len, flag = draw_line_up(orientation, y, x, h, w, img)
            img, line_len, flag = draw_line_down(orientation, y, x, h, w, img)
    return img,elements_list
def draw_random_line(img):

    orientation = 'v' if np.random.randint(0,10)<5 else 'h'
    line_len = np.random.randint(100,500)
    flag = True
    while flag == True:
        if orientation == 'v':
            y = np.random.randint(0, 1200)
            x = np.random.randint(0, 1200)
            flag = detect_white(y,x,line_len,2,img)
            if flag== False:
                for i in range(line_len):
                    img[y+i,x] = 0
                if np.random.randint(0,2) == 1:
                    for i in range(line_len):
                        img[y + i, x + 1] = random.randint(0,100)
                    if np.random.randint(0,2) == 1:
                        for i in range(line_len):
                            img[y + i, x + 2] = random.randint(0,100)
        if orientation == 'h':
            y = np.random.randint(0, 1200)
            x = np.random.randint(0, 1200)
            flag = detect_white(y, x, 2, line_len, img)
            if flag == False:
                for i in range(line_len):
                    img[y , x+i] = 0
                if np.random.randint(0,2) == 1:
                    for i in range(line_len):
                        img[y + 1, x + i] = random.randint(0,100)
                    if np.random.randint(0,2) == 1:
                        for i in range(line_len):
                            img[y + 2, x + i] = random.randint(0,100)
    return img
def GBK3212():
    head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xfe)
    val1 = f'{head:x} {body:x}'
    head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xfe)
    val2 = f'{head:x} {body:x}'
    str1 = bytes.fromhex(val1).decode('gb18030')
    str2 = bytes.fromhex(val2).decode('gb18030')
    p1 = random.randint(0,100)/100
    if p1<0.2:
        char1 = str1 + str2 + "线"
    elif 0.2<=p1<0.4:
        char1 = str1 + str2 + "一" if p1 <0.5 else str1 + str2 + "I"
    elif 0.4<=p1<0.6:
        char1 = str1 + str2 + "二" if p1 <0.5 else str1 + str2 + "II"
    elif 0.6<=p1<0.8:
        char1 = str1 + str2 + "1"
    else:
        char1 = str1 + str2 + "2"
    char2 = str1 + str2 + "I线" if p1 <0.5 else str1 + str2 + "II线"
    char3 = "IA母线" if p1 <0.5 else "IB母线"
    char4 = "#" + str(random.randint(1,10)) + "变" if p1 <0.5 else "#" + str(random.randint(1,10))
    char5 = str(random.randint(10,10000))
    char6 = "容抗" if p1 <0.5 else "感抗"
    p2 = random.randint(0,100)/100
    if p2 <0.2:
        char = char1
    elif 0.2<=p2<0.4:
        char = char2
    elif 0.4<=p2<0.6:
        char = char3
    elif 0.6<=p2<0.7:
        char = char4
    elif 0.7<=p2<0.8:
        char = char6
    else:
        char = char5
    return char
def add_chinese(img,x,y):
    font_path = './simhei.ttf'
    font = ImageFont.truetype(font_path, random.randint(10,35))
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x,y), GBK3212(), font = font, fill=(0))
    img = np.array(img_pil)

    return img

img = np.ones(shape=(1200, 1200), dtype='uint8') * 255





def draw_img():
    img = np.ones(shape=(1200, 1200), dtype='uint8') * 255
    elements_info = []
    r = np.random.randint(80,120) / 100
    print(r)
    if np. random.randint(0,1000) < 1950:
        for i in range(np.random.randint(2,4)):
            img, y, x, orientation,elements_list = draw_struct_1(img,r)
            elements_info.append(elements_list)
            if orientation == 'v':
                if np.random.randint(0, 10) < 9:
                    x = x + np.random.randint(50, 200)
                    img, y, x, _,elements_list = draw_struct_1(img,r, y, x)
                    elements_info.append(elements_list)
                    if np.random.randint(0, 10) < 7:
                        x = x + np.random.randint(100, 200)
                        img, y, x, _,elements_list = draw_struct_1(img,r, y, x)
                        elements_info.append(elements_list)
                        if np.random.randint(0, 10) < 6:
                            x = x + np.random.randint(100, 200)
                            img, y, x, _,elements_list = draw_struct_1(img,r, y, x)
                            elements_info.append(elements_list)
                            if np.random.randint(0, 10) < 5:
                                x = x + np.random.randint(100, 200)
                                img, y, x, _,elements_list = draw_struct_1(img,r, y, x)
                                elements_info.append(elements_list)
                                if np.random.randint(0, 10) < 2:
                                    x = x + np.random.randint(100, 200)
                                    img, y, x, _,elements_list = draw_struct_1(img,r, y, x)
                                    elements_info.append(elements_list)
            if orientation == 'h':
                if np.random.randint(0, 10) < 9:
                    y = y + np.random.randint(100, 200)
                    img, y, x, _, elements_list = draw_struct_1(img,r, y, x)
                    elements_info.append(elements_list)
                    if np.random.randint(0, 10) < 7:
                        y = y + np.random.randint(100, 200)
                        img, y, x, _, elements_list = draw_struct_1(img,r, y, x)
                        elements_info.append(elements_list)
                        if np.random.randint(0, 10) < 5:
                            y = y + np.random.randint(100, 200)
                            img, y, x, _, elements_list = draw_struct_1(img,r, y, x)
                            elements_info.append(elements_list)
                            if np.random.randint(0, 10) < 3:
                                y = y + np.random.randint(100, 200)
                                img, y, x, _, elements_list = draw_struct_1(img,r, y, x)
                                elements_info.append(elements_list)
                                if np.random.randint(0, 10) < 1:
                                    y = y + np.random.randint(100, 200)
                                    img, y, x, _, elements_list = draw_struct_1(img,r, y, x)
                                    elements_info.append(elements_list)
    if np. random.randint(0,100) < 190:
        for i in range(np.random.randint(4,6)):
            img,elements_list = draw_struct_2(img,r)
            elements_info.append(elements_list)
    if np.random.randint(0,100) < 180:
        for i in range(2):
            img,elements_list = draw_struct_3(img,r)
            elements_info.append(elements_list)
    if np.random.randint(0,100)<150:
        for i in range(2):
            img,elements_list = draw_struct_4(img,r)
            elements_info.append(elements_list)
    if np.random.randint(0,100)<910:
        for i in range(np.random.randint(1,2)):
            img,elements_list = draw_struct_5(img,r)
            elements_info.append(elements_list)
    for i in range(np.random.randint(0,3)):
        img = draw_random_line(img)

    for i in range(1,1199):
        for j in range(1,1199):
            if img[i,j] == 0 :
                if np.random.randint(0,10000)<10:
                    img[i,j] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i - 1, j] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i + 1, j] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i, j - 1] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i, j + 1] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i + 1, j + 1] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i + 1, j - 1] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i - 1, j + 1] = 255
                    if np.random.randint(0, 100) < 50:
                        img[i - 1, j - 1] = 255
            if img[i,j] == 255 and i-1>0 and j - 1>0 and i+1<1199 and j+1<1199:
                if np. random.randint(0,1000000)< 10:
                    img[i,j] = 0
                    if np.random.randint(0,100)<50:
                        img[i-1,j] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i+1,j] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i,j-1] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i,j+1] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i+1,j+1] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i+1,j-1] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i-1,j+1] = 0
                    if np.random.randint(0, 100) < 50:
                        img[i-1,j-1] = 0

    # count = 0
    # while count < 10:
    #     x = np.random.randint(1200)
    #     y = np.random.randint(1200)
    #     flag = detect_white(y, x, 30, 60, img)
    #     if flag == False:
    #         img = add_chinese(img, x, y)
    #         count += 1

    elements_info = [x for i in elements_info for x in i]



    return  img, elements_info
def write_file(elements_info,filename):
    with open(filename,'w') as f:
        for element in elements_info:
            x, y, w, h = element[0]
            element_name = element[1]
            if element_name == 'disconnector':
                id = 0
            elif element_name == 'dr':
                id = 1
            elif element_name == 'dg':
                id = 2
            elif element_name == 'byq':
                id = 3
            elif element_name == 'fdj':
                id = 4
            elif element_name == 'glkg':
                id = 5
            elif element_name == 'jd':
                id = 6
            elif element_name == 'jdkg':
                id = 7
            line = str(id) + ' ' + str(x + w/2) + ' ' + str(y + h/2) + ' ' + str(w) + ' ' + str(h)  + '\n'
            f.write(line)
    f.close()


# img, elements_info = draw_img()
#
# cv.imwrite('test_0.jpg',img)
# write_file(elements_info,'./test_0.txt')

for i in range(1):
    img, elements_info = draw_img()
    # ret, binary_ = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    dir_img = 'C:/Users/92090/Desktop/image_preHandle/img/element/data/images_c' + '/test_' + str(i+1) + '.jpg'
    cv.imwrite(dir_img,img)
    dir_txt = 'C:/Users/92090/Desktop/image_preHandle/img/element/data/labels_c' + '/test_' + str(i+1) + '.txt'
    print(dir_txt)
    write_file(elements_info,dir_txt)








