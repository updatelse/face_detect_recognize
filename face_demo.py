#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import copy
import random
import requests
import numpy as np
from numpy import random
from config.scrfd import SCRFD
from config.pipnet import LandmarkDetect
from config.onnx_helper import ArcFaceORT
from PIL import Image, ImageDraw, ImageFont
from config.onnx_ijbc import AlignedDataSet,extract

class Face_Detector_Recognize(object):
    def __init__(self, fd_weights="./weights/scrfd_34g_shape640x640.onnx",ld_weights="./weights/LandmarkDetect.pth",ld_config="./config/pip_32_16_60_r18_l2_l1_10_1_nb10.py",fr_weights="./weights/face_recog", database_folder="./database_folder", face_name_file="./config/face_name_ID.names", imgsz=[640,640], similarity_score_thres=0.25, device="0" ):
        self.fd_weights = fd_weights                                           # face detector weights
        self.ld_weights = ld_weights                                           # face landmark weights
        self.ld_config = ld_config                                             # face landmark config
        self.fr_weights = fr_weights                                           # face recognition weights
        self.database_folder = database_folder                                 # face database directory
        self.face_name_file = face_name_file                                   # face name ID index
        self.imgsz = imgsz                                                     # preprocess image_size
        self.similarity_score_thres = similarity_score_thres                   # face similarity score threshold
        self.device = device                                                   # device, i.e. 0 or 1 or cpu
        self.facedetector = SCRFD(self.fd_weights)                             # detector model init
        self.facedetector.prepare(1)                                           # detector model prepare
        self.land_detector = LandmarkDetect(self.ld_config, self.ld_weights)   # landmark model init
        self.aligndataset = AlignedDataSet()                                   # AlignedDataSet class init
        self.recog_model = ArcFaceORT(model_path = self.fr_weights)            # recognition model init
        self.recog_model.check()                                               # recognition model check
    
    def load_image(self, image_path):
        
        if image_path.startswith('http://') or image_path.startswith('https://'):  #URL路径
            response = requests.get(image_path)
            response_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(response_array, cv2.IMREAD_COLOR)
            image_name = image_path.split("/")[-1].split(".")[0]
        else:
            img = cv2.imread(image_path)                                           #本地路径
            image_name = image_path.split("/")[-1].split(".")[0]

        return img, image_name
    
    def load_database_face(self, delete_flag=False, delete_face_ID=None):

        
        Name_List, Face_Info_List, ID_dict= [], [], {}                         # 返回人脸库中人脸名和人脸特征信息
        
        if delete_flag == False:                                               # 不是删除操作
            for i in os.listdir(self.database_folder):
                name = i.split(".npy")[0]
                feature = np.load(os.path.join(self.database_folder, i))
                Name_List.append(name)                                         # 存文件名, 是ID
                Face_Info_List.append(feature)                                 # 存特征向量信息
            with open(self.face_name_file, 'r') as f2:                         # 打开表单, 读取表单中id与姓名的对应信息         
                lines = f2.readlines()
                for line in lines:
                    parts = line.strip().split(' ')
                    if len(parts) == 3:
                        ID_dict[parts[0]] = [parts[1], parts[2]]

            return Name_List, Face_Info_List, ID_dict                          # 返回上述三项信息
        
        elif (delete_flag == True and delete_face_ID == None):				   # 如果是删除标注且没有传递要删的人员id,那就执行这个elif, 全删
            for ii in os.listdir(self.database_folder):
                os.remove(os.path.join(self.database_folder, ii))              # 遍历、以删除所有人脸
            with open(self.face_name_file, 'w'):                               # 重头写模式打开该表单
                pass                                                           # pass, 不写入任何东西, 以实现清空功能
            print( "Delete All face success")
                    
        else:                                                                  # 是删除指定人员id操作
            for iii in os.listdir(self.database_folder):
                name = iii.split(".npy")[0]                                    # 遍历库中人脸特征文件, 是ID
                if int(name) in delete_face_ID:                                # 如果当前ID 与传递来的待删除ID匹配上了
                    os.remove(os.path.join(self.database_folder, iii))         # 删除该人脸特征文件
                    with open(self.face_name_file, 'r') as f3:                 # 读模型打开表单, 读取表单中id与姓名的对应信息
                        lines3 = f3.readlines()
                        new_lines = []
                        for line3 in lines3:
                            if int(line3.strip().split(' ')[0]) in delete_face_ID:  # 如果当前ID 与传递来的待删除ID匹配上了
                                continue                                            # 跳过
                            new_lines.append(line3)                                 # 否则不跳过,当前line3信息暂存,待会会写入到新表单中(这些是不要删除的信息)
                        with open(self.face_name_file, 'w') as f4:                  # 写模式打开该表单
                            f4.writelines(new_lines)                                # 把刚刚暂存的 不要删除的人脸信息, 覆盖形式写入表单中,这样来实现删除
                    print( "Delete {} face success".format(name) )

    def cv2_chinese_text(self, img, text, position, textColor=(255, 255, 255), textSize=30):
        
        if (isinstance(img, np.ndarray)):                                                        # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)                                                               # 创建一个可以在给定图像上绘图的对象
        fontStyle = ImageFont.truetype("./config/simsun.ttc", textSize, encoding="utf-8")        # 字体的格式
        draw.text(position, text, textColor, font=fontStyle)                                     # 绘制文本
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)                                  # 转换回OpenCV格式

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            #cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            img = self.cv2_chinese_text(img, label, (c1[0], c1[1] - 25), (255, 0, 0), 25)
        return img
    
    def inference(self, unknow_face_img = None, unknow_face_img_name = None, unknow_face_img_path = None, database_ID_index = None, database_Face_Info = None, ID_dict = None, save_path = None, add_face_flag = False, add_face_ID = None, add_face_name = None):
        
        pointslist = [96,97,54,76,82]
        unknow_face_img_backup = copy.deepcopy(unknow_face_img)
        Results, unknow_num = [], 0
        bboxes, kpss = self.facedetector.detect(unknow_face_img, 0.5, input_size = self.imgsz)
        for j in range(bboxes.shape[0]):
            name_lmk_score = []
            name_lmk_score.append(unknow_face_img_name)
            bbox = bboxes[j]
            x1, y1, x2, y2, score = bbox.astype(int)                                                            # 人脸检测框坐标及分数
            lands, det_width, det_height, det_xmin, det_ymin, score = self.land_detector.demo_image(unknow_face_img, self.land_detector.net, self.land_detector.preprocess, self.land_detector.cfg.input_size, self.land_detector.cfg.net_stride, self.land_detector.cfg.num_nb, self.land_detector.device, bbox.astype(int))

            score = np.array(score.cpu()).sum() / 98                                                            # 关键点分数
            unknow_face_area = unknow_face_img[y1:y2, x1:x2, :]                                                 # 裁出人脸区域
            
            for k in pointslist:                                                                                # 存储好人脸关键点坐标待送入识别模型
                name_lmk_score.append(int(lands[k*2] * det_width) + det_xmin - x1)
                name_lmk_score.append(int(lands[k*2+1] * det_height) + det_ymin - y1)
                # if save_path != None:                                                                         # 往原图画关键点坐标
                #     cv2.circle(unknow_face_img_backup, (int(lands[k*2] * det_width) + det_xmin, int(lands[k*2+1] * det_height) + det_ymin), 1, (0, 0, 255), 2)  
            name_lmk_score.append(score)
            img_input_feats = extract(self.recog_model, self.aligndataset, unknow_face_area, name_lmk_score)    # 提取人脸关键点的特征
            img_input_feats1 = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))       # 对所提特征归一化
            if add_face_flag == True and add_face_name != None:                                                 # 如果是新增人脸, 存到库中,程序至此结束
                with open(self.face_name_file, 'a') as f1:
                    f1.write(f"{add_face_ID} {add_face_name} {unknow_face_img_path}\n")                         # 在表单中存上三元组信息 id name path
                    np.save(os.path.join(self.database_folder, str(add_face_ID) +'.npy'), img_input_feats1)     # 存成 'ID.npy'
                    print( "Save {} {} face success".format(add_face_ID, add_face_name) )
            else:
                inds, rec_dict = [], {}
                for fi in range(len(database_Face_Info)):                                                       # 人脸库做对比
                    similarity_score = np.sum(img_input_feats1 * database_Face_Info[fi], -1)                    # 新人脸与库中人脸计算出分数
                    inds.append(similarity_score)
                matched_score = max(inds)                                                                       # 计算最高分
                
                if matched_score >= self.similarity_score_thres:                                                # 设定阈值, 低于0.25那就跟人脸库中谁也不像，不返回名字
                    matched_ID = database_ID_index[inds.index(matched_score)]                                   # 如果大于0.25,返回跟人脸库里谁最像的ID
                    rec_dict["ID"] = matched_ID
                    rec_dict["name"]  = ID_dict[matched_ID][0]                                                  # 根据ID到表单中查询出名字
                    #rec_dict["score"] = "%.3f" % float(matched_score)
                    Results.append(rec_dict)
                    if save_path != None:                                                                       # 可视化
                        label = f'{ID_dict[matched_ID][0]}'
                        unknow_face_img_backup = self.plot_one_box((x1, y1, x2, y2), unknow_face_img_backup, label=label, color=(0, 0, 255), line_thickness=1)   
                else:
                    unknow_num += 1
                    rec_dict["ID"] = None
                    rec_dict["name"]  = str("{}_{}".format("unknow",unknow_num))
                    #rec_dict["score"] = "%.3f" % float(matched_score)
                    Results.append(rec_dict)
                    if save_path != None:                                                                       # 可视化
                        label = f'{str("{}_{}".format("unknow",unknow_num))}'
                        unknow_face_img_backup = self.plot_one_box((x1, y1, x2, y2), unknow_face_img_backup, label=label, color=(0, 0, 255), line_thickness=1)  
        if save_path != None and add_face_flag == False:                                                        # 保存可视化图
            cv2.imwrite(save_path + unknow_face_img_name + ".jpg", unknow_face_img_backup)
            return Results, save_path + unknow_face_img_name + ".jpg"
        else:
            return Results, None



if __name__ == '__main__':
    
    ##*************工作模式一:录人脸接口*************##
    #image_path = "http://192.168.3.165/a/upload/16975372536532.png"                                             # 支持传url
    # image_path = "/shared/qhb/face_detect_recognize/image/quanhaibo2.jpg"                                      # 支持传一张图
    # id, people_name = 5, "张三"                                                                                # 传一个id值和名字
    # face_model = Face_Detector_Recognize()                                                                     # 模型初始化,这一行是通用
    # img, image_name = face_model.load_image(image_path)                                                        # 加载这张图
    # Results = face_model.inference(unknow_face_img=img, unknow_face_img_name=image_name, unknow_face_img_path=image_path, add_face_flag = True, add_face_ID = id, add_face_name = people_name)   # 录新人脸

  
    ##*************工作模式二:删除人脸库中人脸信息接口*************##
    # face_model = Face_Detector_Recognize()                                                                      # 模型初始化,这一行是通用
    # delete_id = [3, 4, 6]                                                                                       # 传一组要删除的人id值
    # face_model.load_database_face(delete_flag = True, delete_face_ID = delete_id)                               # "删除"人脸库文件夹里数据 


    ##*************工作模式三:作业前人员校验接口*************##
    #image_path = "http://192.168.3.165/a/upload/16975372536532.png"                                            # 支持传url
    image_path = "/shared/qhb/face_detect_recognize/image/hezhao_a.jpg"                                         # 支持传一张图
    save_path  = "/shared/qhb/face_detect_recognize/result/"                                                    # 如果可视化, 需指定可视化路径, 不可视化是None
    face_model = Face_Detector_Recognize()                                                                      # 模型初始化,这一行是通用
    img, image_name = face_model.load_image(image_path)                                                         # 加载这张图
    Name_List, Face_Info_List, ID_dict = face_model.load_database_face()                                        # 加载人脸库里数据
    Results, vis_path = face_model.inference(unknow_face_img=img, unknow_face_img_name=image_name, database_ID_index = Name_List, database_Face_Info = Face_Info_List, ID_dict = ID_dict, save_path = save_path)  # 人脸识别
    print(Results)
    print(vis_path) 


    ##*************工作模式四:清空所有人脸*************##
    # face_model = Face_Detector_Recognize()                                                                      # 模型初始化,这一行是通用
    # face_model.load_database_face(delete_flag = True)                                                           # "清空"人脸库 
