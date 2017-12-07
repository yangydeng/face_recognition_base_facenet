'''
@Date  : 2017-11-28 20:15
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    人脸识别用来工作的入口。
'''


import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

from faceNet.get_facenet_bottleneck import get_bottleneck
from faceNet.get_facenet_bottleneck import calculate_dis
from align.align_image_mtcnn import align_image_mtcnn
from arg_class import argvs

# 添加数据库（内部人员）
def convert_bottleneck2str(bottle_neck):
    bottle_neck = bottle_neck[0]
    bottle_neck_str = ','.join(str(x) for x in bottle_neck)
    return bottle_neck_str


def convert_bottleneck2list(bottle_neck_str):
    bottle_neck = [float(x) for x in bottle_neck_str.split(',')]
    return bottle_neck


def add_new_stuff(person_name):
    #这里的person_name同时也是照片的名称。
    database_path = 'database/stuff.csv'

    db_tmp = pd.DataFrame(columns=['name','bottle_neck'])
    db = pd.read_csv(database_path)
    if person_name in db.name.values:
        print(person_name + ' has add to database')
        return

    # 在这里加入align的部分
    sufix = '.jpeg'
    photo_name = person_name+sufix
    photo_base_path = '/home/hiptonese/project/face_recognition/dataset/photo/'
    argvs.image_path = photo_base_path+photo_name
    argvs.output_dir = '/home/hiptonese/project/face_recognition/dataset/face_align/'
    argvs.class_name = person_name
    align_image_mtcnn(argvs)

    # get_bottle_neck部分
    face_image_path = argvs.output_dir+argvs.class_name+'/'+photo_name
    bottle_neck = get_bottleneck([face_image_path])
    bottle_neck_str = convert_bottleneck2str(bottle_neck)

    db_tmp.name = [person_name]
    db_tmp.bottle_neck = [bottle_neck_str]
    db = pd.concat([db, db_tmp])
    db.to_csv(database_path, index=False)
    print(person_name + ' has add to database')


def load_model(model_name):
    # 加载模型
    bottle_neck_input_name = 'BottleneckInputPlaceholder:0'
    final_tensor_name = 'final_tensor:0'
    model_dir = '/home/hiptonese/project/face_recognition/model_save_path/'
    with gfile.FastGFile(model_dir+model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottle_neck_tensor, final_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[bottle_neck_input_name, final_tensor_name]
    )
    return bottle_neck_tensor, final_tensor


def face_recognition(new_person_name):
    sufix = '.jpeg'
    photo_base = '/home/hiptonese/project/face_recognition/dataset/photo/'
    database_path = 'database/stuff.csv'
    photo_name = new_person_name+sufix
    db = pd.read_csv(database_path)
    photo_path = photo_base + photo_name
    # 得到align
    argvs.image_path = photo_path
    argvs.output_dir = '/home/hiptonese/project/face_recognition/dataset/face_align/'
    argvs.class_name = 'visitor'
    align_image_mtcnn(argvs)
    # 当前相片的bottle neck
    face_image_path = argvs.output_dir+argvs.class_name+'/'+photo_name
    bottle_neck = get_bottleneck([face_image_path])


    for line in db.values:
        name, stuff_bottle_neck_str = line[0], line[1]
        stuff_bottle_neck = convert_bottleneck2list(stuff_bottle_neck_str)
        distance = calculate_dis(bottle_neck, stuff_bottle_neck)
        # print(name, distance)
        if distance<threshold:
            print('\n识别结果：'+name)


if __name__ == '__main__':
    threshold = 0.52
    while True:
        print('\n======================================')
        num = input('输入数字  1 添加新成员; 2 人脸识别; 3 退出  \n')
        if num=='1':
            name = input('enter image name for adding to database: \n')
            add_new_stuff(name)
        elif num=='2':
            name = input('enter image name for recognizing: \n')
            face_recognition(name)
        elif num=='3':
            print('bye ~')
            break
        else:
            print('wrong number, try again.\n')
