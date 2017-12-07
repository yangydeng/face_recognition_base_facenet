基于 facenet 的人脸识别项目

本项目基于 facenet 开源项目实现。在原有基础上做了少量封装。
facenet url: https://github.com/davidsandberg/facenet

本项目依赖facenet上的预训练模型：
https://github.com/davidsandberg/facenet#pre-trained-models

项目介绍：
1. 本项目使用facenet上的预训练集做人脸识别;
2. 人脸识别包括两部分：
    (1)人脸对齐(详见align包)
    (2)人脸对比识别(详见facenet包)
3. 使用前需自行指定文件位置等相关信息，位于App.py文件中;
4. 工程主入口为App.py
5. 流程：
    (1) 运行App.py后，两个可选功能，
        a.添加新人：调用mtcnn通过人脸对齐找到图片中人脸的位置，截取人脸并保存。
                  调用get_facenet_bottleneck计算出该人脸的bottleneck，并保存到database/stuff.csv
        b.人脸识别：给出新图片的名称，调用mtcnn以及facenet，得到该图片的bottleneck。
                   将该bottleneck于csv文件中的bottleneck逐一对比，输出结果小于threshold作为识别结果。


