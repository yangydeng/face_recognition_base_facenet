'''
@Date  : 2017-12-07 11:18
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    将使用在 align 中的参数封装到一个类里。
'''


class argvs:
    # 输入图片的位置
    image_path = ''
    output_dir = ''
    # 人名
    class_name = 'dyy'
    image_size = 182
    margin = 44
    gpu_memory_fraction = 1
    detect_multiple_faces = False
    random_order = False
