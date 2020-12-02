import os
import cv2
import sys
import csv
from random import sample
import numpy as np
import tensorflow as tf
import pickle

# tensorboard之前的版本为from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector



SPRITE_FILE = "spriteimage.jpg"     #向量对应的拼接图片
META_FIEL = "metadata.tsv"          #向量对应的标签文件
LOG_DIR = "log_visual"              #tensorboard work_dir 这里尽量不要写绝对路径

SAMPLE_NUM = 28*28                    #采样向量的个数
IMAGE_SIZE = 224                     #图片size
SPIRATE_IMAGE_SIZE = 28             #拼接图片size

def visualisation(vectors):
    """
    param: 需要可视化的高维向量集合, list, [SAMPLE_NUM, dim]
    return: None
    """
    # PROJECTOR可视化的都是TensorFlow中的变量类型。
    y = tf.Variable(vectors)
    checkpoint = tf.train.Checkpoint(embedding=y)
    checkpoint.save(os.path.join(LOG_DIR, "embedding.ckpt"))

    # 通过project.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    # 增加一个需要可视化的embedding结果
    embedding = config.embeddings.add()
    # 指定这个embedding结果所对应的Tensorflow变量名称
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"

    # 指定embedding结果所对应的数据标签文件,改设置可选, 如果没有提供，可视化结果
    # 每个点颜色都是一样的
    # embedding.metadata_path = META_FIEL

    # 指定sprite 图像。这个也是可选的，如果没有提供sprite 图像，那么可视化的结果
    # 每一个点就是一个小困点，而不是具体的图片。
    embedding.sprite.image_path = SPRITE_FILE
    # 这将用于从sprite图像中截取正确的原始图片。
    embedding.sprite.single_image_dim.extend([IMAGE_SIZE, IMAGE_SIZE])

    # 将PROJECTOR所需要的内容写入日志文件。
    projector.visualize_embeddings(LOG_DIR, config)

def create_sprite_image_and_label():
    """
    param: None
    return: 需要可视化的高维向量集合, list, [SAMPLE_NUM, dim]
    """
    # 加载数据
    with open('DatasetEmbedding.pkl', 'rb') as f:
        embeddingDict = pickle.load(f)
    # _class = list(dataset.classes)
    key_arr = list(embeddingDict.keys())

    # 采样一部分向量可视化
    visual_arr = sample(key_arr, SAMPLE_NUM)
    image_arr = []
    for i in range(SAMPLE_NUM):
        image_arr.append(embeddingDict[visual_arr[i]].tolist())

    # 生成标签文件 metadata.tsv
    # with open(os.path.join(LOG_DIR, META_FIEL), 'w') as f:
    #     f.write('__index__' +'\t' +'Label' +'\n')
    #     for i in range(SAMPLE_NUM):
    #         name = visual_arr[i].split('_')[0]
    #         label = _class.index(name)
    #         # f.write(str(label) +'\n')
    #         f.write(str(i) +'\t' +str(label) +'\n')
    
    # 生成拼接图片文件 spriteimage.jpg
    spriteimage = np.ones((IMAGE_SIZE * SPIRATE_IMAGE_SIZE, IMAGE_SIZE * SPIRATE_IMAGE_SIZE,3))
    for i in range(SPIRATE_IMAGE_SIZE):
        for j in range(SPIRATE_IMAGE_SIZE):
            this_filter = i * SPIRATE_IMAGE_SIZE + j
            imgPath=os.path.join('E:/CULane', visual_arr[this_filter])
            this_img = cv2.imread(imgPath)
            this_img = cv2.resize(this_img,(IMAGE_SIZE,IMAGE_SIZE))
            spriteimage[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = this_img
    cv2.imwrite(os.path.join(LOG_DIR, SPRITE_FILE),spriteimage)
    return image_arr

if __name__=="__main__":
    vectors = create_sprite_image_and_label()
    visualisation(vectors)