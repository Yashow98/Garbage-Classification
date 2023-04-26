import os

'''
    为数据集生成对应的txt文件, 制作图片数据索引，路径及标签
'''
total_txt_path = os.path.join("../..", "total.txt")
total_dir = os.path.join("../..", "Garbage_classification")

train_txt_path = os.path.join("../..", "train.txt")
train_dir = os.path.join("../..", "train")

valid_txt_path = os.path.join("../..", "valid.txt")
valid_dir = os.path.join("../..", "valid")

test_txt_path = os.path.join("../..", "test.txt")
test_dir = os.path.join("../..", "test")


def gen_txt(txt_path, img_dir):
    garbage_class = [cls for cls in os.listdir(img_dir)]  # 垃圾类别
    garbage_class.sort()  # 保证排序一致
    class_indices = dict((k, v) for v, k in enumerate(garbage_class))  # 生成类别标签

    with open(txt_path, 'w') as f:
        for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
            for sub_dir in s_dirs:
                i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹路径
                img_list = os.listdir(i_dir)  # 获取类别文件夹下所有jpg图片的路径
                for i in range(len(img_list)):
                    if not img_list[i].endswith('jpg'):  # 若不是jpg文件，跳过
                        continue
                    label = class_indices[sub_dir]
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + ' ' + str(label) + '\n'
                    f.write(line)


if __name__ == '__main__':
    gen_txt(total_txt_path, total_dir)
    gen_txt(train_txt_path, train_dir)
    gen_txt(valid_txt_path, valid_dir)
    gen_txt(test_txt_path, test_dir)
