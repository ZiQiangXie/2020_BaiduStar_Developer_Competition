#encoding=utf-8
import matplotlib
import matplotlib.pyplot as plt


def Draw(label_name, lable_num, y_max, offset=0.0):
    matplotlib.rcParams['axes.unicode_minus'] = False
    x = list(range(len(label_name)))
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(label_name, lable_num, alpha=0.5, width=0.3, color='blue', label="num")
    plt.ylim(0, y_max)  # y轴取值范围
    plt.ylabel("num")
    plt.xticks([index + offset for index in x], label_name)
    plt.legend()
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    plt.show()


class_index = ['301', '104', '106', '103', '105', '201', '108', '207',
               '202', '102', '110', '107', '203', '204', '109', '112',
               '205', '111', '206']
class_num = [28707, 1770, 1103, 976, 323, 459, 244, 588,
             249, 238, 19, 30, 53, 32, 0, 0, 19, 0, 5]

class_index1 = ['301', '104', '106', '103', '105', '201', '108', '207',
               '202', '102', '110', '107', '203', '204', '109', '112',
               '205', '111', '206']
class_num1 = [93744, 10242, 5671, 5149, 4523, 2397, 1855, 1529,
             1065, 976, 626, 521, 326, 283, 224, 210, 72, 39, 15]

class_index2 = [str((i + 1) * 10) for i in range(10)]
class_num2 = [17708, 11605, 3185, 1091, 548, 270, 167, 113, 55, 73]

class_index3 = [str((i + 1) * 10) for i in range(10)]
class_num3 = [56319, 45079, 15077, 5835, 3132, 1755, 1007, 594, 321, 348]

Draw(class_index, class_num, 30000)
Draw(class_index1, class_num1, 100000)
Draw(class_index2, class_num2, 18000, offset=0.5)
Draw(class_index3, class_num3, 60000, offset=0.5)
