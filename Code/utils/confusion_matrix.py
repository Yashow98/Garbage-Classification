import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, class_labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = class_labels

    def update(self, preds, labels):  # 得到混淆矩阵
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, f1-score
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 对角线总数
            FP = np.sum(self.matrix[i, :]) - TP  # pred总数 - TP
            FN = np.sum(self.matrix[:, i]) - TP  # Truth - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1_score = round(2 * TP / (np.sum(self.matrix) + TP - TN), 3) if np.sum(self.matrix) + TP - TN != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1_score])
        print(table)

    def plot(self):
        """
        绘制混淆矩阵
        Returns
        -------
        """
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]， 注意坐标轴和矩阵行列的关系，正好相反
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
