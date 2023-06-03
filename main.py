import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from io import BytesIO
from scipy.signal import stft
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models


def predict_image_class(modelPath, imagePath):
    # 读取CSV文件,取消索引
    data = pd.read_csv(imagePath, index_col=0)

    # 使用stft变换
    ir_fault_data = data['cDAQ9181-171B2ECMod1/ai0'].values
    fs = 10240  # 采样频率为10240KHz

    # 采用汉宁窗口，窗口大小为1024，信号重叠大小为512
    f_ir7, t_ir7, Zxx_ir7 = stft(ir_fault_data, fs, window='hann', nperseg=1024, noverlap=512)

    # 筛选频率范围为0 Hz到1000 Hz的数据
    f_mask = (f_ir7 >= 0) & (f_ir7 <= 500)
    f_filtered = f_ir7[f_mask]
    Zxx_filtered = Zxx_ir7[f_mask, :]

    # 生成图片
    fig, ax = plt.subplots(figsize=(2.24, 2.24))  # 设置图形大小为224x224
    ax.pcolormesh(t_ir7, f_filtered, np.abs(Zxx_filtered), shading='gouraud')  # 使用Gouraud着色进行阴影处理。
    # 移除坐标轴
    ax.axis('off')
    # 调整图像边界
    fig.tight_layout(pad=0)

    # 图片预处理，转换为tensor向量
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理图片
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # 将图像转换为RGB格式
    image = Image.open(buffer).convert('RGB')

    # image.show()  # 显示图片
    image = transform(image)
    image = image.unsqueeze(0)
    plt.close(fig)

    # 预定义模型
    model_state = torch.load(modelPath)
    model = models.resnet34()

    # 替换最后一层全连接层
    num_classes = 2  # 当前状态类别个数
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 加载模型状态
    model.load_state_dict(model_state)

    # 冻结模型参数
    model.eval()

    # 执行预测
    with torch.no_grad():
        output = model(image)

    # 获取预测结果
    _, predicted = torch.max(output, 1)

    # 返回预测的图片类别
    return predicted.item()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    model_path = 'D:/数据/data/model/model.pth'        # 模型文件路径
    image_path = 'D:/数据/data/测试数据/testData/ok/tryok-05-30-10-56-38.csv'  # 状态文件路径（文件大小为5s的信号数据，采样频率为10240KHz）

    # 得到状态参数，0为故障，1为正常
    predicted_class = predict_image_class(model_path, image_path)

    # 状态判断
    if predicted_class == 0:
        print("当前状态：故障")
    if predicted_class == 1:
        print("当前状态：正常")
