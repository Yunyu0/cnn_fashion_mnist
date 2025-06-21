#运行说明
##环境要求
Python 3.7+
PyTorch 1.8+
torchvision
matplotlib
scikit-learn
seaborn

##安装依赖
pip install torch torchvision matplotlib scikit-learn seaborn
##执行命令
python fashion_mnist_cnn.py

#预期输出
训练过程日志（每100个batch打印一次损失）
每个epoch结束时的训练损失和准确率
测试集上的准确率和分类报告
自动生成的可视化图像文件

#文件清单
主程序文件：
fashion_mnist_cnn.py：完整训练和测试代码
##生成图像文件：
fashion_samples.png：训练样本示例
training_history.png：训练损失和准确率曲线
confusion_matrix.png：混淆矩阵
test_predictions.png：测试集预测结果示例
##模型文件：
fashion_mnist_cnn.pth：训练好的模型权重
##数据目录：
./data/FashionMNIST/：下载的数据集文件
