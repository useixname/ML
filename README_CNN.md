# 中文电影评论情感分析系统 (TextCNN)

## 项目概述
这是一个基于改进的 TextCNN 模型的中文电影评论情感分析系统。该系统使用深度学习技术对电影评论进行二分类（正面/负面），并包含多项改进特性以提高分类性能。

## 主要特性
- 改进的 TextCNN 架构
- 多尺度卷积和注意力机制
- Focal Loss 用于处理类别不平衡
- 自适应学习率调整
- 完整的数据预处理流程
- 详细的训练过程可视化
- 模型性能评估和监控

## 技术栈
- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- pandas
- numpy
- jieba
- matplotlib
- tqdm

## 项目结构
-  CNN.py 主程序文件
-  CNN_predict 使用模型进行预测文件
-  stopwords.txt # 停用词表
-  saved_models/ # 模型保存目录
-  best_model.pth # 最佳模型
-  preprocessor.pkl # 预处理器
-  training_history.pkl # 训练历史
-  data/ # 数据目录
-  douban_movie/ # 豆瓣电影评论数据

## 核心组件

### 1. 文本预处理器 (TextPreprocessor)
- 停用词过滤
- 文本清洗
- 分词处理
- 序列填充
- 词汇表构建

### 2. 改进的 TextCNN 模型
- class ImprovedTextCNN(nn.Module):
  - def init(self, vocab_size, embedding_dim=300, max_length=100):
# 多尺度卷积层
- self.convs = nn.ModuleList([
- nn.Conv2d(1, 128, (k, embedding_dim))
- for k in [3, 4, 5]
- ])
# 注意力机制
- self.attention = nn.Sequential(...)
# 批归一化和dropout
- self.batch_norm = nn.BatchNorm1d(384)
- self.dropout = nn.Dropout(0.5)
### 3. Focal Loss
- class FocalLoss(nn.Module):
  - def init(self, alpha=None, gamma=2):
  - self.alpha = alpha # 类别权重
  - self.gamma = gamma # 聚焦参数


## 模型参数

### 1. 预处理参数
- max_length: 100 (序列最大长度)
- min_word_freq: 2 (最小词频)

### 2. 模型参数
- embedding_dim: 300 (词向量维数)
- num_filters: 128  (卷积核数量)
- filter_sizes: [3, 4, 5] (卷积核大小)
- dropout: 0.5 (随机剃除50%的数据)

### 3. 训练参数
- batch_size: 32
- learning_rate: 2e-5
- weight_decay: 0.01
- num_epochs: 20
- patience: 5
- min_delta: 0.001

## 训练过程

### 1. 数据处理
- 加载原始数据
- 数据清洗和平衡
- 文本预处理
- 训练集/验证集划分

### 2. 模型训练
- 使用 Focal Loss 处理类别不平衡
- AdamW 优化器
- ReduceLROnPlateau 学习率调度
- 早停机制

### 3. 模型评估
- 准确率
- F1分数
- 混淆矩阵

## 输出和可视化

### 1. 训练日志
- saved_models/training_[timestamp].log
### 2. 可视化结果
- training_curves.png (损失和准确率曲线)

### 3. 模型文件
- best_model.pth (最佳模型权重)
- preprocessor.pkl (预处理器)
- training_history.pkl (训练历史)

## 性能优化

### 1. 数据处理优化
- 批处理机制
- 多进程数据加载
- PIN_MEMORY 启用

### 2. 训练优化
- 梯度裁剪
- 学习率自适应调整
- 早停机制

## 注意事项
1. 确保 CUDA 可用性
2. 监控 GPU 内存使用
3. 检查数据集平衡性
4. 注意模型过拟合

## 错误处理
系统包含完整的错误处理机制：
- 数据加载异常
- GPU 内存不足
- 模型保存失败
- 训练中断恢复

## 扩展功能
1. 支持模型导出
2. 批量预测接口
3. 模型解释性分析
4. 增量学习支持

