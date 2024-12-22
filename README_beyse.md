# 基于朴素贝叶斯的中文电影评论数据综合分析系统

## 项目概述
这是一个使用朴素贝叶斯进行中文电影评论数据综合分析的系统。该系统采用了 TF-IDF 特征提取和朴素贝叶斯分类器，并包含了完整的数据处理、模型优化和结果可视化流程。

## 主要特性
- 自动数据平衡处理
- 分阶段参数优化
- 文本预处理和特征提取
- 模型性能可视化
- 关键词影响分析
- 批量处理支持
- 详细的日志记录

## 技术栈
- Python 3.8+
- scikit-learn
- pandas
- numpy
- jieba
- matplotlib
- seaborn
- tqdm

## 项目结构
- main.py # 主程序文件
- stopwords.txt # 停用词表
- saved_models/ # 模型和结果保存目录
- bayes_model_optimized.pkl
- optimization_results.json
-  analysis_results/
- data/ # 数据目录
-  douban_movie/ # 豆瓣电影评论数据
## 核心组件

### 1. 文本预处理器 (TextPreprocessor)
- class TextPreprocessor:
  - def init(self, max_length=100):
  - self.max_length = max_length
  - self.stopwords = self.load_stopwords()
### 功能：
- 文本清洗
- 停用词过滤
- 中文分词
- 序列处理

### 2. 模型优化器 (staged_optimization)
- def staged_optimization(X_train, y_train, X_val, y_val, save_dir):
# 第一阶段参数网格
- param_grid_stage1 = {
- 'vectorizermax_features': [7000, 8000],
- 'vectorizerngram_range': [(1, 2), (1, 3)],
- 'classifierestimator_C': [0.1, 10.0]
- }
# 第二阶段参数网格
- param_grid_stage2 = {
- 'vectorizermax_features': [...],
- 'vectorizerngram_range': [(1, 2), (1, 3)],
- 'vectorizermin_df': [2, 3],
- 'classifierestimator_C': [1.0, 2.0]
- }

### 3. 可视化组件 (OptimizationVisualizer)
- class OptimizationVisualizer:
  - def plot_optimization_progress(self, grid_search, stage="")
  - def plot_learning_curves(self, train_scores, val_scores, param_name)
  - def plot_confusion_matrix(self, y_true, y_pred, labels)
  - def plot_roc_curve(self, y_true, y_prob)

## 参数配置

### 1. 数据处理参数
- 文本预处理
  - max_length = 100
  - batch_size = 1000
- 数据集划分
  - test_size = 0.2
  - random_state = 42
### 2. 模型参数
- TF-IDF 参数
  - max_features = 8000
  - min_df = 2
  - max_df = 0.95
- SVM 参数
  - C = 2.0
  - class_weight = 'balanced'
  - dual = 'auto'
  - max_iter = 1000
## 输出结果

### 1. 模型文件
- `svm_model_optimized.pkl`: 优化后的模型
- `optimization_results.json`: 优化过程记录

### 2. 可视化结果
- `sentiment_distribution.png`: 情感分布图
- `key_words.png`: 关键词影响力图
- `confusion_matrix.png`: 混淆矩阵
- `roc_curve.png`: ROC曲线

### 3. 分析报告
- 情感分析结果
- 关键词影响分析
- 极端评论示例

## 性能优化

### 1. 数据处理优化
- 批量处理机制
- 多进程支持
- 内存优化

### 2. 模型优化
- 分阶段参数搜索
- 交叉验证
- 早停机制

## 错误处理
系统包含完整的错误处理机制：
- 数据加载异常
- 预处理错误
- 模型训练异常
- 结果保存失败

## 扩展功能
1. 支持新数据源
2. 自定义评估指标
3. 模型导出功能
4. 增量更新支持

## 维护说明
- 定期更新停用词表
- 监控模型性能
- 优化参数配置
- 更新依赖包