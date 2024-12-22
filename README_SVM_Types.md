# 基于 SVM 的中文电影类型分类系统

## 项目概述
这是一个使用支持向量机（SVM）的多标签分类系统，用于根据电影简介自动预测电影类型。系统采用 TF-IDF 特征提取和 OneVsRest 分类策略，支持多类型标签预测。

## 主要特性
- 多标签分类支持
- 自动参数优化
- TF-IDF 特征提取
- 概率校准
- 完整的错误处理
- 详细的日志记录

## 技术栈
- Python 3.8+
- scikit-learn
- pandas
- numpy
- joblib

## 项目结构
project/
- `main.py` # 主程序文件
- `saved_models/` # 模型保存目录
- `movie_classifier.joblib`
- `label_encoder.joblib`
- `training.log`
- `data/` # 数据目录
- `movie_item.json` # 电影数据
## 核心组件

### 1. 数据加载器
def load_movie_data(file_path: str) -> List[Dict]:
- 加载和预处理电影数据
- 文本清洗
- 数据验证
- 错误处理

### 2. 模型管道
def create_pipeline() -> Pipeline:
- 创建优化的模型管道
- TF-IDF 向量化
- OneVsRest 分类器
- LinearSVC 基础分类器

### 3. 参数优化器
def optimize_pipeline(X_train: List[str], y_train: np.ndarray) -> Pipeline:
- 网格搜索最优参数
- 特征数量优化
- n-gram 范围选择
- 正则化参数调整
## 参数配置

### 1. 特征提取参数
- TfidfVectorizer(
- max_features=10000,
- ngram_range=(1, 3),
- min_df=2,
- max_df=0.95,
- sublinear_tf=True
- )
### 2. 分类器参数
- LinearSVC(
- C=1.0,
- class_weight='balanced',
- dual=False,
- max_iter=1000
- )
### 3. 优化参数网格
- param_grid = {
- 'vectorizermax_features': [5000, 10000, 15000],
- 'vectorizerngram_range': [(1, 2), (1, 3)],
- 'vectorizermin_df': [2, 3],
- 'classifierestimator_C': [0.1, 1.0, 10.0]
- }
## 模型使用

### 1. 加载模型
- pipeline = joblib.load('saved_models/movie_classifier.joblib')
- mlb = joblib.load('saved_models/label_encoder.joblib')
### 2. 预测示例
- def predict_movie_type(text: str, pipeline: Pipeline, mlb: MultiLabelBinarizer) -> Dict[str, Any]:
  - result = predict_movie_type(movie_intro, pipeline, mlb)
  - print(f"预测类型: {result['predicted_types']}")
  - print(f"类型概率: {result['type_probabilities']}")
## 输出结果

### 1. 模型文件
- `movie_classifier.joblib`: 训练好的分类器
- `label_encoder.joblib`: 标签编码器

### 2. 日志文件
- `training.log`: 训练过程日志

### 3. 预测结果
- {
- "predicted_types": ["动作", "冒险"],
- "type_probabilities": {
- "动作": 0.85,
- "冒险": 0.72,
- "剧情": 0.45,
- ...
- }
- }
## 性能优化

### 1. 数据处理优化
- 批量处理
- 并行计算支持
- 内存优化

### 2. 模型优化
- 参数网格搜索
- 交叉验证
- 类别权重平衡

## 错误处理
系统包含完整的错误处理机制：
- 数据加载异常
- JSON 解析错误
- 模型训练异常
- 预测错误处理

## 扩展功能
1. 支持新数据源
2. 自定义评估指标
3. 模型导出功能
4. 增量更新支持

## 维护说明
- 定期更新模型
- 监控预测性能
- 优化参数配置
- 更新依赖包
