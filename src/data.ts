import { ModelDoc, Competition, LearningPath } from "./types";

export const MODELS_DATA: ModelDoc[] = [
  // ==================== 1. 数据预处理与特征工程 (preprocess) ====================
  {
    id: "data-labeling",
    name: "数据标签 (Data Labeling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "将原始数据赋予语义信息，是监督学习模型训练的前提与基石。",
    principles: `1. 手动标注：人工判读并为每条样本赋予真实标签，如图像框选、文本情感判定。
2. 半自动标注：先由规则或轻量模型预标，再由人工校正与审核。
3. 主动学习标注：优先标注不确定性高的样本，以最小标注量换取最大模型增益。
4. 弱监督标注：利用启发式规则或外部知识库批量生成标签。`,
    scenarios: [
      "医学影像中肿瘤区域的像素级标注",
      "电商评论文本情感极性（正/负/中性）标注"
    ],
    limitations: [
      "人工标注成本高、耗时长，不同标注者之间存在主观差异",
      "弱监督标签噪声较大，需后续清洗"
    ],
    caseStudy: {
      title: "工业零件缺陷检测数据集标注",
      description: "某精密零件生产线需对一万张零件图片进行缺陷类型标注，传统人工逐张标注耗时约 200 小时。",
      solution: `1. 先利用规则（明暗对比度阈值）预标 60% 数据；
2. 剩余 40% 高疑似区交由人工精标；
3. 结合主动学习，最终仅标注 3200 张即达 95% 模型精度。`
    },
    code: {
      python: `import pandas as pd
import numpy as np

# 模拟手工标注数据
data = pd.DataFrame({
    'text': ['这家店很好', '一般般', '非常差'],
    'label': [1, 0, -1]  # 1=正面, 0=中性, -1=负面
})
print(data)
# 利用主动学习筛选高不确定性样本
# Uncertainty = max(prob) 越接近0.5越不确定`,
      matlab: `% MATLAB 文本情感半自动标注
texts = {'这家店很好', '一般般', '非常差'};
labels = [1, 0, -1];  % 人工标注`
    },
    resources: [
      { title: "Label Studio 开源标注平台", url: "https://labelstud.io", type: "link" }
    ]
  },
  {
    id: "data-encoding",
    name: "数据编码 (Data Encoding)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "将分类变量转换为模型可理解的数值形式，是所有机器学习流水线的前置必修环节。",
    principles: `1. 标签编码 (Label Encoding)：将类别直接映射为 0, 1, 2, ... 整数，适合有序类别。
2. 独热编码 (One-Hot Encoding)：生成 M 个 0/1 列，适合无序类别，避免引入虚假顺序。
3. 目标编码 (Target Encoding)：用目标变量的均值替代类别，融合标签信息，需防止信息泄露。
4. 频数编码 (Count Encoding)：用类别在训练集中出现的频率替代类别。`,
    scenarios: [
      "城市名、颜色等无序类别特征参与线性回归",
      "高基数类别变量（如邮编、用户 ID）参与随机森林"
    ],
    limitations: [
      "独热编码在高基数类别下会产生维度爆炸",
      "目标编码若在训练集外使用会引入信息泄露偏差"
    ],
    caseStudy: {
      title: "客户职业类型编码建模",
      description: "某银行客户职业字段有 52 种类别，直接标签编码会让树模型误认为「程序员(51)」比「教师(5)」数值更大。",
      solution: `1. 对高基数职业使用目标编码，以逾期率均值替代原始类别；
2. 对低基数有序类别使用 Ordinal Encoding；
3. 留一法交叉验证防止编码信息泄露。`
    },
    code: {
      python: `import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码
le = LabelEncoder()
encoded = le.fit_transform(['教师', '医生', '程序员', '教师'])
print(encoded)  # [2 0 1 2]

# 独热编码 (pandas 简洁方式)
df = pd.DataFrame({'job': ['教师', '医生', '程序员']})
onehot = pd.get_dummies(df, columns=['job'], drop_first=True)
print(onehot)`,
      matlab: `% MATLAB 类别编码
jobs = {'教师','医生','程序员'};
categories = unique(jobs);
label_enc = zeros(size(jobs));
for i = 1:length(categories)
    label_enc(strcmp(jobs, categories{i})) = i - 1;
end`
    },
    resources: [
      { title: "sklearn 编码器官方文档", url: "https://scikit-learn.org/stable/modules/preprocessing.html", type: "link" }
    ]
  },
  {
    id: "outlier-handling",
    name: "异常值处理 (Outlier Detection & Treatment)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "检测并处理数据中的极端或不合理值，是保障模型稳定性的关键前置步骤。",
    principles: `1. 3σ 准则：服从正态分布时，[μ-3σ, μ+3σ] 区间外视为异常，概率约 0.27%。
2. 箱线图 IQR 法：[Q1-1.5·IQR, Q3+1.5·IQR] 之外判定为异常。
3. 孤立森林 (Isolation Forest)：无监督树集成，通过随机切分路径长度识别离群点。
4. Z-score 法：|z| > 3 的样本标记为异常。`,
    scenarios: [
      "工业传感器毛刺脉冲检测与剔除",
      "金融交易异常金额监控"
    ],
    limitations: [
      "3σ 准则极度依赖正态假设，偏态分布下失效严重",
      "IQR 法对中等偏态数据亦会漏判或误判"
    ],
    caseStudy: {
      title: "污水处理流量异常清洗",
      description: "某污水处理厂 1000 条流量数据中存在传感器瞬时高压冲击（异常值 99.0）和间歇断流（缺失值）。",
      solution: `1. 采用 IQR 剔除 5 组超界异常点，以局部中位数替代；
2. 对剩余缺失点用随机森林回归插补；
3. Z-score 二次扫描确认无残留离群点。`
    },
    code: {
      python: `import numpy as np
import pandas as pd
from scipy import stats

data = pd.Series([1.2, 1.5, 99.0, 1.4, 1.6, np.nan])
# IQR 法剔除
q1, q3 = np.percentile(data.dropna(), [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
cleaned = data.clip(lower=lower, upper=upper)`,
      matlab: `% MATLAB 异常值剔除与填充
raw = [1.2; 1.5; 99.0; 1.4; 1.6];
cleaned = filloutliers(raw, 'clip', 'movmedian', 3);
filled = fillmissing(cleaned, 'linear');`
    },
    resources: [
      { title: "sklearn 异常值检测文档", url: "https://scikit-learn.org/stable/modules/outlier_detection.html", type: "link" }
    ]
  },
  {
    id: "invalid-sample-handling",
    name: "无效样本处理 (Invalid Sample Handling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "识别并处理重复记录、矛盾样本和无效输入，是数据清洗的最后一道防线。",
    principles: `1. 重复检测：精确匹配（所有列相同）或近似匹配（关键列子集相同），可基于哈希或编辑距离。
2. 矛盾样本：相同特征向量对应不同标签，在监督学习中产生噪声。
3. 无效输入过滤：剔除标签为空、特征全为零、明显错误格式的记录。
4. 业务规则过滤：依据领域知识排除不可能的样本（如身高 10 米）。`,
    scenarios: [
      "爬虫数据中大量重复抓取的新闻文章去重",
      "医学数据中前后矛盾的患者诊断记录清洗"
    ],
    limitations: [
      "过于严格的去重会丢失有价值的多角度记录",
      "矛盾样本有时反映真实数据分布，不应全部丢弃"
    ],
    caseStudy: {
      title: "问卷数据质量清洗",
      description: "某调研问卷 5000 份中存在：同一 IP 重复提交（120 条）、选项全选同一答案（80 条）、必填字段为空（35 条）。",
      solution: `1. 基于用户 ID 与时间戳哈希精确去重，剔除 120 条；
2. 剔除全选同一选项的 80 条无效问卷；
3. 对缺失必填项的 35 条使用众数填充或删除；
4. 最终保留 4765 条高质量样本。`
    },
    code: {
      python: `import pandas as pd

df = pd.DataFrame({
    'user_id': [1, 1, 2, 3, 3],
    'answer': ['A', 'A', 'B', 'C', 'C'],
    'score': [85, 85, 90, 70, 75]
})
# 精确去重
df_clean = df.drop_duplicates(subset=['user_id', 'answer'], keep='first')
print(f"去重后样本: {len(df_clean)}")`,
      matlab: `% MATLAB 重复检测与剔除
[~, idx] = unique(data(:, [1 3]), 'rows');
clean_data = data(idx, :);`
    },
    resources: [
      { title: "pandas 数据清洗实战指南", url: "https://pandas.pydata.org", type: "link" }
    ]
  },
  {
    id: "feature-generation",
    name: "生成变量 (Feature Generation / Feature Engineering)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "通过领域知识与数学变换从原始变量中构造新特征，是提升模型效果的最有效手段之一。",
    principles: `1. 交叉特征：两特征相乘、相除、差值，如「客单价=销售额/购买次数」。
2. 多项式特征：x1², x1·x2, x2² 等，捕获非线性关系。
3. 分箱特征 (Binning)：将连续变量离散化，如年龄段、收入等级。
4. 时间特征：从日期提取年/月/日/星期/是否节假日等语义特征。
5. 聚合特征：groupby 后计算 count/mean/std/max 等统计量。`,
    scenarios: [
      "用户行为序列中提取「首次购买距注册天数」",
      "房价预测中加入「房间数×装修标准」交叉特征"
    ],
    limitations: [
      "过多特征会产生维度灾难，需配合特征筛选使用",
      "交叉特征可能导致过拟合，尤其是低频组合"
    ],
    caseStudy: {
      title: "电商用户购买行为特征构建",
      description: "原始数据仅有用户 ID、购买时间、商品类别，需预测用户下次购买时间间隔。",
      solution: `1. 从时间戳提取小时段（0-6/7-12/13-18/19-23）和是否周末；
2. 计算用户历史购买频率、活跃天数跨度；
3. 构造「商品类别交叉购买次数」矩阵；
4. 最终特征从 5 维扩展到 38 维，模型 AUC 从 0.71 提升至 0.89。`
    },
    code: {
      python: `import pandas as pd
import numpy as np

df = pd.DataFrame({
    'price': [100, 250, 80, 300],
    'quantity': [2, 1, 3, 1]
})
# 交叉特征
df['total_spend'] = df['price'] * df['quantity']
# 多项式特征
df['price_sq'] = df['price'] ** 2
# 分箱
df['price_bin'] = pd.cut(df['price'], bins=[0, 100, 200, 300], labels=['低', '中', '高'])
print(df)`,
      matlab: `% MATLAB 特征工程
price = [100; 250; 80; 300];
quantity = [2; 1; 3; 1];
total_spend = price .* quantity;
price_bin = discretize(price, [0 100 200 300], 'categorical', {'低','中','高'});`
    },
    resources: [
      { title: "Feature Engineering for Machine Learning", url: "https://github.com", type: "link" }
    ]
  },
  {
    id: "data-standardization",
    name: "数据标准化 (Data Standardization)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "消除特征间量纲差异，使不同物理意义的变量在同尺度下参与建模，是距离度量与梯度下降类算法的必要前提。",
    principles: `1. Z-score 标准化：x' = (x - μ) / σ，结果均值为0、方差为1。
2. Min-Max 归一化：x' = (x - x_min) / (x_max - x_min)，映射到 [0,1]。
3. RobustScaler：中位数和 IQR 替代均值方差，对异常值更稳健。
4. 对数变换：y = ln(x+1)，压缩极偏态右拖尾分布。`,
    scenarios: [
      "KNN、SVM、神经网络等距离敏感算法的预处理",
      "多地区 GDP 与人口规模差异悬殊的经济指标融合"
    ],
    limitations: [
      "Min-Max 对极值极其敏感，一个异常大会严重压缩其余区间",
      "对数变换仅适用于正值数据"
    ],
    caseStudy: {
      title: "多地区经济指标等权融合",
      description: "人口数量达百万级，GDP 增长率为小数级，直接参与回归会导致增长率权重接近于0。",
      solution: `1. 对人口和 GDP 增长率分别做 Z-score 标准化；
2. 标准化后两特征均值为0、方差为1，消除量纲影响；
3. 在等尺度下参与回归，两类指标贡献度均衡。`
    },
    code: {
      python: `from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

X = np.array([[1e6, 0.02], [2e6, 0.05], [1.5e6, 0.03]])
# Z-score 标准化
X_std = StandardScaler().fit_transform(X)
# Min-Max 归一化
X_norm = MinMaxScaler().fit_transform(X)`,
      matlab: `% MATLAB 标准化与归一化
X = [1e6 0.02; 2e6 0.05; 1.5e6 0.03];
X_std = zscore(X);
X_norm = (X - min(X)) ./ (max(X) - min(X));`
    },
    resources: [
      { title: "特征缩放对距离度量算法的重要性", url: "https://arxiv.org", type: "paper" }
    ]
  },
  {
    id: "dummy-variable",
    name: "虚拟变量转换 (Dummy Variable Encoding)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "将分类变量转换为 0/1 指示变量，使其能够参与线性回归等数值模型，是哑变量化的核心方法。",
    principles: `1. 独热哑变量：M 个类别的变量生成 M 个 0/1 列（或 M-1 个以避免多重共线性）。
2. 标签指示编码：将每个类别映射为唯一的二进制向量。
3. 约束哑变量：引入「参照组」，其余类别相对于参照组的变化量即为哑变量系数。
4. 与连续变量交互：构造「类别×数值」交叉哑变量，捕获斜率差异。`,
    scenarios: [
      "回归模型中纳入「季节（春夏秋冬）」作为四个哑变量",
      "政策评估中「实验组/对照组」的二元哑变量"
    ],
    limitations: [
      "过多独热编码产生高维稀疏矩阵，增加计算成本",
      "完全编码会导致「虚拟变量陷阱」，需删除一列"
    ],
    caseStudy: {
      title: "产品质量等级与销售额关系回归",
      description: "产品等级为「优/良/合格/不合格」四个档次，回归中引入季节和区域哑变量。",
      solution: `1. 季节用四个哑变量，但删除「春季」作为参照组；
2. 区域同样处理；
3. 回归结果显示「等级优」相比「等级合格」平均销售额高 23.5 万元（p<0.01）。`
    },
    code: {
      python: `import pandas as pd

df = pd.DataFrame({
    'grade': ['优', '良', '合格', '不合格', '优'],
    'sales': [100, 80, 60, 30, 95]
})
# 哑变量编码（自动删除第一列避免共线性）
dummies = pd.get_dummies(df['grade'], drop_first=True, prefix='grade')
df = pd.concat([df, dummies], axis=1)
print(df)`,
      matlab: `% MATLAB 虚拟变量矩阵
grade = categorical({'优';'良';'合格';'不合格'});
dummy_matrix = dummyvar(grp2idx(grade));`
    },
    resources: [
      { title: "哑变量回归分析详解", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "missing-value-treatment",
    name: "缺失值处理 (Missing Value Imputation)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "对数据中的缺失值进行填补，是保证样本完整性和模型正常运行的必要步骤。",
    principles: `1. 删除法：直接删除含缺失值的行（MCAR/MAR 条件下无偏）。
2. 均值/中位数填充：简单快速，但会压缩方差、扭曲分布。
3. 多重插补 (MICE)：迭代地用其他特征预测缺失值，重复多次取均值，保留不确定性。
4. KNN 插补：用缺失样本最近的 K 个完整样本均值填充。
5. 随机森林插补：迭代预测缺失值，适合非线性关系。`,
    scenarios: [
      "问卷调查中的非随机缺失值填补",
      "传感器时序数据中偶发性断流记录填补"
    ],
    limitations: [
      "简单均值填充会低估方差，降低特征间相关性",
      "多重插补需数据满足 MAR（随机缺失）假设"
    ],
    caseStudy: {
      title: "临床医学多项指标缺失填补",
      description: "某临床数据集 12 项血液指标中，约 8% 单元格存在缺失，且缺失模式显示与患者年龄相关（MAR）。",
      solution: `1. 采用 MICE 方法，以其余 11 项指标预测缺失值；
2. 迭代 20 轮后取均值；
3. 填补后数据分布与原始完整数据分布的 KL 散度从 0.18 降至 0.04。`
    },
    code: {
      python: `import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

data = pd.DataFrame({'val': [1.2, np.nan, 1.4, np.nan, 1.6]})
# 均值填充
imp_mean = SimpleImputer(strategy='mean')
data_mean = imp_mean.fit_transform(data)
# KNN 填充
imp_knn = KNNImputer(n_neighbors=2)
data_knn = imp_knn.fit_transform(data)`,
      matlab: `% MATLAB 缺失值处理
raw = [1.2; NaN; 1.4; NaN; 1.6];
filled_mean = fillmissing(raw, 'movmean', 2);`
    },
    resources: [
      { title: "sklearn 缺失值填补官方文档", url: "https://scikit-learn.org/stable/modules/impute.html", type: "link" }
    ]
  },
  {
    id: "time-series-smoothing",
    name: "时序数据消颤转换 (Time Series Smoothing & Denoising)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "消除时序数据中的高频噪声与毛刺，揭示数据背后的平滑趋势与周期规律。",
    principles: `1. 滑动平均 (Moving Average)：y_t = (1/(2k+1)) Σ x_{t+i}，简单平滑。
2. 指数加权移动平均 (EWMA)：近期数据权重更高，α 控制衰减速度。
3. Savitzky-Golay 滤波：保持峰值和宽度的高阶多项式拟合平滑。
4. 小波去噪 (Wavelet Denoising)：将信号分解为多尺度小波系数，阈值滤波后重构。`,
    scenarios: [
      "传感器高频毛刺信号清洗",
      "心电图 (ECG) 噪声滤除"
    ],
    limitations: [
      "滑动平均会引入滞后延迟，平滑窗口过大丢失信息",
      "小波去噪需正确选取小波基和阈值，否则会过度平滑"
    ],
    caseStudy: {
      title: "某精密仪器位移传感器去噪",
      description: "某精密仪器 10kHz 高频位移传感器采集的原始信号中混入了工频（50Hz）电磁干扰和随机热噪声。",
      solution: `1. 采用 Daubechies-4 小波分解到第 6 层；
2. 硬阈值滤除低能量高频小波系数；
3. 重构后信噪比 (SNR) 从 12dB 提升至 28dB，位移曲线毛刺完全消失。`
    },
    code: {
      python: `import pandas as pd
import numpy as np

signal = pd.Series([1.2, 1.5, 99.0, 1.4, 1.6, 1.3])
# 移动平均
smoothed = signal.rolling(window=3, center=True).mean()
# 指数加权移动平均
ewma = signal.ewm(span=3).mean()
print("滑动平均:", smoothed.values)
print("EWMA:", ewma.values)`,
      matlab: `% MATLAB 时序平滑
raw = [1.2 1.5 99.0 1.4 1.6 1.3];
smoothed = smoothdata(raw, 'movmean', 'WindowSize', 3);
ewma = tsmovavg(raw, 'e', 3);`
    },
    resources: [
      { title: "小波分析理论及其工程应用", url: "https://www.mathworks.com", type: "link" }
    ]
  },
  {
    id: "winsorizing-trimming",
    name: "缩尾/截尾处理 (Winsorizing & Trimming)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "将极端值替换为指定百分位数值，在保留样本量的同时控制异常值影响，是截断异常的稳健方法。",
    principles: `1. 缩尾 (Winsorizing)：将小于第 p 百分位的值替换为第 p 百分位，大于第 (100-p) 百分位的替换为第 (100-p) 百分位。
2. 截尾 (Trimming)：直接删除超出百分位区间的样本。
3. 双侧缩尾：对称处理上下尾，常用 p=1% 或 p=5%。
4. 单侧缩尾：仅处理上尾或下尾，适用于仅存在单侧异常的场景。`,
    scenarios: [
      "金融收益率分布的极值处理（防止极端值主导回归）",
      "学生成绩分布中极端低分处理"
    ],
    limitations: [
      "缩尾会将极端值的真实信息截断，可能扭曲尾部统计特性",
      "截尾会减少样本量，影响统计检验功效"
    ],
    caseStudy: {
      title: "上市公司高管薪酬回归分析",
      description: "某高管薪酬数据集存在个别天价年薪（>1亿元），严重扭曲 OLS 回归系数，导致 95% 样本的系数失去意义。",
      solution: `1. 对薪酬变量做 1% 双侧缩尾；
2. 缩尾后 R² 从 0.12 提升至 0.67；
3. 核心解释变量回归系数从 0.003 变为 0.31，方向一致但更稳定。`
    },
    code: {
      python: `import numpy as np
from scipy.stats import mstats

data = np.array([10, 15, 20, 25, 30, 500])
# 缩尾至 5%-95% 区间
winsorized = mstats.winsorize(data, limits=[0.05, 0.05])
print("原始:", data)
print("缩尾后:", winsorized)`,
      matlab: `% MATLAB 缩尾处理
x = [10 15 20 25 30 500];
p05 = prctile(x, 5); p95 = prctile(x, 95);
winsorized = min(max(x, p05), p95);`
    },
    resources: [
      { title: "scipy.stats.mstats winsorize 文档", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html", type: "link" }
    ]
  },
  {
    id: "data-sampling",
    name: "数据采样 (Data Sampling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "按特定策略从总体中抽取样本，是统计推断、模型训练和大型数据集高效处理的核心技术。",
    principles: `1. 简单随机抽样：每个样本等概率抽取，保证无偏代表性。
2. 分层抽样 (Stratified Sampling)：按类别比例从各层独立抽样，保证各层在样本中比例一致。
3. 系统抽样：每隔 k 个抽取一个，适合有序总体。
4. 聚类抽样：将总体划分为若干簇，随机抽取部分簇后纳入全部个体。
5. Bootstrap：有放回重复抽样，用于估计统计量的分布与置信区间。`,
    scenarios: [
      "大规模调查中从全国百万用户中抽取代表性样本",
      "模型训练时 GPU 显存不足下的 mini-batch 随机采样"
    ],
    limitations: [
      "随机抽样在小样本下仍可能产生抽样偏差",
      "分层抽样需事先了解各层比例，盲目分层可能适得其反"
    ],
    caseStudy: {
      title: "全国消费者满意度调查抽样设计",
      description: "全国消费者满意度调查总体包含 4.9 亿用户，直接调查成本极高。城市/农村、不同年龄段、不同收入层差异显著。",
      solution: `1. 将总体按城市等级（超一线/一线/二线/其他）和年龄段分层，共 16 层；
2. 按各层实际人口比例分配样本量；
3. 层内采用简单随机抽样，最终 5000 样本的均值估计精度达到 95% 置信区间 ±1.2%。`
    },
    code: {
      python: `import numpy as np
import pandas as pd

df = pd.DataFrame({'id': range(10000), 'value': np.random.randn(10000)})
# 简单随机抽样
sample = df.sample(n=500, random_state=42)
# 分层抽样
strat_sample = df.groupby('category', group_keys=False).apply(lambda x: x.sample(frac=0.1))
# Bootstrap
boot_indices = np.random.choice(len(df), size=1000, replace=True)
boot_sample = df.iloc[boot_indices]`,
      matlab: `% MATLAB 分层抽样
idx = dividem Groups(pop, groupID, 0.1);
sample = pop(idx, :);
% Bootstrap
boot_idx = datasample(1:size(data,1), 1000, 'Replace', true);`
    },
    resources: [
      { title: "sklearn model_selection 采样文档", url: "https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection", type: "link" }
    ]
  },
  {
    id: "sample-weighting",
    name: "样本加权 (Sample Weighting)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "为不同样本赋予不同重要性权重，以平衡分布、反映业务价值或纠正抽样偏差。",
    principles: `1. 逆概率加权 (IPW)：样本权重 = 1 / 入样概率，纠正非随机抽样偏差。
2. 类别平衡加权：正样本权重 = 负样本数 / 正样本数，直接输入模型类别权重参数。
3. 基于损失敏感性加权：对高损失样本或高价值样本赋予更高权重。
4. 时间衰减加权：近期样本权重更高，如 α^t 指数衰减。`,
    scenarios: [
      "因果推断中纠正选择性偏差",
      "欺诈检测中对少数欺诈样本赋予更高权重"
    ],
    limitations: [
      "权重设置依赖业务经验，不当权重可能放大噪声影响",
      "逆概率加权在入样概率接近0时权重趋于无穷大，极不稳定"
    ],
    caseStudy: {
      title: "在线广告点击率模型加权",
      description: "某广告平台收集的曝光-点击日志中，高价值广告曝光量远少于低价值广告，导致模型偏向低价值广告优化。",
      solution: `1. 按广告价值计算逆概率权重：高价广告给 5 倍权重；
2. 同时引入时间衰减权重，近期样本权重为 1.2^天数差；
3. 加权后模型的 CTR 预估偏差降低 18%，高价值广告曝光率提升 32%。`
    },
    code: {
      python: `import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame({
    'feature': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})
# 类别平衡权重
class_weights = {0: 1, 1: 5}  # 少数类权重5
clf = RandomForestClassifier(class_weight=class_weights, random_state=42)
clf.fit(df[['feature']], df['target'])`,
      matlab: `% MATLAB 样本加权分类
classWeights = [1 5];
template = templateTree('ClassWeights', classWeights);
clf = fitcensemble(data, 'Method', 'Bag', 'Learners', template);`
    },
    resources: [
      { title: "sklearn 类别权重文档", url: "https://scikit-learn.org/stable/modules/svm.html#weighted-samples", type: "link" }
    ]
  },
  {
    id: "data-transformation",
    name: "数据变换 (Data Transformation)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "通过非线性数学变换使数据满足模型假设或改善分布形态，是连接原始数据与统计模型的桥梁。",
    principles: `1. 对数变换 y = ln(x+1)：压缩右偏分布，降低异方差。
2. Box-Cox 变换：(x^λ - 1)/λ，λ 由最大似然估计，自动找到最优幂次。
3. 平方根变换：y = √x，适合轻度右偏。
4. 幂变换 (Yeo-Johnson)：支持含零和负值的通用化 Box-Cox。
5. 分位数变换：将数据映射到均匀或正态分布，消除偏态和异常值影响。`,
    scenarios: [
      "收入、房价等严重右偏分布的正态化",
      "方差随均值增大的异方差数据调和"
    ],
    limitations: [
      "Box-Cox 仅适用于严格正值数据",
      "变换后的变量单位失去原始物理含义，解释性下降"
    ],
    caseStudy: {
      title: "城市家庭年收入分布正态化",
      description: "某城市 5000 户家庭年收入分布严重右偏（均值 15 万，中位数 8 万，最大值 800 万），直接回归残差图呈明显喇叭形异方差。",
      solution: `1. 对年收入取自然对数，分布接近正态；
2. 对数变换后残差方差齐性检验通过 (White's test p=0.73)；
3. 回归系数解释为「收入每增加 1%，因变量变化约 β%」。`
    },
    code: {
      python: `import numpy as np
import pandas as pd
from scipy import stats

income = np.array([5e4, 8e4, 1e5, 5e5, 2e6, 8e6])
# 对数变换
log_income = np.log1p(income)
# Box-Cox 变换 (自动求最优 lambda)
bc_income, lam = stats.boxcox(income + 1)
print(f"Box-Cox lambda={lam:.3f}")
print(f"变换后偏度: {stats.skew(bc_income):.3f}")`,
      matlab: `% MATLAB Box-Cox 变换
income = [50000 80000 1e5 5e5 2e6 8e6];
[bc_income, lambda] = boxcox(income + 1);
fprintf('最优 lambda=%.3f\\n', lambda);`
    },
    resources: [
      { title: "scipy.stats Box-Cox 变换文档", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html", type: "link" }
    ]
  },
  {
    id: "sample-balance-smote",
    name: "样本均衡-过采样-SMOTE (SMOTE Oversampling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "通过在少数类样本邻域内合成新样本，解决分类任务中正负样本极度不平衡的问题。",
    principles: `1. 在少数类样本 x 的 k 近邻（通常 k=5）中随机选择一个近邻 x_nn。
2. 在连线上作随机插值：x_new = x + α·(x_nn - x)，α ∈ [0,1] 随机。
3. 重复上述过程直到各类样本量均衡。
4. Borderline-SMOTE：仅对边界附近少数样本合成，增强分类边界识别能力。`,
    scenarios: [
      "信用卡欺诈检测（欺诈率仅 0.1%）",
      "罕见疾病医学诊断"
    ],
    limitations: [
      "在类别严重重叠区域合成会制造虚假样本，加剧类别重叠",
      "对噪声样本敏感，噪声点周围也会合成大量合成样本"
    ],
    caseStudy: {
      title: "信用卡欺诈检测样本均衡",
      description: "某银行 50 万条交易记录中，欺诈交易仅 487 条（占 0.097%），直接训练准确率 99.9% 但对欺诈完全无识别能力。",
      solution: `1. 使用 SMOTE 将 487 条欺诈样本扩增到 25 万条（与正常样本 1:1 均衡）；
2. 用 Borderline-SMOTE 对边界欺诈样本加强合成；
3. 均衡后模型的欺诈召回率从 12% 提升至 89%，同时保持精度 78%。`
    },
    code: {
      python: `from imblearn.over_sampling import SMOTE
import numpy as np

np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.concatenate([np.zeros(990), np.ones(10)])
# SMOTE 过采样
sm = SMOTE(k_neighbors=5, random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(f"原始: {len(y)} | 均衡后: {len(y_res)}, 少数类: {sum(y_res)}")`,
      matlab: `% MATLAB SMOTE 近似实现
X_maj = X(y==0,:); X_min = X(y==1,:);
% 参考 GitHub: imbalanced-algorithms`
    },
    resources: [
      { title: "SMOTE 原始论文", url: "https://arxiv.org/abs/1106.1813", type: "paper" }
    ]
  },
  {
    id: "sample-balance-adasyn",
    name: "样本均衡-过采样-ADASYN (ADASYN Adaptive Oversampling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "根据样本周围少数类密度自适应调节合成量，优先合成难以学习的边界区域样本。",
    principles: `1. 计算每个少数类样本 x_i 的多数类邻居占比 r_i（密集度指标）。
2. 计算需合成样本总数：G = (m_l - m_s)·β，β 为目标平衡度。
3. 按比例 r_i = r_i / Σ r_i 分配各样本需合成的数量。
4. 在高 r_i（低密度、难学）区域合成更多样本。`,
    scenarios: [
      "医疗影像早期肿瘤检测（肿瘤样本稀少且特征模糊）",
      "网络入侵检测中的新型攻击模式识别"
    ],
    limitations: [
      "对噪声样本极为敏感，在噪声密集区域会生成大量无效合成样本",
      "计算复杂度高于 SMOTE，尤其在高维数据下"
    ],
    caseStudy: {
      title: "早期肺结节 CT 影像恶性判别",
      description: "1000 例 CT 影像中仅 35 例确诊恶性，且恶性样本特征空间分布稀疏，与良性样本高度重叠。",
      solution: `1. ADASYN 发现恶性样本的边界区域（局部密度低）占比高达 60%；
2. 在这些低密度区域合成约 200 个新恶性样本；
3. 模型在恶性样本召回率从 45% 提升至 82%，同时假阳性率控制在 15%。`
    },
    code: {
      python: `from imblearn.over_sampling import ADASYN
import numpy as np

X = np.random.randn(500, 4)
y = np.concatenate([np.zeros(450), np.ones(50)])
# ADASYN 自适应合成
ada = ADASYN(n_neighbors=5, random_state=42)
X_res, y_res = ada.fit_resample(X, y)
print(f"少数类: 原始={sum(y==1)}, 均衡后={sum(y_res==1)}")`,
      matlab: `% MATLAB ADASYN 近似实现
% ADASYN 本质是对低密度少数类区域加权 SMOTE
% 参考 GitHub: imbalanced-algorithms`
    },
    resources: [
      { title: "ADASYN 论文", url: "https://www.cs.cmu.edu/~qyj/DRSA/adasyn.pdf", type: "paper" }
    ]
  },
  {
    id: "sample-balance-random-under",
    name: "样本均衡-下采样-随机 (Random Under-sampling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "随机丢弃多数类样本以平衡类别分布，方法简单直接，适用于多数类样本严重过剩的场景。",
    principles: `1. 从多数类中随机抽取与少数类等量（或设定比例）的样本。
2. 完全随机丢弃，未利用任何样本特征信息，可能删除重要样本。
3. 常与集成方法结合（如 EasyEnsemble：多次随机下采样后分别训练基分类器再集成）。
4. 欠采样率 = 少数类样本数 / 多数类样本数。`,
    scenarios: [
      "工业异常检测（正常样本充足，缺陷样本稀缺）",
      "文本分类中的类别极度不平衡"
    ],
    limitations: [
      "随机丢弃可能删除包含关键信息的多数类样本，导致欠拟合",
      "大幅减少训练样本量，损失数据多样性"
    ],
    caseStudy: {
      title: "印刷电路板 (PCB) AOI 缺陷检测",
      description: "某 PCB 工厂 10 万张 AOI 检测图片中，仅 800 张含缺陷（0.8%），且缺陷类型多样。",
      solution: `1. 随机下采样 800 张正常样本，与 800 张缺陷样本构成均衡训练集；
2. 结合集成学习：重复随机下采样 10 次，每次训练一个 CNN 分类器；
3. 10 个基分类器投票集成，缺陷召回率达 91%，误报率 <3%。`
    },
    code: {
      python: `import numpy as np
from sklearn.utils import resample

X = np.random.randn(1000, 3)
y = np.concatenate([np.zeros(950), np.ones(50)])
# 随机下采样至 1:3 比例
n_minority = sum(y == 1)
X_minority = X[y == 1]; y_minority = y[y == 1]
X_majority = X[y == 0]; y_majority = y[y == 0]
X_majority_downsampled, y_majority_downsampled = resample(
    X_majority, y_majority, n_samples=n_minority * 3, random_state=42
)
X_balanced = np.vstack([X_minority, X_majority_downsampled])
y_balanced = np.concatenate([y_minority, y_majority_downsampled])`,
      matlab: `% MATLAB 随机下采样
idx_majority = find(y == 0);
idx_minority = find(y == 1);
n_target = length(idx_minority) * 3;
selected = datasample(idx_majority, n_target, 'Replace', false);
X_balanced = [X(idx_minority,:); X(selected,:)];`
    },
    resources: [
      { title: "EasyEnsemble 集成欠采样论文", url: "https://cs.nju.edu.cn/wuxx/docs/papers/Liu+08.pdf", type: "paper" }
    ]
  },
  {
    id: "sample-balance-cluster-centroids",
    name: "样本均衡-下采样-Cluster Centroids (Cluster Centroids Under-sampling)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "利用聚类将多数类样本压缩为 K 个聚类中心，以少数类样本数为 K，在保留分布特征的同时大幅缩减样本量。",
    principles: `1. 对多数类样本运行 K-Means（或 MiniBatchKMeans），聚成 K 个簇。
2. 用各簇的质心（Centroid）代替该簇内所有原始样本。
3. K 通常设为与少数类样本数相近，从而自然实现类别均衡。
4. 每个质心保留了该簇样本的分布特征，避免随机丢弃造成的关键样本损失。`,
    scenarios: [
      "大规模不平衡数据集的高效下采样",
      "保留多数类分布形态的智能压缩"
    ],
    limitations: [
      "聚类本身可能丢失原始样本的细节特征",
      "K 值选择需要领域知识或交叉验证"
    ],
    caseStudy: {
      title: "信用卡交易异常检测数据压缩",
      description: "某银行 500 万条交易记录中正常交易 499.5 万条，异常仅 5000 条。普通随机下采样会丢失大量正常行为模式信息。",
      solution: `1. 对 499.5 万正常交易做 K-Means，聚成 5000 个簇；
2. 用 5000 个簇质心替代全部正常交易；
3. 与 5000 条异常样本构成 1:1 均衡训练集；
4. 相比随机下采样，Cluster Centroids 保留了正常交易的分布特征，模型对新型欺诈识别率提升 27%。`
    },
    code: {
      python: `from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import MiniBatchKMeans
import numpy as np

X = np.random.randn(5000, 4)
y = np.concatenate([np.zeros(4500), np.ones(500)])
# ClusterCentroids 欠采样
cc = ClusterCentroids(
    estimator=MiniBatchKMeans(n_init=3, random_state=42),
    sampling_strategy='auto', random_state=42
)
X_res, y_res = cc.fit_resample(X, y)
print(f"原始样本量: {len(y)} → 均衡后: {len(y_res)}")`,
      matlab: `% MATLAB Cluster Centroids 近似实现
X_maj = X(y==0,:);  n = sum(y==1);
[~, C] = kmeans(X_maj, n, 'MaxIter', 100);
X_res = [X(y==1,:); C];`
    },
    resources: [
      { title: "imbalanced-learn ClusterCentroids 文档", url: "https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html", type: "link" }
    ]
  },
  {
    id: "sample-balance-smote-tomek",
    name: "样本均衡-组合采样-SMOTE+Tomek Links (SMOTE + Tomek Links)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "先用 SMOTE 过采样增加少数类样本，再用 Tomek Links 删除类别边界上的噪声样本，双管齐下改善分类边界质量。",
    principles: `1. SMOTE 阶段：在少数类邻域内合成新样本，初步缓解类别失衡。
2. Tomek Link 检测：若两个不同类样本互为最近邻，则构成一个 Tomek Link，代表类别边界模糊区域。
3. 删除策略：删除 Tomek Link 中的多数类样本（或同时删除两侧），以清理类别边界。
4. 优势：既增加了少数类样本量，又清理了边界噪声，比单独使用任一方法效果更优。`,
    scenarios: [
      "疾病早期筛查（需增加少数正类样本且清理边界噪声）",
      "高精度制造缺陷检测"
    ],
    limitations: [
      "SMOTE 可能在线性插值时产生位于多数类区域内的虚构样本（需配合 Tomek 删除）",
      "Tomek Links 仅删除边界样本，对类别严重重叠情况效果有限"
    ],
    caseStudy: {
      title: "高精度光纤连接器瑕疵检测",
      description: "某光纤连接器生产线上，正常品 9800 件，轻微瑕疵 150 件，严重缺陷 50 件，三类高度重叠，边界模糊。",
      solution: `1. 先对轻微瑕疵和严重缺陷做 SMOTE 插值扩增到各类 3000 件；
2. 检测并删除 Tomek Links 中的 420 对边界样本；
3. 均衡后 SVM 分类器边界清晰度提升，召回率从 62% 提升至 91%，同时误报率降低 40%。`
    },
    code: {
      python: `from imblearn.combine import SMOTETomek
import numpy as np

X = np.random.randn(2000, 6)
y = np.concatenate([np.zeros(1800), np.ones(200)])
# SMOTE + Tomek Links 组合
smt = SMOTETomek(sampling_strategy='auto', random_state=42)
X_res, y_res = smt.fit_resample(X, y)
print(f"原始少数类: {sum(y==1)} → 处理后: {sum(y_res==1)}")`,
      matlab: `% MATLAB SMOTE+Tomek 近似实现
% 1) SMOTE 过采样
% 2) 检测 Tomek Links (互为最近邻的不同类样本对)
% 3) 删除多数类端点
% 参考 imbalanced-learn 组合采样器`
    },
    resources: [
      { title: "imbalanced-learn SMOTETomek 文档", url: "https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html", type: "link" }
    ]
  },
  {
    id: "sample-balance-smoteenn",
    name: "样本均衡-组合采样-SMOTE+ENN (SMOTE + Edited Nearest Neighbors)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "先用 SMOTE 过采样，再用 ENN 删除被多数类邻居包围的少数类样本，进一步清理噪声和边界异常。",
    principles: `1. SMOTE 阶段：合成少数类样本以解决类别失衡。
2. ENN (Edited Nearest Neighbors)：用 KNN 重新分类，删除被超过半数邻居与自己类别不同的样本。
3. ENN 会删除两类交界处被错误邻居包围的样本（含 SMOTE 生成的噪声合成样本）。
4. 组合效果优于单独使用，常见搭配为 SMOTETomek 和 SMOTEENN。`,
    scenarios: [
      "不平衡医疗诊断数据集的清洗与均衡",
      "高精度工业缺陷检测中的噪声样本剔除"
    ],
    limitations: [
      "ENN 可能过度删除有用样本，尤其在类别重叠严重时",
      "组合方法计算成本最高"
    ],
    caseStudy: {
      title: "芯片焊点缺陷检测数据清洗",
      description: "某芯片封装生产线 2 万张焊点图片中，正常 19500 张，缺陷 500 张，SMOTE 过采样后出现大量边界模糊合成样本。",
      solution: `1. SMOTE 将缺陷扩增到 8000 张；
2. ENN 删除 3NN 中超过半数与自身类别不一致的样本，共剔除 1200 张；
3. 清洗后模型对边界模糊缺陷的识别精确率从 61% 提升至 88%。`
    },
    code: {
      python: `from imblearn.combine import SMOTEENN
import numpy as np

X = np.random.randn(2000, 6)
y = np.concatenate([np.zeros(1800), np.ones(200)])
# SMOTE + ENN 组合
smoteenn = SMOTEENN(sampling_strategy='auto', random_state=42)
X_res, y_res = smoteenn.fit_resample(X, y)
print(f"均衡清洗后样本: {len(y_res)}, 少数类: {sum(y_res==1)}")`,
      matlab: `% MATLAB SMOTE+ENN 近似实现
% 1) SMOTE 过采样
% 2) ENN: 用 KNN 重新分类，删除被错误邻居包围的样本
% 参考 imbalanced-learn SMOTEENN`
    },
    resources: [
      { title: "imbalanced-learn SMOTEENN 文档", url: "https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html", type: "link" }
    ]
  },
  {
    id: "feature-selection-variance",
    name: "特征筛选-方差选择法 (Variance Threshold Feature Selection)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "剔除方差低于阈值的准常数特征，是最基础、最高效的预筛选步骤，可大幅减少特征维度。",
    principles: `1. 计算各特征方差：Var(X) = E[(X - μ)²]。
2. 设定方差阈值 τ，剔除 Var(X) < τ 的特征。
3. 默认阈值 τ=0（剔除零方差特征），或按业务设定如 τ=0.01。
4. 注意：方差受量纲影响，需先做标准化再计算方差。`,
    scenarios: [
      "大规模特征工程后快速预筛选（10万维→1万维）",
      "剔除问卷中全选同一选项的无效问题"
    ],
    limitations: [
      "仅考虑单特征方差，不衡量特征与目标变量的相关性",
      "高方差特征未必有预测能力，可能引入噪声"
    ],
    caseStudy: {
      title: "基因表达数据快速预筛选",
      description: "某基因表达数据集包含 20000 个基因特征，其中约 40% 基因在所有样本中表达量几乎相同（方差接近零）。",
      solution: `1. 对表达量做 Z-score 标准化；
2. 设定方差阈值 0.1，剔除 7840 个低方差基因；
3. 剩余 12160 个特征进入后续相关性筛选；
4. 预筛选使后续模型训练速度提升 3.2 倍。`
    },
    code: {
      python: `import numpy as np
from sklearn.feature_selection import VarianceThreshold

X = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 0], [0, 1, 0]])
# 方差阈值筛选
selector = VarianceThreshold(threshold=0.5)
X_selected = selector.fit_transform(X)
print(f"保留特征索引: {selector.get_support(indices=True)}")`,
      matlab: `% MATLAB 方差阈值筛选
X_std = zscore(X);
variances = var(X_std, 0, 1);
kept = variances > 0.5;
X_selected = X(:, kept);`
    },
    resources: [
      { title: "sklearn VarianceThreshold 文档", url: "https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance", type: "link" }
    ]
  },
  {
    id: "feature-selection-rf-importance",
    name: "特征筛选-随机森林特征重要度 (Random Forest Feature Importance)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "利用随机森林中每棵树分裂时各特征带来的信息增益评估特征重要性，是最通用的嵌入式特征筛选方法。",
    principles: `1. 平均不纯度下降 (MDI)：每棵树分裂时统计各特征的加权不纯度下降量，求平均。
2. 置换重要度 (Permutation Importance)：打乱某特征后模型精度下降量，下降越多越重要，不受特征量纲影响。
3. 递归特征消除 (RFE)：从所有特征出发，递归删除最不重要特征，逐步评估子集性能。
4. Gini 重要性 vs. Mean Decrease Accuracy：前者有偏向高基数特征的偏差，后者更可靠但计算更慢。`,
    scenarios: [
      "生物标志物筛选（从数百个候选指标中找出最相关生物标志物）",
      "客户流失预测特征归因"
    ],
    limitations: [
      "MDI 有偏向高基数特征和连续特征的偏差",
      "特征重要性反映的是对当前模型的价值，不代表因果重要性"
    ],
    caseStudy: {
      title: "肺功能检验指标重要性排序",
      description: "某临床数据集包含 23 项肺功能检验指标，希望找出对 COPD 诊断最重要的前 5 项指标以降低检测成本。",
      solution: `1. 训练 500 棵随机森林，用 Permutation Importance 评估；
2. 选出前 5 关键指标：FEV1/FVC 比值、一秒用力呼气容积、残气量、DLCO、RV/TLC；
3. 仅用这 5 项构建精简诊断模型，AUC=0.93，与全特征模型 AUC=0.95 相当，但检测成本降低 78%。`
    },
    code: {
      python: `from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

X = np.random.randn(200, 10)
y = (X[:, 0] * 2 + X[:, 3] > 0).astype(int)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
# 置换重要性（更可靠）
perm_imp = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
top_idx = perm_imp.importances_mean.argsort()[-5:][::-1]
print("Top 5 特征索引:", top_idx)`,
      matlab: `% MATLAB 随机森林特征重要度
B = TreeBagger(100, X, y, 'Method', 'classification');
featImp = oobPermutedPredictorImportance(B);
[~, topIdx] = sort(featImp, 'descend');
topIdx(1:5)`
    },
    resources: [
      { title: "sklearn 随机森林特征重要度文档", url: "https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html", type: "link" }
    ]
  },
  {
    id: "feature-selection-xgboost",
    name: "特征筛选-XGBoost 特征重要度 (XGBoost Feature Importance)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "基于梯度提升树分裂各特征的频次和增益量评估重要性，是 XGBoost 内置的可靠特征筛选方法。",
    principles: `1. weight（分裂频次）：特征在所有树中被选作分裂节点的总次数。
2. gain（分裂增益）：特征带来目标函数下降的总增益量，最常用。
3. cover（覆盖量）：特征分裂所影响的样本数加权总和。
4. 按重要性排序后取 Top K 或设定阈值（如 importance > 0.01）筛选特征。`,
    scenarios: [
      "金融风控模型特征工程后的重要性排序与精简",
      "推荐系统用户画像特征重要性归因"
    ],
    limitations: [
      "高相关特征会分散重要性（如特征 A 和 B 高度相关，单独看都不高）",
      "XGBoost 默认仅输出相对重要度，需结合业务阈值做绝对判断"
    ],
    caseStudy: {
      title: "银行贷款违约风险特征筛选",
      description: "某银行贷款风控模型包含 89 个特征（包括 20 个强相关衍生特征），模型可解释性和推理速度均不理想。",
      solution: `1. 训练 XGBoost 模型，提取 gain 重要度排序；
2. 发现前 15 个特征已包含 92% 的信息量；
3. 精简后模型 AUC 仅从 0.891 降至 0.883，推理速度提升 6 倍；
4. 客户经理可直接解读 Top 15 特征的决策影响。`
    },
    code: {
      python: `import xgboost as xgb
import numpy as np

X = np.random.randn(500, 15)
y = (X[:, 0] * 1.5 + X[:, 5] * -0.8 > 0).astype(int)
dtrain = xgb.DMatrix(X, label=y)
params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
model = xgb.train(params, dtrain, num_boost_round=100)
# 获取 gain 重要度
importance = model.get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("Top 5:", sorted_imp[:5])`,
      matlab: `% MATLAB XGBoost 特征重要度
model = fitcensemble(X, y, 'Method', 'LogitBoost');
featImp = predictorImportance(model);
[~, idx] = sort(featImp, 'descend');
fprintf('Top features: %d %d %d\\n', idx(1:5));`
    },
    resources: [
      { title: "XGBoost 官方文档", url: "https://xgboost.readthedocs.io", type: "link" }
    ]
  },
  {
    id: "feature-selection-correlation",
    name: "特征筛选-相关系数法 (Correlation-based Feature Selection)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "通过计算特征与目标变量的相关系数，保留高相关特征，剔除与目标无关的冗余特征。",
    principles: `1. Pearson 相关系数：衡量线性相关，r = Cov(X,Y) / (σX·σY)，取值 [-1, 1]。
2. Spearman 等级相关系数：基于秩次，适用于单调非线性关系，不受极端值影响。
3. 互相关矩阵：剔除特征间高度相关的冗余特征（|r| > 0.9 通常视为冗余）。
4. 阈值筛选：|r| > 0.3（或业务设定）保留，否则剔除。`,
    scenarios: [
      "回归建模前的线性相关性快速探测",
      "剔除多指标体系中的高度共线性冗余变量"
    ],
    limitations: [
      "Pearson 仅捕捉线性关系，非线性相关的强特征可能被漏选",
      "高相关特征间相互遮掩，需结合其他方法综合判断"
    ],
    caseStudy: {
      title: "宏观经济指标对 GDP 增长率相关性分析",
      description: "某省 8 个宏观经济指标（GDP、固投、消费、出口等）均与 GDP 增长率高度相关（r>0.85），存在严重冗余。",
      solution: `1. 计算 Pearson 相关系数矩阵，发现固投与 GDP 相关系数高达 0.97；
2. 剔除相关系数 > 0.9 的一对冗余特征中信息量更少者；
3. 从 8 个特征精简至 5 个独立指标，VIF 从平均 12.3 降至 2.8，OLS 回归不再发散。`
    },
    code: {
      python: `import pandas as pd
import numpy as np

df = pd.DataFrame({
    'X1': np.random.randn(100),
    'X2': np.random.randn(100) * 0.5 + 1,
    'X3': np.random.randn(100) + 2
})
# 计算与目标变量相关
target = pd.Series(np.random.randn(100))
corr_with_target = df.apply(lambda col: col.corr(target))
print("与目标相关性:", corr_with_target.sort_values(ascending=False))`,
      matlab: `% MATLAB 相关系数分析
R = corrcoef([X target]);
corr_with_target = R(1:end-1, end);
keep = abs(corr_with_target) > 0.3;`
    },
    resources: [
      { title: "scipy.stats 相关系数文档", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html", type: "link" }
    ]
  },
  {
    id: "feature-selection-mutual-info",
    name: "特征筛选-互信息法 (Mutual Information Feature Selection)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "利用互信息衡量特征与目标变量之间的统计依赖强度，可捕获任意非线性关系，是相关性分析的强大补充。",
    principles: `1. 互信息 I(X; Y) = H(X) + H(Y) - H(X, Y) = E[log(p(x,y) / (p(x)p(y))]。
2. I(X; Y) = 0 当且仅当 X 与 Y 统计独立（完全无关系）。
3. 相比 Pearson，互信息可捕获任意非线性关系（单调、周期性等）。
4. 估计方法：连续特征用 k 近邻估计（sklearn 默认）。`,
    scenarios: [
      "基因-表型非线性关联分析",
      "文本分类中词频与情感极性的非线性关系捕捉"
    ],
    limitations: [
      "连续变量的互信息估计有方差，需谨慎设置 k 近邻参数",
      "不能直接衡量因果关系，只能检测统计依赖"
    ],
    caseStudy: {
      title: "广告点击率预测特征筛选",
      description: "某 DSP 广告平台收集了 200 多个特征，其中包含用户设备类型、浏览时间段、广告尺寸等。部分特征与 CTR 存在非线性关系。",
      solution: `1. 使用 sklearn 的 mutual_info_classif 计算各特征与 CTR 的互信息；
2. 发现「时段×设备」交叉特征的 MI=0.38，远超其他单特征（最高 0.12）；
3. 按 MI 排序取 Top 30 特征后，LR + 交叉特征模型的 AUC=0.82，超越 Pearson 筛选版本的 AUC=0.75。`
    },
    code: {
      python: `from sklearn.feature_selection import mutual_info_classif
import numpy as np

X = np.random.randn(200, 10)
y = (X[:, 0]**2 + X[:, 3] > 0).astype(int)  # 非线性依赖
mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
mi_scores = dict(zip(range(10), mi))
sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
print("互信息排序:", sorted_mi[:5])`,
      matlab: `% MATLAB 互信息估计 (Information Theory Toolbox)
mi_est = mutualinfo(X, y, 'k', 5);
[~, idx] = sort(mi_est, 'descend');`
    },
    resources: [
      { title: "sklearn 互信息特征选择文档", url: "https://scikit-learn.org/stable/modules/feature_selection.html#mutual-information-for-classification", type: "link" }
    ]
  },
  {
    id: "feature-selection-chi2",
    name: "特征筛选-卡方检验法 (Chi-Square Feature Selection)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "利用卡方检验评估类别型特征与目标变量的独立性，是分类任务中最经典的统计特征筛选方法。",
    principles: `1. 构建列联表：统计每个特征类别与目标类别组合的观测频数 O_ij。
2. 计算期望频数 E_ij：在零假设（特征与目标独立）下各格的期望值。
3. 卡方统计量：χ² = Σ((O_ij - E_ij)² / E_ij)，自由度 (m-1)(n-1)。
4. χ² 越大，p 值越小，说明特征与目标越不独立（即越有预测价值）。`,
    scenarios: [
      "文本分类中的词频/文档词类与情感标签的独立性检验",
      "医学诊断中症状类型与疾病诊断的关联分析"
    ],
    limitations: [
      "仅适用于类别型特征（离散变量），连续特征需先离散化",
      "对低频类别极度敏感，期望频数 < 5 时结果不可靠"
    ],
    caseStudy: {
      title: "网购用户购买意向特征重要性评估",
      description: "某电商平台收集了用户职业、年龄段、登录设备类型等 12 个类别特征，与「是否购买」标签做卡方检验筛选。",
      solution: `1. 对连续特征（年龄段、购买频次）做分箱离散化；
2. 计算 12 个特征的卡方值；
3. 发现「商品浏览深度」（χ²=142.3）和「历史购买次数」（χ²=98.7）的 p<0.001，高度相关；
4. 保留 Top 8 特征后，朴素贝叶斯分类器精度从 61% 提升至 79%。`
    },
    code: {
      python: `from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd

X_cat = pd.DataFrame({'job': ['老师','医生','程序员']*100, 'city': ['北京','上海','深圳']*100})
y = np.concatenate([np.zeros(150), np.ones(150)])
X_enc = OrdinalEncoder().fit_transform(X_cat)
# 卡方检验
chi2_scores, p_values = chi2(X_enc, y)
print("卡方值:", chi2_scores, "p值:", p_values)`,
      matlab: `% MATLAB 卡方检验
obs = [30 10; 15 45];
[chi2_stat, p_val] = chi2test(obs);`
    },
    resources: [
      { title: "sklearn chi2 特征选择文档", url: "https://scikit-learn.org/stable/modules/feature_selection.html#chi-square-feature-selection", type: "link" }
    ]
  },
  {
    id: "feature-selection-vif",
    name: "特征筛选-VIF 方差膨胀因子法 (Variance Inflation Factor)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "通过量化特征间的多重共线性程度，剔除高度冗余的共线性特征，是回归建模前最重要的降维手段之一。",
    principles: `1. 对特征 X_j，以 X_j 为因变量，对其余所有特征做多元回归，得 R²_j。
2. 计算 VIF_j = 1 / (1 - R²_j)，R²_j 越接近 1（与其他特征高度相关），VIF 越大。
3. VIF > 10 通常认为存在严重共线性，应剔除或合并。
4. 逐步剔除最高 VIF 特征，迭代直到所有 VIF < 阈值。`,
    scenarios: [
      "多元回归中消除高度共线性（GDP、固投、消费三指数高度联动）",
      "经济指标体系中多重共线性检测与剔除"
    ],
    limitations: [
      "VIF 仅衡量线性共线性，无法检测非线性关系",
      "高 VIF 特征若与目标强相关，可能不宜直接剔除，需权衡"
    ],
    caseStudy: {
      title: "房地产价格影响因素共线性诊断",
      description: "某城市 8 个经济指标预测房价，其中「建筑面积」「套内面积」「公摊面积」三者高度共线性，OLS 回归系数符号出现倒号。",
      solution: `1. 计算所有特征 VIF，发现建筑面积=42.3、套内面积=38.1、公摊面积=19.7，均超过阈值 10；
2. 剔除「建筑面积」（VIF 最高），保留套内和公摊；
3. 重新回归后所有 VIF < 5，系数符号正常，AIC 从 342 降至 287。`
    },
    code: {
      python: `import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame({
    'area': [100, 120, 80, 90],
    'rooms': [3, 4, 2, 3],
    'price': [500, 600, 400, 450]
})
# 计算 VIF
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif[vif['VIF'] > 10])`,
      matlab: `% MATLAB VIF 计算
X_std = zscore(X);
R = corrcoef(X_std);
VIF = diag(inv(R))';
highVIF = VIF > 10;`
    },
    resources: [
      { title: "statsmodels VIF 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html", type: "link" }
    ]
  },
  {
    id: "feature-selection-rfe",
    name: "特征筛选-递归消除特征法 (Recursive Feature Elimination, RFE)",
    category: "preprocess",
    categoryName: "🛠️ 数据预处理与特征工程",
    summary: "从所有特征出发递归删除最不重要特征，通过评估每个子集的性能找到最优特征子集，是包裹式特征选择的代表方法。",
    principles: `1. 在全特征上训练基模型（如 SVM、随机森林、逻辑回归）。
2. 根据模型内置重要性排序，删除最不重要的特征。
3. 在剩余特征上重新训练，重复步骤 1-2。
4. 输出各特征被消除的顺序，以及各子集大小对应的交叉验证性能曲线。
5. 可通过 cv 参数直接返回达到最高性能的推荐特征数。`,
    scenarios: [
      "在候选特征数量较多时系统搜索最优子集（如 50→5 逐步筛选）",
      "结合稳定性选择降低随机波动影响"
    ],
    limitations: [
      "计算成本高，每次迭代需重新训练模型",
      "对噪声特征较多的数据集，可能在早期错误删除重要特征"
    ],
    caseStudy: {
      title: "乳腺癌生物标志物递归筛选",
      description: "某蛋白组学数据集包含 100 个候选生物标志物，希望通过 RFE 找到最具诊断价值的子集。",
      solution: `1. 使用 SVM-RFE，从 100 个标志物出发递归删除；
2. 10 折交叉验证显示，Top 12 标志物子集达最高 AUC=0.94；
3. 这 12 个标志物中包括已知的 ER、PR、HER2 核心标志物，以及 4 个新发现的潜在标志物。`
    },
    code: {
      python: `from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(200, 15)
y = (X[:, 0] + X[:, 5] > 0).astype(int)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=clf, n_features_to_select=5, step=1)
rfe.fit(X, y)
print("最优特征索引:", rfe.support_)
print("特征排名:", rfe.ranking_)`,
      matlab: `% MATLAB 递归特征消除
model = fitcensemble(X, y, 'Method', 'LogitBoost');
[featIdx, score] = fsrfec(model, X, y);
topFeatures = featIdx(score > 0.8);`
    },
    resources: [
      { title: "sklearn RFE 文档", url: "https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination", type: "link" }
    ]
  },

  // ==================== 2. 数据降维 (reduction) ====================
  {
    id: "pca-dimension-reduction",
    name: "数据降维-PCA 降维 (Principal Component Analysis)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "利用正交投影手段提取最大方差特征组合，是最经典且应用最广泛的线性降维方法。",
    principles: `1. 标准化输入矩阵并求出协方差矩阵 C = Cov(X)。
2. 解出特征值方程 C·u_i = λ_i·u_i。
3. 按特征值从大到小排序，取前 k 个主分量，使累加贡献率达 85% 以上。
4. 将数据投影到前 k 个主分量构成的低维子空间。`,
    scenarios: [
      "高维度多源光谱或电磁检测特征合并",
      "去除各变量间由高相似及强协同效应产生的共线性"
    ],
    limitations: [
      "PCA 只作全局线性投影，无法精确识别局部流形边界",
      "降维后的主成分含义宽泛，缺乏原始业务对应性"
    ],
    caseStudy: {
      title: "某地区多维度水体指标综合降维",
      description: "共包含 15 项物理、无机、重金属指标，它们高度联动，希望浓缩为 3 个主指标进行可视化。",
      solution: `1. 通过 Z-score 规范各项数值；
2. 降维发现前三个主分量累加贡献达 89.4%；
3. 借此计算省市水质空间流形排名。`
    },
    code: {
      python: `from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(50, 6)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"各主成分贡献率: {pca.explained_variance_ratio_}")
print(f"累加贡献率: {sum(pca.explained_variance_ratio_):.2%}")`,
      matlab: `% MATLAB 执行标准 PCA 降维
data = randn(50, 6);
[coefs, score, latent, tsquare, explained] = pca(data);
cum_contribution = cumsum(explained);`
    },
    resources: [
      { title: "sklearn PCA 官方文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html", type: "link" }
    ]
  },
  {
    id: "lda-dimension-reduction",
    name: "数据降维-线性判别法 LDA (Linear Discriminant Analysis)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "在保证类别可分性的前提下最大化类间方差与类内方差之比，是有监督降维的代表方法。",
    principles: `1. 计算类内散度矩阵 S_W = Σ Σ (x - μ_i)(x - μ_i)^T。
2. 计算类间散度矩阵 S_B = Σ n_i (μ_i - μ)(μ_i - μ)^T。
3. 求解广义特征值问题 S_W⁻¹ S_B v = λ v。
4. 取前 L-1 个最大特征值对应的特征向量（L 为类别数），投影数据。`,
    scenarios: [
      "人脸识别中的特征降维（如 Eigenfaces）",
      "文本分类中的类别可分性增强降维"
    ],
    limitations: [
      "最多只能降到 L-1 维，对于二分类最多降至 1 维",
      "当训练样本不足时，类内散度矩阵奇异会导致求解不稳定"
    ],
    caseStudy: {
      title: "鸢尾花品种分类特征降维",
      description: "鸢尾花数据集有 4 个特征和 3 个类别，希望降到 2 维以便可视化，同时保证类别可分性。",
      solution: `1. 使用 LDA 将 4 维降到 2 维（3 类→最多 2 维）；
2. 前两个判别轴累加解释了 97.8% 的类间方差；
3. 二维可视化下三个品种完全可分，分类准确率达 98%。`
    },
    code: {
      python: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

X = np.random.randn(150, 4)
y = np.array([0]*50 + [1]*50 + [2]*50)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
print(f"各判别轴解释方差比: {lda.explained_variance_ratio_}")`,
      matlab: `% MATLAB LDA 降维
X = randn(150, 4);
Y = [zeros(50,1); ones(50,1); 2*ones(50,1)];
[W, Z] =lda(X, Y, 2);`
    },
    resources: [
      { title: "sklearn LDA 官方文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html", type: "link" }
    ]
  },
  {
    id: "isomap-dimension-reduction",
    name: "数据降维-ISOMap (Isometric Mapping)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "通过测地距离保持数据的内在几何结构，是等距映射的非线性流形学习方法。",
    principles: `1. 构建 k 近邻图：每个点连接其 k 个最近邻。
2. 计算所有点对之间的最短路径（近似测地距离）。
3. 在距离矩阵上运行 MDS（多维缩放）得到低维嵌入。
4. 关键：测地距离沿流形表面绕行，比欧氏距离更能反映曲面数据的真实结构。`,
    scenarios: [
      "手写数字图像的流形结构可视化（如 MNIST 卷曲展平）",
      "基因表达数据在疾病亚型分类中的流形发现"
    ],
    limitations: [
      "近邻图构建对 k 值敏感，过小图不连通，过大图短路失效",
      "对高维数据计算最短路径复杂度为 O(n² log n)，大规模数据极慢"
    ],
    caseStudy: {
      title: "MNIST 手写数字流形可视化",
      description: "70000 张 MNIST 手写数字图像（784 维），希望降到 2 维可视化并观察各数字类别的分布结构。",
      solution: `1. 在 784 维原始图像上计算 10-NN 图；
2. Floyd-Warshall 计算所有点对测地距离；
3. MDS 降维到 2 维；
4. 二维可视化中数字 0 和 1 明显分离，数字 3/5/8 在边缘区域高度重叠。`
    },
    code: {
      python: `# sklearn 没有内置 ISOMap，使用 Manifold 包
from sklearn.manifold import Isomap
import numpy as np

X = np.random.randn(200, 10)
# ISOMap 降维至 2 维
iso = Isomap(n_neighbors=10, n_components=2)
X_iso = iso.fit_transform(X)
print(f"重构误差: {iso.reconstruction_error():.4f}")`,
      matlab: `% MATLAB ISOMap 降维
% 使用自定义实现或 Toolbox
% 1) 构建 k-NN 图
% 2) 最短路径计算 (graphallshortestpaths)
% 3) MDS 嵌入`
    },
    resources: [
      { title: "ISOMap 原始论文", url: "https://doi.org/10.1126/science.1077202", type: "paper" }
    ]
  },
  {
    id: "lle-dimension-reduction",
    name: "数据降维-LLE 局部线性嵌入 (Locally Linear Embedding)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "假设数据在局部是线性的，通过保持局部邻域的重构系数来实现降维，是流形学习的经典方法。",
    principles: `1. 对每个样本点找出其 k 个最近邻。
2. 用最小二乘求出每个点在局部邻域内的线性重构系数 W_ij。
3. 在低维空间中找到同样满足重构系数 W 的嵌入坐标 Y_i，最小化 Σ |Y_i - Σ W_ij Y_j|²。
4. 约束 Y 中心化且协方差为单位阵，防止退化。`,
    scenarios: [
      "面部表情图像的流形结构分析",
      "语音信号在低维空间的特征提取"
    ],
    limitations: [
      "对噪声极为敏感，噪声点会破坏局部线性结构",
      "无法处理样本外新数据的嵌入（需用 out-of-sample extension）"
    ],
    caseStudy: {
      title: "人脸图像流形降维",
      description: "某研究收集了 1000 张同一人的不同光照和角度人脸图像（每张 256 维），希望提取内在 3 维参数（光照方向、水平旋转、俯仰角）。",
      solution: `1. 设 k=12，构造每个图像的 12 个最近邻图；
2. 最小二乘计算局部重构系数；
3. LLE 降到 3 维；
4. 三个嵌入维度与真实光照和旋转角度高度相关（r>0.9）。`
    },
    code: {
      python: `from sklearn.manifold import LocallyLinearEmbedding
import numpy as np

X = np.random.randn(200, 20)
# LLE 降维至 2 维
lle = LocallyLinearEmbedding(n_neighbors=12, n_components=2, method='standard')
X_lle = lle.fit_transform(X)
print(f"重构误差: {lle.reconstruction_error_:.4f}")`,
      matlab: `% MATLAB LLE 降维
X = randn(200, 20);
n_neighbors = 12;
% 1) 找 k 近邻
% 2) 最小二乘求 W
% 3) 特征分解求 Y`
    },
    resources: [
      { title: "LLE 原始论文", url: "https://science.sciencemag.org/content/290/5500/2323", type: "paper" }
    ]
  },
  {
    id: "kpca-dimension-reduction",
    name: "数据降维-KPCA 核主成分分析 (Kernel PCA)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "通过核函数将数据映射到高维特征空间后执行 PCA，实现非线性降维。",
    principles: `1. 选择核函数 K(x_i, x_j)：常用 RBF 核 K(x,y)=exp(-||x-y||²/(2σ²))、多项式核。
2. 计算核矩阵 K，不显式计算映射 Φ(x)。
3. 对核矩阵做中心化处理。
4. 解核矩阵的特征值分解，取前 k 个特征向量作为降维结果。
5. 降维后数据 Z_i = Σ α_j^k K(x_i, x_j)。`,
    scenarios: [
      "手写数字识别的非线性流形特征提取",
      "非线性相关工业过程监控与故障检测"
    ],
    limitations: [
      "核函数选择和超参数（σ）调优困难，缺乏通用指导原则",
      "高维核矩阵存储和特征分解复杂度为 O(n²)，不适合超大规模数据"
    ],
    caseStudy: {
      title: "齿轮箱故障模式非线性特征提取",
      description: "某齿轮箱振动信号经 FFT 变换得到 128 维频域特征，但故障模式与特征呈非线性关系，普通 PCA 无法有效区分。",
      solution: `1. 采用 RBF 核（γ=0.05）执行 KPCA 降维到 3 维；
2. 前 3 个核主分量解释了 85% 的非线性特征方差；
3. 在 3 维核主分量空间中，不同故障模式形成明显分离的簇，AUC=0.97。`
    },
    code: {
      python: `from sklearn.decomposition import KernelPCA
import numpy as np

X = np.random.randn(100, 5)
# RBF 核 PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)
print(f"核主成分形状: {X_kpca.shape}")`,
      matlab: `% MATLAB KPCA 降维
X = randn(100, 5);
K = exp(-pdist2(X, X).^2 / (2*sigma^2));  % RBF 核
K = center(K);  % 中心化
[V, ~] = eigs(K, 2);  % 取前2个特征向量`
    },
    resources: [
      { title: "sklearn KernelPCA 官方文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html", type: "link" }
    ]
  },
  {
    id: "sne-dimension-reduction",
    name: "数据降维-SNE t-SNE (t-Distributed Stochastic Neighbor Embedding)",
    category: "reduction",
    categoryName: "🎯 数据降维",
    summary: "通过保持高维和低维概率分布相似性，将高维数据映射到 2-3 维空间进行可视化，是最流行的流形可视化方法。",
    principles: `1. 高维空间：计算点 i 和 j 的条件概率 p_{j|i} = exp(-||x_i - x_j||² / (2σ_i²)) / Σ_{k≠i} exp(-||x_i - x_k||² / (2σ_i²))。
2. 低维空间：用 t 分布 q_{ij} = (1 + ||y_i - y_j||²)⁻¹ / Σ_{k≠l} (1 + ||y_k - y_l||²)⁻¹。
3. 最小化 KL 散度 KL(P||Q) = Σ_i Σ_j p_{ij} log(p_{ij}/q_{ij})，使低维分布 Q 逼近高维分布 P。
4. t-SNE 用 t 分布替代高斯分布解决「拥挤问题」（高维距离差异在低维被压缩）。`,
    scenarios: [
      "单细胞 RNA-seq 数据的高维细胞类型可视化",
      "高维文本嵌入（如 BERT）的二维可视化聚类"
    ],
    limitations: [
      "无法保留全局距离结构，仅适合局部邻域可视化",
      "计算复杂度 O(n²)，且无法直接处理新样本（需重算全部）"
    ],
    caseStudy: {
      title: "单细胞基因表达数据可视化",
      description: "某单细胞 RNA-seq 数据集包含 20000 个基因（高维）和 5000 个细胞，需识别细胞亚群并可视化。",
      solution: `1. 先用 PCA 将 20000 维降到 50 维（加速 t-SNE）；
2. 运行 t-SNE（perplexity=30）降到 2 维；
3. 可视化中清晰可见 6 个细胞亚群，其中 2 个亚群在空间上邻近（提示分化连续性）。`
    },
    code: {
      python: `from sklearn.manifold import TSNE
import numpy as np

X = np.random.randn(500, 50)
# t-SNE 降维至 2 维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
print(f"t-SNE 降维结果形状: {X_tsne.shape}")`,
      matlab: `% MATLAB t-SNE (Statistics and Machine Learning Toolbox R2019a+)
Y = tsne(X, 'NumDimensions', 2, 'Perplexity', 30);`
    },
    resources: [
      { title: "t-SNE 原始论文", url: "https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf", type: "paper" }
    ]
  },

  // ==================== 3. 预测模型 (prediction) ====================
  {
    id: "linear-regression-ols",
    name: "线性回归（最小二乘法） (Linear Regression, OLS)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "通过最小二乘准则拟合自变量与因变量之间的线性关系，是连续型结果预测与解释中最基础的模型。",
    principles: `1. 假设响应变量与特征之间满足线性关系：y = X\\beta + \\varepsilon。
2. 最小二乘法通过最小化残差平方和 \\sum_i (y_i - \\hat{y}_i)^2 求解参数。
3. 闭式解为 \\hat{\\beta} = (X^T X)^{-1} X^T y。
4. 可用于预测、影响因素分析和边际效应解释。`,
    scenarios: [
      "房价、销量、能耗、产量等连续变量预测",
      "分析多个影响因素对目标变量的线性作用"
    ],
    limitations: [
      "对多重共线性、异常值和非线性关系较敏感",
      "需要关注残差独立性、同方差性和线性假设"
    ],
    caseStudy: {
      title: "广告投入与销售额预测",
      description: "基于广告投入、折扣和客流量等指标预测月销售额。",
      solution: `1. 构建自变量矩阵与销售额响应变量；
2. 用最小二乘估计回归系数；
3. 输出预测值并解释主要影响因素。`
    },
    code: {
      python: `from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1, 10], [2, 12], [3, 15], [4, 18]])
y = np.array([20, 24, 31, 36])
model = LinearRegression().fit(X, y)
print(model.coef_, model.predict([[5, 20]]))`,
      matlab: `% MATLAB 最小二乘线性回归
X = [1 10; 2 12; 3 15; 4 18];
y = [20; 24; 31; 36];
b = regress(y, [ones(size(X,1),1), X]);`
    },
    resources: [
      { title: "LinearRegression 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html", type: "link" }
    ]
  },
  {
    id: "ridge-regression",
    name: "岭回归 (Ridge Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "在线性回归损失函数中加入 L2 正则项，以缓解多重共线性并提升模型稳定性。",
    principles: `1. 目标函数为 \\|y - X\\beta\\|_2^2 + \\lambda \\|\\beta\\|_2^2。
2. L2 正则化会收缩回归系数，但通常不会将系数压到 0。
3. 适合特征间高度相关的回归预测问题。
4. \\lambda 越大，系数收缩越明显，偏差增大但方差降低。`,
    scenarios: [
      "高相关指标下的连续变量预测",
      "希望保留全部变量但增强模型稳健性的场景"
    ],
    limitations: [
      "不能自动做特征筛选",
      "正则参数需要交叉验证或经验调整"
    ],
    caseStudy: {
      title: "多指标能耗预测中的共线性处理",
      description: "建筑面积、空调功率、照明强度等指标高度相关，普通线性回归不稳定。",
      solution: `1. 构建标准化后的特征矩阵；
2. 通过 Ridge 回归拟合；
3. 用交叉验证选择合适正则强度。`
    },
    code: {
      python: `from sklearn.linear_model import Ridge
import numpy as np
X = np.random.randn(20, 5)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(20)
model = Ridge(alpha=1.0).fit(X, y)
print(model.coef_)`,
      matlab: `% MATLAB 岭回归
X = randn(20,5); y = X(:,1)*2 + X(:,2)*1.5 + randn(20,1);
k = 1; b = ridge(y, X, k, 0);`
    },
    resources: [
      { title: "Ridge regression", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html", type: "link" }
    ]
  },
  {
    id: "hierarchical-regression",
    name: "分层回归 (Hierarchical Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "按理论或业务逻辑分批次加入自变量，比较模型解释力的增量变化，用于分层解释与预测建模。",
    principles: `1. 将变量按层次分组，如控制变量、核心变量、交互变量。
2. 逐步拟合多个嵌套回归模型。
3. 通过比较 R^2、调整后 R^2 或 F 检验分析新增变量的增量解释力。
4. 强调变量进入顺序背后的理论意义。`,
    scenarios: [
      "社会科学、教育、医学等强调分层解释的回归分析",
      "比较新增变量是否显著提升预测能力"
    ],
    limitations: [
      "结果依赖变量进入顺序",
      "更强调解释结构，不是纯自动化建模方法"
    ],
    caseStudy: {
      title: "学生成绩影响因素的分层回归分析",
      description: "先纳入家庭背景变量，再纳入学习行为变量，比较模型解释力的提升。",
      solution: `1. 拟合基础控制模型；
2. 分层加入核心解释变量；
3. 比较 R^2 增量并解释新增贡献。`
    },
    code: {
      python: `import statsmodels.api as sm
import pandas as pd

df = pd.DataFrame({
    'y':[80,82,78,85,88,90],
    'age':[16,16,15,17,17,16],
    'income':[5,6,4,7,8,6],
    'study':[2,3,1,4,5,3]
})
X1 = sm.add_constant(df[['age', 'income']])
X2 = sm.add_constant(df[['age', 'income', 'study']])
print(sm.OLS(df['y'], X1).fit().rsquared)
print(sm.OLS(df['y'], X2).fit().rsquared)`,
      matlab: `% MATLAB 分层回归
mdl1 = fitlm(tbl, 'y ~ age + income');
mdl2 = fitlm(tbl, 'y ~ age + income + study');`
    },
    resources: [
      { title: "Statsmodels OLS", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "grey-prediction-gm11",
    name: "灰色预测模型GM(1,1) (Grey Prediction GM(1,1))",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "适用于小样本、贫信息、近似单调序列的经典灰色预测方法，是数学建模中常见的短期趋势预测模型。",
    principles: `1. 对原始序列做一次累加生成（1-AGO）以削弱随机波动。
2. 建立灰微分方程：dx^{(1)}/dt + a x^{(1)} = b。
3. 通过最小二乘估计发展系数 a 与灰作用量 b。
4. 再通过逆累加还原得到原始序列预测值。`,
    scenarios: [
      "样本量很小的中短期趋势预测",
      "产量、磨损、成本等近似单调序列预测"
    ],
    limitations: [
      "对剧烈震荡序列不适用",
      "长期外推精度通常会明显下降"
    ],
    caseStudy: {
      title: "设备磨损量短期预测",
      description: "仅有 5 期设备磨损监测数据，希望预测下一期磨损程度。",
      solution: `1. 对原始序列进行 1-AGO；
2. 构建 B 矩阵估计参数；
3. 输出下一期预测值。`
    },
    code: {
      python: `import numpy as np
x0 = np.array([15, 18, 22, 26, 31], dtype=float)
x1 = np.cumsum(x0)
B = np.column_stack((-0.5 * (x1[:-1] + x1[1:]), np.ones(len(x0)-1)))
Y = x0[1:]
a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
pred = [(x0[0] - b/a) * np.exp(-a*k) + b/a for k in range(len(x0)+1)]
print(np.diff(pred))`,
      matlab: `% MATLAB GM(1,1)
x0 = [15 18 22 26 31]';
x1 = cumsum(x0);
B = [-0.5*(x1(1:end-1)+x1(2:end)), ones(length(x0)-1,1)];
u = B \ x0(2:end);`
    },
    resources: [
      { title: "GM(1,1) 教程", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "kmeans-clustering",
    name: "聚类分析(K-Means) (K-Means Clustering)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "通过最小化簇内平方和将样本划分为 K 个簇，常用于用户分群、样本画像和模式发现。",
    principles: `1. 预先指定簇数 K。
2. 初始化 K 个中心并将样本分配到最近质心。
3. 更新各簇质心并迭代直至收敛。
4. 目标是最小化簇内平方误差和。`,
    scenarios: [
      "客户分群、区域分类、模式识别",
      "对无标签样本做结构发现和分段建模"
    ],
    limitations: [
      "需要预先给定 K 值",
      "对初始中心和异常值较敏感，不适合非凸簇"
    ],
    caseStudy: {
      title: "电商用户消费分群",
      description: "根据消费金额和购买频次将用户划分为不同类型。",
      solution: `1. 标准化用户特征；
2. 使用 K-Means 聚类；
3. 输出各类用户画像。`
    },
    code: {
      python: `from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1,2],[1,3],[8,8],[9,8]])
km = KMeans(n_clusters=2, random_state=42).fit(X)
print(km.labels_)`,
      matlab: `% MATLAB K-Means 聚类
X = [1 2; 1 3; 8 8; 9 8];
idx = kmeans(X, 2);`
    },
    resources: [
      { title: "KMeans 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html", type: "link" }
    ]
  },
  {
    id: "dbscan-clustering",
    name: "密度聚类(DBSCAN) (DBSCAN)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "基于密度可达思想识别任意形状簇，并能自动识别噪声点，是经典的密度聚类方法。",
    principles: `1. 设定邻域半径 eps 和最小样本数 min_samples。
2. 核心点邻域内样本数达到阈值，可向外扩展形成簇。
3. 边界点附属于某簇，离群点则被标记为噪声。
4. 不需要预先指定簇数。`,
    scenarios: [
      "空间点云聚类、异常点识别、任意形状簇划分",
      "含噪数据下的稳健无监督分组"
    ],
    limitations: [
      "对参数 eps 和 min_samples 较敏感",
      "不同密度簇同时存在时效果可能变差"
    ],
    caseStudy: {
      title: "地理位置热点区域识别",
      description: "从用户经纬度轨迹中识别自然聚集区域并剔除噪声点。",
      solution: `1. 构造坐标特征；
2. 设定密度参数运行 DBSCAN；
3. 提取聚类簇与离群点。`
    },
    code: {
      python: `from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1,2],[1,2.1],[8,8],[8.1,8], [20,20]])
labels = DBSCAN(eps=0.3, min_samples=2).fit_predict(X)
print(labels)`,
      matlab: `% MATLAB DBSCAN
X = [1 2; 1 2.1; 8 8; 8.1 8; 20 20];
idx = dbscan(X, 0.3, 2);`
    },
    resources: [
      { title: "DBSCAN 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html", type: "link" }
    ]
  },
  {
    id: "logistic-regression",
    name: "逻辑回归 (Logistic Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "用于二分类或多分类概率预测，通过逻辑函数将线性组合映射到概率空间。",
    principles: `1. 设二分类概率 P(Y=1|X) = 1/(1+e^{-X\\beta})。
2. 对数几率满足 logit(p) = \\log(p/(1-p)) = X\\beta。
3. 参数通常通过极大似然估计求解。
4. 输出的是类别概率与分类决策边界。`,
    scenarios: [
      "违约预测、疾病诊断、用户流失判断等分类任务",
      "需要输出概率解释的分类问题"
    ],
    limitations: [
      "默认决策边界为线性",
      "对特征共线性和异常值仍需关注"
    ],
    caseStudy: {
      title: "客户流失预测",
      description: "根据消费频率、投诉次数和使用时长预测用户是否流失。",
      solution: `1. 构造二分类标签与解释变量；
2. 拟合逻辑回归模型；
3. 输出流失概率并进行分类。`
    },
    code: {
      python: `from sklearn.linear_model import LogisticRegression
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
model = LogisticRegression().fit(X, y)
print(model.predict_proba([[4,6]]))`,
      matlab: `% MATLAB 逻辑回归
X = [1 10; 2 8; 8 2; 9 1];
y = [0;0;1;1];
b = glmfit(X, y, 'binomial', 'link', 'logit');`
    },
    resources: [
      { title: "LogisticRegression 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html", type: "link" }
    ]
  },
  {
    id: "lasso-regression",
    name: "Lasso回归 (Lasso Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "在线性回归中加入 L1 正则项，既能拟合连续变量，又能自动完成变量筛选。",
    principles: `1. 目标函数为 \\|y - X\\beta\\|_2^2 + \\lambda \\|\\beta\\|_1。
2. L1 正则会将部分系数压缩为 0，实现特征选择。
3. 适合高维稀疏建模与变量筛选场景。
4. 正则强度由 \\lambda 控制。`,
    scenarios: [
      "高维回归预测与自动筛选关键变量",
      "希望获得稀疏可解释模型的场景"
    ],
    limitations: [
      "对高度相关变量的选择可能不稳定",
      "正则参数选择会显著影响结果"
    ],
    caseStudy: {
      title: "高维营销因子筛选与销量预测",
      description: "从大量营销特征中筛选关键变量并预测销量。",
      solution: `1. 标准化特征；
2. 用 Lasso 回归拟合；
3. 根据非零系数识别关键因子。`
    },
    code: {
      python: `from sklearn.linear_model import Lasso
import numpy as np
X = np.random.randn(30, 8)
y = 2*X[:,0] - 1.5*X[:,3] + np.random.randn(30)*0.2
model = Lasso(alpha=0.1).fit(X, y)
print(model.coef_)`,
      matlab: `% MATLAB Lasso 回归
X = randn(30,8); y = 2*X(:,1)-1.5*X(:,4)+0.2*randn(30,1);
[B, FitInfo] = lasso(X, y);`
    },
    resources: [
      { title: "Lasso 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html", type: "link" }
    ]
  },
  {
    id: "hierarchical-clustering",
    name: "分层聚类 (Hierarchical Clustering)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "通过逐步合并或拆分样本形成树状层次结构，适合探索样本之间的嵌套关系。",
    principles: `1. 自底向上凝聚法从每个样本单独成簇开始。
2. 根据样本或簇之间的距离不断合并。
3. 常见连接方式包括 single、complete、average、ward。
4. 最终输出树状图以辅助选择聚类层级。`,
    scenarios: [
      "样本层次结构探索、类别谱系分析",
      "希望可视化聚类过程而非直接指定簇数"
    ],
    limitations: [
      "计算复杂度较高，不适合超大规模数据",
      "早期错误合并难以撤销"
    ],
    caseStudy: {
      title: "地区经济结构分层聚类",
      description: "根据多个经济指标探索各地区的层次相似关系。",
      solution: `1. 标准化地区指标；
2. 计算距离矩阵并执行凝聚聚类；
3. 通过树状图确定分组层级。`
    },
    code: {
      python: `from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
X = np.array([[1,2],[1,3],[8,8],[9,8]])
Z = linkage(X, method='ward')
print(Z)`,
      matlab: `% MATLAB 分层聚类
X = [1 2; 1 3; 8 8; 9 8];
Z = linkage(X, 'ward');`
    },
    resources: [
      { title: "SciPy hierarchical clustering", url: "https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html", type: "link" }
    ]
  },
  {
    id: "ordered-logistic-regression",
    name: "有序逻辑回归 (Ordered Logistic Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "适用于因变量为有序分类的预测问题，如满意度等级、风险等级和教育层次等。",
    principles: `1. 因变量类别具有自然顺序，但相邻类别间距不一定相等。
2. 建模的是累计概率的对数几率。
3. 常见形式为比例优势模型（proportional odds model）。
4. 输出各等级的概率预测与影响方向。`,
    scenarios: [
      "满意度等级、信用评级、风险等级预测",
      "结果是有序分类而非连续值或无序分类的场景"
    ],
    limitations: [
      "比例优势假设不成立时模型解释会受影响",
      "实现和诊断比普通逻辑回归更复杂"
    ],
    caseStudy: {
      title: "客户满意度等级预测",
      description: "根据响应速度、价格和服务体验预测客户满意度等级。",
      solution: `1. 构造有序标签；
2. 拟合有序逻辑回归；
3. 输出各等级概率并分析关键影响因素。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.DataFrame({
    'y':[0,1,1,2,2,2],
    'x1':[1,2,3,4,5,6],
    'x2':[6,5,4,3,2,1]
})
model = OrderedModel(df['y'], df[['x1','x2']], distr='logit')
print(model.fit(method='bfgs', disp=False).params)`,
      matlab: `% MATLAB 有序逻辑回归可通过 mnrfit 等方法扩展实现`
    },
    resources: [
      { title: "OrderedModel 文档", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "plsr-regression",
    name: "偏最小二乘回归(PLSR) (Partial Least Squares Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "在降维的同时最大化自变量与因变量之间的协方差，适合高维、强共线性和小样本回归预测。",
    principles: `1. 从 X 和 y 中提取潜在成分。
2. 成分提取同时考虑解释 X 方差和与 y 的相关性。
3. 可在降维后完成回归预测。
4. 特别适用于变量多、样本少、共线性强的场景。`,
    scenarios: [
      "光谱分析、化学计量学、高维小样本预测",
      "普通线性回归因共线性而不稳定的情形"
    ],
    limitations: [
      "主成分物理含义不如原始变量直观",
      "成分数选择会影响结果表现"
    ],
    caseStudy: {
      title: "高维光谱数据浓度预测",
      description: "用大量高相关光谱变量预测样品浓度。",
      solution: `1. 提取少量潜在成分；
2. 用潜在成分回归目标变量；
3. 兼顾降维与预测精度。`
    },
    code: {
      python: `from sklearn.cross_decomposition import PLSRegression
import numpy as np
X = np.random.randn(40, 10)
y = X[:,0]*2 - X[:,1] + np.random.randn(40)*0.3
pls = PLSRegression(n_components=2).fit(X, y)
print(pls.coef_)`,
      matlab: `% MATLAB PLSR
X = randn(40,10); y = X(:,1)*2 - X(:,2) + 0.3*randn(40,1);
[XL,YL,XS,YS,beta] = plsregress(X, y, 2);`
    },
    resources: [
      { title: "PLSRegression 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html", type: "link" }
    ]
  },
  {
    id: "probit-regression",
    name: "二分类概率单位回归(Probit) (Binary Probit Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "与逻辑回归类似的二分类概率模型，但使用正态分布累积分布函数作为链接函数。",
    principles: `1. 模型形式为 P(Y=1|X) = \\Phi(X\\beta)，其中 \\Phi 为标准正态分布函数。
2. 参数通常通过极大似然法估计。
3. 适合潜变量服从正态误差结构的二分类问题。
4. 与 logit 模型相比，链接函数不同但解释逻辑类似。`,
    scenarios: [
      "违约、是否购买、是否通过等二分类预测",
      "希望采用正态链接函数的计量经济学建模"
    ],
    limitations: [
      "与逻辑回归相比解释门槛稍高",
      "对变量关系形式的假设仍然较强"
    ],
    caseStudy: {
      title: "用户购买行为 Probit 预测",
      description: "根据价格、收入和促销信息预测用户是否购买。",
      solution: `1. 构造二元响应变量；
2. 拟合 Probit 模型；
3. 输出购买概率与边际影响方向。`
    },
    code: {
      python: `import statsmodels.api as sm
import numpy as np
X = sm.add_constant(np.array([[1,10],[2,8],[8,2],[9,1]], dtype=float))
y = np.array([0,0,1,1])
model = sm.Probit(y, X).fit(disp=False)
print(model.params)`,
      matlab: `% MATLAB Probit 回归
b = glmfit(X, y, 'binomial', 'link', 'probit');`
    },
    resources: [
      { title: "Statsmodels Probit", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "demings-regression",
    name: "Deming's 回归 (Deming Regression)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "当自变量和因变量都存在测量误差时，用于替代普通最小二乘回归的误差变量模型。",
    principles: `1. 普通最小二乘默认 X 无误差，而 Deming 回归允许 X 与 Y 都有测量误差。
2. 通过最小化点到拟合直线的加权正交距离进行估计。
3. 需要指定 X 与 Y 的误差方差比。
4. 常用于方法学比较和仪器一致性分析。`,
    scenarios: [
      "两种测量方法比对、仪器校准与实验方法一致性研究",
      "解释变量本身含明显测量误差的线性关系建模"
    ],
    limitations: [
      "需要已知或假设误差方差比",
      "不如普通回归那样常见，软件支持相对有限"
    ],
    caseStudy: {
      title: "两台传感器测量结果校准",
      description: "两种设备都存在测量误差，希望建立更合理的校准关系。",
      solution: `1. 收集两台设备的配对测量值；
2. 设定误差方差比；
3. 拟合 Deming 回归并建立校准方程。`
    },
    code: {
      python: `import numpy as np
x = np.array([1.0, 2.1, 3.0, 4.2])
y = np.array([1.2, 2.0, 3.3, 4.1])
print('Deming regression often uses specialized packages or custom implementation.')`,
      matlab: `% MATLAB Deming 回归通常需自定义实现或使用外部函数`
    },
    resources: [
      { title: "Deming regression overview", url: "https://en.wikipedia.org/wiki/Deming_regression", type: "link" }
    ]
  },
  {
    id: "biclustering",
    name: "二阶聚类 (Biclustering)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "同时在样本维度和变量维度上寻找局部共现结构，适合发现子矩阵模式与局部簇结构。",
    principles: `1. 传统聚类只聚样本或变量，二阶聚类同时聚两者。
2. 目标是在数据矩阵中寻找具有相似模式的子矩阵块。
3. 常用于表达谱分析、推荐系统和局部结构发现。
4. 能识别只在部分特征上相似的样本群。`,
    scenarios: [
      "基因表达数据中的局部共表达模式发现",
      "用户-商品矩阵中的局部兴趣群体分析"
    ],
    limitations: [
      "模型复杂度较高，结果解释门槛较大",
      "对参数和数据预处理较敏感"
    ],
    caseStudy: {
      title: "用户-商品偏好局部结构发现",
      description: "希望同时找出某类用户与某类商品之间的局部偏好关系。",
      solution: `1. 构建用户-商品矩阵；
2. 使用二阶聚类寻找局部块结构；
3. 输出样本与变量的联合簇。`
    },
    code: {
      python: `from sklearn.cluster import SpectralCoclustering
import numpy as np
X = np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
model = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
print(model.row_labels_, model.column_labels_)`,
      matlab: `% MATLAB 二阶聚类通常需自定义实现或借助外部工具箱`
    },
    resources: [
      { title: "Spectral Co-clustering", url: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html", type: "link" }
    ]
  },

  {
    id: "ahp-pro",
    name: "层次分析法（AHP专业版） (Analytic Hierarchy Process - Professional)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "适用于多层级、多准则决策的经典主观赋权方法，通过构造判断矩阵、计算特征向量与一致性检验得到稳定权重。",
    principles: `1. 构造判断矩阵 A = (a_{ij})，满足 a_{ij} > 0, a_{ji} = 1/a_{ij}, a_{ii}=1。
2. 计算最大特征值与权重向量：A w = \\lambda_{max} w，并对 w 归一化。
3. 一致性指标：CI = (\\lambda_{max} - n)/(n-1)。
4. 一致性比率：CR = CI/RI，当 CR < 0.1 时认为判断矩阵通过一致性检验。
5. 多层结构下继续进行层次总排序，得到方案层综合权重。`,
    scenarios: [
      "区域创新能力、营商环境、城市宜居性等多指标综合评价",
      "数学建模中含有专家经验判断与主观偏好的方案优选"
    ],
    limitations: [
      "高度依赖专家主观判断，若打分随意会直接影响结果可靠性",
      "指标数量很多时，判断矩阵构造成本高且一致性较难保证"
    ],
    caseStudy: {
      title: "城市交通治理方案优选",
      description: "对 4 个交通治理备选方案，从成本、效率、可实施性、公众满意度四方面进行综合决策。",
      solution: `1. 构建目标层—准则层—方案层三级结构；
2. 通过专家两两比较构造判断矩阵；
3. 计算权重并完成一致性检验；
4. 最终得到各方案综合得分并排序。`
    },
    code: {
      python: `import numpy as np

A = np.array([
    [1, 3, 5],
    [1/3, 1, 2],
    [1/5, 1/2, 1]
])
vals, vecs = np.linalg.eig(A)
max_idx = np.argmax(vals.real)
lambda_max = vals[max_idx].real
w = vecs[:, max_idx].real
w = w / w.sum()
CI = (lambda_max - A.shape[0]) / (A.shape[0] - 1)
print("weights=", w, "CI=", CI)`,
      matlab: `% MATLAB AHP 权重求解
A = [1 3 5; 1/3 1 2; 1/5 1/2 1];
[V, D] = eig(A);
[lambda_max, idx] = max(diag(D));
w = V(:, idx) / sum(V(:, idx));
CI = (lambda_max - size(A,1)) / (size(A,1)-1);`
    },
    resources: [
      { title: "AHP 层次分析法教程", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "factor-analysis-exploratory",
    name: "因子分析（探索性） (Exploratory Factor Analysis)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "通过少量潜在公共因子解释多项观测指标间的相关结构，常用于综合评价中的降维、指标归类与结构提炼。",
    principles: `1. 基本模型：X = \\mu + \\Lambda F + \\varepsilon，其中 F 为公共因子，\\Lambda 为因子载荷矩阵。
2. 利用相关系数矩阵提取特征根与特征向量，保留特征根大于 1 或累计贡献率较高的因子。
3. 通过 Varimax 等旋转方法提升因子解释性。
4. 计算因子得分后可作为后续评价、聚类或排序的核心变量。`,
    scenarios: [
      "问卷量表结构分析与综合素质评价",
      "多个高度相关指标的降维整合与评价建模"
    ],
    limitations: [
      "因子个数选择存在主观性，不同旋转方式会影响解释结果",
      "数据若相关性不足，则因子分析意义较弱"
    ],
    caseStudy: {
      title: "大学生综合能力评价维度提取",
      description: "对学习能力、创新能力、表达能力、组织能力等 20 个二级指标做探索性因子分析。",
      solution: `1. 构建相关矩阵并进行 KMO/Bartlett 检验；
2. 提取 4 个公共因子并做方差最大旋转；
3. 将因子得分作为学生综合能力评价的核心依据。`
    },
    code: {
      python: `from factor_analyzer import FactorAnalyzer
import numpy as np

X = np.random.rand(100, 8)
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(X)
print("loadings=\n", fa.loadings_)`,
      matlab: `% MATLAB 探索性因子分析
X = rand(100, 8);
[Lambda, Psi, T, stats, F] = factoran(X, 3, 'rotate', 'varimax');`
    },
    resources: [
      { title: "探索性因子分析入门", url: "https://www.mathworks.com", type: "link" }
    ]
  },
  {
    id: "dea",
    name: "数据包络分析（DEA） (Data Envelopment Analysis)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "一种以线性规划为基础的相对效率评价方法，适合比较多个决策单元在多输入多输出条件下的效率水平。",
    principles: `1. 设决策单元 DMU_j 的输入为 x_{ij}，输出为 y_{rj}。
2. 构建 CCR 或 BCC 模型，求取效率值 \\theta。
3. 若 \\theta = 1 且无松弛变量，则 DMU 有效；否则为非 DEA 有效。
4. 可进一步分析规模报酬、冗余投入与不足产出。`,
    scenarios: [
      "银行网点、医院、地区、企业等多投入多产出效率评价",
      "教育资源配置、创新产出效率、公共服务绩效分析"
    ],
    limitations: [
      "DEA 只能做相对效率评价，不能直接给出绝对最优标准",
      "对异常样本较敏感，且决策单元数过少会削弱区分能力"
    ],
    caseStudy: {
      title: "医院运营效率评价",
      description: "比较 15 家医院在人力、床位、经费投入与门诊量、治愈率等输出上的相对效率。",
      solution: `1. 构建输入输出指标体系；
2. 使用 CCR/BCC 模型计算效率值；
3. 识别有效医院与低效医院，并分析冗余投入来源。`
    },
    code: {
      python: `import pulp

# 简化版 DEA 输入导向示意
prob = pulp.LpProblem('DEA', pulp.LpMinimize)
theta = pulp.LpVariable('theta', lowBound=0)
prob += theta
prob.solve()
print('theta=', theta.value())`,
      matlab: `% MATLAB DEA 可使用 linprog 自行构造
f = [1; zeros(5,1)];
% 根据 CCR/BCC 模型补充约束矩阵后调用 linprog`
    },
    resources: [
      { title: "DEA 效率评价模型综述", url: "https://www.sciencedirect.com", type: "paper" }
    ]
  },
  {
    id: "fuzzy-eval",
    name: "模糊综合评价 (Fuzzy Comprehensive Evaluation)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "利用模糊隶属度刻画难以精确划分的评价对象，适合处理边界模糊、定性较强的综合评价问题。",
    principles: `1. 建立因素集 U 与评价集 V。
2. 对每个指标构造隶属函数并形成隶属度矩阵 R。
3. 通过权重向量 A 与 R 进行模糊合成：B = A \\circ R 或 B = A · R。
4. 根据最大隶属度原则或加权得分完成最终评价。`,
    scenarios: [
      "环境质量、安全风险、服务满意度等模糊边界评价问题",
      "数学建模中定性与定量混合的综合评分场景"
    ],
    limitations: [
      "隶属函数与权重设计存在较强主观性",
      "当指标过多且差异小，结果可能出现均衡化"
    ],
    caseStudy: {
      title: "供电站安全等级评价",
      description: "根据温湿度、设备老化、周边环境等指标对供电站安全状态进行综合判定。",
      solution: `1. 构建指标集与等级集；
2. 根据专家经验建立隶属函数；
3. 计算模糊综合评价向量并判定安全等级。`
    },
    code: {
      python: `import numpy as np
A = np.array([0.4, 0.3, 0.3])
R = np.array([
    [0.8, 0.2, 0.0],
    [0.3, 0.5, 0.2],
    [0.2, 0.4, 0.4]
])
B = A @ R
print(B)`,
      matlab: `% MATLAB 模糊综合评价
A = [0.4 0.3 0.3];
R = [0.8 0.2 0; 0.3 0.5 0.2; 0.2 0.4 0.4];
B = A * R;`
    },
    resources: [
      { title: "模糊综合评价基础讲解", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "topsis",
    name: "优劣解距离法（TOPSIS） (Technique for Order Preference by Similarity to Ideal Solution)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "通过比较各方案与正理想解和负理想解的距离完成排序，是最常用的多指标综合评价方法之一。",
    principles: `1. 对原始指标矩阵做标准化处理。
2. 乘以指标权重得到加权规范化矩阵。
3. 确定正理想解 Z^+ 和负理想解 Z^-。
4. 计算各方案到理想解的距离 D_i^+, D_i^-。
5. 计算贴近度 C_i = D_i^- / (D_i^+ + D_i^-) 并排序。`,
    scenarios: [
      "方案优选、城市排名、企业绩效评价",
      "数学建模竞赛中的多指标排序与决策问题"
    ],
    limitations: [
      "结果依赖权重设置和标准化方式",
      "极端值会影响理想解位置，进而影响排序稳定性"
    ],
    caseStudy: {
      title: "城市创新能力排序",
      description: "对多个城市从科研投入、人才密度、产业产出等维度进行综合排名。",
      solution: `1. 对各指标进行无量纲化；
2. 引入权重构建加权矩阵；
3. 计算贴近度并输出城市综合排名。`
    },
    code: {
      python: `import numpy as np
X = np.array([[80, 65, 0.5], [70, 80, 0.3], [90, 75, 0.6]])
Xn = X / np.sqrt((X**2).sum(axis=0))
w = np.array([0.3, 0.4, 0.3])
Z = Xn * w
zp, zn = Z.max(axis=0), Z.min(axis=0)
Dp = np.sqrt(((Z - zp)**2).sum(axis=1))
Dn = np.sqrt(((Z - zn)**2).sum(axis=1))
C = Dn / (Dp + Dn)
print(C)`,
      matlab: `% MATLAB TOPSIS
X = [80 65 0.5; 70 80 0.3; 90 75 0.6];
Xn = X ./ sqrt(sum(X.^2));
w = [0.3 0.4 0.3];
Z = Xn .* w;`
    },
    resources: [
      { title: "TOPSIS 方法讲解", url: "https://www.mathworks.com", type: "link" }
    ]
  },
  {
    id: "rsr",
    name: "秩和比综合评价法（RSR） (Rank-Sum Ratio)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "将多指标值转化为秩次后综合，适合分布不规则、量纲不统一且强调相对排序的评价场景。",
    principles: `1. 对各指标按优劣方向转化为秩次 R_{ij}。
2. 计算对象的秩和比：RSR_i = \\sum_j R_{ij} / (n \\times m)。
3. 可进一步结合 Probit 或回归分析完成分档。
4. 方法本质为非参数综合评价，强调相对顺序。`,
    scenarios: [
      "医院、学校、地区等多对象绩效排位",
      "数据分布不规则、无法直接线性加权的综合评价问题"
    ],
    limitations: [
      "丢失了原始数值差距信息，仅保留顺序信息",
      "当样本量较小或并列值较多时区分度有限"
    ],
    caseStudy: {
      title: "社区卫生服务中心综合考核",
      description: "对多个社区卫生中心从服务质量、效率、满意度等维度进行秩次综合评价。",
      solution: `1. 各指标转为秩次；
2. 计算各对象 RSR 值；
3. 结合分档规则输出优秀、良好、一般等级。`
    },
    code: {
      python: `import numpy as np
from scipy.stats import rankdata
X = np.array([[8,90],[6,80],[9,95]])
ranks = np.column_stack([rankdata(X[:,j]) for j in range(X.shape[1])])
rsr = ranks.mean(axis=1) / X.shape[0]
print(rsr)`,
      matlab: `% MATLAB RSR
X = [8 90; 6 80; 9 95];
ranks = tiedrank(X);
rsr = mean(ranks, 2) / size(X,1);`
    },
    resources: [
      { title: "RSR 综合评价法应用", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "coupling-coordination",
    name: "耦合协调度 (Coupling Coordination Degree)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "用于分析多个子系统之间的互动耦合关系及其协调发展程度，常见于区域经济、生态、产业协同研究。",
    principles: `1. 分别计算各子系统综合发展指数 U_1, U_2, ..., U_m。
2. 构造耦合度 C 衡量系统间相互作用强度。
3. 构造综合发展指数 T = \\sum \\alpha_i U_i。
4. 计算协调度 D = \\sqrt{C \\times T}。
5. 根据 D 的大小划分失调、勉强协调、良好协调、高度协调等等级。`,
    scenarios: [
      "区域经济—生态环境协同发展评价",
      "产业结构—创新能力—资源环境耦合分析"
    ],
    limitations: [
      "耦合公式选择存在一定经验性，不同文献口径不完全一致",
      "若子系统指数构造不合理，会直接影响协调度解释"
    ],
    caseStudy: {
      title: "城市经济与生态系统协调评价",
      description: "研究某地区经济增长与生态保护是否同步改善。",
      solution: `1. 分别计算经济子系统与生态子系统综合得分；
2. 计算耦合度与协调度；
3. 判断各年份由失调走向协调的发展阶段。`
    },
    code: {
      python: `import numpy as np
U1 = np.array([0.3, 0.5, 0.7])
U2 = np.array([0.2, 0.45, 0.68])
C = 2 * np.sqrt(U1 * U2) / (U1 + U2)
T = 0.5 * U1 + 0.5 * U2
D = np.sqrt(C * T)
print(D)`,
      matlab: `% MATLAB 耦合协调度
U1 = [0.3 0.5 0.7];
U2 = [0.2 0.45 0.68];
C = 2*(U1.*U2).^0.5./(U1+U2);
T = 0.5*U1 + 0.5*U2;
D = sqrt(C.*T);`
    },
    resources: [
      { title: "耦合协调度模型综述", url: "https://www.sciencedirect.com", type: "paper" }
    ]
  },
  {
    id: "ahp-simple",
    name: "层次分析法（AHP简化版） (Simplified AHP)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "在保留层次分析法核心思想的前提下，使用简化权重求解方式快速获得主观赋权结果，适合竞赛中的快速建模。",
    principles: `1. 仍然构造正互反判断矩阵 A。
2. 用几何平均法近似权重：w_i = (\\prod_j a_{ij})^{1/n} / \\sum_k (\\prod_j a_{kj})^{1/n}。
3. 可选做一致性近似检验或直接用作快速赋权。
4. 适合指标不多、强调速度与可实现性的场景。`,
    scenarios: [
      "建模竞赛时间紧张时的快速主观赋权",
      "指标层次清晰但不需要完整特征值法求解的综合评价问题"
    ],
    limitations: [
      "理论严谨性略弱于标准 AHP 特征值法",
      "若判断矩阵一致性很差，简化法结果也会失真"
    ],
    caseStudy: {
      title: "竞赛中城市宜居性快速评分",
      description: "在有限时间内对多个城市从教育、医疗、收入、环境四维度做主观赋权。",
      solution: `1. 构造 4×4 判断矩阵；
2. 用几何平均法快速求权重；
3. 结合加权得分完成最终排序。`
    },
    code: {
      python: `import numpy as np
A = np.array([[1,2,4],[1/2,1,3],[1/4,1/3,1]])
g = np.prod(A, axis=1)**(1/A.shape[0])
w = g / g.sum()
print(w)`,
      matlab: `% MATLAB AHP 简化法
A = [1 2 4; 1/2 1 3; 1/4 1/3 1];
g = prod(A, 2).^(1/size(A,1));
w = g / sum(g);`
    },
    resources: [
      { title: "AHP 简化算法讲解", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "entropy-weight",
    name: "熵值法 (Entropy Weight Method)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "基于指标离散程度进行客观赋权，指标变异越大，提供的信息越多，权重通常越高。",
    principles: `1. 对指标矩阵做标准化处理。
2. 计算比重 p_{ij} = x_{ij} / \\sum_i x_{ij}。
3. 熵值 e_j = -k \\sum_i p_{ij} \\ln p_{ij}，其中 k = 1/\\ln n。
4. 差异系数 d_j = 1 - e_j。
5. 权重 w_j = d_j / \\sum_j d_j。`,
    scenarios: [
      "希望避免主观赋权的客观评价问题",
      "与 TOPSIS、灰色关联等方法联合使用进行综合排序"
    ],
    limitations: [
      "只依赖数据离散程度，不能体现业务重要性或专家偏好",
      "若指标波动极小，权重区分度会较弱"
    ],
    caseStudy: {
      title: "产业竞争力客观赋权",
      description: "对多个产业从投入、产出、创新、绿色发展等指标上进行客观权重计算。",
      solution: `1. 对指标矩阵标准化；
2. 计算熵值与差异系数；
3. 输出客观权重后用于后续综合评价。`
    },
    code: {
      python: `import numpy as np
X = np.array([[1,2],[2,4],[3,6]], dtype=float)
P = X / X.sum(axis=0)
k = 1 / np.log(X.shape[0])
e = -k * np.sum(P * np.log(P + 1e-12), axis=0)
d = 1 - e
w = d / d.sum()
print(w)`,
      matlab: `% MATLAB 熵值法
X = [1 2; 2 4; 3 6];
P = X ./ sum(X);
k = 1 / log(size(X,1));
e = -k * sum(P .* log(P + eps));
d = 1 - e;
w = d / sum(d);`
    },
    resources: [
      { title: "熵值法赋权原理", url: "https://www.mathworks.com", type: "link" }
    ]
  },
  {
    id: "critic-weight",
    name: "CRITIC权重法 (CRITIC Weighting)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "同时考虑指标波动性和指标间冲突性进行客观赋权，是比单纯熵值法更强调信息独立性的权重方法。",
    principles: `1. 对指标标准化并计算标准差 \\sigma_j，衡量对比强度。
2. 计算指标相关系数 r_{jk}，衡量冲突性。
3. 信息量：C_j = \\sigma_j \\sum_k (1 - r_{jk})。
4. 权重：w_j = C_j / \\sum_j C_j。`,
    scenarios: [
      "指标之间相关性较强的客观赋权场景",
      "希望同时考虑波动性与冗余性的综合评价问题"
    ],
    limitations: [
      "对相关矩阵较敏感，数据质量差时结果不稳定",
      "仍属于纯客观赋权，不能体现专家偏好"
    ],
    caseStudy: {
      title: "城市发展指标 CRITIC 赋权",
      description: "对经济、科技、教育、环境等多个指标进行客观赋权，并避免高度相关指标权重虚高。",
      solution: `1. 计算标准差与相关矩阵；
2. 得到每个指标的信息量 C_j；
3. 归一化后形成 CRITIC 权重。`
    },
    code: {
      python: `import numpy as np
X = np.random.rand(5,4)
std = X.std(axis=0, ddof=1)
R = np.corrcoef(X, rowvar=False)
C = std * np.sum(1 - R, axis=0)
w = C / C.sum()
print(w)`,
      matlab: `% MATLAB CRITIC 权重法
X = rand(5,4);
s = std(X);
R = corrcoef(X);
C = s .* sum(1 - R, 1);
w = C / sum(C);`
    },
    resources: [
      { title: "CRITIC 赋权方法研究", url: "https://www.sciencedirect.com", type: "paper" }
    ]
  },
  {
    id: "independence-weight-coefficient",
    name: "独立性权系数法 (Independence Weight Coefficient Method)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "依据指标之间的独立程度分配权重，强调信息不重复、相关性低的指标应在综合评价中获得更高权重。",
    principles: `1. 建立指标相关性或相似性矩阵。
2. 衡量每个指标与其余指标的独立程度，独立性越高说明信息重复越少。
3. 根据独立性系数构建权重向量。
4. 常与客观赋权或组合赋权联合使用。`,
    scenarios: [
      "指标间相关关系复杂、需要控制信息冗余的评价问题",
      "构建组合赋权模型中的独立性修正环节"
    ],
    limitations: [
      "方法形式在不同文献中存在差异，需明确具体定义口径",
      "仅从独立性角度出发，不能全面体现重要性"
    ],
    caseStudy: {
      title: "科技评价中的独立性赋权",
      description: "多个创新指标高度相关，希望抑制信息重复对总分的影响。",
      solution: `1. 计算指标间相关矩阵；
2. 根据独立性系数构造客观权重；
3. 将其用于综合评分模型，提升指标区分性。`
    },
    code: {
      python: `import numpy as np
X = np.random.rand(8,4)
R = np.corrcoef(X, rowvar=False)
ind = 1 / (np.sum(np.abs(R), axis=0) + 1e-12)
w = ind / ind.sum()
print(w)`,
      matlab: `% MATLAB 独立性权系数法示意
X = rand(8,4);
R = corrcoef(X);
ind = 1 ./ sum(abs(R), 1);
w = ind / sum(ind);`
    },
    resources: [
      { title: "组合赋权中的独立性思想", url: "https://www.cnki.net", type: "paper" }
    ]
  },
  {
    id: "cv-weight",
    name: "变异系数法 (Coefficient of Variation Weighting)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "根据指标相对波动程度进行客观赋权，波动越大说明区分能力越强，因此权重通常越高。",
    principles: `1. 对每个指标计算均值 \\mu_j 与标准差 \\sigma_j。
2. 变异系数：CV_j = \\sigma_j / \\mu_j。
3. 权重：w_j = CV_j / \\sum_j CV_j。
4. 常用于多指标评价中的客观赋权模块。`,
    scenarios: [
      "快速客观赋权与综合评分",
      "指标量纲不同但波动性能够体现辨识度的评价问题"
    ],
    limitations: [
      "若均值接近 0，则变异系数不稳定",
      "仅考虑离散性，忽视指标间相关性与业务意义"
    ],
    caseStudy: {
      title: "区域发展指标客观权重计算",
      description: "对就业、收入、教育、医疗等指标通过变异系数法给出客观权重。",
      solution: `1. 分别计算各指标均值与标准差；
2. 得到 CV 值并归一化；
3. 将权重输入综合评价模型。`
    },
    code: {
      python: `import numpy as np
X = np.random.rand(6,4)
cv = X.std(axis=0, ddof=1) / X.mean(axis=0)
w = cv / cv.sum()
print(w)`,
      matlab: `% MATLAB 变异系数法
X = rand(6,4);
cv = std(X) ./ mean(X);
w = cv / sum(cv);`
    },
    resources: [
      { title: "变异系数法赋权应用", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "grey-relational-analysis",
    name: "灰色关联分析 (Grey Relational Analysis)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "通过比较序列几何形态相近程度衡量关联强弱，适合样本较少、信息不完全条件下的综合评价与排序。",
    principles: `1. 设参考序列 x_0(k) 与比较序列 x_i(k)。
2. 对序列做无量纲化处理。
3. 计算灰色关联系数：\\xi_i(k) = (\\Delta_{min} + \\rho \\Delta_{max})/(\\Delta_i(k) + \\rho \\Delta_{max})。
4. 对各时刻关联系数取平均得到灰色关联度。
5. 关联度越大，说明方案越接近参考序列。`,
    scenarios: [
      "方案优选、影响因素分析、综合排序",
      "小样本、弱信息建模问题中的评价与关联识别"
    ],
    limitations: [
      "结果受无量纲化方式与分辨系数 \\rho 选择影响",
      "主要反映形态相似性，不直接反映因果关系"
    ],
    caseStudy: {
      title: "产品方案与理想标准的接近度分析",
      description: "比较多个产品方案相对理想指标序列的接近程度。",
      solution: `1. 构造理想参考序列；
2. 计算各方案与参考序列的灰色关联度；
3. 按关联度大小完成优选排序。`
    },
    code: {
      python: `import numpy as np
x0 = np.array([1,2,3,4])
X = np.array([[1.1,1.9,3.2,3.8],[0.8,2.2,2.9,4.1]])
Delta = np.abs(X - x0)
rho = 0.5
xi = (Delta.min() + rho * Delta.max()) / (Delta + rho * Delta.max())
gra = xi.mean(axis=1)
print(gra)`,
      matlab: `% MATLAB 灰色关联分析
x0 = [1 2 3 4];
X = [1.1 1.9 3.2 3.8; 0.8 2.2 2.9 4.1];
D = abs(X - x0);
rho = 0.5;
xi = (min(D(:)) + rho*max(D(:))) ./ (D + rho*max(D(:)));`
    },
    resources: [
      { title: "灰色关联分析教程", url: "https://www.mathworks.com", type: "link" }
    ]
  },
  {
    id: "vikor",
    name: "多准则妥协解排序法（VIKOR） (VIKOR)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "面向多准则决策的折中排序方法，强调群体效益与个体最大遗憾之间的妥协解。",
    principles: `1. 对每个指标确定最优值 f_j^* 与最劣值 f_j^-。
2. 计算群体效益指标 S_i = \\sum_j w_j (f_j^* - f_{ij})/(f_j^* - f_j^-)。
3. 计算最大遗憾指标 R_i = \\max_j [w_j (f_j^* - f_{ij})/(f_j^* - f_j^-)]。
4. 计算综合指标 Q_i = v(S_i-S^*)/(S^- - S^*) + (1-v)(R_i-R^*)/(R^- - R^*)。
5. 根据 Q 值排序并判断是否满足妥协解条件。`,
    scenarios: [
      "多方案排序且希望平衡整体最优与局部最差表现",
      "政府、企业、工程决策中的折中最优选择"
    ],
    limitations: [
      "结果受参数 v 选择影响，不同偏好下排序可能变化",
      "需额外判断优势条件与稳定性条件才可确认妥协解"
    ],
    caseStudy: {
      title: "基础设施建设方案妥协排序",
      description: "多个建设方案在成本、效益、环保、社会接受度之间存在显著权衡。",
      solution: `1. 构建标准化评价矩阵与权重；
2. 计算 S、R、Q 三类指标；
3. 输出妥协解与备选方案序列。`
    },
    code: {
      python: `import numpy as np
X = np.array([[80,70,90],[75,85,88],[90,60,80]], dtype=float)
w = np.array([0.3,0.4,0.3])
f_star = X.max(axis=0)
f_minus = X.min(axis=0)
F = w * (f_star - X) / (f_star - f_minus + 1e-12)
S = F.sum(axis=1)
R = F.max(axis=1)
print(S, R)`,
      matlab: `% MATLAB VIKOR
X = [80 70 90; 75 85 88; 90 60 80];
w = [0.3 0.4 0.3];
fstar = max(X); fminus = min(X);
F = (fstar - X) ./ (fstar - fminus + eps) .* w;`
    },
    resources: [
      { title: "VIKOR 方法综述", url: "https://www.sciencedirect.com", type: "paper" }
    ]
  },
  {
    id: "ism",
    name: "解释结构模型（ISM） (Interpretive Structural Modeling)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "用于识别复杂系统中因素之间的层次递进与结构关系，尤其适合综合评价中的指标影响路径分析。",
    principles: `1. 明确系统因素集合 S = {s_1, s_2, ..., s_n}。
2. 根据专家判断建立邻接矩阵 A，描述因素间是否存在直接影响关系。
3. 计算可达矩阵 M = (A + I)^k，直到稳定。
4. 通过可达集、先行集与交集确定各因素层级。
5. 最终绘制多层递阶有向图，解释系统结构。`,
    scenarios: [
      "复杂评价体系中的指标层级关系分析",
      "风险因素、障碍因素、关键影响链条识别"
    ],
    limitations: [
      "高度依赖专家对关系的判断，主观性较强",
      "更偏结构解释，不能直接替代数值评分模型"
    ],
    caseStudy: {
      title: "企业创新能力影响因素层级分析",
      description: "分析政策支持、人才储备、研发投入、成果转化等因素之间的结构关系。",
      solution: `1. 建立因素集合与邻接矩阵；
2. 计算可达矩阵；
3. 分层提取根源层、中间层和表层因素，形成 ISM 结构图。`
    },
    code: {
      python: `import numpy as np
A = np.array([
    [0,1,0],
    [0,0,1],
    [0,0,0]
], dtype=int)
I = np.eye(A.shape[0], dtype=int)
M = A + I
print(M)`,
      matlab: `% MATLAB ISM 邻接矩阵示意
A = [0 1 0; 0 0 1; 0 0 0];
I = eye(size(A));
M = A + I;`
    },
    resources: [
      { title: "ISM 解释结构模型入门", url: "https://www.bilibili.com", type: "video" }
    ]
  },

  // ==================== 6. 差异性分析 (difference-analysis) ====================
  {
    id: "one-sample-t-test",
    name: "单样本T检验 (One-Sample T-Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "检验单个样本均值是否与某个理论值或目标值存在显著差异，是最基础的参数差异分析方法。",
    principles: `1. 原假设 H0：总体均值 \\mu = \\mu_0。
2. 统计量：t = (\\bar{x} - \\mu_0)/(s/\\sqrt{n})。
3. 在样本近似正态或样本量较大时，t 统计量服从自由度 n-1 的 t 分布。
4. 根据 p 值判断样本均值是否显著偏离目标值。`,
    scenarios: [
      "检验产品平均寿命是否达到标准值",
      "比较实验测量均值是否偏离理论常数"
    ],
    limitations: [
      "对正态性较敏感，小样本下尤其需要先做分布检验",
      "只能比较一个总体均值与单个基准值"
    ],
    caseStudy: {
      title: "检验灯泡寿命是否达到标称 1000 小时",
      description: "抽取一批灯泡测得寿命样本，判断平均寿命是否显著低于标称值。",
      solution: `1. 设定 H0: \\mu = 1000；
2. 计算样本均值、样本标准差与 t 值；
3. 根据 p 值判断是否达标。`
    },
    code: {
      python: `import scipy.stats as stats
x = [980, 995, 1002, 988, 991, 1005, 997]
t, p = stats.ttest_1samp(x, popmean=1000)
print(t, p)`,
      matlab: `% MATLAB 单样本T检验
x = [980 995 1002 988 991 1005 997];
[h,p,ci,stats] = ttest(x, 1000);`
    },
    resources: [
      { title: "One-sample t-test tutorial", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html", type: "link" }
    ]
  },
  {
    id: "paired-sample-t-test",
    name: "配对样本T检验 (Paired-Sample T-Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较同一对象在两种条件下的均值差异，常见于前后测、左右手、实验前后对照等配对设计。",
    principles: `1. 将配对观测差值记为 d_i = x_i - y_i。
2. 原假设 H0：差值总体均值 \\mu_d = 0。
3. 统计量：t = \\bar{d}/(s_d/\\sqrt{n})。
4. 本质上是对差值序列做单样本 T 检验。`,
    scenarios: [
      "干预前后成绩、血压、产量变化显著性检验",
      "同一批样本在两种方案下表现差异比较"
    ],
    limitations: [
      "要求样本必须一一配对，且差值近似正态",
      "若配对关系错误会直接破坏结论有效性"
    ],
    caseStudy: {
      title: "培训前后测试分数差异分析",
      description: "比较同一批学生培训前后的成绩是否显著提升。",
      solution: `1. 计算每位学生的前后分差；
2. 对差值执行配对 T 检验；
3. 判断培训效果是否显著。`
    },
    code: {
      python: `import scipy.stats as stats
before = [60, 65, 70, 68, 72]
after = [66, 70, 78, 72, 80]
t, p = stats.ttest_rel(after, before)
print(t, p)`,
      matlab: `% MATLAB 配对样本T检验
before = [60 65 70 68 72];
after = [66 70 78 72 80];
[h,p,ci,stats] = ttest(after, before);`
    },
    resources: [
      { title: "Paired t-test guide", url: "https://www.mathworks.com/help/stats/ttest.html", type: "link" }
    ]
  },
  {
    id: "independent-sample-t-test",
    name: "独立样本T检验 (Independent-Sample T-Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较两个独立总体的均值是否存在显著差异，是两组实验对照设计中最常用的方法。",
    principles: `1. 原假设 H0：\\mu_1 = \\mu_2。
2. 若满足方差齐性，可采用 pooled variance 形式的 t 检验。
3. 若不满足方差齐性，则使用 Welch T 检验。
4. 根据 p 值判断两组均值差异是否显著。`,
    scenarios: [
      "实验组与对照组效果差异分析",
      "男女性、南北方、不同机器批次均值差异比较"
    ],
    limitations: [
      "默认假设两组独立，若存在配对关系则不能使用",
      "需关注正态性与方差齐性假设"
    ],
    caseStudy: {
      title: "两种肥料对作物产量的差异比较",
      description: "比较两种肥料下两组独立试验田的平均产量是否不同。",
      solution: `1. 检查方差齐性；
2. 选择普通 t 检验或 Welch 检验；
3. 输出显著性结论。`
    },
    code: {
      python: `import scipy.stats as stats
a = [82, 85, 88, 84, 90]
b = [78, 80, 79, 81, 77]
t, p = stats.ttest_ind(a, b, equal_var=False)
print(t, p)`,
      matlab: `% MATLAB 独立样本T检验
a = [82 85 88 84 90];
b = [78 80 79 81 77];
[h,p,ci,stats] = ttest2(a, b, 'Vartype', 'unequal');`
    },
    resources: [
      { title: "Welch t-test reference", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html", type: "link" }
    ]
  },
  {
    id: "one-way-anova",
    name: "单因素方差分析 (One-Way ANOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "比较三个及以上组别均值是否存在显著差异，是多组均值差异分析的经典方法。",
    principles: `1. 原假设 H0：各组总体均值相等。
2. 将总变异分解为组间变异与组内变异。
3. 构造统计量 F = MSB / MSW。
4. 若 p 值较小，则说明至少有一组均值与其他组不同。`,
    scenarios: [
      "多种处理方案、多个品牌、多个地区均值差异检验",
      "建模中三组及以上实验条件的显著性比较"
    ],
    limitations: [
      "只能判断是否存在差异，不能直接说明哪几组不同",
      "需满足独立性、正态性与方差齐性假设"
    ],
    caseStudy: {
      title: "三种教学方法成绩差异比较",
      description: "比较三种教学方式下学生平均分数是否有显著差异。",
      solution: `1. 计算组间与组内平方和；
2. 得到 F 统计量与 p 值；
3. 若显著，再继续做事后多重比较。`
    },
    code: {
      python: `import scipy.stats as stats
g1 = [82, 85, 84]
g2 = [76, 78, 80]
g3 = [88, 90, 91]
F, p = stats.f_oneway(g1, g2, g3)
print(F, p)`,
      matlab: `% MATLAB 单因素方差分析
y = [82 85 84 76 78 80 88 90 91]';
g = [ones(3,1); 2*ones(3,1); 3*ones(3,1)];
[p,tbl,stats] = anova1(y, g);`
    },
    resources: [
      { title: "One-way ANOVA", url: "https://www.mathworks.com/help/stats/anova1.html", type: "link" }
    ]
  },
  {
    id: "post-hoc-multiple-comparison",
    name: "事后多重比较 (Post Hoc Multiple Comparison)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "在 ANOVA 显著后进一步确定具体哪几组之间存在差异，常用 Tukey、Bonferroni、Dunnett 等方法。",
    principles: `1. 先由 ANOVA 判断整体均值差异是否显著。
2. 再对组间两两均值差进行校正比较。
3. Tukey HSD 适合所有组间两两比较。
4. Bonferroni 更保守，可控制多重检验的一类错误率。`,
    scenarios: [
      "单因素或多因素方差分析后的分组差异定位",
      "多组实验处理效果的精细比较"
    ],
    limitations: [
      "多重校正会降低检验功效",
      "应根据研究目的选择合适的校正方法"
    ],
    caseStudy: {
      title: "不同药物组疗效两两比较",
      description: "ANOVA 显著后，进一步判断究竟哪些药物组之间存在显著差异。",
      solution: `1. 完成整体方差分析；
2. 使用 Tukey HSD 进行两两比较；
3. 输出显著差异组对。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.DataFrame({
    'score': [82,85,84,76,78,80,88,90,91],
    'group': ['A','A','A','B','B','B','C','C','C']
})
print(pairwise_tukeyhsd(df['score'], df['group']))`,
      matlab: `% MATLAB 事后多重比较
[c,m,h,gnames] = multcompare(stats);`
    },
    resources: [
      { title: "Tukey HSD in statsmodels", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "two-way-anova",
    name: "双因素方差分析 (Two-Way ANOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "同时考察两个因素及其交互作用对响应变量的影响，是实验设计中非常常见的差异分析方法。",
    principles: `1. 将总变异分解为因素 A、因素 B、交互项 AB 与误差项。
2. 分别计算各来源的均方与 F 统计量。
3. 同时检验主效应和交互效应。
4. 若交互显著，应优先解释交互而非单独主效应。`,
    scenarios: [
      "温度与湿度、药物与性别等双因素实验设计",
      "研究两个因素是否共同影响结果变量"
    ],
    limitations: [
      "模型解释较单因素更复杂，交互项显著时需谨慎解释",
      "设计不平衡时分析会更复杂"
    ],
    caseStudy: {
      title: "温度与肥料对产量的双因素分析",
      description: "研究温度水平和肥料类型是否共同影响作物产量。",
      solution: `1. 构建含交互项的线性模型；
2. 分析两个主效应及交互效应；
3. 判断最重要的影响来源。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.DataFrame({
    'y':[10,12,11,14,15,13,16,18,17,19,20,18],
    'A':['L','L','L','L','H','H','H','H','L','L','H','H'],
    'B':['X','X','Y','Y','X','X','Y','Y','X','Y','X','Y']
})
model = ols('y ~ C(A) * C(B)', data=df).fit()
print(sm.stats.anova_lm(model, typ=2))`,
      matlab: `% MATLAB 双因素方差分析
[p,tbl,stats] = anovan(y, {A, B}, 'model', 'interaction');`
    },
    resources: [
      { title: "Two-way ANOVA", url: "https://www.mathworks.com/help/stats/anovan.html", type: "link" }
    ]
  },
  {
    id: "three-way-anova",
    name: "三因素方差分析 (Three-Way ANOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "研究三个因素及其多重交互项对结果变量的影响，适用于更复杂的实验设计。",
    principles: `1. 同时检验三个主效应 A、B、C。
2. 考察两两交互 AB、AC、BC 以及三重交互 ABC。
3. 通过 F 检验判断各效应是否显著。
4. 若高阶交互显著，应优先从交互结构角度解释结果。`,
    scenarios: [
      "多条件实验设计、工业过程控制、多因素配方实验",
      "同时研究多个控制因子影响的场景"
    ],
    limitations: [
      "模型复杂度高，对样本量要求更高",
      "高阶交互项解释难度较大"
    ],
    caseStudy: {
      title: "温度、压力和催化剂三因素实验",
      description: "分析三个工艺因素对反应收率的综合影响。",
      solution: `1. 构建三因素含交互模型；
2. 检验主效应与各阶交互项；
3. 输出显著因素组合。`
    },
    code: {
      python: `# 可使用 statsmodels 通过公式 y ~ A*B*C 拟合
print('Use statsmodels formula: y ~ C(A)*C(B)*C(Cf)')`,
      matlab: `% MATLAB 三因素方差分析
[p,tbl,stats] = anovan(y, {A, B, C}, 'model', 'interaction');`
    },
    resources: [
      { title: "Higher-order ANOVA overview", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "multi-way-anova",
    name: "多因素方差分析 (N-Way ANOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "将方差分析推广到三个以上因素的场景，用于系统考察多个分类因素对响应变量的影响。",
    principles: `1. 将响应变量表示为多个因素及其交互项的线性模型。
2. 对每个主效应与交互项构造 F 检验。
3. 通过型别平方和（Type I/II/III）处理不平衡设计。
4. 输出显著因素、交互结构与解释贡献。`,
    scenarios: [
      "复杂试验设计和多控制因素工业实验",
      "竞赛中分类因素较多的差异显著性分析"
    ],
    limitations: [
      "因素越多，模型越复杂且结果解释成本显著上升",
      "高维交互项会造成样本量需求急剧增加"
    ],
    caseStudy: {
      title: "复杂工艺参数组合分析",
      description: "同时考虑多个工艺因素对产品性能的影响。",
      solution: `1. 建立多因素 ANOVA 模型；
2. 选用合适平方和类型；
3. 输出显著因素并进行降阶解释。`
    },
    code: {
      python: `print('Use statsmodels with formula including multiple categorical factors and interactions')`,
      matlab: `% MATLAB 多因素方差分析
[p,tbl,stats] = anovan(y, group_cells, 'model', 'interaction');`
    },
    resources: [
      { title: "ANOVA with multiple factors", url: "https://www.mathworks.com/help/stats/anovan.html", type: "link" }
    ]
  },
  {
    id: "manova",
    name: "多变量方差分析 (MANOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "当存在多个相关响应变量时，同时检验组间在多维均值向量上的差异，避免分别检验带来的信息割裂。",
    principles: `1. 将多个响应变量组成向量 Y。
2. 检验不同组在均值向量上的整体差异。
3. 常见统计量包括 Wilks' Lambda、Pillai's Trace 等。
4. 若整体显著，再进一步查看各变量的后续分析。`,
    scenarios: [
      "同时考察成绩、满意度、效率等多个相关结果变量",
      "多维输出实验的整体差异检验"
    ],
    limitations: [
      "对协方差结构、样本量和多元正态性有要求",
      "结果解释比单变量 ANOVA 更复杂"
    ],
    caseStudy: {
      title: "不同教学法对多项学习指标的综合影响",
      description: "比较三种教学方法对成绩、兴趣和参与度三项指标的联合影响。",
      solution: `1. 构建多响应变量矩阵；
2. 进行 MANOVA；
3. 若显著，再分解到单个指标。`
    },
    code: {
      python: `from statsmodels.multivariate.manova import MANOVA
import pandas as pd

df = pd.DataFrame({
    'score':[80,82,78,88,90,87],
    'interest':[70,72,68,85,87,84],
    'group':['A','A','A','B','B','B']
})
print(MANOVA.from_formula('score + interest ~ group', data=df).mv_test())`,
      matlab: `% MATLAB MANOVA 可使用 manova1
% [d,p,stats] = manova1(X, group);`
    },
    resources: [
      { title: "Statsmodels MANOVA", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "ancova",
    name: "协方差分析 (ANCOVA)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "在比较组均值差异时控制协变量影响，是 ANOVA 与回归结合的经典方法。",
    principles: `1. 在分类因素模型中加入连续协变量。
2. 先用协变量解释一部分波动，再比较各组调整后的均值差异。
3. 模型常写作：Y = 因子效应 + 协变量效应 + 误差。
4. 能降低误差方差，提高组间差异检验的灵敏度。`,
    scenarios: [
      "控制基线水平后比较实验组与对照组差异",
      "教育、医学、农业试验中剔除背景变量影响"
    ],
    limitations: [
      "要求协变量与响应变量线性相关，且各组回归斜率关系需合理",
      "协变量若受处理本身影响，解释会变复杂"
    ],
    caseStudy: {
      title: "控制入学成绩后的教学法差异比较",
      description: "研究不同教学法对期末成绩的影响，同时控制入学基线成绩。",
      solution: `1. 将期末成绩作为因变量；
2. 将教学法作为因素，入学成绩作为协变量；
3. 比较校正后的组间差异。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.DataFrame({
    'post':[82,84,80,88,90,86],
    'pre':[70,72,69,73,74,71],
    'group':['A','A','A','B','B','B']
})
model = ols('post ~ pre + C(group)', data=df).fit()
print(sm.stats.anova_lm(model, typ=2))`,
      matlab: `% MATLAB ANCOVA 可通过 fitlm / anovan 实现
mdl = fitlm(tbl, 'post ~ pre + group');`
    },
    resources: [
      { title: "ANCOVA overview", url: "https://www.mathworks.com/help/stats/fitlm.html", type: "link" }
    ]
  },
  {
    id: "summary-one-way-anova",
    name: "摘要单因素方差分析 (One-Way ANOVA from Summary Data)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "当只有各组样本量、均值和标准差等摘要统计量时，可基于摘要信息完成单因素方差分析。",
    principles: `1. 仅依赖各组 n_i、\\bar{x}_i、s_i 等摘要信息。
2. 利用摘要统计量重构组间平方和与组内平方和。
3. 计算 F 值和 p 值完成检验。
4. 适合文献复现、二次分析和数据不可得场景。`,
    scenarios: [
      "只有论文表格摘要结果而没有原始数据时的差异检验",
      "公开统计报告的二次复核分析"
    ],
    limitations: [
      "缺乏原始数据时无法做更细致的分布诊断",
      "对异常值、偏态等细节无法直接检查"
    ],
    caseStudy: {
      title: "根据论文摘要表复算组间差异",
      description: "某研究只给出了三组的样本量、均值和标准差，希望复核其单因素 ANOVA 结果。",
      solution: `1. 由摘要量重构组内与组间变异；
2. 计算 F 和 p；
3. 与文献结果比对一致性。`
    },
    code: {
      python: `print('Compute SSB and SSW from group n, mean, std, then F = MSB / MSW')`,
      matlab: `% MATLAB 摘要单因素方差分析需自行根据摘要量计算平方和`
    },
    resources: [
      { title: "ANOVA from summary statistics", url: "https://en.wikipedia.org/wiki/F-test", type: "link" }
    ]
  },
  {
    id: "summary-t-test",
    name: "摘要T检验 (T-Test from Summary Data)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "当没有原始样本，仅有均值、标准差和样本量时，可基于摘要统计量完成单样本或双样本 T 检验。",
    principles: `1. 使用样本量 n、样本均值 \\bar{x}、标准差 s 构造 t 统计量。
2. 双样本场景可根据是否方差齐性选择 pooled 或 Welch 形式。
3. 通过自由度与 p 值判断显著性。
4. 常用于文献复现与元分析前的基础统计。`,
    scenarios: [
      "没有原始数据但需要复现论文统计结论",
      "公开报告数据的二次检验"
    ],
    limitations: [
      "无法检查原始分布与异常值情况",
      "配对结构若缺失则无法准确还原配对检验"
    ],
    caseStudy: {
      title: "根据论文表格数据复算两组差异",
      description: "文献仅提供两组的均值、标准差与样本量，需要验证其显著性结论。",
      solution: `1. 由摘要量计算 t 值；
2. 根据自由度计算 p 值；
3. 输出复核结论。`
    },
    code: {
      python: `print('Use mean, std, n to manually compute t statistic and p-value')`,
      matlab: `% MATLAB 摘要T检验可根据均值、标准差、样本量手动计算`
    },
    resources: [
      { title: "t-test from summary statistics", url: "https://www.bmj.com", type: "paper" }
    ]
  },
  {
    id: "one-sample-equivalence-test",
    name: "单样本等价检验 (One-Sample Equivalence Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于检验样本均值是否落在给定等价区间内，强调“足够接近”而不是“是否不同”。",
    principles: `1. 设等价区间为 [-\\Delta, \\Delta] 或围绕目标值的容许偏差区间。
2. 常用 TOST（Two One-Sided Tests）方法。
3. 分别检验均值差是否大于下界、是否小于上界。
4. 两个单侧检验都通过时，可判定等价成立。`,
    scenarios: [
      "验证产品指标是否与标准值等价",
      "药物、生物等效性和工艺替代性分析"
    ],
    limitations: [
      "需要预先合理设定等价界值，主观性较强",
      "等价检验与差异检验结论逻辑不同，不能混用"
    ],
    caseStudy: {
      title: "新传感器读数是否与标准仪器等价",
      description: "检验新设备平均读数是否落在标准值允许误差范围内。",
      solution: `1. 设定允许误差边界；
2. 执行 TOST 单样本等价检验；
3. 判断新设备是否可替代。`
    },
    code: {
      python: `from statsmodels.stats.weightstats import ttost_1samp
x = [9.8, 10.1, 10.0, 9.9, 10.2]
print(ttost_1samp(x, low=9.7, upp=10.3))`,
      matlab: `% MATLAB 单样本等价检验可手动实现 TOST`
    },
    resources: [
      { title: "TOST equivalence testing", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "two-sample-equivalence-test",
    name: "双样本等价检验 (Two-Sample Equivalence Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于判断两个独立样本总体均值差是否处于预设等价范围内，是差异显著性检验的重要补充。",
    principles: `1. 原假设不是“相等”，而是“差异超过等价界值”。
2. 常用 TOST 方法分别检验均值差大于下界、小于上界。
3. 若两个单侧检验均显著，则认为两总体等价。
4. 特别适用于替代方案、替代材料、替代药物比较。`,
    scenarios: [
      "比较新旧方案是否在可接受误差范围内等价",
      "工艺替代、设备替代和生物等效性分析"
    ],
    limitations: [
      "等价边界的设定直接影响结论",
      "等价并不意味着完全相同，只表示差异小到可接受"
    ],
    caseStudy: {
      title: "新旧电池容量是否等价",
      description: "比较新工艺电池与旧工艺电池容量是否在允许偏差范围内保持等价。",
      solution: `1. 设定容量差异容许区间；
2. 对两独立样本执行 TOST；
3. 输出是否满足等价。`
    },
    code: {
      python: `from statsmodels.stats.weightstats import ttost_ind
x1 = [100, 101, 99, 102]
x2 = [100, 100, 98, 101]
print(ttost_ind(x1, x2, low=-3, upp=3))`,
      matlab: `% MATLAB 双样本等价检验可手动实现 TOST`
    },
    resources: [
      { title: "Two-sample equivalence test", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "paired-equivalence-test",
    name: "配对样本等价检验 (Paired Equivalence Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于检验配对观测的平均差值是否位于可接受等价区间内，本质上是对配对差值进行等价检验。",
    principles: `1. 先计算配对差值 d_i = x_i - y_i。
2. 将等价区间作用于差值均值 \\mu_d。
3. 使用 TOST 对差值均值执行双单侧检验。
4. 适合前后测或同一对象双方案比较的等价性分析。`,
    scenarios: [
      "同一对象在两台设备上的测量是否等价",
      "新旧算法对同一批样本结果是否可替代"
    ],
    limitations: [
      "需要严格一一配对",
      "差值的分布假设和等价边界设定都需谨慎"
    ],
    caseStudy: {
      title: "两台仪器对同一批样品的测量等价性",
      description: "比较两台仪器对同一样品的输出差异是否在可接受范围内。",
      solution: `1. 计算每对样本的差值；
2. 对差值序列执行 TOST；
3. 判断两仪器能否互换使用。`
    },
    code: {
      python: `print('Paired equivalence test can be implemented by applying TOST to paired differences')`,
      matlab: `% MATLAB 配对等价检验可对差值手动做 TOST`
    },
    resources: [
      { title: "Equivalence testing concepts", url: "https://www.ncbi.nlm.nih.gov", type: "paper" }
    ]
  },
  {
    id: "chi-square-test",
    name: "卡方检验 (Chi-Square Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "面向分类变量的经典差异分析工具，可用于独立性检验、同质性检验与列联表关联分析。",
    principles: `1. 构造列联表并记录观测频数 O_{ij}。
2. 在原假设下计算期望频数 E_{ij}。
3. 统计量：\\chi^2 = \\sum (O_{ij} - E_{ij})^2 / E_{ij}。
4. 根据卡方分布与 p 值判断分类变量之间是否相关。`,
    scenarios: [
      "性别与品牌偏好、地区与选择行为等分类变量差异分析",
      "问卷调查与列联表数据显著性分析"
    ],
    limitations: [
      "若期望频数过小，卡方近似会失效",
      "只能反映统计关联，不能直接说明因果"
    ],
    caseStudy: {
      title: "地区与产品偏好差异检验",
      description: "比较不同地区用户对产品型号偏好的分布是否一致。",
      solution: `1. 构造地区 × 型号列联表；
2. 计算卡方统计量；
3. 判断分布差异是否显著。`
    },
    code: {
      python: `import numpy as np
import scipy.stats as stats
obs = np.array([[30, 20], [15, 35]])
chi2, p, dof, ex = stats.chi2_contingency(obs)
print(chi2, p)`,
      matlab: `% MATLAB 卡方检验
obs = [30 20; 15 35];
% 可用 crosstab 或 chi2gof 等函数辅助实现`
    },
    resources: [
      { title: "Chi-square test", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html", type: "link" }
    ]
  },
  {
    id: "one-sample-wilcoxon-signed-rank",
    name: "单样本Wilcoxon符号秩检验 (One-Sample Wilcoxon Signed-Rank Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于检验单样本中位数是否等于某个给定值，是单样本 T 检验的非参数替代方法。",
    principles: `1. 将样本与假设中位数作差。
2. 去掉为 0 的差值，对绝对差值排序并赋秩。
3. 分别求正负差值的秩和。
4. 根据秩和统计量判断中位数是否显著偏离目标值。`,
    scenarios: [
      "分布偏态时检验中位数是否等于理论值",
      "非正态小样本的一样本差异分析"
    ],
    limitations: [
      "检验的是位置差异，通常解释为中位数差异而非均值差异",
      "若数据大量并列或离散化严重，检验效能会下降"
    ],
    caseStudy: {
      title: "用户评分中位数是否高于 4 分",
      description: "评分数据偏态明显，使用非参数方法检验总体位置。",
      solution: `1. 以 4 分为比较基准；
2. 计算差值并做符号秩检验；
3. 判断评分是否显著高于目标水平。`
    },
    code: {
      python: `import numpy as np
from scipy.stats import wilcoxon
x = np.array([4.2, 4.5, 4.0, 4.8, 4.6])
stat, p = wilcoxon(x - 4.0)
print(stat, p)`,
      matlab: `% MATLAB 单样本 Wilcoxon 符号秩检验
x = [4.2 4.5 4.0 4.8 4.6];
p = signrank(x, 4.0);`
    },
    resources: [
      { title: "Wilcoxon signed-rank test", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html", type: "link" }
    ]
  },
  {
    id: "paired-wilcoxon-signed-rank",
    name: "配对样本Wilcoxon符号秩检验 (Paired Wilcoxon Signed-Rank Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较两组配对样本的中位数差异，是配对样本 T 检验的非参数替代方法。",
    principles: `1. 计算配对差值 d_i = x_i - y_i。
2. 对非零绝对差值排序并赋秩。
3. 计算正负秩和并构造检验统计量。
4. 判断配对条件下两次测量位置是否显著不同。`,
    scenarios: [
      "治疗前后、实验前后、左右手等配对非参数差异分析",
      "配对数据但差值不服从正态时的稳健检验"
    ],
    limitations: [
      "只能用于配对设计，不能替代独立样本检验",
      "大量相同值会影响秩次区分"
    ],
    caseStudy: {
      title: "干预前后满意度差异分析",
      description: "配对满意度评分分布偏态，使用 Wilcoxon 检验分析干预效果。",
      solution: `1. 计算配对差值；
2. 进行符号秩检验；
3. 输出显著性结论。`
    },
    code: {
      python: `from scipy.stats import wilcoxon
before = [3, 4, 4, 5, 3]
after = [4, 4, 5, 5, 4]
stat, p = wilcoxon(after, before)
print(stat, p)`,
      matlab: `% MATLAB 配对 Wilcoxon 检验
before = [3 4 4 5 3];
after = [4 4 5 5 4];
p = signrank(after, before);`
    },
    resources: [
      { title: "Paired Wilcoxon", url: "https://www.mathworks.com/help/stats/signrank.html", type: "link" }
    ]
  },
  {
    id: "mann-whitney-u-test",
    name: "独立样本MannWhitney检验 (Mann-Whitney U Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较两个独立样本的位置分布差异，是独立样本 T 检验的经典非参数替代方法。",
    principles: `1. 将两组样本合并排序并赋秩。
2. 计算各组秩和并构造 U 统计量。
3. 原假设为两组分布位置相同。
4. 当分布形态相近时，常解释为中位数差异检验。`,
    scenarios: [
      "两独立组偏态数据差异比较",
      "非正态或存在异常值时的双样本对照分析"
    ],
    limitations: [
      "若两组分布形态差别很大，结果不宜简单解释为中位数差异",
      "主要基于秩而非原始数值差"
    ],
    caseStudy: {
      title: "两种界面下用户停留时长差异",
      description: "停留时长明显偏态，比较两独立用户组的使用差异。",
      solution: `1. 合并两组数据并计算秩次；
2. 计算 U 值和 p 值；
3. 判断两组是否有显著差异。`
    },
    code: {
      python: `from scipy.stats import mannwhitneyu
a = [12, 15, 18, 20]
b = [8, 9, 10, 13]
stat, p = mannwhitneyu(a, b, alternative='two-sided')
print(stat, p)`,
      matlab: `% MATLAB Mann-Whitney 检验
p = ranksum(a, b);`
    },
    resources: [
      { title: "Mann-Whitney U", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html", type: "link" }
    ]
  },
  {
    id: "friedman-test",
    name: "多配对样本Friedman检验 (Friedman Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较三个及以上配对条件下的差异，是重复测量 ANOVA 的非参数替代方法。",
    principles: `1. 每个对象在多个条件下都有观测值。
2. 对每个对象内部的多个条件赋秩。
3. 汇总各条件秩和并构造 Friedman 统计量。
4. 判断多个配对条件的位置差异是否显著。`,
    scenarios: [
      "同一批对象在多种方案、多个时间点下的非参数比较",
      "重复测量但不满足正态性假设的场景"
    ],
    limitations: [
      "只能判断整体是否有差异，显著后仍需做后续两两比较",
      "要求每个对象在所有条件下都有观测"
    ],
    caseStudy: {
      title: "三种算法在同一批数据集上的表现比较",
      description: "同一批任务分别由三种算法处理，比较其效果差异。",
      solution: `1. 按任务内部对三算法结果赋秩；
2. 做 Friedman 检验；
3. 若显著再开展后续比较。`
    },
    code: {
      python: `from scipy.stats import friedmanchisquare
a = [8, 7, 9, 6]
b = [6, 5, 7, 4]
c = [9, 8, 10, 7]
stat, p = friedmanchisquare(a, b, c)
print(stat, p)`,
      matlab: `% MATLAB Friedman 检验
p = friedman(X, 1);`
    },
    resources: [
      { title: "Friedman test", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html", type: "link" }
    ]
  },
  {
    id: "kruskal-wallis-test",
    name: "多独立样本Kruskal-Wallis检验 (Kruskal-Wallis Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较三个及以上独立样本的分布位置差异，是单因素方差分析的非参数替代方法。",
    principles: `1. 将所有组样本合并排序后赋秩。
2. 比较各组平均秩是否存在系统差异。
3. 构造 H 统计量并近似服从卡方分布。
4. 当显著时说明至少有一组分布位置不同。`,
    scenarios: [
      "多独立组偏态数据差异检验",
      "多组样本不满足正态性和方差齐性时的稳健比较"
    ],
    limitations: [
      "显著后不能直接指出具体哪几组不同，仍需事后比较",
      "主要基于秩次信息，损失了部分数值细节"
    ],
    caseStudy: {
      title: "三种客服流程满意度差异分析",
      description: "三组独立用户满意度分布明显偏态，希望比较其整体差异。",
      solution: `1. 合并数据并排序赋秩；
2. 执行 Kruskal-Wallis 检验；
3. 根据结果决定是否继续做事后比较。`
    },
    code: {
      python: `from scipy.stats import kruskal
g1 = [82, 84, 86]
g2 = [76, 79, 81]
g3 = [88, 90, 91]
stat, p = kruskal(g1, g2, g3)
print(stat, p)`,
      matlab: `% MATLAB Kruskal-Wallis 检验
p = kruskalwallis(y, group);`
    },
    resources: [
      { title: "Kruskal-Wallis test", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html", type: "link" }
    ]
  },
  {
    id: "chi-square-goodness-of-fit",
    name: "卡方拟合优度检验 (Chi-Square Goodness-of-Fit Test)",
    category: "difference-analysis",
    categoryName: "📊 差异性分析",
    summary: "用于比较观测频数分布与理论分布是否一致，是分类数据分布适配性检验的经典方法。",
    principles: `1. 给定理论分布下的期望频数 E_i。
2. 统计观测频数 O_i。
3. 计算统计量：\\chi^2 = \\sum (O_i - E_i)^2 / E_i。
4. 判断样本分布是否显著偏离理论分布。`,
    scenarios: [
      "检验骰子是否公平、到达频数是否符合泊松分布",
      "分类频数是否符合既定比例结构"
    ],
    limitations: [
      "期望频数过小会影响卡方近似准确性",
      "理论分布若来自样本估计，自由度需相应调整"
    ],
    caseStudy: {
      title: "检验抽奖结果是否均匀分布",
      description: "比较不同奖项的出现次数是否与理论设定比例一致。",
      solution: `1. 统计各奖项观测频数；
2. 根据理论概率得到期望频数；
3. 计算卡方拟合优度统计量并判断显著性。`
    },
    code: {
      python: `import numpy as np
from scipy.stats import chisquare
obs = np.array([18, 22, 20, 19, 21])
exp = np.array([20, 20, 20, 20, 20])
stat, p = chisquare(obs, f_exp=exp)
print(stat, p)`,
      matlab: `% MATLAB 卡方拟合优度检验
obs = [18 22 20 19 21];
exp = [20 20 20 20 20];
chi2 = sum((obs-exp).^2 ./ exp);`
    },
    resources: [
      { title: "Chi-square goodness of fit", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html", type: "link" }
    ]
  },
  {
    id: "pearson-correlation",
    name: "Pearson相关性分析 (Pearson Correlation)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "衡量两个连续变量之间线性相关强度与方向，是最常用的相关性分析方法。",
    principles: `1. Pearson 相关系数定义为 r = Cov(X,Y)/(\\sigma_X \\sigma_Y)。
2. 取值范围为 [-1, 1]，绝对值越大线性相关越强。
3. r > 0 表示正相关，r < 0 表示负相关。
4. 常配合显著性检验判断相关系数是否显著偏离 0。`,
    scenarios: [
      "分析收入与消费、温度与产量等连续变量之间的线性关系",
      "回归分析前的变量初筛与共线性检查"
    ],
    limitations: [
      "仅适合衡量线性关系，对非线性但单调关系可能低估关联程度",
      "对异常值较敏感"
    ],
    caseStudy: {
      title: "广告投入与销售额线性相关分析",
      description: "希望判断广告预算增加是否与销售额提升呈显著线性关系。",
      solution: `1. 收集多期广告投入与销售额数据；
2. 计算 Pearson 相关系数 r；
3. 根据 p 值判断相关性是否显著。`
    },
    code: {
      python: `from scipy.stats import pearsonr
x = [10, 12, 15, 18, 20]
y = [100, 120, 150, 170, 210]
r, p = pearsonr(x, y)
print(r, p)`,
      matlab: `% MATLAB Pearson相关分析
x = [10 12 15 18 20]';
y = [100 120 150 170 210]';
[R,P] = corr(x, y, 'Type', 'Pearson');`
    },
    resources: [
      { title: "Pearson correlation in SciPy", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html", type: "link" }
    ]
  },
  {
    id: "spearman-correlation",
    name: "Spearman相关性分析 (Spearman Rank Correlation)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "基于秩次衡量两个变量之间单调关系的强弱，适用于偏态数据、等级数据和非线性单调关系。",
    principles: `1. 先将原始观测值转化为秩次。
2. 计算秩变量之间的 Pearson 相关，得到 Spearman 系数 \\rho_s。
3. 其本质衡量的是单调相关而不要求严格线性。
4. 对异常值的敏感性低于 Pearson。`,
    scenarios: [
      "等级评分、满意度、排名等有序变量相关分析",
      "连续变量非正态或仅呈单调关系时的稳健分析"
    ],
    limitations: [
      "只反映单调关系，不适合复杂非单调结构",
      "大量并列秩时解释力会下降"
    ],
    caseStudy: {
      title: "满意度排名与复购意愿相关分析",
      description: "问卷评分偏态明显，希望用秩相关评估满意度与复购意愿之间的关系。",
      solution: `1. 将两个变量转为秩次；
2. 计算 Spearman 系数；
3. 判断单调关系是否显著。`
    },
    code: {
      python: `from scipy.stats import spearmanr
x = [1, 2, 3, 4, 5]
y = [3, 4, 4, 5, 5]
rho, p = spearmanr(x, y)
print(rho, p)`,
      matlab: `% MATLAB Spearman相关分析
[R,P] = corr(x, y, 'Type', 'Spearman');`
    },
    resources: [
      { title: "Spearman rank correlation", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html", type: "link" }
    ]
  },
  {
    id: "kendall-tau-b-correlation",
    name: "Kendall's tau-b相关性分析 (Kendall's Tau-b)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "基于一致对与不一致对的比较来衡量两个变量的等级相关性，尤其适合小样本和存在大量并列值的等级数据。",
    principles: `1. 比较任意两对观测在两个变量上的排序是否一致。
2. 统计一致对数 C 与不一致对数 D。
3. Tau-b 会对并列值进行修正。
4. 系数取值范围为 [-1, 1]，越接近两端表示等级相关越强。`,
    scenarios: [
      "小样本等级数据的相关性研究",
      "存在较多并列值时的稳健秩相关分析"
    ],
    limitations: [
      "计算复杂度高于 Pearson 和 Spearman",
      "主要适用于有序变量或秩信息场景"
    ],
    caseStudy: {
      title: "专家排序意见一致性中的等级相关分析",
      description: "比较两位专家对若干项目的排序结果是否一致。",
      solution: `1. 记录两位专家的排序；
2. 计算 Kendall tau-b 系数；
3. 根据显著性结果判断排序相关程度。`
    },
    code: {
      python: `from scipy.stats import kendalltau
x = [1, 2, 3, 4, 5]
y = [1, 2, 2, 4, 5]
tau, p = kendalltau(x, y)
print(tau, p)`,
      matlab: `% MATLAB Kendall相关分析
[R,P] = corr(x, y, 'Type', 'Kendall');`
    },
    resources: [
      { title: "Kendall tau correlation", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html", type: "link" }
    ]
  },
  {
    id: "cochrans-q-test",
    name: "Cochran's Q 检验 (Cochran's Q Test)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "用于比较三个及以上配对二分类处理结果是否存在系统差异，本质上是 McNemar 检验的多组扩展。",
    principles: `1. 数据要求为同一对象在多个条件下的二分类结果（如是/否、成功/失败）。
2. 原假设为各处理条件下阳性概率相同。
3. 基于行列边际统计量构造 Q 统计量。
4. 在样本量较大时近似服从卡方分布。`,
    scenarios: [
      "同一批受试者在多种二分类条件下结果比较",
      "多个诊断方法对同一样本判断结果的差异分析"
    ],
    limitations: [
      "仅适用于配对二分类数据",
      "显著后仍需进一步定位具体差异来源"
    ],
    caseStudy: {
      title: "三种诊断工具阳性判定差异检验",
      description: "同一批患者分别使用三种诊断工具，比较其阳性判定率是否一致。",
      solution: `1. 构造受试者 × 工具 的二分类矩阵；
2. 计算 Cochran's Q 统计量；
3. 判断各工具判定结果是否存在系统差异。`
    },
    code: {
      python: `from statsmodels.stats.contingency_tables import cochrans_q
x = [[1,1,0],[1,0,0],[0,1,0],[1,1,1]]
print(cochrans_q(x))`,
      matlab: `% MATLAB 可自行实现 Cochran's Q 或借助自定义函数`
    },
    resources: [
      { title: "Cochran's Q test in statsmodels", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "kappa-consistency-test",
    name: "Kappa一致性检验 (Cohen's Kappa)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "用于评估两名评价者对分类结果的一致程度，并剔除随机一致的影响。",
    principles: `1. 先构造两个评价者的分类列联表。
2. 计算观察一致率 P_o 与随机一致率 P_e。
3. Kappa 系数：\\kappa = (P_o - P_e)/(1 - P_e)。
4. \\kappa 越大表示超越随机因素后的一致性越强。`,
    scenarios: [
      "两位医生、两位标注员、两种分类工具结果一致性评价",
      "人工标注质量评估"
    ],
    limitations: [
      "受类别不平衡影响较大",
      "主要适合两位评价者情形，多评价者需扩展方法"
    ],
    caseStudy: {
      title: "两位标注员图像分类一致性评估",
      description: "希望判断两位标注员对图像分类标签的一致性是否可靠。",
      solution: `1. 构造两位标注员的标签对应表；
2. 计算 Kappa 系数；
3. 判断一致性是否达到可接受水平。`
    },
    code: {
      python: `from sklearn.metrics import cohen_kappa_score
a = [0,1,1,0,1]
b = [0,1,0,0,1]
print(cohen_kappa_score(a, b))`,
      matlab: `% MATLAB Kappa 可根据列联表手动计算`
    },
    resources: [
      { title: "Cohen kappa score", url: "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html", type: "link" }
    ]
  },
  {
    id: "kendall-coefficient-of-concordance",
    name: "Kendall一致性检验 (Kendall's W)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "用于衡量多名评价者对若干对象排序结果的一致性程度，是多专家排序一致性分析的经典方法。",
    principles: `1. 多位评价者分别对同一组对象进行排序。
2. 统计每个对象的总秩和。
3. 由总秩和离散程度计算 Kendall 一致性系数 W。
4. W 越接近 1，说明多评价者排序越一致。`,
    scenarios: [
      "专家打分排序一致性分析",
      "德尔菲法、层次分析前的专家意见一致性检验"
    ],
    limitations: [
      "只适用于排序或等级打分数据",
      "若存在大量并列值，需要做相应修正"
    ],
    caseStudy: {
      title: "专家评审排序一致性检验",
      description: "多位专家对候选方案进行排序，希望判断专家意见是否足够一致。",
      solution: `1. 汇总专家排序矩阵；
2. 计算 Kendall's W；
3. 检验专家意见是否具有统计一致性。`
    },
    code: {
      python: `print('Kendall W can be implemented from rank matrix by computing rank-sum dispersion')`,
      matlab: `% MATLAB Kendall一致性检验可基于秩矩阵手动计算 W`
    },
    resources: [
      { title: "Kendall coefficient of concordance", url: "https://en.wikipedia.org/wiki/Kendall%27s_W", type: "link" }
    ]
  },
  {
    id: "intraclass-correlation-coefficient",
    name: "组内相关系数 (Intraclass Correlation Coefficient, ICC)",
    category: "correlation-analysis",
    categoryName: "🔗 相关性分析",
    summary: "用于衡量同一组对象在不同评价者或重复测量条件下的一致性与可靠性，是连续型一致性分析的重要工具。",
    principles: `1. ICC 基于方差分解思想，将总变异拆分为对象差异、评价者差异与误差。
2. 不同设计对应不同 ICC 类型，如 ICC(1)、ICC(2)、ICC(3)。
3. ICC 越接近 1，说明组内一致性越高。
4. 常用于评分者一致性、重复测量可靠性和量表稳定性评估。`,
    scenarios: [
      "多位评分者对同一批对象打分的一致性评价",
      "医学、教育、心理测量中的重复测量可靠性分析"
    ],
    limitations: [
      "需根据研究设计正确选择 ICC 类型",
      "对方差结构和样本设计较敏感"
    ],
    caseStudy: {
      title: "多位教师评分一致性评估",
      description: "多位教师对同一批学生作文打分，希望判断评分体系是否稳定可靠。",
      solution: `1. 构建对象 × 评分者得分矩阵；
2. 根据研究设计选择 ICC 类型；
3. 计算 ICC 并解释评分可靠性。`
    },
    code: {
      python: `import pingouin as pg
import pandas as pd

df = pd.DataFrame({
    'targets':[1,1,2,2,3,3],
    'raters':['A','B','A','B','A','B'],
    'scores':[80,82,75,76,90,91]
})
print(pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='scores'))`,
      matlab: `% MATLAB ICC 可通过方差分解或统计工具箱相关流程实现`
    },
    resources: [
      { title: "Intraclass correlation in Pingouin", url: "https://pingouin-stats.org", type: "link" }
    ]
  },

  // ==================== 8. 机器学习分类 (machine-learning) ====================
  {
    id: "decision-tree-classification",
    name: "决策树分类 (Decision Tree Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过递归划分特征空间生成树状判别规则，是最直观、可解释性较强的分类模型之一。",
    principles: `1. 在每个节点选择最优特征进行划分。
2. 常用划分标准包括信息增益、信息增益率和基尼指数。
3. 递归生长子树直到满足停止条件。
4. 预测时按规则从根节点走到叶节点输出类别。`,
    scenarios: [
      "客户分类、风险识别、医学诊断等监督分类任务",
      "需要较强可解释性的分类场景"
    ],
    limitations: [
      "容易过拟合，对数据扰动较敏感",
      "单棵树的泛化能力通常不如集成模型"
    ],
    caseStudy: {
      title: "用户流失分类识别",
      description: "根据活跃度、投诉次数和消费频率判断用户是否流失。",
      solution: `1. 构建训练样本；
2. 用决策树学习分类规则；
3. 输出是否流失的分类结果。`
    },
    code: {
      python: `from sklearn.tree import DecisionTreeClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB 决策树分类
Mdl = fitctree(X, y);`
    },
    resources: [
      { title: "DecisionTreeClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html", type: "link" }
    ]
  },
  {
    id: "random-forest-classification",
    name: "随机森林分类 (Random Forest Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过多棵决策树投票完成分类，能有效降低单树过拟合并提升稳定性与泛化能力。",
    principles: `1. 对样本做 Bootstrap 重采样生成多组训练集。
2. 每棵树在节点划分时随机选择部分特征。
3. 训练多棵决策树形成森林。
4. 通过多数投票输出最终分类结果。`,
    scenarios: [
      "中小规模表格数据分类",
      "需要兼顾精度、稳定性和一定解释性的任务"
    ],
    limitations: [
      "模型体积较大，推理速度较单棵树慢",
      "整体解释性弱于单棵决策树"
    ],
    caseStudy: {
      title: "信用风险客户分类",
      description: "利用收入、负债、交易行为等特征识别高风险客户。",
      solution: `1. 构建多棵决策树；
2. 用随机森林训练分类器；
3. 用投票结果判定风险类别。`
    },
    code: {
      python: `from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB 随机森林分类可使用 TreeBagger
Mdl = TreeBagger(100, X, y, 'Method', 'classification');`
    },
    resources: [
      { title: "RandomForestClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html", type: "link" }
    ]
  },
  {
    id: "adaboost-classification",
    name: "adaboost分类 (AdaBoost Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过迭代训练弱分类器并重点关注前轮错分样本，不断提升整体分类性能。",
    principles: `1. 初始化所有样本权重相同。
2. 训练弱分类器并计算分类误差。
3. 提高错分样本权重，降低正确样本权重。
4. 加权组合多个弱分类器得到强分类器。`,
    scenarios: [
      "二分类或多分类表格数据任务",
      "希望在简单基分类器基础上提升精度的场景"
    ],
    limitations: [
      "对噪声点和异常值较敏感",
      "弱分类器选择不当时效果有限"
    ],
    caseStudy: {
      title: "营销响应用户分类",
      description: "根据历史行为判断用户是否会响应活动。",
      solution: `1. 用浅层树作为弱分类器；
2. 迭代更新样本权重；
3. 组合成强分类器。`
    },
    code: {
      python: `from sklearn.ensemble import AdaBoostClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB AdaBoost 可通过 fitcensemble 实现
Mdl = fitcensemble(X, y, 'Method', 'AdaBoostM1');`
    },
    resources: [
      { title: "AdaBoostClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html", type: "link" }
    ]
  },
  {
    id: "gbdt-classification",
    name: "梯度提升树(GBDT)分类 (Gradient Boosting Decision Tree Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过逐轮拟合前一轮损失函数的负梯度来增强模型，是经典的 Boosting 树分类方法。",
    principles: `1. 从一个弱模型开始建立初始预测。
2. 每轮拟合当前损失函数的负梯度残差。
3. 将新树按学习率叠加到已有模型上。
4. 迭代后形成强分类器。`,
    scenarios: [
      "表格数据分类",
      "需要较强非线性拟合能力的二分类和多分类任务"
    ],
    limitations: [
      "训练速度相对较慢",
      "参数较多，不当设置时易过拟合"
    ],
    caseStudy: {
      title: "贷款违约分类",
      description: "根据财务与行为特征判断借款人是否违约。",
      solution: `1. 构造训练样本；
2. 逐轮拟合损失梯度；
3. 输出分类概率与结果。`
    },
    code: {
      python: `from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = GradientBoostingClassifier(random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB GBDT 分类
Mdl = fitcensemble(X, y, 'Method', 'LogitBoost');`
    },
    resources: [
      { title: "GradientBoostingClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html", type: "link" }
    ]
  },
  {
    id: "catboost-classification",
    name: "CatBoost分类 (CatBoost Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "专为类别特征友好设计的梯度提升树模型，能在表格分类任务中取得稳定且高质量的表现。",
    principles: `1. 基于梯度提升树框架进行迭代建模。
2. 原生支持类别特征编码与有序提升策略。
3. 有效缓解目标泄漏和预测偏移问题。
4. 特别适合含大量类别变量的表格分类。`,
    scenarios: [
      "含类别特征较多的业务分类任务",
      "用户画像、金融风控、营销预测等表格场景"
    ],
    limitations: [
      "依赖额外库支持",
      "参数调优和训练耗时在大数据下需关注"
    ],
    caseStudy: {
      title: "电商用户转化分类",
      description: "利用地区、渠道、设备类型等类别变量预测用户是否转化。",
      solution: `1. 保留原始类别特征；
2. 使用 CatBoost 训练分类器；
3. 输出转化概率与类别。`
    },
    code: {
      python: `from catboost import CatBoostClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = CatBoostClassifier(verbose=0, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB CatBoost 通常需调用 Python 接口或外部工具`
    },
    resources: [
      { title: "CatBoost 文档", url: "https://catboost.ai/docs/", type: "link" }
    ]
  },
  {
    id: "extratrees-classification",
    name: "ExtraTrees分类 (Extra Trees Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过更随机的特征和切分点选择构建多棵树，常比随机森林具有更快训练速度与更低方差。",
    principles: `1. 构建多棵极端随机树。
2. 每个节点随机选择候选特征与随机切分点。
3. 综合多棵树投票给出分类结果。
4. 以更强随机性降低方差。`,
    scenarios: [
      "高维表格分类",
      "希望快速训练树集成模型的场景"
    ],
    limitations: [
      "单棵树偏差较大",
      "整体可解释性一般"
    ],
    caseStudy: {
      title: "设备故障分类",
      description: "根据多个传感器指标判断设备是否故障。",
      solution: `1. 用多棵极端随机树建模；
2. 集成投票输出故障类别；
3. 对重要特征做辅助解释。`
    },
    code: {
      python: `from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB ExtraTrees 通常可借助 Python 接口实现`
    },
    resources: [
      { title: "ExtraTreesClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html", type: "link" }
    ]
  },
  {
    id: "knn-classification",
    name: "K近邻(KNN)分类 (K-Nearest Neighbors Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "依据待分类样本附近的 K 个近邻标签进行投票，是最基础的实例学习分类方法之一。",
    principles: `1. 选择距离度量方式，如欧氏距离。
2. 对新样本寻找最近的 K 个训练样本。
3. 根据近邻类别多数投票输出结果。
4. K 值控制平滑程度和偏差方差权衡。`,
    scenarios: [
      "小中规模分类任务",
      "需要简单直接的非参数分类方法"
    ],
    limitations: [
      "预测时计算量大",
      "对特征缩放和噪声点敏感"
    ],
    caseStudy: {
      title: "客户等级分类",
      description: "根据活跃度和消费金额将客户分为不同等级。",
      solution: `1. 标准化特征；
2. 选择 K 值；
3. 基于邻近样本投票完成分类。`
    },
    code: {
      python: `from sklearn.neighbors import KNeighborsClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB KNN 分类
Mdl = fitcknn(X, y, 'NumNeighbors', 3);`
    },
    resources: [
      { title: "KNeighborsClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html", type: "link" }
    ]
  },
  {
    id: "bp-neural-network-classification",
    name: "bp神经网络分类 (BP Neural Network Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "利用反向传播算法训练多层前馈神经网络，适合处理复杂非线性分类边界。",
    principles: `1. 构建输入层、隐藏层和输出层。
2. 前向传播得到输出结果。
3. 通过损失函数反向传播更新权重。
4. 多轮迭代后完成分类模型训练。`,
    scenarios: [
      "非线性模式明显的分类问题",
      "需要比传统浅层模型更强表达能力的场景"
    ],
    limitations: [
      "训练对参数和样本规模较敏感",
      "可解释性通常较弱"
    ],
    caseStudy: {
      title: "图像特征分类",
      description: "根据提取后的特征向量判断样本所属类别。",
      solution: `1. 建立多层感知机结构；
2. 使用 BP 算法训练；
3. 输出分类标签。`
    },
    code: {
      python: `from sklearn.neural_network import MLPClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=1000, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB BP 神经网络分类
net = patternnet(10);
net = train(net, X', dummyvar(y)');`
    },
    resources: [
      { title: "MLPClassifier 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html", type: "link" }
    ]
  },
  {
    id: "svm-classification",
    name: "支持向量机(SVM)分类 (Support Vector Machine Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "通过寻找最大间隔超平面完成分类，并可借助核函数处理非线性分类问题。",
    principles: `1. 在线性可分情形下寻找最大间隔分类超平面。
2. 通过支持向量决定决策边界位置。
3. 软间隔参数 C 控制分类错误与间隔大小的平衡。
4. 核函数可将数据映射到高维空间处理非线性分类。`,
    scenarios: [
      "中小样本、高维分类问题",
      "文本分类、图像识别和生物信息分类任务"
    ],
    limitations: [
      "大规模数据上训练成本较高",
      "核函数和参数选择对结果影响大"
    ],
    caseStudy: {
      title: "文本情感分类",
      description: "根据文本特征向量判断评论是正向还是负向。",
      solution: `1. 构造向量化特征；
2. 选择核函数训练 SVM；
3. 输出情感类别。`
    },
    code: {
      python: `from sklearn.svm import SVC
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = SVC(kernel='rbf', probability=True, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB SVM 分类
Mdl = fitcsvm(X, y, 'KernelFunction', 'rbf');`
    },
    resources: [
      { title: "SVC 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html", type: "link" }
    ]
  },
  {
    id: "xgboost-classification",
    name: "XGBoost分类 (XGBoost Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "高性能梯度提升树框架，具有正则化、并行化和缺失值处理能力，在表格分类任务中表现出色。",
    principles: `1. 基于 Boosting 逐轮拟合新树。
2. 在目标函数中加入正则项控制模型复杂度。
3. 使用二阶梯度信息提升优化效率。
4. 支持缺失值自动处理与并行训练。`,
    scenarios: [
      "各类结构化数据分类任务",
      "比赛和工业场景中的高精度表格分类"
    ],
    limitations: [
      "参数较多，调参成本较高",
      "依赖额外库环境"
    ],
    caseStudy: {
      title: "欺诈交易分类",
      description: "根据交易行为特征识别交易是否存在欺诈风险。",
      solution: `1. 构建结构化特征；
2. 训练 XGBoost 分类器；
3. 输出风险类别与概率。`
    },
    code: {
      python: `from xgboost import XGBClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42)
clf.fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB XGBoost 通常需借助 Python 接口或外部工具`
    },
    resources: [
      { title: "XGBoost 文档", url: "https://xgboost.readthedocs.io/", type: "link" }
    ]
  },
  {
    id: "lightgbm-classification",
    name: "LightGBM分类 (LightGBM Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "基于直方图算法和叶子优先生长策略的高效梯度提升框架，适合大规模分类任务。",
    principles: `1. 使用直方图算法加速特征分桶与分裂。
2. 采用叶子优先生长策略提升损失下降效率。
3. 支持类别特征和大规模数据训练。
4. 在保持精度的同时兼顾训练速度。`,
    scenarios: [
      "大规模表格数据分类",
      "对训练效率和预测精度要求都较高的任务"
    ],
    limitations: [
      "叶子优先生长可能在小数据集上过拟合",
      "需额外安装库并调参"
    ],
    caseStudy: {
      title: "金融客户分类评分",
      description: "基于大量交易与行为特征对客户风险等级进行分类。",
      solution: `1. 构建大规模特征矩阵；
2. 使用 LightGBM 分类器训练；
3. 输出类别与概率结果。`
    },
    code: {
      python: `from lightgbm import LGBMClassifier
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = LGBMClassifier(n_estimators=50, random_state=42).fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB LightGBM 通常需借助 Python 接口或外部工具`
    },
    resources: [
      { title: "LightGBM 文档", url: "https://lightgbm.readthedocs.io/", type: "link" }
    ]
  },
  {
    id: "naive-bayes-classification",
    name: "朴素贝叶斯分类 (Naive Bayes Classification)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "基于贝叶斯定理和条件独立假设进行分类，是高效且经典的概率分类模型。",
    principles: `1. 根据贝叶斯公式计算后验概率。
2. 假设给定类别后各特征条件独立。
3. 比较不同类别的后验概率大小。
4. 选择概率最大的类别作为输出。`,
    scenarios: [
      "文本分类、垃圾邮件识别、医学初筛等任务",
      "需要快速训练和推理的分类场景"
    ],
    limitations: [
      "条件独立假设通常较强",
      "特征相关性强时效果可能下降"
    ],
    caseStudy: {
      title: "邮件垃圾分类",
      description: "根据词频和文本特征判断邮件是否为垃圾邮件。",
      solution: `1. 构造词频特征；
2. 训练朴素贝叶斯分类器；
3. 输出邮件类别。`
    },
    code: {
      python: `from sklearn.naive_bayes import GaussianNB
import numpy as np
X = np.array([[1,10],[2,8],[8,2],[9,1]])
y = np.array([0,0,1,1])
clf = GaussianNB().fit(X, y)
print(clf.predict([[3,6]]))`,
      matlab: `% MATLAB 朴素贝叶斯分类
Mdl = fitcnb(X, y);`
    },
    resources: [
      { title: "GaussianNB 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html", type: "link" }
    ]
  },
  {
    id: "logistic-regression-gradient-descent",
    name: "逻辑回归（梯度下降法） (Logistic Regression with Gradient Descent)",
    category: "machine-learning",
    categoryName: "🤖 机器学习分类",
    summary: "使用梯度下降优化逻辑回归参数的分类方法，适合从算法角度理解二分类概率模型训练过程。",
    principles: `1. 使用 Sigmoid 函数将线性组合映射到概率空间。
2. 构造对数损失函数衡量预测误差。
3. 通过梯度下降不断更新参数以最小化损失。
4. 最终根据概率阈值输出分类结果。`,
    scenarios: [
      "二分类建模教学与算法原理展示",
      "需要解释概率输出和参数更新过程的任务"
    ],
    limitations: [
      "默认线性决策边界，对复杂非线性分类能力有限",
      "学习率选择不当会影响收敛速度与稳定性"
    ],
    caseStudy: {
      title: "用户是否购买分类",
      description: "根据浏览时长、点击次数等特征预测用户是否购买。",
      solution: `1. 构造二分类标签；
2. 用梯度下降优化逻辑回归参数；
3. 根据预测概率输出类别。`
    },
    code: {
      python: `import numpy as np
X = np.array([[1,1,10],[1,2,8],[1,8,2],[1,9,1]], dtype=float)
y = np.array([0,0,1,1], dtype=float)
w = np.zeros(X.shape[1])
lr = 0.1
for _ in range(1000):
    z = X @ w
    p = 1 / (1 + np.exp(-z))
    grad = X.T @ (p - y) / len(y)
    w -= lr * grad
pred = 1 / (1 + np.exp(-(np.array([1,3,6]) @ w)))
print(pred > 0.5)`,
      matlab: `% MATLAB 逻辑回归梯度下降示意
X = [ones(4,1), [1 10; 2 8; 8 2; 9 1]];
y = [0;0;1;1];
w = zeros(size(X,2),1);
alpha = 0.1;
for i = 1:1000
    p = 1 ./ (1 + exp(-X*w));
    grad = X' * (p - y) / length(y);
    w = w - alpha * grad;
end`
    },
    resources: [
      { title: "Logistic regression overview", url: "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression", type: "link" }
    ]
  },
  {
    id: "decision-tree-regression",
    name: "决策树回归 (Decision Tree Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "通过递归划分特征空间并在叶节点输出连续值，是结构直观、可解释性较强的非线性回归方法。",
    principles: `1. 选择最优特征和切分点划分样本空间。
2. 以均方误差等指标衡量划分质量。
3. 递归生成回归树直到满足停止条件。
4. 在叶节点输出样本均值或局部拟合结果。`,
    scenarios: [
      "房价、销量、能耗等连续变量预测",
      "希望捕捉非线性关系且保持一定可解释性的场景"
    ],
    limitations: [
      "容易过拟合，对数据扰动敏感",
      "单棵树预测稳定性通常一般"
    ],
    caseStudy: {
      title: "房价回归预测",
      description: "根据面积、楼层和装修情况预测房屋价格。",
      solution: `1. 构建连续型标签数据；
2. 训练决策树回归模型；
3. 根据叶节点输出预测价格。`
    },
    code: {
      python: `from sklearn.tree import DecisionTreeRegressor
import numpy as np
X = np.array([[50,3],[60,5],[80,8],[100,10]])
y = np.array([120,150,210,260])
reg = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
print(reg.predict([[70,6]]))`,
      matlab: `% MATLAB 决策树回归
Mdl = fitrtree(X, y);`
    },
    resources: [
      { title: "DecisionTreeRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html", type: "link" }
    ]
  },
  {
    id: "random-forest-regression",
    name: "随机森林回归 (Random Forest Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "通过多棵回归树的平均输出进行预测，能有效提升稳定性并降低单树过拟合风险。",
    principles: `1. 对样本做 Bootstrap 重采样形成多组训练集。
2. 每棵树在节点处随机选择部分特征进行划分。
3. 构建多棵回归树。
4. 对所有树的预测结果取平均作为最终输出。`,
    scenarios: [
      "连续型表格数据预测",
      "需要鲁棒、稳定回归结果的业务任务"
    ],
    limitations: [
      "模型体积较大",
      "解释性弱于单棵树"
    ],
    caseStudy: {
      title: "门店销量预测",
      description: "根据促销、天气和客流等因素预测门店销量。",
      solution: `1. 构造训练样本；
2. 训练随机森林回归器；
3. 对多棵树输出求平均得到预测值。`
    },
    code: {
      python: `from sklearn.ensemble import RandomForestRegressor
import numpy as np
X = np.array([[50,3],[60,5],[80,8],[100,10]])
y = np.array([120,150,210,260])
reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
print(reg.predict([[70,6]]))`,
      matlab: `% MATLAB 随机森林回归可使用 TreeBagger
Mdl = TreeBagger(100, X, y, 'Method', 'regression');`
    },
    resources: [
      { title: "RandomForestRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html", type: "link" }
    ]
  },
  {
    id: "adaboost-regression",
    name: "adaboost回归 (AdaBoost Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "通过迭代组合多个弱回归器并动态调整样本权重，逐步提升整体回归拟合能力。",
    principles: `1. 初始化样本权重。
2. 训练弱回归器并计算预测误差。
3. 提高误差较大样本的权重。
4. 加权组合多个弱回归器输出最终预测。`,
    scenarios: [
      "中小规模回归任务",
      "希望利用简单基学习器提升回归性能的场景"
    ],
    limitations: [
      "对异常点较敏感",
      "基学习器和损失形式会影响效果"
    ],
    caseStudy: {
      title: "二手车价格预测",
      description: "根据车龄、里程和排量预测二手车价格。",
      solution: `1. 使用浅层树作为弱回归器；
2. 迭代更新样本权重；
3. 集成多个弱学习器得到最终预测。`
    },
    code: {
      python: `from sklearn.ensemble import AdaBoostRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = AdaBoostRegressor(n_estimators=50, random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB AdaBoost 回归可通过 fitrensemble 实现
Mdl = fitrensemble(X, y, 'Method', 'LSBoost');`
    },
    resources: [
      { title: "AdaBoostRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html", type: "link" }
    ]
  },
  {
    id: "gbdt-regression",
    name: "梯度提升树（GBDT）回归 (Gradient Boosting Decision Tree Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "通过逐轮拟合前一轮残差或损失梯度来增强回归模型，是经典的非线性集成回归方法。",
    principles: `1. 从初始常数模型开始。
2. 每轮拟合当前损失函数的负梯度或残差。
3. 将新树乘以学习率后叠加到已有模型上。
4. 多轮迭代后形成强回归器。`,
    scenarios: [
      "房价、销量、风险分数等连续变量预测",
      "需要较强非线性表达能力的表格回归任务"
    ],
    limitations: [
      "训练速度相对较慢",
      "参数较多且易过拟合"
    ],
    caseStudy: {
      title: "电力负荷预测",
      description: "根据温度、时段和历史负荷预测未来电力负荷。",
      solution: `1. 构建回归特征；
2. 使用 GBDT 逐轮拟合残差；
3. 输出连续负荷预测值。`
    },
    code: {
      python: `from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = GradientBoostingRegressor(random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB GBDT 回归
Mdl = fitrensemble(X, y, 'Method', 'LSBoost');`
    },
    resources: [
      { title: "GradientBoostingRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html", type: "link" }
    ]
  },
  {
    id: "extratrees-regression",
    name: "ExtraTrees回归 (Extra Trees Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "利用更强随机性的多棵极端随机树完成回归，常在速度和方差控制上具有优势。",
    principles: `1. 构建多棵极端随机回归树。
2. 节点划分时随机选择特征和切分点。
3. 每棵树独立生成回归预测。
4. 对所有树结果取平均作为最终输出。`,
    scenarios: [
      "高维表格数据回归",
      "希望快速训练树集成回归模型的场景"
    ],
    limitations: [
      "单棵树偏差较大",
      "整体解释性一般"
    ],
    caseStudy: {
      title: "工业指标回归预测",
      description: "根据多维传感器数据预测产线输出指标。",
      solution: `1. 使用极端随机树构建回归集成；
2. 对多树结果取平均；
3. 输出最终预测值。`
    },
    code: {
      python: `from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = ExtraTreesRegressor(n_estimators=100, random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB ExtraTrees 回归通常可借助 Python 接口实现`
    },
    resources: [
      { title: "ExtraTreesRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html", type: "link" }
    ]
  },
  {
    id: "catboost-regression",
    name: "CatBoost回归 (CatBoost Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "适合含类别特征的表格回归任务，利用有序提升策略在稳定性和精度之间取得良好平衡。",
    principles: `1. 基于梯度提升树框架构建回归模型。
2. 原生支持类别特征处理。
3. 通过有序提升减少目标泄漏风险。
4. 输出连续数值预测结果。`,
    scenarios: [
      "含类别变量较多的价格、评分、销量预测",
      "表格回归任务中的高质量建模"
    ],
    limitations: [
      "依赖额外库",
      "在大规模任务中需要关注训练资源与调参成本"
    ],
    caseStudy: {
      title: "商品销量回归预测",
      description: "根据品类、地区、促销方式等特征预测商品销量。",
      solution: `1. 保留原始类别特征；
2. 训练 CatBoost 回归器；
3. 输出销量预测值。`
    },
    code: {
      python: `from catboost import CatBoostRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = CatBoostRegressor(verbose=0, random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB CatBoost 回归通常需调用 Python 接口或外部工具`
    },
    resources: [
      { title: "CatBoost 文档", url: "https://catboost.ai/docs/", type: "link" }
    ]
  },
  {
    id: "knn-regression",
    name: "K近邻(KNN)回归 (K-Nearest Neighbors Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "通过待预测样本附近 K 个近邻样本的目标值平均或加权平均进行回归预测，是经典的非参数回归方法。",
    principles: `1. 选择距离度量方式。
2. 对新样本寻找最近的 K 个邻居。
3. 对这些邻居的目标值做平均或距离加权平均。
4. 输出连续预测结果。`,
    scenarios: [
      "小中规模非线性回归任务",
      "希望使用简单直观的实例学习回归方法"
    ],
    limitations: [
      "预测时计算量大",
      "对特征缩放和噪声点较敏感"
    ],
    caseStudy: {
      title: "租金回归预测",
      description: "根据面积、位置和楼层预测房屋租金。",
      solution: `1. 标准化特征；
2. 选择合适的 K 值；
3. 通过邻近样本平均得到预测值。`
    },
    code: {
      python: `from sklearn.neighbors import KNeighborsRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = KNeighborsRegressor(n_neighbors=2).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB KNN 回归
Mdl = fitrknn(X, y, 'NumNeighbors', 2);`
    },
    resources: [
      { title: "KNeighborsRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html", type: "link" }
    ]
  },
  {
    id: "bp-neural-network-regression",
    name: "bp神经网络回归 (BP Neural Network Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "利用反向传播算法训练多层前馈神经网络，适合逼近复杂非线性连续映射关系。",
    principles: `1. 构建输入层、隐藏层和输出层。
2. 前向传播得到回归输出。
3. 使用损失函数反向传播更新权重。
4. 多轮迭代后学习复杂连续函数关系。`,
    scenarios: [
      "复杂非线性连续值预测",
      "传统浅层模型难以拟合的回归任务"
    ],
    limitations: [
      "训练对样本规模和参数设置较敏感",
      "解释性较弱"
    ],
    caseStudy: {
      title: "设备寿命回归预测",
      description: "根据运行状态特征预测设备剩余寿命。",
      solution: `1. 设计多层神经网络结构；
2. 用 BP 算法训练模型；
3. 输出寿命预测值。`
    },
    code: {
      python: `from sklearn.neural_network import MLPRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = MLPRegressor(hidden_layer_sizes=(16,8), max_iter=2000, random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB BP 神经网络回归
net = fitnet(10);
net = train(net, X', y');`
    },
    resources: [
      { title: "MLPRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html", type: "link" }
    ]
  },
  {
    id: "svr-regression",
    name: "支持向量机(SVR)回归 (Support Vector Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "在保留间隔最大化思想的基础上，通过 ε-不敏感损失实现连续值预测，是经典的核方法回归模型。",
    principles: `1. 使用 ε-不敏感损失忽略小范围预测误差。
2. 在高维特征空间中寻找平滑回归函数。
3. 支持向量决定最终回归边界形状。
4. 核函数可处理复杂非线性关系。`,
    scenarios: [
      "小样本、高维回归问题",
      "需要核方法建模非线性连续关系的任务"
    ],
    limitations: [
      "大规模样本训练成本较高",
      "核函数和超参数选择对效果影响明显"
    ],
    caseStudy: {
      title: "空气质量指标回归预测",
      description: "根据气象和排放特征预测空气质量指数。",
      solution: `1. 构建输入特征；
2. 选择核函数训练 SVR；
3. 输出连续预测值。`
    },
    code: {
      python: `from sklearn.svm import SVR
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = SVR(kernel='rbf').fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB SVR 回归
Mdl = fitrsvm(X, y, 'KernelFunction', 'rbf');`
    },
    resources: [
      { title: "SVR 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html", type: "link" }
    ]
  },
  {
    id: "xgboost-regression",
    name: "XGBoost回归 (XGBoost Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "高性能梯度提升回归框架，具备正则化、并行化和缺失值处理能力，在表格回归任务中非常常用。",
    principles: `1. 基于 Boosting 逐轮添加新树。
2. 使用目标函数与正则项控制模型复杂度。
3. 利用一阶和二阶梯度提升优化效率。
4. 输出连续型预测结果。`,
    scenarios: [
      "结构化数据的价格、评分、时长等回归问题",
      "对精度要求高的表格回归任务"
    ],
    limitations: [
      "参数较多，调参成本高",
      "依赖额外库环境"
    ],
    caseStudy: {
      title: "订单时长回归预测",
      description: "根据业务特征预测订单处理时长。",
      solution: `1. 构建结构化特征；
2. 训练 XGBoost 回归器；
3. 输出时长预测值。`
    },
    code: {
      python: `from xgboost import XGBRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = XGBRegressor(n_estimators=50, max_depth=3, eval_metric='rmse', random_state=42)
reg.fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB XGBoost 回归通常需借助 Python 接口或外部工具`
    },
    resources: [
      { title: "XGBoost 文档", url: "https://xgboost.readthedocs.io/", type: "link" }
    ]
  },
  {
    id: "lightgbm-regression",
    name: "LightGBM回归 (LightGBM Regression)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "基于直方图和叶子优先生长策略的高效梯度提升回归框架，适合大规模连续值预测任务。",
    principles: `1. 使用直方图算法加速特征分桶与分裂。
2. 采用叶子优先生长方式提高损失下降效率。
3. 支持类别特征与大规模数据训练。
4. 输出高效率的连续值预测结果。`,
    scenarios: [
      "大规模表格回归",
      "需要兼顾训练效率和预测精度的任务"
    ],
    limitations: [
      "在小数据集上可能过拟合",
      "需要额外安装库并调参"
    ],
    caseStudy: {
      title: "用户消费金额回归预测",
      description: "根据用户画像和行为特征预测未来消费金额。",
      solution: `1. 构建高维表格特征；
2. 使用 LightGBM 回归器训练；
3. 输出消费金额预测值。`
    },
    code: {
      python: `from lightgbm import LGBMRegressor
import numpy as np
X = np.array([[2,3],[3,5],[5,8],[8,10]])
y = np.array([12,15,21,30])
reg = LGBMRegressor(n_estimators=50, random_state=42).fit(X, y)
print(reg.predict([[4,6]]))`,
      matlab: `% MATLAB LightGBM 回归通常需借助 Python 接口或外部工具`
    },
    resources: [
      { title: "LightGBM 文档", url: "https://lightgbm.readthedocs.io/", type: "link" }
    ]
  },
  {
    id: "linear-regression-gradient-descent",
    name: "线性回归（梯度下降法） (Linear Regression with Gradient Descent)",
    category: "machine-learning-regression",
    categoryName: "🧠 机器学习回归",
    summary: "使用梯度下降优化线性回归参数的回归方法，适合从算法视角理解连续变量预测模型的训练过程。",
    principles: `1. 建立线性模型 y = X\\beta。
2. 定义均方误差损失函数衡量预测误差。
3. 通过梯度下降迭代更新参数以最小化损失。
4. 训练完成后输出连续数值预测结果。`,
    scenarios: [
      "回归建模教学与算法推导展示",
      "需要理解参数更新机制的连续变量预测任务"
    ],
    limitations: [
      "默认只能拟合线性关系",
      "学习率设置不当会导致收敛缓慢或震荡"
    ],
    caseStudy: {
      title: "房价回归预测",
      description: "根据面积、楼层和房龄等特征预测房价。",
      solution: `1. 构造连续型目标变量；
2. 使用梯度下降优化线性回归参数；
3. 输出最终预测价格。`
    },
    code: {
      python: `import numpy as np
X = np.array([[1,50,3],[1,60,5],[1,80,8],[1,100,10]], dtype=float)
y = np.array([120,150,210,260], dtype=float)
w = np.zeros(X.shape[1])
lr = 0.0001
for _ in range(5000):
    pred = X @ w
    grad = X.T @ (pred - y) / len(y)
    w -= lr * grad
print(np.array([1,70,6]) @ w)`,
      matlab: `% MATLAB 线性回归梯度下降
X = [ones(4,1), [50 3; 60 5; 80 8; 100 10]];
y = [120;150;210;260];
w = zeros(size(X,2),1);
alpha = 1e-4;
for i = 1:5000
    grad = X' * (X*w - y) / length(y);
    w = w - alpha * grad;
end`
    },
    resources: [
      { title: "Linear models overview", url: "https://scikit-learn.org/stable/modules/linear_model.html", type: "link" }
    ]
  },

  // ==================== 8. 统计分析 (stats-analysis) ====================
  {
    id: "principal-component-analysis",
    name: "主成分分析(PCA) (Principal Component Analysis)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "通过线性变换将多个相关变量压缩为少数互不相关的主成分，是最经典的降维与综合信息提取方法。",
    principles: `1. 对原始变量标准化后构造协方差矩阵或相关系数矩阵。
2. 求解特征值与特征向量，按解释方差大小排序主成分。
3. 前几个主成分保留大部分原始信息，实现降维。
4. 主成分彼此正交，常用于综合得分、可视化和后续建模。`,
    scenarios: [
      "高维指标降维与综合评价",
      "解决变量相关性强时的信息压缩问题"
    ],
    limitations: [
      "主成分通常缺乏直观业务解释",
      "只能捕捉线性结构，对非线性结构无能为力"
    ],
    caseStudy: {
      title: "城市综合发展指标降维",
      description: "将多个高度相关的经济、教育和环境指标压缩为少数综合因子。",
      solution: `1. 对原始指标做标准化；
2. 提取累计贡献率较高的前几个主成分；
3. 用主成分得分进行综合排序。`
    },
    code: {
      python: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
X = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
Xz = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xz)
print(pca.explained_variance_ratio_)`,
      matlab: `% MATLAB PCA
X = [1 2 3; 2 3 4; 3 4 5; 4 5 6];
[coeff, score, latent, ~, explained] = pca(X);`
    },
    resources: [
      { title: "PCA 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html", type: "link" }
    ]
  },
  {
    id: "canonical-correlation-analysis",
    name: "典型相关分析 (Canonical Correlation Analysis)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于研究两组变量之间整体线性相关结构，通过寻找两组变量的线性组合使其相关性最大。",
    principles: `1. 将变量分成两组 X 与 Y。
2. 分别寻找线性组合 U = a^T X 和 V = b^T Y。
3. 使 corr(U, V) 最大，得到第一对典型变量。
4. 后续典型变量在与前面对正交约束下继续提取。`,
    scenarios: [
      "研究一组投入指标与一组产出指标之间的整体关联",
      "心理、教育、医学和经济学中的多变量关联分析"
    ],
    limitations: [
      "样本量不足或变量组内部共线性强时结果不稳定",
      "解释难度高于普通相关分析"
    ],
    caseStudy: {
      title: "学习行为与成绩结构关系分析",
      description: "分析出勤、作业、阅读等行为变量与数学、英语、物理成绩变量之间的整体关联。",
      solution: `1. 将行为和成绩分别作为两组变量；
2. 提取典型变量对；
3. 解释最强关联结构。`
    },
    code: {
      python: `from sklearn.cross_decomposition import CCA
import numpy as np
X = np.random.randn(20, 3)
Y = np.random.randn(20, 2)
cca = CCA(n_components=1).fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
print(np.corrcoef(X_c.T, Y_c.T)[0, 1])`,
      matlab: `% MATLAB 典型相关分析
[A,B,r,U,V,stats] = canoncorr(X, Y);`
    },
    resources: [
      { title: "CCA 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html", type: "link" }
    ]
  },
  {
    id: "poisson-distribution-test",
    name: "泊松分布检验 (Poisson Distribution Test)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于检验计数型数据是否服从泊松分布，常见于事件次数、缺陷数、到达数等离散计数场景。",
    principles: `1. 泊松分布适合描述单位时间或单位区域内随机事件的发生次数。
2. 其参数为 \\lambda，且理论上均值等于方差。
3. 可通过拟合优度检验比较观测频数与泊松理论频数。
4. 也可结合离散度指标初步判断是否偏离泊松假设。`,
    scenarios: [
      "交通事故次数、电话呼入量、设备故障数建模前分布检验",
      "判断计数数据是否适合使用泊松回归或相关计数模型"
    ],
    limitations: [
      "若存在过度离散或零膨胀，泊松假设常常失效",
      "分箱方式会影响拟合优度检验结果"
    ],
    caseStudy: {
      title: "呼叫中心来电次数分布检验",
      description: "希望判断每小时来电次数是否近似服从泊松分布。",
      solution: `1. 估计样本均值作为 \\lambda；
2. 计算各次数的理论频数；
3. 使用拟合优度检验比较观测值与理论值。`
    },
    code: {
      python: `import numpy as np
from scipy.stats import poisson, chisquare
obs = np.array([18, 25, 20, 12, 5])
lam = 1.8
expected = poisson.pmf(np.arange(len(obs)), lam) * obs.sum()
expected *= obs.sum() / expected.sum()
print(chisquare(obs, expected))`,
      matlab: `% MATLAB 泊松分布检验
obs = [18 25 20 12 5];
lambda = 1.8;
expFreq = poisspdf(0:length(obs)-1, lambda) * sum(obs);`
    },
    resources: [
      { title: "Poisson distribution", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html", type: "link" }
    ]
  },
  {
    id: "runs-test",
    name: "游程检验 (Runs Test)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于检验序列的随机性，常见于时间序列预处理、质量控制和二元序列独立性分析。",
    principles: `1. 将观测序列按某种规则转化为二元序列或高低序列。
2. 连续相同符号构成一个游程（run）。
3. 若序列随机，则游程数不会显著偏离理论期望。
4. 通过 Z 统计量或精确检验判断随机性是否成立。`,
    scenarios: [
      "检验时间序列残差是否随机",
      "检验生产质量序列是否存在趋势或聚集现象"
    ],
    limitations: [
      "主要反映顺序随机性，不能替代完整时间序列建模",
      "结果依赖于二元化或阈值划分方式"
    ],
    caseStudy: {
      title: "残差序列随机性检验",
      description: "在回归建模后判断残差是否仍存在明显结构性模式。",
      solution: `1. 将残差按正负划分；
2. 计算游程数；
3. 判断残差是否近似随机。`
    },
    code: {
      python: `from statsmodels.sandbox.stats.runs import runstest_1samp
x = [1.2, -0.5, 0.8, -1.1, 0.6, -0.3, 0.2]
print(runstest_1samp(x))`,
      matlab: `% MATLAB 游程检验通常需自定义实现或使用统计工具箱相关函数`
    },
    resources: [
      { title: "Runs test in statsmodels", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "stepwise-regression",
    name: "逐步回归 (Stepwise Regression)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "通过逐步加入或剔除变量来构建回归模型，常用于变量筛选和简化模型结构。",
    principles: `1. 根据显著性、AIC、BIC 或调整后 R^2 等标准筛选变量。
2. 常见策略包括前向选择、后向剔除和双向逐步法。
3. 每一步都重新评估变量进入或移出后的模型表现。
4. 结果是一个相对精简的回归模型。`,
    scenarios: [
      "自变量较多时的初步变量筛选",
      "希望平衡预测效果与模型简洁性的回归建模"
    ],
    limitations: [
      "容易受样本扰动影响，模型稳定性有限",
      "作为自动筛选方法，可能忽略理论背景"
    ],
    caseStudy: {
      title: "区域经济指标筛选建模",
      description: "从十余个经济指标中筛选出对 GDP 增长最关键的变量。",
      solution: `1. 设定逐步筛选标准；
2. 自动比较变量进出后的模型表现；
3. 保留显著且解释力较强的变量。`
    },
    code: {
      python: `print('Python 可结合 statsmodels 手动实现前向/后向逐步回归')`,
      matlab: `% MATLAB 逐步回归
mdl = stepwiselm(X, y);`
    },
    resources: [
      { title: "stepwiselm 文档", url: "https://www.mathworks.com/help/stats/stepwiselm.html", type: "link" }
    ]
  },
  {
    id: "linear-discriminant-analysis",
    name: "线性判别 (Linear Discriminant Analysis)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "通过寻找最大化类间差异、最小化类内差异的投影方向实现分类与判别，是经典监督判别方法。",
    principles: `1. 计算类内散度矩阵与类间散度矩阵。
2. 寻找投影方向使类间散度与类内散度之比最大。
3. 可用于分类，也可用于有监督降维。
4. 假设各类协方差结构相同且数据近似正态。`,
    scenarios: [
      "已知类别标签下的分类和判别分析",
      "医学诊断、用户分类、样本识别等监督学习问题"
    ],
    limitations: [
      "对协方差同质性假设较敏感",
      "非线性类别边界下效果有限"
    ],
    caseStudy: {
      title: "客户类别判别分析",
      description: "基于消费频率、客单价和活跃度指标区分高价值与普通用户。",
      solution: `1. 计算类间与类内散度；
2. 建立线性判别函数；
3. 根据投影结果完成分类。`
    },
    code: {
      python: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
X = np.array([[1,2],[2,3],[8,8],[9,9]])
y = np.array([0,0,1,1])
lda = LinearDiscriminantAnalysis().fit(X, y)
print(lda.predict([[3,4]]))`,
      matlab: `% MATLAB 线性判别
Mdl = fitcdiscr(X, y);`
    },
    resources: [
      { title: "LDA 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html", type: "link" }
    ]
  },
  {
    id: "range-analysis",
    name: "极差分析 (Range Analysis)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "常用于正交试验中比较各因素不同水平对结果的影响大小，通过极差大小快速识别主次因素。",
    principles: `1. 分别计算每个因素在各水平下的平均响应值。
2. 对同一因素求最大均值与最小均值之差，得到极差 R。
3. 极差越大，说明该因素对结果影响越显著。
4. 常与正交试验设计结合使用。`,
    scenarios: [
      "正交试验结果的快速主效应分析",
      "多因素多水平实验的初步影响排序"
    ],
    limitations: [
      "只能做较粗略的影响排序，不能替代严格显著性检验",
      "对交互作用刻画能力较弱"
    ],
    caseStudy: {
      title: "工艺参数正交试验极差分析",
      description: "在温度、时间和浓度多因素试验后，快速识别最重要控制因素。",
      solution: `1. 计算各因素各水平下均值；
2. 得到各因素极差；
3. 按极差排序判断主次因素。`
    },
    code: {
      python: `import pandas as pd
print('极差分析通常根据正交试验表手工计算各水平平均值与极差')`,
      matlab: `% MATLAB 极差分析可按正交试验表手动汇总计算`
    },
    resources: [
      { title: "正交试验极差分析介绍", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "curve-fitting-toolbox",
    name: "拟合工具箱 (Curve Fitting Toolbox)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于快速完成线性、非线性、多项式和自定义函数拟合，是工程建模与经验公式提取中的常用工具。",
    principles: `1. 根据散点数据选择拟合函数形式，如线性、多项式、指数或自定义模型。
2. 通过最小二乘等方法估计参数。
3. 用 R^2、残差图和置信区间评估拟合质量。
4. 可用于经验公式构建和预测。`,
    scenarios: [
      "实验数据经验公式拟合",
      "非线性关系建模与曲线近似"
    ],
    limitations: [
      "拟合函数选择带有经验性",
      "过高阶模型容易过拟合"
    ],
    caseStudy: {
      title: "材料应力-应变曲线拟合",
      description: "根据实验测得的应力应变散点提取经验公式。",
      solution: `1. 选择合适拟合模型；
2. 估计参数并画出拟合曲线；
3. 评估拟合优度与残差。`
    },
    code: {
      python: `import numpy as np
from scipy.optimize import curve_fit

def f(x, a, b):
    return a * np.exp(b * x)
x = np.array([1,2,3,4])
y = np.array([2.2,2.8,3.9,5.1])
params, _ = curve_fit(f, x, y)
print(params)`,
      matlab: `% MATLAB 拟合工具箱常用 fit 函数
x = [1 2 3 4]'; y = [2.2 2.8 3.9 5.1]';
fitobj = fit(x, y, 'exp1');`
    },
    resources: [
      { title: "curve_fit 文档", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html", type: "link" }
    ]
  },
  {
    id: "generalized-linear-model",
    name: "广义线性模型 (Generalized Linear Model, GLM)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "将线性模型推广到非正态响应变量场景，通过分布族与链接函数统一处理多种响应类型。",
    principles: `1. GLM 由随机成分、系统成分和链接函数组成。
2. 响应变量可服从正态、二项、泊松、Gamma 等分布族。
3. 链接函数将 E(Y|X) 与线性预测子 X\\beta 关联起来。
4. 统一覆盖线性回归、逻辑回归、泊松回归等模型。`,
    scenarios: [
      "二分类、计数型和偏态连续响应变量建模",
      "需要比普通线性回归更灵活的统计建模场景"
    ],
    limitations: [
      "模型指定错误会导致解释偏差",
      "链接函数和分布族选择需要经验与诊断支持"
    ],
    caseStudy: {
      title: "医院就诊次数建模",
      description: "响应变量为计数型就诊次数，不适合用普通线性回归。",
      solution: `1. 选择泊松或其他合适分布族；
2. 指定链接函数建立 GLM；
3. 评估系数和预测效果。`
    },
    code: {
      python: `import statsmodels.api as sm
import numpy as np
X = sm.add_constant(np.array([[1],[2],[3],[4]], dtype=float))
y = np.array([1,2,2,4])
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(model.params)`,
      matlab: `% MATLAB GLM
b = glmfit(X, y, 'poisson');`
    },
    resources: [
      { title: "Statsmodels GLM", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "generalized-estimating-equations",
    name: "广义估计方程 (Generalized Estimating Equations, GEE)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于处理重复测量或聚类相关数据，在存在组内相关性的情况下估计总体平均效应。",
    principles: `1. GEE 是对 GLM 的扩展，用于相关样本数据。
2. 通过指定工作相关结构刻画组内观测相关性。
3. 重点估计总体平均效应，而非个体随机效应。
4. 常用于纵向数据与聚类样本分析。`,
    scenarios: [
      "重复测量、面板数据、群组抽样数据分析",
      "医学随访和社会调查中的相关数据建模"
    ],
    limitations: [
      "主要得到边际效应解释，不适合所有个体层随机结构问题",
      "相关结构选取会影响效率"
    ],
    caseStudy: {
      title: "患者多次随访结果建模",
      description: "同一批患者在多个时间点重复测量，需要控制组内相关性。",
      solution: `1. 设定患者 ID 作为聚类单位；
2. 指定工作相关结构；
3. 拟合 GEE 得到总体平均效应。`
    },
    code: {
      python: `import statsmodels.api as sm
import pandas as pd

df = pd.DataFrame({
    'y':[1,0,1,1,0,0],
    'x':[1,2,1,2,1,2],
    'id':[1,1,2,2,3,3]
})
model = sm.GEE.from_formula('y ~ x', groups='id', data=df, family=sm.families.Binomial())
print(model.fit().params)`,
      matlab: `% MATLAB GEE 一般需借助相关统计函数或自定义流程实现`
    },
    resources: [
      { title: "Statsmodels GEE", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "log-linear-model",
    name: "对数线性模型 (Log-Linear Model)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "用于分析多维列联表中分类变量之间的关联结构，是分类频数数据建模的重要工具。",
    principles: `1. 将列联表单元频数的对数表示为分类变量主效应与交互效应之和。
2. 适合分析多个分类变量之间的关联关系。
3. 可检验变量独立性、条件独立性及高阶交互作用。
4. 是列联表分析与离散频数建模的经典框架。`,
    scenarios: [
      "多维分类变量关系分析",
      "问卷、医学分组、社会统计列联表建模"
    ],
    limitations: [
      "维度高时参数较多，解释复杂",
      "小样本稀疏表格下估计会不稳定"
    ],
    caseStudy: {
      title: "多分类问卷列联表建模",
      description: "分析性别、地区、产品偏好三个分类变量之间的交互关系。",
      solution: `1. 构造多维列联表；
2. 建立对数线性模型；
3. 检验主效应和交互项显著性。`
    },
    code: {
      python: `print('对数线性模型可用 Poisson GLM 对列联表频数进行建模')`,
      matlab: `% MATLAB 对数线性模型可借助 glmfit / fitglm 对频数建模`
    },
    resources: [
      { title: "Log-linear models overview", url: "https://online.stat.psu.edu", type: "link" }
    ]
  },
  {
    id: "mixed-effects-model",
    name: "混合模型 (Mixed-Effects Model)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "同时包含固定效应和随机效应，用于处理层级结构、重复测量和组间差异的复杂数据。",
    principles: `1. 固定效应描述总体平均关系，随机效应描述个体或群组特异偏差。
2. 常用于层级数据、嵌套数据和纵向数据分析。
3. 可写作 y = X\\beta + Zb + \\varepsilon，其中 b 为随机效应。
4. 比普通回归更适合处理组内相关和异质性。`,
    scenarios: [
      "学生嵌套在班级、患者嵌套在医院等层级数据建模",
      "重复测量和个体差异显著的统计分析"
    ],
    limitations: [
      "模型设定和参数估计更复杂",
      "随机效应结构选择会显著影响结果"
    ],
    caseStudy: {
      title: "班级层级下的学生成绩建模",
      description: "研究学习时间对成绩的影响，同时考虑不同班级之间的随机差异。",
      solution: `1. 指定学习时间为固定效应；
2. 指定班级为随机效应；
3. 估计总体趋势和班级差异。`
    },
    code: {
      python: `import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({
    'score':[80,82,78,88,90,85],
    'study':[2,3,1,4,5,3],
    'class':['A','A','A','B','B','B']
})
model = smf.mixedlm('score ~ study', df, groups=df['class']).fit()
print(model.params)`,
      matlab: `% MATLAB 混合模型
lme = fitlme(tbl, 'score ~ study + (1|class)');`
    },
    resources: [
      { title: "MixedLM 文档", url: "https://www.statsmodels.org", type: "link" }
    ]
  },

  // ==================== 7. 规划求解 (programming-solvers) ====================
  {
    id: "linear-programming",
    name: "线性规划 (Linear Programming)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "在线性目标函数和线性约束条件下寻求最优解，是运筹优化中最基础也最常用的规划模型之一。",
    principles: `1. 建立线性目标函数与线性约束组。
2. 将可行域表示为凸多面体。
3. 最优解通常出现在可行域顶点处。
4. 可结合单纯形法、内点法等算法进行求解。`,
    scenarios: [
      "生产计划、资源分配、运输调度、成本最小化",
      "存在明确线性约束和线性收益/成本结构的优化问题"
    ],
    limitations: [
      "只能处理线性目标与线性约束",
      "难以直接表达复杂非线性关系"
    ],
    caseStudy: {
      title: "原料分配收益最大化",
      description: "在原料、工时和库存限制下，确定各产品产量组合以最大化总利润。",
      solution: `1. 构建利润最大化目标函数；
2. 写出原料与产能约束；
3. 用线性规划求最优生产方案。`
    },
    code: {
      python: `from scipy.optimize import linprog
c = [-3, -5]
A = [[1, 2], [3, 2]]
b = [8, 12]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
print(res.x, -res.fun)`,
      matlab: `% MATLAB 线性规划
f = [-3; -5];
A = [1 2; 3 2];
b = [8; 12];
[x, fval] = linprog(f, A, b);`
    },
    resources: [
      { title: "SciPy linprog 文档", url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html", type: "link" }
    ]
  },
  {
    id: "interior-point-method",
    name: "内点法 (Interior Point Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "从可行域内部沿势垒路径逼近最优解，适合大规模线性规划、二次规划与部分非线性规划问题。",
    principles: `1. 将边界约束转化为势垒项加入目标函数。
2. 从可行域内部出发进行连续迭代。
3. 随着势垒参数减小逐步逼近边界最优点。
4. 对大规模稀疏问题通常具有较好数值稳定性。`,
    scenarios: [
      "大规模线性规划和二次规划",
      "稀疏约束结构明显的工程优化问题"
    ],
    limitations: [
      "实现与调参相对复杂",
      "需要较好的初始可行性或数值处理策略"
    ],
    caseStudy: {
      title: "大规模物流网络流优化",
      description: "在数千条运输边和容量约束下，优化整体运输成本。",
      solution: `1. 建立网络流线性规划模型；
2. 采用内点法在可行域内部迭代；
3. 输出满足约束的低成本运输方案。`
    },
    code: {
      python: `from scipy.optimize import linprog
c = [-3, -5]
A = [[1, 2], [3, 2]]
b = [8, 12]
res = linprog(c, A_ub=A, b_ub=b, method='interior-point')
print(res.x, -res.fun)`,
      matlab: `% MATLAB 内点法
f = [-3; -5];
A = [1 2; 3 2];
b = [8; 12];
opts = optimoptions('linprog','Algorithm','interior-point');
[x, fval] = linprog(f, A, b, [], [], zeros(2,1), [], opts);`
    },
    resources: [
      { title: "Interior-point overview", url: "https://en.wikipedia.org/wiki/Interior-point_method", type: "link" }
    ]
  },
  {
    id: "simplex-method",
    name: "单纯形法 (Simplex Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过在可行域顶点间迭代移动寻找最优解，是线性规划中最经典的精确求解算法。",
    principles: `1. 将线性规划转化为标准形式。
2. 从一个基础可行解出发。
3. 通过换入换出变量在相邻顶点间移动。
4. 当检验数不再改进时得到最优解。`,
    scenarios: [
      "中小规模线性规划精确求解",
      "教学、建模竞赛和传统运筹优化问题"
    ],
    limitations: [
      "在极端病态问题上可能效率较低",
      "不适合特别大规模稀疏问题"
    ],
    caseStudy: {
      title: "产品产量组合优化",
      description: "根据利润和资源约束选择最优产品组合。",
      solution: `1. 构建标准线性规划模型；
2. 从初始基可行解开始迭代；
3. 输出收益最大的顶点解。`
    },
    code: {
      python: `from scipy.optimize import linprog
c = [-2, -3]
A = [[1, 2], [3, 1]]
b = [8, 12]
res = linprog(c, A_ub=A, b_ub=b, method='simplex')
print(res.x, -res.fun)`,
      matlab: `% MATLAB 单纯形法
f = [-2; -3];
A = [1 2; 3 1];
b = [8; 12];
opts = optimoptions('linprog','Algorithm','simplex');
[x, fval] = linprog(f, A, b, [], [], zeros(2,1), [], opts);`
    },
    resources: [
      { title: "Simplex algorithm", url: "https://en.wikipedia.org/wiki/Simplex_algorithm", type: "link" }
    ]
  },
  {
    id: "revised-simplex-method",
    name: "修正单纯形法 (Revised Simplex Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过矩阵更新而不是显式维护整个单纯形表，提高了单纯形法在中大规模问题中的计算效率。",
    principles: `1. 只维护基矩阵及其逆或分解信息。
2. 每步根据检验数选择入基变量。
3. 通过比值检验确定出基变量。
4. 使用矩阵更新减少重复运算成本。`,
    scenarios: [
      "中大规模线性规划",
      "希望保留单纯形法思想但提升计算效率的场景"
    ],
    limitations: [
      "实现复杂度高于普通单纯形法",
      "仍属于顶点迭代法"
    ],
    caseStudy: {
      title: "仓储配送线性优化",
      description: "在多仓多点供需结构下优化运输方案。",
      solution: `1. 建立运输线性规划模型；
2. 使用修正单纯形法维护基矩阵；
3. 输出低成本配送决策。`
    },
    code: {
      python: `from scipy.optimize import linprog
c = [-2, -3]
A = [[1, 2], [3, 1]]
b = [8, 12]
res = linprog(c, A_ub=A, b_ub=b, method='revised simplex')
print(res.x, -res.fun)`,
      matlab: `% MATLAB 修正单纯形法通常由底层求解器自动实现`
    },
    resources: [
      { title: "Revised simplex method", url: "https://encyclopediaofmath.org/wiki/Revised_simplex_method", type: "link" }
    ]
  },
  {
    id: "nonlinear-programming",
    name: "非线性规划 (Nonlinear Programming)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "处理目标函数或约束条件中存在非线性关系的优化问题，是连续优化中的核心建模框架。",
    principles: `1. 允许目标函数或约束为非线性形式。
2. 常通过梯度、Hessian 或近似信息迭代更新。
3. 可结合罚函数、拉格朗日乘子和约束变换处理复杂限制。
4. 通常只能保证局部最优，需要关注初值影响。`,
    scenarios: [
      "结构设计、参数估计、能耗最小化等连续优化任务",
      "具有非线性物理规律或业务关系的约束优化问题"
    ],
    limitations: [
      "可能存在多个局部最优点",
      "对初始值和模型光滑性较敏感"
    ],
    caseStudy: {
      title: "设备参数能耗优化",
      description: "在非线性安全约束下最小化设备运行能耗。",
      solution: `1. 构建非线性能耗目标函数；
2. 写出运行边界和安全约束；
3. 使用非线性规划算法求局部最优。`
    },
    code: {
      python: `from scipy.optimize import minimize
fun = lambda x: (x[0]-1)**2 + (x[1]-2)**2 + x[0]*x[1]
cons = ({'type': 'ineq', 'fun': lambda x: 3 - x[0] - x[1]})
res = minimize(fun, [0, 0], constraints=cons)
print(res.x, res.fun)`,
      matlab: `% MATLAB 非线性规划
fun = @(x) (x(1)-1)^2 + (x(2)-2)^2 + x(1)*x(2);
nonlcon = @(x) deal([], x(1)+x(2)-3);
[x, fval] = fmincon(fun, [0;0], [], [], [], [], [], [], nonlcon);`
    },
    resources: [
      { title: "Nonlinear programming", url: "https://en.wikipedia.org/wiki/Nonlinear_programming", type: "link" }
    ]
  },
  {
    id: "nelder-mead-method",
    name: "下山单纯形法 (Nelder-Mead Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "一种不依赖梯度信息的局部搜索方法，适用于不可导或梯度难以获取的连续优化问题。",
    principles: `1. 在 n 维空间构造 n+1 个顶点形成单纯形。
2. 通过反射、扩张、收缩和缩减操作移动单纯形。
3. 逐步向目标函数更优区域靠近。
4. 适合低维、噪声或不可微目标优化。`,
    scenarios: [
      "黑箱函数、小规模参数调优",
      "缺乏解析梯度的局部连续优化问题"
    ],
    limitations: [
      "容易陷入局部最优",
      "高维问题效率通常较差"
    ],
    caseStudy: {
      title: "实验参数无梯度调优",
      description: "目标函数由实验结果给出，无法显式求导。",
      solution: `1. 初始化一个参数单纯形；
2. 通过反射和收缩不断更新候选点；
3. 收敛到局部较优实验参数。`
    },
    code: {
      python: `from scipy.optimize import minimize
import numpy as np
fun = lambda x: (x[0]-2)**2 + abs(x[1]-3)
res = minimize(fun, [0, 0], method='Nelder-Mead')
print(res.x, res.fun)`,
      matlab: `% MATLAB 下山单纯形法
fun = @(x) (x(1)-2)^2 + abs(x(2)-3);
[x, fval] = fminsearch(fun, [0, 0]);`
    },
    resources: [
      { title: "Nelder-Mead method", url: "https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method", type: "link" }
    ]
  },
  {
    id: "improved-bfgs-quasi-newton",
    name: "改进的BFGS拟牛顿法 (Improved BFGS Quasi-Newton Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过对 Hessian 逆矩阵进行近似更新来加速收敛，是求解光滑无约束或弱约束优化问题的重要方法。",
    principles: `1. 用一阶梯度信息近似二阶曲率信息。
2. 通过 BFGS 更新公式迭代修正近似 Hessian 逆矩阵。
3. 结合线搜索决定步长。
4. 改进版本通常增强数值稳定性和收敛鲁棒性。`,
    scenarios: [
      "光滑连续优化、参数估计、机器学习损失优化",
      "希望比梯度下降更快收敛的中等规模问题"
    ],
    limitations: [
      "要求目标函数相对光滑",
      "在强非凸问题中仍可能陷入局部极值"
    ],
    caseStudy: {
      title: "连续参数损失最小化",
      description: "通过优化连续参数使模型损失函数最小。",
      solution: `1. 计算目标函数梯度；
2. 用 BFGS 更新曲率近似；
3. 借助线搜索快速收敛到局部最优。`
    },
    code: {
      python: `from scipy.optimize import minimize
fun = lambda x: (x[0]-1)**2 + (x[1]-2.5)**2
res = minimize(fun, [0.0, 0.0], method='BFGS')
print(res.x, res.fun)`,
      matlab: `% MATLAB BFGS 拟牛顿法
fun = @(x) (x(1)-1)^2 + (x(2)-2.5)^2;
opts = optimoptions('fminunc','Algorithm','quasi-newton');
[x, fval] = fminunc(fun, [0; 0], opts);`
    },
    resources: [
      { title: "BFGS method", url: "https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm", type: "link" }
    ]
  },
  {
    id: "improved-conjugate-direction-method",
    name: "改进的共轭方向法 (Improved Conjugate Direction Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过构造彼此共轭的搜索方向减少重复搜索，在二次优化和大规模连续优化中具有较好效率。",
    principles: `1. 选择一组互相共轭的搜索方向。
2. 在每个方向上执行一维搜索。
3. 更新方向集以减少重复下降。
4. 改进方法通常增强非二次情形下的稳定性。`,
    scenarios: [
      "二次优化、大规模光滑目标优化",
      "希望减少 Hessian 显式计算的连续优化问题"
    ],
    limitations: [
      "对非光滑目标不适合",
      "方向更新策略会显著影响效果"
    ],
    caseStudy: {
      title: "结构参数二次优化",
      description: "在二次近似目标下优化多个设计参数。",
      solution: `1. 初始化共轭方向；
2. 轮流在各方向做一维搜索；
3. 通过方向更新提高收敛效率。`
    },
    code: {
      python: `import numpy as np
Q = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x = np.zeros(2)
r = b - Q @ x
p = r.copy()
for _ in range(2):
    alpha = (r @ r) / (p @ Q @ p)
    x = x + alpha * p
    r_new = r - alpha * (Q @ p)
    beta = (r_new @ r_new) / (r @ r)
    p = r_new + beta * p
    r = r_new
print(x)`,
      matlab: `% MATLAB 共轭方向法可自定义实现或借助优化工具箱相关流程`
    },
    resources: [
      { title: "Conjugate gradient method", url: "https://en.wikipedia.org/wiki/Conjugate_gradient_method", type: "link" }
    ]
  },
  {
    id: "truncated-newton-boundary-method",
    name: "(边界) 截断牛顿法 ((Boundary) Truncated Newton Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过在牛顿法框架下近似求解子问题并截断迭代，适合大规模光滑约束优化与边界优化问题。",
    principles: `1. 在每轮迭代中构造牛顿方向相关子问题。
2. 不精确求解子问题，而是用截断迭代近似方向。
3. 可结合边界约束处理可行域限制。
4. 适合 Hessian 难以显式存储的大规模问题。`,
    scenarios: [
      "大规模连续优化",
      "存在边界限制且 Hessian 规模很大的问题"
    ],
    limitations: [
      "实现复杂度较高",
      "通常需要较好的预条件或方向近似策略"
    ],
    caseStudy: {
      title: "高维参数边界优化",
      description: "在变量上下界约束下优化高维连续参数。",
      solution: `1. 构造局部二次子问题；
2. 用截断迭代近似牛顿方向；
3. 结合边界控制更新参数。`
    },
    code: {
      python: `from scipy.optimize import minimize
fun = lambda x: (x[0]-1)**2 + (x[1]-2)**2
res = minimize(fun, [0, 0], method='TNC', bounds=[(0, 2), (0, 3)])
print(res.x, res.fun)`,
      matlab: `% MATLAB 截断牛顿边界法可通过相关优化函数和边界约束配置实现`
    },
    resources: [
      { title: "Truncated Newton method", url: "https://en.wikipedia.org/wiki/Truncated_Newton_method", type: "link" }
    ]
  },
  {
    id: "linear-approximation-bundle-method",
    name: "线性近似束优化方法 (Linear Approximation Bundle Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过收集子梯度信息构造线性近似束，是求解非光滑凸优化问题的重要方法。",
    principles: `1. 在当前点收集目标函数的子梯度信息。
2. 用这些信息构造局部线性下近似束。
3. 通过求解辅助问题决定下一步迭代点。
4. 逐步逼近非光滑优化问题的最优解。`,
    scenarios: [
      "非光滑凸优化",
      "存在 max、绝对值、分段函数等结构的规划问题"
    ],
    limitations: [
      "实现较复杂",
      "主要适用于凸或近凸结构"
    ],
    caseStudy: {
      title: "分段成本函数优化",
      description: "成本函数含多个分段线性片段，目标不可光滑。",
      solution: `1. 提取当前点的子梯度；
2. 构造局部线性束模型；
3. 迭代求解非光滑最优决策。`
    },
    code: {
      python: `import numpy as np
x = 2.0
subgrad = 1 if x > 0 else -1
print('bundle methods typically require custom implementation, subgradient =', subgrad)`,
      matlab: `% MATLAB 束优化方法通常需自定义实现`
    },
    resources: [
      { title: "Bundle method", url: "https://optimization-online.org/tag/bundle-method/", type: "link" }
    ]
  },
  {
    id: "slsqp-method",
    name: "序贯最小二乘规划算法 (Sequential Least Squares Programming, SLSQP)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过逐步求解二次近似子问题处理带约束的非线性优化，是常用的通用型约束优化算法。",
    principles: `1. 将原非线性规划在当前点附近近似为二次规划子问题。
2. 在每一步同时考虑目标下降和约束满足。
3. 通过序贯迭代不断更新解。
4. 适合中小规模光滑约束优化。`,
    scenarios: [
      "带等式/不等式约束的非线性优化",
      "建模竞赛中常见的连续约束优化问题"
    ],
    limitations: [
      "对初值和模型光滑性较敏感",
      "大规模问题上效率可能有限"
    ],
    caseStudy: {
      title: "成本与性能平衡优化",
      description: "在预算和安全约束下优化设计参数。",
      solution: `1. 建立非线性目标和约束；
2. 用 SLSQP 逐步求二次子问题；
3. 输出满足约束的局部最优设计。`
    },
    code: {
      python: `from scipy.optimize import minimize
fun = lambda x: (x[0]-1)**2 + (x[1]-2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 2})
res = minimize(fun, [0, 0], method='SLSQP', constraints=cons)
print(res.x, res.fun)`,
      matlab: `% MATLAB SLSQP 可近似对应约束非线性优化流程
fun = @(x) (x(1)-1)^2 + (x(2)-2.5)^2;
[x, fval] = fmincon(fun, [0;0]);`
    },
    resources: [
      { title: "SLSQP 文档", url: "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html", type: "link" }
    ]
  },
  {
    id: "trust-region-method",
    name: "信赖域算法 (Trust Region Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "在当前点附近构造局部模型并限定搜索半径，通过动态调整信赖域实现稳定的连续优化。",
    principles: `1. 在当前迭代点构造局部近似模型。
2. 在一个有限半径的信赖域内求子问题。
3. 根据近似模型与真实下降效果调整信赖域大小。
4. 平衡收敛速度与数值稳定性。`,
    scenarios: [
      "无约束或弱约束连续优化",
      "需要较强稳定性的二阶优化问题"
    ],
    limitations: [
      "需要局部模型信息",
      "实现和调参相对复杂"
    ],
    caseStudy: {
      title: "连续设计参数优化",
      description: "在可微目标函数下寻找稳定的局部最优设计参数。",
      solution: `1. 建立局部二次近似模型；
2. 在信赖域内求解子问题；
3. 动态调整步长半径直至收敛。`
    },
    code: {
      python: `from scipy.optimize import minimize
fun = lambda x: (x[0]-1)**2 + (x[1]-2)**2
res = minimize(fun, [0, 0], method='trust-constr')
print(res.x, res.fun)`,
      matlab: `% MATLAB 信赖域算法
fun = @(x) (x(1)-1)^2 + (x(2)-2)^2;
opts = optimoptions('fminunc','Algorithm','trust-region');
[x, fval] = fminunc(fun, [0;0], opts);`
    },
    resources: [
      { title: "Trust region", url: "https://en.wikipedia.org/wiki/Trust_region", type: "link" }
    ]
  },
  {
    id: "integer-programming",
    name: "整数规划 (Integer Programming)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "要求部分或全部决策变量取整数值，适合处理离散决策、选址、排班和组合优化问题。",
    principles: `1. 在线性或非线性规划框架中增加整数约束。
2. 可分为纯整数规划和混合整数规划。
3. 常用分支定界、割平面等方法求解。
4. 解空间离散，通常比连续规划更难。`,
    scenarios: [
      "选址、排班、装箱、路径设计、投资组合离散决策",
      "变量必须取整数或离散值的优化问题"
    ],
    limitations: [
      "计算复杂度高",
      "大规模问题求解时间可能很长"
    ],
    caseStudy: {
      title: "仓库选址优化",
      description: "决定哪些仓库开启以及客户由哪个仓库服务。",
      solution: `1. 构建成本最小化目标；
2. 对建仓变量施加整数约束；
3. 求解最优启用方案。`
    },
    code: {
      python: `import pulp
x = pulp.LpVariable('x', lowBound=0, cat='Integer')
y = pulp.LpVariable('y', lowBound=0, cat='Integer')
prob = pulp.LpProblem('ip', pulp.LpMaximize)
prob += 3*x + 2*y
prob += x + 2*y <= 8
prob.solve()
print(x.value(), y.value())`,
      matlab: `% MATLAB 整数规划
f = [-3; -2];
intcon = [1 2];
A = [1 2]; b = 8;
[x, fval] = intlinprog(f, intcon, A, b);`
    },
    resources: [
      { title: "Integer programming", url: "https://en.wikipedia.org/wiki/Integer_programming", type: "link" }
    ]
  },
  {
    id: "branch-and-bound",
    name: "分支定界法 (Branch and Bound)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过对子问题递归分支并利用上下界剪枝，是求解整数规划和组合优化问题的经典框架。",
    principles: `1. 将原问题拆分为若干子问题进行分支。
2. 对每个子问题计算上下界。
3. 若某子问题不可能优于当前最优解则剪枝。
4. 重复直到所有有效分支处理完毕。`,
    scenarios: [
      "整数规划、0-1规划、组合优化",
      "需要精确求解离散最优解的问题"
    ],
    limitations: [
      "最坏情况下搜索树可能很大",
      "依赖良好的界估计和剪枝策略"
    ],
    caseStudy: {
      title: "设施选址离散优化",
      description: "在多个候选点中选择最优设施组合。",
      solution: `1. 对候选决策变量分支；
2. 计算每个子问题的松弛下界；
3. 剪枝并保留最优组合。`
    },
    code: {
      python: `print('Branch and bound is usually implemented inside MILP solvers such as CBC, Gurobi, or CPLEX.')`,
      matlab: `% MATLAB 分支定界法通常由 intlinprog 等求解器内部实现`
    },
    resources: [
      { title: "Branch and bound", url: "https://en.wikipedia.org/wiki/Branch_and_bound", type: "link" }
    ]
  },
  {
    id: "zero-one-programming",
    name: "0-1规划 (0-1 Programming)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "整数规划的特殊形式，变量只能取 0 或 1，常用于表示是否选择、是否执行等二元决策。",
    principles: `1. 决策变量仅取 0 或 1。
2. 适合将“选/不选”“开/不开”等逻辑离散化。
3. 常与线性目标和约束结合形成 0-1 线性规划。
4. 多借助分支定界或启发式方法求解。`,
    scenarios: [
      "项目选择、设备启停、站点开闭、路径选择",
      "需要二元逻辑决策建模的问题"
    ],
    limitations: [
      "组合规模增长快",
      "大规模问题求精确解成本高"
    ],
    caseStudy: {
      title: "投资项目选择",
      description: "在预算约束下选择收益最高的项目组合。",
      solution: `1. 用 0/1 变量表示项目是否入选；
2. 构建预算约束和收益目标；
3. 求解最优组合方案。`
    },
    code: {
      python: `import itertools
profits = [8, 10, 6]
costs = [4, 6, 3]
budget = 8
best = None
for pick in itertools.product([0,1], repeat=3):
    if sum(c*p for c,p in zip(costs, pick)) <= budget:
        value = sum(v*p for v,p in zip(profits, pick))
        best = max(best or (0, None), (value, pick))
print(best)`,
      matlab: `% MATLAB 0-1 规划
f = [-8; -10; -6];
intcon = [1 2 3];
A = [4 6 3]; b = 8;
lb = [0;0;0]; ub = [1;1;1];
[x, fval] = intlinprog(f, intcon, A, b, [], [], lb, ub);`
    },
    resources: [
      { title: "0-1 integer programming", url: "https://en.wikipedia.org/wiki/Integer_programming#Special_cases", type: "link" }
    ]
  },
  {
    id: "enumeration-method",
    name: "枚举法 (Enumeration Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过遍历全部或部分候选解并逐个比较目标值求最优解，是最直接的离散优化求解思路。",
    principles: `1. 列举全部候选解或可行组合。
2. 对每个候选解计算目标函数值。
3. 筛选满足约束的可行解。
4. 比较所有可行解后得到最优方案。`,
    scenarios: [
      "规模较小的组合优化问题",
      "教学演示、精确验证和暴力搜索基准构造"
    ],
    limitations: [
      "组合数爆炸，难以扩展到大规模问题",
      "计算成本随变量数快速上升"
    ],
    caseStudy: {
      title: "小规模组合选取",
      description: "在少量候选方案中寻找满足约束的最优方案。",
      solution: `1. 列举全部可选组合；
2. 过滤不满足约束的方案；
3. 比较目标值得到最优解。`
    },
    code: {
      python: `import itertools
best = None
for x in itertools.product([0,1], repeat=4):
    val = sum(x)
    best = max(best or (0, None), (val, x))
print(best)`,
      matlab: `% MATLAB 枚举法通常通过循环或 combinatorics 函数手动实现`
    },
    resources: [
      { title: "Brute-force search", url: "https://en.wikipedia.org/wiki/Brute-force_search", type: "link" }
    ]
  },
  {
    id: "heuristic-algorithm",
    name: "启发式算法 (Heuristic Algorithm)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "利用经验规则、局部搜索或问题结构快速寻找较优解，适合精确算法难以高效处理的大规模复杂优化问题。",
    principles: `1. 利用问题结构设计近似搜索规则。
2. 不保证全局最优，但追求较好可行解。
3. 常结合局部改进、邻域搜索或贪心思想。
4. 可作为大规模问题的高效近似求解策略。`,
    scenarios: [
      "大规模组合优化、排班、路径、装箱等问题",
      "需要在有限时间内得到高质量可行解的任务"
    ],
    limitations: [
      "通常不保证最优性",
      "算法质量依赖具体启发规则设计"
    ],
    caseStudy: {
      title: "快速路径近似求解",
      description: "在时间限制下快速生成一条较短配送路径。",
      solution: `1. 根据最近邻规则构建初始解；
2. 用局部交换进一步改进；
3. 输出较优但不一定全局最优的方案。`
    },
    code: {
      python: `print('Heuristic algorithms are usually custom-designed according to problem structure.')`,
      matlab: `% MATLAB 启发式算法多为针对具体问题自定义实现`
    },
    resources: [
      { title: "Heuristic", url: "https://en.wikipedia.org/wiki/Heuristic", type: "link" }
    ]
  },
  {
    id: "genetic-algorithm",
    name: "遗传算法 (GA) (Genetic Algorithm)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "模拟生物进化中的选择、交叉和变异过程，在复杂非凸搜索空间中寻找高质量解。",
    principles: `1. 将候选解编码成染色体。
2. 根据适应度选择优良个体。
3. 通过交叉和变异生成新一代群体。
4. 迭代进化后得到较优解。`,
    scenarios: [
      "路径规划、参数优化、排程与多峰全局搜索",
      "传统梯度法难以处理的复杂组合/连续混合优化问题"
    ],
    limitations: [
      "收敛速度可能较慢",
      "参数设置对结果影响较大"
    ],
    caseStudy: {
      title: "TSP 路径优化",
      description: "在多个城市之间寻找总路程较短的巡回路径。",
      solution: `1. 编码城市访问顺序；
2. 通过选择、交叉和变异演化种群；
3. 输出最终高适应度路径。`
    },
    code: {
      python: `import numpy as np
np.random.seed(0)
pop = np.random.randint(0, 2, (6, 8))
fitness = pop.sum(axis=1)
print(pop[fitness.argmax()])`,
      matlab: `% MATLAB 遗传算法
fun = @(x) x(1)^2 + x(2)^2;
[x, fval] = ga(fun, 2);`
    },
    resources: [
      { title: "Genetic algorithm", url: "https://en.wikipedia.org/wiki/Genetic_algorithm", type: "link" }
    ]
  },
  {
    id: "particle-swarm-optimization",
    name: "粒子群算法 (PSO) (Particle Swarm Optimization)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "模拟群体协同搜索机制，通过粒子位置和速度更新寻找连续或离散空间中的优良解。",
    principles: `1. 每个粒子代表一个候选解。
2. 粒子根据个体最优和全局最优更新速度与位置。
3. 群体协作推动整体搜索向优良区域移动。
4. 迭代后得到较优解。`,
    scenarios: [
      "连续参数优化、控制参数整定、路径规划",
      "希望用群体智能做全局近似搜索的任务"
    ],
    limitations: [
      "后期可能早熟收敛",
      "参数配置会影响探索和开发平衡"
    ],
    caseStudy: {
      title: "连续参数寻优",
      description: "通过粒子群搜索连续参数空间中的低损失区域。",
      solution: `1. 初始化粒子群位置与速度；
2. 更新个体最优和全局最优；
3. 迭代得到较优参数组合。`
    },
    code: {
      python: `import numpy as np
np.random.seed(0)
p = np.random.uniform(-5, 5, (10, 2))
v = np.zeros_like(p)
pbest = p.copy()
gbest = p[0]
for _ in range(5):
    v = 0.5*v + 1.5*np.random.rand()*(pbest - p)
    p += v
print(p[0])`,
      matlab: `% MATLAB 粒子群算法
fun = @(x) x(1)^2 + x(2)^2;
[x, fval] = particleswarm(fun, 2, [-5 -5], [5 5]);`
    },
    resources: [
      { title: "Particle swarm optimization", url: "https://en.wikipedia.org/wiki/Particle_swarm_optimization", type: "link" }
    ]
  },
  {
    id: "simulated-annealing",
    name: "模拟退火算法 (SA) (Simulated Annealing)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "通过逐步降温并以一定概率接受较差解来跳出局部最优，是经典的随机全局优化方法。",
    principles: `1. 将优化过程类比为物理退火降温。
2. 在高温阶段允许接受劣解以增强探索。
3. 随着温度降低逐步收敛到稳定区域。
4. 平衡全局搜索与局部精修。`,
    scenarios: [
      "组合优化、路径规划、参数调优",
      "存在多个局部最优的复杂优化问题"
    ],
    limitations: [
      "收敛速度可能较慢",
      "温度计划设计会显著影响效果"
    ],
    caseStudy: {
      title: "复杂路径优化",
      description: "在多峰搜索空间中寻找更短的访问路径。",
      solution: `1. 初始化路径和温度；
2. 通过扰动生成新解；
3. 按退火准则接受或拒绝新解并逐步降温。`
    },
    code: {
      python: `import numpy as np
np.random.seed(0)
T = 10.0
x = 3.0
for _ in range(20):
    x_new = x + np.random.randn()*0.5
    if np.exp(-(x_new**2 - x**2)/T) > np.random.rand():
        x = x_new
    T *= 0.9
print(x)`,
      matlab: `% MATLAB 模拟退火
fun = @(x) x(1)^2 + 4*sin(x(1));
[x, fval] = simulannealbnd(fun, 3, -10, 10);`
    },
    resources: [
      { title: "Simulated annealing", url: "https://en.wikipedia.org/wiki/Simulated_annealing", type: "link" }
    ]
  },
  {
    id: "monte-carlo-method",
    name: "蒙特卡洛 (Monte Carlo Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "利用大量随机采样估计复杂系统的期望、概率或近似最优值，是处理高维不确定性问题的重要数值方法。",
    principles: `1. 构造随机样本模拟系统状态或决策结果。
2. 通过大量采样估计目标函数、积分或概率量。
3. 利用统计平均逼近真实结果。
4. 可结合搜索策略用于随机优化与仿真分析。`,
    scenarios: [
      "随机模拟、风险评估、高维积分、近似优化",
      "解析求解困难且不确定性显著的问题"
    ],
    limitations: [
      "收敛速度通常较慢",
      "高精度估计需要大量样本"
    ],
    caseStudy: {
      title: "随机收益仿真评估",
      description: "通过大量随机样本估计不同策略下的平均收益水平。",
      solution: `1. 生成随机扰动或需求样本；
2. 计算每次仿真的策略收益；
3. 用均值和分布评估策略优劣。`
    },
    code: {
      python: `import numpy as np
np.random.seed(0)
samples = np.random.rand(100000, 2)
inside = ((samples[:,0]-0.5)**2 + (samples[:,1]-0.5)**2 <= 0.25).mean()
print(inside)`,
      matlab: `% MATLAB 蒙特卡洛模拟
n = 1e5;
X = rand(n,2);
inside = mean((X(:,1)-0.5).^2 + (X(:,2)-0.5).^2 <= 0.25);
disp(inside);`
    },
    resources: [
      { title: "Monte Carlo method", url: "https://en.wikipedia.org/wiki/Monte_Carlo_method", type: "link" }
    ]
  },
  {
    id: "econometrics-adf-unit-root-test",
    name: "单位根检验(ADF) (ADF Unit Root Test)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "用于检验时间序列是否平稳，是建立 ARIMA、VAR 和协整模型前最常用的平稳性诊断方法。",
    principles: `1. 构造增广 Dickey-Fuller 回归：Δy_t = α + βt + γy_{t-1} + Σδ_iΔy_{t-i} + ε_t。
2. 原假设通常为序列存在单位根，即非平稳。
3. 通过对 γ 的 t 统计量与临界值比较判断是否拒绝原假设。
4. 可按是否包含常数项与趋势项选择不同检验形式。`,
    scenarios: [
      "宏观经济指标建模前检验 GDP、CPI、汇率等序列是否平稳",
      "金融收益率与价格序列的平稳性识别"
    ],
    limitations: [
      "对结构突变较敏感，突变存在时容易误判",
      "滞后阶数选择不当会影响检验效能"
    ],
    caseStudy: {
      title: "检验月度销售额序列是否可直接建模",
      description: "某企业有 60 个月的销售额数据，希望判断是否需要先做差分再建立预测模型。",
      solution: `1. 绘制原序列观察趋势；
2. 对原序列执行 ADF 检验；
3. 若不能拒绝单位根，则进行一阶差分并再次检验；
4. 以平稳后的序列进入后续建模。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.stattools import adfuller

series = pd.Series([102, 105, 107, 110, 115, 118, 121, 125])
stat, pvalue, *_ = adfuller(series)
print("ADF统计量:", stat)
print("p值:", pvalue)`,
      matlab: `% MATLAB ADF 单位根检验
series = [102 105 107 110 115 118 121 125]';
[h,pValue,stat] = adftest(series);`
    },
    resources: [
      { title: "statsmodels adfuller 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html", type: "link" }
    ]
  },
  {
    id: "econometrics-differencing-analysis",
    name: "差分分析 (Differencing Analysis)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "通过对时间序列做一阶或高阶差分去除趋势与单位根，是非平稳序列转化为平稳序列的核心操作。",
    principles: `1. 一阶差分定义为 Δy_t = y_t - y_{t-1}。
2. 二阶差分可表示为 Δ²y_t = Δy_t - Δy_{t-1}。
3. 差分能削弱趋势和累积效应，使均值与方差更趋稳定。
4. 在 ARIMA 中，差分阶数 d 反映序列平稳化所需次数。`,
    scenarios: [
      "对存在明显趋势的销量、人口、价格序列进行平稳化处理",
      "为 ARIMA、协整、VAR 等模型准备输入序列"
    ],
    limitations: [
      "过度差分会破坏原有信息并引入额外噪声",
      "差分后序列解释性通常弱于原始水平值"
    ],
    caseStudy: {
      title: "对年度游客数量序列进行平稳化",
      description: "景区游客量逐年上升且波动随水平放大，直接建模不满足平稳性要求。",
      solution: `1. 先做一阶差分消除线性趋势；
2. 对差分序列做平稳性检验；
3. 如仍存在趋势，再考虑二阶差分；
4. 选取最小必要差分阶数。`
    },
    code: {
      python: `import pandas as pd
series = pd.Series([120, 128, 137, 149, 160, 174])
first_diff = series.diff().dropna()
second_diff = first_diff.diff().dropna()
print(first_diff)
print(second_diff)`,
      matlab: `% MATLAB 差分分析
series = [120 128 137 149 160 174]';
firstDiff = diff(series);
secondDiff = diff(series, 2);`
    },
    resources: [
      { title: "时间序列差分介绍", url: "https://otexts.com/fpp3/stationarity.html", type: "link" }
    ]
  },
  {
    id: "econometrics-acf-pacf",
    name: "（偏）自相关分析(pacf/acf) (ACF/PACF Analysis)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "通过自相关函数与偏自相关函数识别时间序列的依赖结构，是判断 AR、MA、ARIMA 阶数的重要依据。",
    principles: `1. ACF 衡量序列与其不同滞后值之间的相关程度。
2. PACF 衡量控制中间滞后后，当前值与某一滞后值的净相关。
3. AR(p) 模型常表现为 PACF 截尾、ACF 拖尾。
4. MA(q) 模型常表现为 ACF 截尾、PACF 拖尾。`,
    scenarios: [
      "识别 ARIMA 模型中的 p、q 阶数",
      "分析经济指标或金融收益序列的滞后依赖关系"
    ],
    limitations: [
      "样本量较小时图形判断存在较大主观性",
      "季节性或结构突变会干扰传统 ACF/PACF 识别"
    ],
    caseStudy: {
      title: "为 ARIMA 选阶提供图形依据",
      description: "对一组平稳后的月度订单量数据，需要快速判断应优先尝试 AR 还是 MA 成分。",
      solution: `1. 先对序列平稳化；
2. 绘制 ACF 与 PACF 图；
3. 根据截尾和拖尾特征初步确定 p、q 候选范围；
4. 再结合信息准则比较模型。`
    },
    code: {
      python: `import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
series = [3.2, 3.4, 3.3, 3.8, 4.0, 4.1, 4.2, 4.0]
plot_acf(series, lags=5)
plot_pacf(series, lags=5)
plt.show()`,
      matlab: `% MATLAB ACF/PACF 分析
series = [3.2 3.4 3.3 3.8 4.0 4.1 4.2 4.0]';
autocorr(series);
parcorr(series);`
    },
    resources: [
      { title: "statsmodels ACF/PACF 绘图文档", url: "https://www.statsmodels.org/stable/tsa.html", type: "link" }
    ]
  },
  {
    id: "econometrics-arima",
    name: "时间序列分析(ARIMA) (ARIMA Time Series Model)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "ARIMA 将自回归、差分和移动平均统一起来，是处理中短期平稳或可差分平稳序列的经典预测模型。",
    principles: `1. ARIMA(p,d,q) 由自回归项 AR(p)、差分项 I(d)、移动平均项 MA(q) 组成。
2. d 用于去除趋势并使序列平稳。
3. p 与 q 通常结合 ACF/PACF 和 AIC/BIC 进行选择。
4. 建模后需检验残差是否接近白噪声。`,
    scenarios: [
      "销售额、客流量、产量等单变量时间序列短期预测",
      "缺少外生解释变量时的纯序列建模"
    ],
    limitations: [
      "主要处理线性结构，对复杂非线性关系捕捉有限",
      "对异常值与结构变化较敏感"
    ],
    caseStudy: {
      title: "月度订单量短期预测",
      description: "平台仅掌握过去 3 年的月订单量，希望预测未来 3 个月走势。",
      solution: `1. 对序列做平稳性检验和必要差分；
2. 依据 ACF/PACF 初步设定 p、q；
3. 比较 AIC/BIC 选出最优模型；
4. 输出未来 3 个月预测值和置信区间。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

series = pd.Series([120, 125, 123, 130, 136, 140, 145, 147, 150, 155])
model = ARIMA(series, order=(1, 1, 1)).fit()
print(model.forecast(steps=3))`,
      matlab: `% MATLAB ARIMA
series = [120 125 123 130 136 140 145 147 150 155]';
mdl = arima(1,1,1);
estMdl = estimate(mdl, series, 'Display', 'off');
forecast(estMdl, 3, series);`
    },
    resources: [
      { title: "statsmodels ARIMA 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html", type: "link" }
    ]
  },
  {
    id: "econometrics-garch",
    name: "GARCH模型 (GARCH Model)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "用于描述波动率聚集现象，尤其适合金融收益率序列中条件异方差结构的建模与预测。",
    principles: `1. GARCH(p,q) 同时建模条件均值与条件方差。
2. 常见 GARCH(1,1) 形式为 σ_t² = ω + αε_{t-1}² + βσ_{t-1}²。
3. α 反映新冲击对波动的即时影响，β 反映波动持续性。
4. 若 α + β 接近 1，说明波动具有较强持久性。`,
    scenarios: [
      "股票、汇率、利率收益率波动预测",
      "风险度量、VaR 估计和金融压力分析"
    ],
    limitations: [
      "主要适合金融类高频波动问题，其他场景解释性有限",
      "模型设定不当时参数估计可能不稳定"
    ],
    caseStudy: {
      title: "股票收益率波动率预测",
      description: "研究者希望根据过去收益率序列预测未来短期风险水平。",
      solution: `1. 先对收益率序列拟合均值方程；
2. 检验 ARCH 效应是否显著；
3. 建立 GARCH(1,1) 模型；
4. 输出条件波动率路径用于风险管理。`
    },
    code: {
      python: `import numpy as np
from arch import arch_model

returns = np.array([0.01, -0.02, 0.015, -0.005, 0.018, -0.01, 0.012])
model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit(disp='off')
print(result.params)`,
      matlab: `% MATLAB GARCH
returns = [0.01 -0.02 0.015 -0.005 0.018 -0.01 0.012]';
mdl = garch(1,1);
estMdl = estimate(mdl, returns, 'Display', 'off');`
    },
    resources: [
      { title: "arch Python 包文档", url: "https://arch.readthedocs.io", type: "link" }
    ]
  },
  {
    id: "econometrics-granger-causality",
    name: "格兰杰因果检验 (Granger Causality Test)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "检验一个变量的过去信息是否能显著提升另一个变量的预测能力，是时序因果方向探索的经典工具。",
    principles: `1. 核心思想是“先发生且能提升预测”而非严格哲学意义上的因果。
2. 比较包含与不包含变量 X 滞后项时，对变量 Y 的预测效果差异。
3. 常通过 F 检验或卡方检验判断滞后项联合显著性。
4. 一般要求参与检验的序列平稳或已做平稳化处理。`,
    scenarios: [
      "分析广告投放是否领先影响销量变化",
      "研究货币供应、利率、产出等宏观变量之间的动态关系"
    ],
    limitations: [
      "结论依赖滞后阶数与变量选取，不能替代真正因果识别",
      "遗漏变量和共同趋势可能导致伪因果结论"
    ],
    caseStudy: {
      title: "检验搜索热度是否领先销量变化",
      description: "企业希望验证网络搜索指数是否对未来销量有预测增益。",
      solution: `1. 对搜索指数与销量序列做平稳化处理；
2. 设定多个滞后阶数；
3. 执行格兰杰因果检验；
4. 根据显著性判断是否存在预测领先关系。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

df = pd.DataFrame({
    'sales': [100, 102, 101, 105, 108, 110, 112, 115],
    'search': [50, 51, 53, 54, 57, 58, 60, 62]
})
grangercausalitytests(df[['sales', 'search']], maxlag=2, verbose=False)`,
      matlab: `% MATLAB 可通过 VAR 结果与联合显著性检验间接实现格兰杰因果分析`
    },
    resources: [
      { title: "statsmodels 格兰杰因果检验文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html", type: "link" }
    ]
  },
  {
    id: "econometrics-var",
    name: "VAR向量自回归模型 (VAR Model)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "将多个内生时间序列同时纳入系统建模，刻画变量之间的动态相互影响，是宏观与金融时序分析的重要框架。",
    principles: `1. VAR(p) 模型中，每个变量都由自身和其他变量的滞后项共同解释。
2. 一般形式为 Y_t = c + A_1Y_{t-1} + ... + A_pY_{t-p} + ε_t。
3. 可结合脉冲响应函数分析冲击传导路径。
4. 也可通过方差分解评估各变量对预测误差的贡献。`,
    scenarios: [
      "研究利率、通胀、产出之间的联动关系",
      "分析销量、价格、投放之间的动态反馈结构"
    ],
    limitations: [
      "参数数量随变量数和滞后阶数快速膨胀，对样本量要求较高",
      "结果解释依赖识别设定和变量排序"
    ],
    caseStudy: {
      title: "分析价格与销量的双向动态关系",
      description: "企业想同时研究价格调整和销量波动之间是否存在反馈机制。",
      solution: `1. 构建价格与销量平稳序列；
2. 以信息准则选择 VAR 滞后阶数；
3. 估计 VAR 模型；
4. 通过脉冲响应分析价格冲击对销量的持续影响。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.api import VAR

df = pd.DataFrame({
    'price': [10, 10.2, 10.1, 10.4, 10.3, 10.5, 10.6],
    'sales': [100, 98, 101, 97, 99, 96, 95]
})
model = VAR(df)
result = model.fit(maxlags=2)
print(result.summary())`,
      matlab: `% MATLAB VAR
Y = [10 100; 10.2 98; 10.1 101; 10.4 97; 10.3 99; 10.5 96; 10.6 95];
Mdl = varm(2, 2);
estMdl = estimate(Mdl, Y);`
    },
    resources: [
      { title: "statsmodels VAR 文档", url: "https://www.statsmodels.org/stable/vector_ar.html", type: "link" }
    ]
  },
  {
    id: "econometrics-sarima",
    name: "季节性ARIMA模型 (Seasonal ARIMA, SARIMA)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "在 ARIMA 基础上加入季节性自回归、差分与移动平均成分，适合具有明确周期规律的时间序列预测。",
    principles: `1. SARIMA 记作 ARIMA(p,d,q)(P,D,Q)_s。
2. 其中 (P,D,Q)_s 描述周期长度为 s 的季节结构。
3. 常需同时考虑常规差分与季节差分来消除趋势和季节性。
4. 最终模型需通过残差诊断确认拟合是否充分。`,
    scenarios: [
      "月度、季度销售额和客流量等具有明显季节波动的序列预测",
      "节假日周期性需求分析"
    ],
    limitations: [
      "模型参数较多，选阶复杂度高于普通 ARIMA",
      "季节结构变化较快时模型稳定性下降"
    ],
    caseStudy: {
      title: "月度客流量季节性预测",
      description: "景区客流受寒暑假和节假日影响明显，希望预测下一年度各月客流。",
      solution: `1. 检查 12 期季节性规律；
2. 结合常规与季节差分实现平稳化；
3. 拟合 SARIMA 模型；
4. 输出未来 12 个月预测结果。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

series = pd.Series([120, 135, 150, 180, 220, 260, 240, 210, 190, 170, 150, 130] * 2)
model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(model.forecast(steps=3))`,
      matlab: `% MATLAB SARIMA
series = repmat([120 135 150 180 220 260 240 210 190 170 150 130], 1, 2)';
mdl = arima('ARLags',1,'D',1,'MALags',1,'Seasonality',12,'SARLags',12,'SMALags',12);
estMdl = estimate(mdl, series, 'Display', 'off');`
    },
    resources: [
      { title: "SARIMAX 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html", type: "link" }
    ]
  },
  {
    id: "econometrics-cointegration-test",
    name: "协整检验 (Cointegration Test)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "用于判断多个非平稳序列之间是否存在长期稳定均衡关系，是处理中长期经济联动关系的关键方法。",
    principles: `1. 若多个序列各自非平稳，但某线性组合平稳，则称它们协整。
2. Engle-Granger 两步法先回归再检验残差是否平稳。
3. Johansen 方法可同时处理多个变量和多条协整关系。
4. 协整成立时，可进一步构建误差修正模型 ECM。`,
    scenarios: [
      "分析消费与收入、汇率与价格、股价与指数之间的长期均衡关系",
      "处理共同趋势明显但短期偏离存在的宏观经济变量"
    ],
    limitations: [
      "样本量偏小或结构突变存在时，检验结果不稳定",
      "协整不代表严格因果，只表示长期联动"
    ],
    caseStudy: {
      title: "检验房价与收入是否存在长期均衡关系",
      description: "研究者希望判断收入增长能否长期解释房价变动趋势。",
      solution: `1. 先检验房价与收入序列均为同阶单整；
2. 采用 Engle-Granger 方法回归并提取残差；
3. 对残差做平稳性检验；
4. 若协整成立，再建立误差修正模型。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.stattools import coint

x = pd.Series([100, 104, 108, 113, 117, 121, 126, 130])
y = pd.Series([80, 83, 87, 91, 94, 98, 101, 105])
stat, pvalue, _ = coint(x, y)
print(stat, pvalue)`,
      matlab: `% MATLAB 协整检验
x = [100 104 108 113 117 121 126 130]';
y = [80 83 87 91 94 98 101 105]';
[h,pValue] = egcitest([x y]);`
    },
    resources: [
      { title: "statsmodels coint 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html", type: "link" }
    ]
  },
  {
    id: "econometrics-moving-average-method",
    name: "移动平均法 (Moving Average Method)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "通过对相邻时期数据求平均平滑短期波动，是最基础也最直观的时间序列平滑与短期预测方法。",
    principles: `1. k 期移动平均以最近 k 个观测值的平均数作为当前平滑值。
2. 能有效削弱随机扰动，突出序列的趋势变化。
3. 窗口越大，平滑效果越强，但响应新变化越慢。
4. 常用于趋势展示、初步预测和季节调整前预处理。`,
    scenarios: [
      "对销量、库存、流量等高波动序列做基础平滑",
      "作为复杂时序模型前的趋势观察工具"
    ],
    limitations: [
      "对拐点反应滞后，无法捕捉突发结构变化",
      "不适合长期预测和复杂季节模式建模"
    ],
    caseStudy: {
      title: "用 3 期移动平均平滑周销量",
      description: "店铺周销量噪声较大，希望先观察潜在趋势后再做运营决策。",
      solution: `1. 设定 3 周滑动窗口；
2. 计算移动平均序列；
3. 比较原序列与平滑序列；
4. 用最近移动平均值做短期参考预测。`
    },
    code: {
      python: `import pandas as pd
series = pd.Series([50, 53, 49, 60, 64, 63, 66])
ma3 = series.rolling(window=3).mean()
print(ma3)`,
      matlab: `% MATLAB 移动平均法
series = [50 53 49 60 64 63 66]';
ma3 = movmean(series, 3);`
    },
    resources: [
      { title: "移动平均法介绍", url: "https://otexts.com/fpp3/moving-averages.html", type: "link" }
    ]
  },
  {
    id: "econometrics-simple-exponential-smoothing",
    name: "单指数平滑 (Simple Exponential Smoothing)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "对近期观测赋予更高权重的加权平滑方法，适合无明显趋势和季节性的稳定时间序列短期预测。",
    principles: `1. 平滑公式为 S_t = αy_t + (1-α)S_{t-1}。
2. α ∈ (0,1) 控制新数据权重，越大表示越重视最新观测。
3. 预测值通常取为最新平滑值。
4. 适合围绕某稳定水平上下波动的序列。`,
    scenarios: [
      "稳定需求量、日常库存消耗量等无趋势序列预测",
      "短期业务监控中的快速平滑更新"
    ],
    limitations: [
      "不能处理明显趋势和季节性结构",
      "参数选择不合理时容易过度平滑或过度跟随噪声"
    ],
    caseStudy: {
      title: "平稳耗材需求短期预测",
      description: "医院某耗材日消耗量整体较稳定，仅存在随机波动。",
      solution: `1. 设定初始水平与平滑系数；
2. 递推更新平滑值；
3. 将最新平滑值作为下一期预测；
4. 根据误差优化 α。`
    },
    code: {
      python: `from statsmodels.tsa.holtwinters import SimpleExpSmoothing
series = [80, 82, 81, 83, 84, 82, 83]
model = SimpleExpSmoothing(series).fit(smoothing_level=0.3, optimized=False)
print(model.forecast(2))`,
      matlab: `% MATLAB 单指数平滑可通过 smoothdata 或自定义递推实现`
    },
    resources: [
      { title: "Simple Exponential Smoothing", url: "https://www.statsmodels.org/stable/tsa.html", type: "link" }
    ]
  },
  {
    id: "econometrics-double-exponential-smoothing",
    name: "双指数平滑 (Double Exponential Smoothing)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "在单指数平滑基础上同时估计水平项和趋势项，适合存在明显上升或下降趋势但无季节性的时间序列。",
    principles: `1. 又称 Holt 线性趋势法。
2. 分别递推更新水平项 l_t 和趋势项 b_t。
3. 预测式一般为 ŷ_{t+h} = l_t + hb_t。
4. 通过两个平滑参数分别控制水平与趋势更新速度。`,
    scenarios: [
      "持续增长的订单量、产量、用户数等趋势序列预测",
      "无明显季节性但存在线性趋势的业务数据"
    ],
    limitations: [
      "难以处理复杂季节性波动",
      "趋势突变时短期预测可能偏差较大"
    ],
    caseStudy: {
      title: "门店订单量趋势预测",
      description: "门店订单量整体持续增长，希望构建一个比移动平均更灵敏的短期趋势预测模型。",
      solution: `1. 使用 Holt 模型分解水平和趋势；
2. 估计最优平滑参数；
3. 对未来若干期做线性外推；
4. 根据预测误差持续调整参数。`
    },
    code: {
      python: `from statsmodels.tsa.holtwinters import Holt
series = [120, 126, 131, 140, 146, 152, 159]
model = Holt(series).fit(optimized=True)
print(model.forecast(3))`,
      matlab: `% MATLAB 双指数平滑可用 Holt 方法或自定义水平-趋势递推实现`
    },
    resources: [
      { title: "Holt trend method", url: "https://otexts.com/fpp3/holt.html", type: "link" }
    ]
  },
  {
    id: "econometrics-winters-method",
    name: "Winters法 (Holt-Winters Method)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "在水平与趋势之外进一步加入季节项，是处理带趋势和季节性的经典指数平滑模型。",
    principles: `1. Holt-Winters 同时估计水平项、趋势项和季节项。
2. 可分为加法季节模型和乘法季节模型。
3. 当季节波动幅度近似恒定时常用加法形式；随水平变化时常用乘法形式。
4. 预测时综合三类成分进行外推。`,
    scenarios: [
      "月度销售、季度营收、节假日客流等趋势+季节并存的数据预测",
      "比 SARIMA 更强调平滑递推和业务可解释性的场景"
    ],
    limitations: [
      "需要较稳定的季节模式，季节结构剧烈变化时效果下降",
      "参数较多，初始化不佳会影响预测质量"
    ],
    caseStudy: {
      title: "月度销售额季节性预测",
      description: "零售企业销售额既长期增长又存在年度旺淡季循环。",
      solution: `1. 判断季节性是加法还是乘法；
2. 拟合 Holt-Winters 模型；
3. 分解水平、趋势和季节成分；
4. 预测未来若干月销售走势。`
    },
    code: {
      python: `from statsmodels.tsa.holtwinters import ExponentialSmoothing
series = [120, 135, 150, 180, 220, 260, 240, 210, 190, 170, 150, 130] * 2
model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit()
print(model.forecast(3))`,
      matlab: `% MATLAB Winters 法可按三参数指数平滑递推实现，或借助 Econometrics Toolbox 相关函数`
    },
    resources: [
      { title: "Holt-Winters exponential smoothing", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html", type: "link" }
    ]
  },
  {
    id: "econometrics-time-series-decomposition",
    name: "时间序列分解 (Time Series Decomposition)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "将时间序列拆分为趋势、季节和随机扰动成分，有助于理解数据结构并为后续预测模型选择提供依据。",
    principles: `1. 加法模型常写为 y_t = T_t + S_t + R_t。
2. 乘法模型常写为 y_t = T_t × S_t × R_t。
3. 可使用经典分解、STL 分解等方法提取各成分。
4. 分解后可分别分析长期趋势、季节规律和异常扰动。`,
    scenarios: [
      "探索销量、客流、气象等时序数据的结构组成",
      "为季节调整、异常检测和模型选型提供前置分析"
    ],
    limitations: [
      "分解结果依赖方法和周期设定，参数不当会误导解释",
      "强突变和复杂非线性结构下经典分解可能不足"
    ],
    caseStudy: {
      title: "分析月度客流的趋势与季节因素",
      description: "景区希望知道客流增长究竟来自长期趋势还是季节性旺季影响。",
      solution: `1. 指定 12 期季节长度；
2. 对客流序列做加法或 STL 分解；
3. 分别查看趋势、季节和残差成分；
4. 结合结果选择后续预测模型。`
    },
    code: {
      python: `import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

series = pd.Series([120, 135, 150, 180, 220, 260, 240, 210, 190, 170, 150, 130] * 2)
result = seasonal_decompose(series, model='additive', period=12)
print(result.trend.dropna().head())`,
      matlab: `% MATLAB 时间序列分解可结合 movmean、detrend 或 seasonal decomposition 思路实现`
    },
    resources: [
      { title: "seasonal_decompose 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html", type: "link" }
    ]
  },
  {
    id: "econometrics-ransac-robust-regression",
    name: "稳健回归(RANSAC) (RANSAC Robust Regression)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "通过反复随机抽样寻找受异常值影响较小的拟合结果，是含离群点数据中常用的稳健回归方法。",
    principles: `1. RANSAC 反复从样本中随机抽取最小子集拟合基础模型。
2. 根据残差阈值划分内点与外点。
3. 选择拥有最多内点的模型作为候选最优解。
4. 最终常基于内点集合重新估计参数。`,
    scenarios: [
      "含明显异常值的线性关系拟合",
      "传感器误测、图像匹配和工业检测中的稳健建模"
    ],
    limitations: [
      "对阈值和迭代次数较敏感",
      "若内点比例过低，算法稳定性会下降"
    ],
    caseStudy: {
      title: "含异常订单的广告投入回归",
      description: "广告费与销量大致呈线性关系，但促销日产生极端点，普通最小二乘回归受影响明显。",
      solution: `1. 使用 RANSAC 随机拟合多个候选直线；
2. 依据残差筛选内点；
3. 在内点上重新估计回归模型；
4. 比较与 OLS 的稳健性差异。`
    },
    code: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

X = np.array([[1], [2], [3], [4], [20]])
y = np.array([2, 4, 6, 8, 5])
model = RANSACRegressor(estimator=LinearRegression(), random_state=0).fit(X, y)
print(model.estimator_.coef_, model.estimator_.intercept_)`,
      matlab: `% MATLAB RANSAC 稳健回归通常需自定义实现或借助鲁棒拟合工具箱函数`
    },
    resources: [
      { title: "sklearn RANSACRegressor 文档", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html", type: "link" }
    ]
  },
  {
    id: "econometrics-quantile-regression",
    name: "分位数回归 (Quantile Regression)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "直接建模条件分位数而非条件均值，能够揭示自变量对不同分布位置样本的异质性影响。",
    principles: `1. 与 OLS 最小化平方损失不同，分位数回归最小化分位损失函数。
2. τ=0.5 时对应中位数回归。
3. 不同 τ 值下可刻画高位、中位、低位样本受解释变量影响的差异。
4. 对异方差和偏态分布更具适应性。`,
    scenarios: [
      "收入分布、房价分布、风险损失等异质性分析",
      "希望同时研究低分位和高分位群体响应差异的场景"
    ],
    limitations: [
      "结果解释比均值回归更复杂",
      "高维特征和多分位同时估计时计算成本较高"
    ],
    caseStudy: {
      title: "研究教育对不同收入层级的影响",
      description: "教育年限对低收入与高收入群体的边际影响可能不同。",
      solution: `1. 分别设定 0.25、0.5、0.75 分位；
2. 拟合多组分位数回归；
3. 比较教育变量在不同分位上的系数差异；
4. 识别异质性效应。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    'income': [3, 4, 5, 6, 8, 10],
    'edu': [9, 10, 12, 12, 16, 18],
    'exp': [1, 2, 3, 5, 8, 10]
})
model = smf.quantreg('income ~ edu + exp', df).fit(q=0.5)
print(model.params)`,
      matlab: `% MATLAB 分位数回归可借助线性规划思路或第三方工具实现`
    },
    resources: [
      { title: "statsmodels QuantReg 文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html", type: "link" }
    ]
  },
  {
    id: "econometrics-panel-model",
    name: "面板模型 (Panel Data Model)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "结合截面和时间两个维度分析个体差异与时间变化，是处理企业、地区、国家等纵向数据的核心计量框架。",
    principles: `1. 面板数据同时包含个体维度 i 和时间维度 t。
2. 常见模型包括混合回归、固定效应模型和随机效应模型。
3. 固定效应通过个体内变动识别参数，控制不随时间变化的个体异质性。
4. 随机效应适用于个体效应与解释变量不相关的情形，可结合 Hausman 检验选择模型。`,
    scenarios: [
      "研究多地区经济增长、多企业绩效、多学校成绩的纵向变化",
      "需要控制个体不可观测异质性的实证分析"
    ],
    limitations: [
      "数据整理要求较高，缺失和不平衡面板会增加处理难度",
      "固定效应无法估计时间不变变量的系数"
    ],
    caseStudy: {
      title: "分析数字化投入对企业利润的影响",
      description: "研究者拥有多家企业多年财务与投入数据，希望控制企业固有差异后估计投入效果。",
      solution: `1. 整理企业-年份面板数据；
2. 分别拟合固定效应和随机效应模型；
3. 用 Hausman 检验比较两者；
4. 解释核心变量的面板回归系数。`
    },
    code: {
      python: `import pandas as pd
from linearmodels.panel import PanelOLS

panel = pd.DataFrame({
    'firm': ['A', 'A', 'B', 'B'],
    'year': [2022, 2023, 2022, 2023],
    'profit': [10, 12, 8, 9],
    'digital': [3, 4, 2, 3]
}).set_index(['firm', 'year'])
model = PanelOLS.from_formula('profit ~ 1 + digital + EntityEffects', panel)
print(model.fit())`,
      matlab: `% MATLAB 面板模型可通过 fitlm、自定义虚拟变量或计量工具箱扩展实现`
    },
    resources: [
      { title: "linearmodels PanelOLS 文档", url: "https://bashtage.github.io/linearmodels/", type: "link" }
    ]
  },
  {
    id: "econometrics-two-stage-regression",
    name: "两阶段回归 (Two-Stage Regression / 2SLS)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "利用工具变量分两阶段估计内生解释变量的净效应，是解决反向因果和遗漏变量偏误的经典方法。",
    principles: `1. 当解释变量与误差项相关时，OLS 估计会有偏。
2. 第一阶段用工具变量预测内生变量。
3. 第二阶段用预测得到的外生部分解释因变量。
4. 工具变量需满足相关性和外生性两个核心条件。`,
    scenarios: [
      "教育回报、价格弹性、政策评估等存在内生性的问题",
      "需要借助外生冲击或制度变量进行识别的实证研究"
    ],
    limitations: [
      "弱工具变量会导致估计不稳定甚至更差",
      "工具变量外生性往往难以完全验证"
    ],
    caseStudy: {
      title: "估计培训时长对收入的净影响",
      description: "培训时长可能受个人能力影响而内生，研究者选用距离培训机构远近作为工具变量。",
      solution: `1. 用距离变量预测培训时长；
2. 获取培训时长的拟合值；
3. 在第二阶段回归收入；
4. 检查第一阶段强度和识别合理性。`
    },
    code: {
      python: `import pandas as pd
from linearmodels.iv import IV2SLS

df = pd.DataFrame({
    'income': [10, 12, 13, 15, 16],
    'train': [2, 3, 3, 4, 5],
    'distance': [8, 7, 6, 4, 3],
    'age': [22, 25, 27, 30, 34]
})
model = IV2SLS.from_formula('income ~ 1 + age + [train ~ distance]', df).fit()
print(model.summary)`,
      matlab: `% MATLAB 两阶段最小二乘可通过 ivregress 2sls 实现`
    },
    resources: [
      { title: "linearmodels IV2SLS 文档", url: "https://bashtage.github.io/linearmodels/iv/introduction.html", type: "link" }
    ]
  },
  {
    id: "econometrics-gmm-estimation",
    name: "GMM估计 (Generalized Method of Moments)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "利用样本矩条件估计参数，兼具较强的理论通用性与异方差稳健性，是现代计量经济学的重要估计框架。",
    principles: `1. 设模型满足一组矩条件 E[g(Z_t, θ)] = 0。
2. GMM 通过最小化样本矩与理论矩偏离构造目标函数。
3. 权重矩阵的选择影响估计效率，常采用两步 GMM。
4. 可用于过度识别约束检验与复杂结构模型估计。`,
    scenarios: [
      "资产定价、动态面板、工具变量扩展估计",
      "存在异方差或难以写出完整似然函数的模型"
    ],
    limitations: [
      "矩条件设定不当会直接导致估计失真",
      "小样本下表现可能不如大样本理论理想"
    ],
    caseStudy: {
      title: "利用矩条件估计消费模型参数",
      description: "研究者基于欧拉方程构造若干正交条件，希望在不完全指定分布的情况下估计参数。",
      solution: `1. 根据经济理论写出矩条件；
2. 计算样本矩并构造目标函数；
3. 执行一步或两步 GMM；
4. 做 Hansen 过度识别检验。`
    },
    code: {
      python: `import numpy as np
from statsmodels.sandbox.regression.gmm import IV2SLS

# 简化示意：很多 GMM 问题会借助 IV/GMM 类接口实现
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=float)
y = np.array([3, 4, 6, 7], dtype=float)
Z = X.copy()
model = IV2SLS(y, X, Z).fit()
print(model.params)`,
      matlab: `% MATLAB GMM 可通过 gmm 或自定义矩条件优化实现`
    },
    resources: [
      { title: "statsmodels GMM 相关接口", url: "https://www.statsmodels.org/stable/gmm.html", type: "link" }
    ]
  },
  {
    id: "econometrics-did",
    name: "双重差分DID（倍差法） (Difference-in-Differences, DID)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "通过比较处理组与对照组在政策前后的变化差异识别政策效应，是准实验评估中最常见的方法之一。",
    principles: `1. 核心思想是“差中之差”，即处理组变化减去对照组变化。
2. 典型回归项为 Treat、Post 和 Treat×Post。
3. 交互项系数通常解释为平均处理效应。
4. 关键识别假设是平行趋势，即若无政策干预，两组趋势应大体一致。`,
    scenarios: [
      "政策效果评估、平台规则调整评估、门店改造前后效果分析",
      "无法随机试验但拥有前后期与对照组数据的研究"
    ],
    limitations: [
      "平行趋势假设若不成立，结论会失真",
      "同期其他冲击可能污染政策效应识别"
    ],
    caseStudy: {
      title: "评估补贴政策对就业率的影响",
      description: "部分地区实施就业补贴，研究者希望比较试点地区和非试点地区政策前后的就业变化。",
      solution: `1. 构造地区是否试点与政策后虚拟变量；
2. 建立 DID 回归；
3. 解读交互项系数；
4. 用事件研究或图形辅助检验平行趋势。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    'y': [60, 68, 62, 63],
    'treat': [1, 1, 0, 0],
    'post': [0, 1, 0, 1]
})
model = smf.ols('y ~ treat + post + treat:post', data=df).fit()
print(model.params)`,
      matlab: `% MATLAB DID 可通过包含交互项的线性回归实现`
    },
    resources: [
      { title: "Difference-in-Differences 介绍", url: "https://mixtape.scunning.com/09-difference_in_differences", type: "link" }
    ]
  },
  {
    id: "econometrics-tobit-regression",
    name: "Tobit回归 (Tobit Regression)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "用于因变量存在截断或删失现象的回归建模，能同时处理大量堆积在边界上的观测值。",
    principles: `1. 假设潜在变量 y_t^* = X_tβ + ε_t。
2. 实际观测值在阈值内时等于潜在变量，超出阈值则被截断或删失。
3. 参数一般通过极大似然估计获得。
4. 适合支出额、评分下限或上限等边界受限变量。`,
    scenarios: [
      "家庭消费支出中大量零值数据分析",
      "满意度或损失额度存在上下界截断的经济行为研究"
    ],
    limitations: [
      "对分布假设较依赖，误差项设定不当会影响结果",
      "删失与截断机制若更复杂，基础 Tobit 可能不足"
    ],
    caseStudy: {
      title: "分析家庭教育支出决定因素",
      description: "部分家庭教育支出为 0，普通线性回归难以合理处理边界堆积。",
      solution: `1. 将教育支出视为左删失变量；
2. 建立 Tobit 模型；
3. 估计收入、家庭规模等变量影响；
4. 区分对参与概率与支出水平的综合作用。`
    },
    code: {
      python: `print('Python 可用 statsmodels 自定义 MLE 或其他专用包实现 Tobit 回归')`,
      matlab: `% MATLAB Tobit 回归可通过 mle、自定义似然或相关工具箱实现`
    },
    resources: [
      { title: "Tobit model overview", url: "https://en.wikipedia.org/wiki/Tobit_model", type: "link" }
    ]
  },
  {
    id: "econometrics-count-data-regression",
    name: "计数数据回归 (Count Data Regression)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "针对取值为非负整数的计数型因变量建模，常见形式包括 Poisson 回归和负二项回归。",
    principles: `1. Poisson 回归假设 E(Y|X)=Var(Y|X)，并常用对数链接函数。
2. 若数据存在过度离散，可改用负二项回归。
3. 系数通常解释为对期望计数的对数影响。
4. 可扩展到零膨胀模型处理过多零值情况。`,
    scenarios: [
      "事故次数、就诊次数、购买次数、点击次数等计数结果建模",
      "分析单位时间或单位空间内事件发生频次"
    ],
    limitations: [
      "Poisson 的均值等于方差假设常被现实数据违反",
      "过多零值或聚集性较强时需更复杂扩展模型"
    ],
    caseStudy: {
      title: "预测用户月度购买次数",
      description: "平台希望研究会员等级和活动触达次数对月购买笔数的影响。",
      solution: `1. 先拟合 Poisson 回归；
2. 检查是否过度离散；
3. 若有需要改为负二项回归；
4. 解释各变量对期望购买次数的影响。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
    'count': [1, 0, 2, 3, 1, 4],
    'touch': [2, 1, 3, 5, 2, 6],
    'vip': [0, 0, 1, 1, 0, 1]
})
X = sm.add_constant(df[['touch', 'vip']])
model = sm.GLM(df['count'], X, family=sm.families.Poisson()).fit()
print(model.params)`,
      matlab: `% MATLAB 计数数据回归可通过 fitglm(...,'Distribution','poisson') 实现`
    },
    resources: [
      { title: "statsmodels Poisson 回归文档", url: "https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html", type: "link" }
    ]
  },
  {
    id: "econometrics-psm",
    name: "倾向得分匹配分组回归(PSM) (Propensity Score Matching)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "先估计样本接受处理的倾向得分，再进行匹配或分组比较，以缓解样本选择偏差带来的估计偏误。",
    principles: `1. 倾向得分是个体在观测协变量条件下接受处理的概率。
2. 常先用 Logit/Probit 模型估计倾向得分。
3. 再进行最近邻、半径、核匹配或分层匹配。
4. 匹配后比较处理组与对照组结果变量差异以估计处理效应。`,
    scenarios: [
      "项目参与效果评估、用户干预效果分析、医疗方案比较",
      "非随机实验中希望提升处理组与对照组可比性"
    ],
    limitations: [
      "只能控制可观测变量导致的选择偏差，无法解决不可观测偏误",
      "倾向得分模型设定错误会影响匹配质量"
    ],
    caseStudy: {
      title: "评估培训项目对就业率的影响",
      description: "参加培训的人可能本身更积极，直接比较会高估项目效果。",
      solution: `1. 用年龄、学历、经验等变量估计参与培训概率；
2. 对处理组和对照组做匹配；
3. 检验匹配后的协变量平衡性；
4. 在匹配样本上估计平均处理效应。`
    },
    code: {
      python: `import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'treat': [1, 1, 0, 0, 1, 0],
    'age': [24, 28, 23, 27, 31, 30],
    'edu': [12, 16, 12, 14, 18, 15]
})
X = df[['age', 'edu']]
y = df['treat']
ps_model = LogisticRegression().fit(X, y)
df['pscore'] = ps_model.predict_proba(X)[:, 1]
print(df[['treat', 'pscore']])`,
      matlab: `% MATLAB PSM 可先估计倾向得分，再按最近邻或分层策略手动匹配`
    },
    resources: [
      { title: "倾向得分匹配简介", url: "https://mixtape.scunning.com/05-matching_and_subclassification", type: "link" }
    ]
  },
  {
    id: "econometrics-association-analysis",
    name: "关联分析 (Association Analysis)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "用于识别变量或事件之间的共现关系与关联强度，可服务于经济行为模式挖掘和政策联动分析。",
    principles: `1. 关联分析可基于列联表、规则挖掘或相关统计量展开。
2. 在规则挖掘中，支持度、置信度和提升度是核心指标。
3. 在计量场景中也可通过列联分析或离散事件共现结构识别关联模式。
4. 其重点是发现关系结构，而非直接进行因果识别。`,
    scenarios: [
      "消费行为组合分析、政策工具共现分析、问卷选项联动分析",
      "希望从离散事件或分类变量中发现潜在模式"
    ],
    limitations: [
      "强关联不代表因果关系",
      "当变量很多时容易产生大量冗余规则，需要进一步筛选"
    ],
    caseStudy: {
      title: "识别用户购买组合模式",
      description: "平台希望发现哪些服务会被同一类客户同时购买，以支持组合推荐。",
      solution: `1. 整理交易或选择记录；
2. 计算支持度、置信度和提升度；
3. 保留具有业务意义的高质量规则；
4. 转化为营销与推荐策略。`
    },
    code: {
      python: `import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

basket = pd.DataFrame({
    'A': [1, 1, 0, 1],
    'B': [1, 0, 1, 1],
    'C': [0, 1, 1, 1]
})
freq = apriori(basket.astype(bool), min_support=0.5, use_colnames=True)
rules = association_rules(freq, metric='confidence', min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])`,
      matlab: `% MATLAB 关联分析可基于频繁项集与规则指标自行实现`
    },
    resources: [
      { title: "mlxtend association_rules 文档", url: "https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/", type: "link" }
    ]
  },
  {
    id: "econometrics-regression-discontinuity",
    name: "断点回归 (Regression Discontinuity Design, RDD)",
    category: "econometrics",
    categoryName: "📊 计量经济模型",
    summary: "利用处理分配在某阈值处发生跳变的制度规则识别局部因果效应，是高可信度准实验方法之一。",
    principles: `1. 当处理资格由某运行变量是否超过阈值决定时，可比较阈值附近样本。
2. 核心识别思想是阈值附近个体在其他方面近似可比。
3. 可分为锐断点和模糊断点两类设计。
4. 常通过局部线性回归与不同带宽稳健性检验估计处理效应。`,
    scenarios: [
      "奖学金分数线、补贴门槛、年龄政策线等制度阈值分析",
      "难以随机试验但存在明确规则边界的政策评估"
    ],
    limitations: [
      "识别的是阈值附近的局部效应，外推性有限",
      "若样本可操纵运行变量，识别假设会被破坏"
    ],
    caseStudy: {
      title: "评估奖学金政策对升学率的影响",
      description: "学生达到某成绩线即可获得奖学金，研究者希望分析奖学金对后续升学率的局部影响。",
      solution: `1. 以成绩作为运行变量并确定阈值；
2. 选取阈值附近样本；
3. 建立局部线性断点回归；
4. 检查带宽与多项式阶数下结果稳健性。`
    },
    code: {
      python: `import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    'score': [58, 59, 60, 61, 62, 63],
    'y': [70, 71, 78, 80, 81, 82]
})
df['treat'] = (df['score'] >= 60).astype(int)
df['running'] = df['score'] - 60
model = smf.ols('y ~ treat + running + treat:running', data=df).fit()
print(model.params)`,
      matlab: `% MATLAB 断点回归可通过构造阈值虚拟变量与局部回归实现`
    },
    resources: [
      { title: "RDD 简介", url: "https://mixtape.scunning.com/06-regression_discontinuity", type: "link" }
    ]
  },
];

export const COMPETITIONS_DATA: Competition[] = [
  {
    id: "mcm-icm",
    name: "美国大学生数学建模竞赛 (MCM/ICM)",
    alias: "美赛",
    logoChar: "🇺🇸",
    month: 2,
    timeline: {
      signup: "每年 10 月 - 次年 1 月底",
      contest: "每年 2 月初 (共100小时，通常由周五至下周一)",
      results: "每年 4 月底"
    },
    description: "由美国数学及其应用联合会 (COMAP) 主办的全球性顶级数学建模比赛，分为 MCM (数学建模竞赛 - A/B/C 题，偏重连续、离散、数据分析) 和 ICM (交叉学科建模竞赛 - D/E/F 题，偏重运筹学、环境、政策/网络决策)。",
    requirements: "在校普通高校全日制本科生及专科生，3人组队（不可跨校。必须配备一名指导老师），全英文撰写论文及支撑代码说明。",
    valueAnalysis: "全球学术声望最高，一等奖以上极其有利于保研中英文加分、申国外名校。注重想法的新颖包容性、图表渲染精美度、以及论文书写的学术范例。",
    pastPapersUrl: "https://www.comap.com/undergraduate/contests/mcm/",
    targetAudience: "全部人员",
    difficulty: 4
  },
  {
    id: "kc-cup",
    name: "「科创杯」大学生数学建模挑战赛",
    alias: "科创杯",
    logoChar: "💡",
    month: 3,
    timeline: {
      signup: "每年 12 月 - 3 月中旬",
      contest: "每年 3 月中下旬 (3 天 72 小时)",
      results: "每年 5 月左右"
    },
    description: "面向全国高校学生的数学建模基础挑战赛，赛题门槛迈入适中，考核范围多集中于常规的统计预测、运筹优化等基础数学应用，非常适合新手打怪积累经验重构团队合作。",
    requirements: "高校在读专科、本科、研究生，3人组队，无跨校限制，中文撰写。",
    valueAnalysis: "主要作用为团队打底、磨合节奏与热身，在保研中加分较少，但能有效增进编程、写作和建模三人的初期默契，是性价比极高的新手练手平台。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "teddy-cup",
    name: "「泰迪杯」大学生数据挖掘挑战赛",
    alias: "泰迪杯",
    logoChar: "🧸",
    month: 3,
    timeline: {
      signup: "每年 3 月初 - 4 月中旬",
      contest: "每年 4 月底 (共15天，超长实战周期)",
      results: "每年 6 月初"
    },
    description: "由广东泰迪科技股份有限公司主办，中国工业与应用数学学会协调的大数据特征提取与商业挖掘深度对抗赛。主攻真实企业运营中的机器学习、大语言模型、深度图分析等核心大数据难题。",
    requirements: "本专科生及硕士研究生均可。3人组队，需要完成包含程序代码、数据表、答辩PPT和完整商业数据挖掘分析报告的综合方案包。",
    valueAnalysis: "在各大保研高校及互联网大厂中含金量非常高！由于其考察周期长、数据量大、实用性极强，获奖同学的数据分析、机器学习实战能力会得到学界与业界一致首选认可。",
    pastPapersUrl: "https://www.tipdm.org/",
    targetAudience: "进阶",
    difficulty: 4
  },
  {
    id: "mathorcup",
    name: "「MathorCup高校杯」数学建模挑战赛",
    alias: "MathorCup",
    logoChar: "📐",
    month: 4,
    timeline: {
      signup: "每年 12 月 - 4 月中旬",
      contest: "每年 4 月中下旬 (4 天 96 小时)",
      results: "每年 6 月初"
    },
    description: "由中国优选法统筹法与应用数学学会主办，赛题在设计上紧密结合了现代大物流、智能仓储、供应链优化、高保真交通流量调度等业界的核心痛点，具有强烈的最优化及运筹学特色。",
    requirements: "全国高校在校博士、硕士、本科以及大专学生，3人组队，不限学校指导老师约束，拼搏实操性能强。",
    valueAnalysis: "国赛/美赛春季「黄金大热身赛」。规模巨大，每年有上万支甚至数万支队伍参赛，是进行正统格式、高强度拉练的一流平台，适合玩/练习，实战性极强。",
    pastPapersUrl: "http://www.mathorcup.org/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "huadong-cup",
    name: "「华东杯」大学生数学建模邀请赛",
    alias: "华东杯",
    logoChar: "🌉",
    month: 4,
    timeline: {
      signup: "每年 3 月 - 4 月中旬",
      contest: "每年 4 月中下旬 (3 天 72 小时)",
      results: "每年 5 月底"
    },
    description: "由华东地区知名高校发起，复旦大学数学科学学院等多家名校联合组织和评审的精品邀请挑战赛。题目既包含经典工程机理题，亦紧密挂钩最前沿的社会发展与气候减排宏观考评。",
    requirements: "主要是华东各省市及全国知名大学在校生，3人组队（不限名额限制），中文撰写。",
    valueAnalysis: "在长三角及华东各省区认可度极佳，学术氛围非常浓郁，题目逻辑极其严密。其考究程度几乎是中国全国大赛(国赛)的完美翻版，对志在保研华东名校的同学是一块耀眼的敲门砖。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "huazhong-cup",
    name: "「华中杯」大学生数学建模挑战赛",
    alias: "华中杯",
    logoChar: "🏞️",
    month: 4,
    timeline: {
      signup: "每年 3 月 - 4 月底",
      contest: "每年 4 月底 - 5 月初 (3 天 72 小时)",
      results: "每年 6 月中旬"
    },
    description: "由湖北省工业与应用数学学会、华中科大、武汉大学等多家单位联合举办，是中部地区级别最高、历史最久的数模标杆赛事。主攻中等难度的工科寻优、多层动态决策及回归预断问题。",
    requirements: "在校本科及专科生。3人一组，配出高纯度中文建模报告。",
    valueAnalysis: "由于在中部各省（鄂、湘、赣、豫等）具有极高的知名度，许多高校直接将其对齐为省部级二级赛事，提供极佳的学分认可与评优保研奖励。属于备战国赛的最佳阶梯赛。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "rz-china-1",
    name: "「认证杯」中国数学建模网络挑战赛",
    alias: "认证杯",
    logoChar: "🏷️",
    month: 4,
    timeline: {
      signup: "每年 2 月 - 4 月中旬",
      contest: "每年 4 月中旬 (一阶段 3 天) & 5 月下旬 (二阶段 3 天)",
      results: "每年 6 月 - 7 月"
    },
    description: "由内蒙古自治区数学学会主办，数学中国网全力承办的多阶段持续性建模大赛。赛题倡导「回归物理、契合机制、重在实操」的宗旨，帮助数模组经历两阶段磨炼脱胎换骨。",
    requirements: "全国高校在校生均可组队，3人成双结对，可在指导老师名下磨合并递交中、英版格式。",
    valueAnalysis: "对团队代码实现、基础物理力学约束及分析机理等底层知识磨练极为深入，两阶段的推进式赛制便于新手在极低风险和多次试错反馈中完成自我救赎与实操攀爬。",
    pastPapersUrl: "http://www.tzmcm.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "may-day",
    name: "五一数学建模竞赛",
    alias: "五一赛",
    logoChar: "🛠️",
    month: 5,
    timeline: {
      signup: "每年 3 月 - 4 月底",
      contest: "每年 5 月 1 日 - 5 月 4 日 (3 天 72 小时)",
      results: "每年 6 月中旬"
    },
    description: "由中国矿业大学、徐州市工业与应用数学学会联合主办的全国大规模传统数模拉练平台。时间完美对齐五一长假，题目极其务实，如高炉冶炼配比、煤炭选矿路径或矿区边坡位移推断等。",
    requirements: "本专科及研究生皆可，3人一队，中文写作，不卡学校边界。",
    valueAnalysis: "被称作「民间国赛二号」。在全国覆盖范围和品牌影响力极其深远，适合玩/练习。其高容量的参评队伍加上极佳的严谨度，甚至其获奖概率和得分模型就是国赛评选概率的最佳对应指标。",
    pastPapersUrl: "https://mcm.cumt.edu.cn/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "zq-cup",
    name: "「中青杯」全国大学生数学建模竞赛",
    alias: "中青杯",
    logoChar: "🌟",
    month: 5,
    timeline: {
      signup: "每年 3 月 - 5 月中旬",
      contest: "每年 5 月中下旬 (3 天 72 小时)",
      results: "每年 7 月初"
    },
    description: "由中国科协科普 activity、吉林省科技工作者服务中心等发起，旨在激发青少年数理科普学习兴趣的在校赛。试题通常趣味性强，偏重经典运筹网点设计及大众生活经济大分析。",
    requirements: "大专、本科、研究生，3人一队，无跨校门槛限制。",
    valueAnalysis: "赛风相对包容，难度对小白组极为柔和，提供高面额、极精美的奖状，适合新手以此搭建第一份简历并跑通第一次数模完整文档闭环写出。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "csg-cup",
    name: "长三角高校数学建模竞赛",
    alias: "长三角赛",
    logoChar: "🚄",
    month: 5,
    timeline: {
      signup: "每年 4 月 - 5 月中旬",
      contest: "每年 5 月中下旬 (3 天 72 小时)",
      results: "每年 7 月初"
    },
    description: "为落实长三角一体化战略，由浙江省、江苏省、上海市、安徽省数学会携手联合倾力打造的区域重磅赛事。题材通常涉及区域高新交通协同、智慧电网融合等典型一体化规划难题。",
    requirements: "长三角及全国各大院校专科、本科、研究生，3人组队无门槛，简体中文撰写。",
    valueAnalysis: "依托长三角发达的工业与教研背景，其评审标杆与奖状分量在江浙沪等东南沿海省市及对应高校认可度极高，是一项极佳的黄金备战赛，能显著提高考研和江浙保研资助评排分。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 2
  },
  {
    id: "sw-cup",
    name: "数维杯大学生数学建模竞赛 (春季赛)",
    alias: "数维杯",
    logoChar: "🔢",
    month: 5,
    timeline: {
      signup: "每年 3 月 - 5 月上旬",
      contest: "每年 5 月中旬 (3 天 72 小时)",
      results: "每年 6 月底"
    },
    description: "由数维杯数学建模培训中心承办、内蒙古数学学会协办的全国高人气社会练客平台。赛题紧密围绕真实的当下热点话题（如碳达峰、新能源车销量拟合、自适应人工智能风险）展开。",
    requirements: "在校研究生、本专科，3人一队，开放中英双重通道报告投递。",
    valueAnalysis: "在同类民间数模赛中名气响亮，注重算法的创新。竞赛组委会提供丰富的参考文档、答卷诊断及全面的模型辅导课程。对新手打稳算法代码桩极有成效。",
    pastPapersUrl: "http://www.nmmcm.org.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "stat-modeling",
    name: "全国大学生统计建模竞赛",
    alias: "统计建模赛",
    logoChar: "📊",
    month: 5,
    timeline: {
      signup: "每年 3 月 - 5 月初",
      contest: "每年 5 月 - 6 月底 (论文大报告撰写并递交)",
      results: "每年 8 月下旬"
    },
    description: "由中国统计学会主办的国家骨干统计应用类竞赛。有别于 3 天的突击拼杀，本项比赛具有 2 个月左右的漫长精雕细琢跨度，重点考究团队在真实复杂的民生统计、国家宏观数据中提取洞察、构建专业多元统计回归与机器学习深度报告的能力。",
    requirements: "本科生、研究生。3人一队，须由指导老师严格认定后通过高校选拔推选，极其强调原始数据支撑和统计计算真实有效性。",
    valueAnalysis: "由于由国家统计权威学会倾情背书，该赛事的学术权威性与保研声誉在各大财经和经济管理类院、理工统计院属于顶级级别，适合玩/练习，含金量极高！",
    pastPapersUrl: "http://www.cos.org.cn/",
    targetAudience: "全部人员",
    difficulty: 4
  },
  {
    id: "electric-cup",
    name: "全国大学生电工数学建模竞赛",
    alias: "电工杯",
    logoChar: "⚡",
    month: 5,
    timeline: {
      signup: "每年 4 月 - 5 月中旬",
      contest: "每年 5 月下旬 (3 天 72 小时)",
      results: "每年 7 月中旬"
    },
    description: "由中国电机工程学会电工数学专委会组织。赛题设计上电工向、物理强、能源浓度极高（例如电网潮汐能潮流求解、高精度储能充放流控制、风光混合发电输出预测等物理偏微分寻优模型），适合玩/练习。",
    requirements: "在校本科生、专科生。3人一组，具备较好物理机理、微积分常微分偏微分推导基础者优先。",
    valueAnalysis: "数模圈知名的「硬核工科杯」。对于电气工程、动力机械、自动化控制等专业的同学，其含金量和行业认可几乎可直接对齐全国大奖，是应聘国家电网和南方电网的一大保研求职杀手锏。",
    pastPapersUrl: "http://djshuo.com/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "ne-provinces",
    name: "东北三省数学建模联赛",
    alias: "东三省联赛",
    logoChar: "❄️",
    month: 5,
    timeline: {
      signup: "每年 3 月 - 4 月底",
      contest: "每年 5 月初 (10 天左右漫长打磨)",
      results: "每年 6 月中旬"
    },
    description: "由黑龙江、吉林、辽宁三省数学会联合鼎力推举，是北方各高校备战全国大赛的核心桥头堡。属于典型的分布式长考核建模，题目更考究论理推导的学术严密。对经典概率回归、流体阻力等课题深挖彻底。",
    requirements: "北方各省乃至全国高校在校学生，3人组队，通过学校教务大组申报。",
    valueAnalysis: "属于省级联赛中的老牌天花板，在东北和华北高校极富盛名。其获奖人数广、极易斩获省级一二等奖，对保研或期末测评加分作用极佳，是奠定信心的坚硬靠山。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "data-stat-analysis",
    name: "全国大学生数据统计与分析竞赛",
    alias: "数据统计赛",
    logoChar: "📈",
    month: 5,
    timeline: {
      signup: "每年 4 月 - 5 月底",
      contest: "每年 5 月底 (3 天 72 小时)",
      results: "每年 7 月中旬"
    },
    description: "由多家信息统计与分析学会全力指导推动，是侧重考究对于复杂多态表格数据去量纲转换、均值插值清洗、分类关联性度量等最前沿统计操作的入门级实操大赛。",
    requirements: "本专科生、研究生、甚至个人形式亦可参赛，3人一队，通过网络大厅提交。",
    valueAnalysis: "难度极其贴近基础，是磨练 Numpy, Pandas 数据清洗及 Scipy, Statsmodels 卡方/方差/相关系数检验的最佳初级课堂，能迅速树立对特征预处理的深刻认识。",
    pastPapersUrl: "http://www.tzmcm.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "school-contest",
    name: "各高校数学建模校内选拔赛",
    alias: "校内赛",
    logoChar: "🏫",
    month: 6,
    timeline: {
      signup: "每年 4 月 - 5 月底",
      contest: "每年 6 月初 (通常安排在端午或周末三天)",
      results: "每年 6 月中下旬"
    },
    description: "各高校教务处，理学院数学系为了筛选 9 月全国高教社杯(国赛)主力首发队员而独立精巧组织的校内全真实战大演习。赛题通常选用往届国一原题或名师校内自创题，对本校学风量身定做。",
    requirements: "本校在校普通本科、专科及研究生，3人一队，不可跨校。",
    valueAnalysis: "国赛的极关键入场券！许多强力数模高校规定，唯有在校内选拔赛中获得省一、省二标准及以上、或校内排名前 30% 档的队伍，方可获得学校全额报销报名的机会并锁定 9 月国赛名额！",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "新手",
    difficulty: 2
  },
  {
    id: "shenzhen-cup",
    name: "「深圳杯」数学建模挑战赛",
    alias: "深圳杯",
    logoChar: "🌴",
    month: 7,
    timeline: {
      signup: "每年 4 月 - 6 月初",
      contest: "每年 7 月初发放预赛论文 (初赛30天) -> 8 月赴深圳进行总决赛终极线下答辩",
      results: "每年 8 月中旬"
    },
    description: "由中国工业与应用数学学会联合深圳市相关部门主办，是数模届中最令人神往的「公费极客游」高难度大赛。赛题彻底源发于深圳市交通管理、高新技术规划及公共卫生真实的顶层大难题。",
    requirements: "不限学历，博士、硕士、本科无障碍竞争，3~4人一队。免报名费，最终专家组评估初赛排前 20 的队伍，将获全免差旅费赴深圳参与夏令营落地答辩！",
    valueAnalysis: "含金量极大，含金量极高！能入围赴深答辩的唯有顶级极客团队。不仅在各大顶尖 985 高校保研中被直接核定为国家级顶尖奖项，更是荣登国内顶尖团队答辩、直接展示自身数理造诣的无上秀场。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "进阶",
    difficulty: 4
  },
  {
    id: "huashu-cup",
    name: "「华数杯」全国大学生数学建模竞赛",
    alias: "华数杯",
    logoChar: "🏆",
    month: 8,
    timeline: {
      signup: "每年 6 月 - 8 月初",
      contest: "每年 8 月初 (3 天 72 小时，正值暑假备战黄金季)",
      results: "每年 9 月初"
    },
    description: "由中国未来研究会科普工作委员会主办。主创团队多为国赛、美赛资深阅评专家，赛题偏重经典微积分方程求解、机器学习分类挖掘或者非线性排队寻优。是全国数模组暑期集训、以赛代练的不二选择。",
    requirements: "大专、本科、研究生，3人组队，不限专业和学校边界，暑期在线提交。",
    valueAnalysis: "暑期集训完结、验证战力和状态的最好全真阅兵式。在9月国赛前半个月出炉成绩，并提供详细的得分雷达与阅卷专家的匿名诊单。对校准论文排版、修改代码短板起到了至关重要的定音作用。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 2
  },
  {
    id: "cumcm",
    name: "「高教社杯」全国大学生数学建模竞赛 (国赛)",
    alias: "国赛",
    logoChar: "🇨🇳",
    month: 9,
    timeline: {
      signup: "每年 5 月 - 9 月底",
      contest: "每年 9 月中旬 (周四18时至周日20时，共74小时)",
      results: "每年 11 月上旬"
    },
    description: "由中国工业与应用数学学会 (CSIAM) 主办的中国最具含金量、规模最大的高校数学建模竞赛。赛题风格稳重严密、注重真凭实据及准确物理机制建模，共设 A、B（物理工程与机理题）、C（运筹商业大数据问题）等三道本科组题目。",
    requirements: "在校专科、本科学生。3人一队，不可跨校，必须在学校统一报名，简体中文撰写。限制提交代码和辅助材料备检。",
    valueAnalysis: "国内竞赛界无可动摇的保研一级利器！本专科生设计含金量极巅。获得国家一等奖、二等奖将直接收获极高的加分评级。对逻辑链条完备自洽、算法代码精确计算校验有着吹毛求疵般的考核，必须苦战打磨论文排版和细节证明。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 5
  },
  {
    id: "huawei",
    name: "「华为杯」中国研究生数学建模竞赛",
    alias: "研赛",
    logoChar: "🧬",
    month: 9,
    timeline: {
      signup: "每年 6 月 - 9 月",
      contest: "每年 9 月下旬 (120 小时)",
      results: "每年 11 月底"
    },
    description: "中国研究生创新实践系列大赛中参赛规模最大、影响力最深远的全国性高水准赛事。由教育部学位管理与研究生教育司指导，冠名华为。题目通常就是科技大厂里真正的难关或学术界真刀实枪的研究卡脖子突破面，极度磨炼全队心气。",
    requirements: "在读博士生、硕士研究生。部分地区放宽校际交叉，基础需要3人成组拼杀。",
    valueAnalysis: "对读研期间谋取知名大厂（如华为、阿里等）顶层算法工程师及博士推选有着顶重加成。许多一等奖甚至二等奖得主都可以直接豁免各大名厂首轮笔试直通算法技术总监综合面试绿通。",
    pastPapersUrl: "https://cpipc.acge.org.cn/",
    targetAudience: "研究生",
    difficulty: 5
  },
  {
    id: "sw-intl",
    name: "「数维杯」国际大学生数学建模挑战赛",
    alias: "数维国赛",
    logoChar: "🌍",
    month: 11,
    timeline: {
      signup: "每年 9 月 - 11 月中旬",
      contest: "每年 11 月中下旬 (4 天 96 小时)",
      results: "次年 1 月"
    },
    description: "由内蒙古数学学会、数维杯数学建模中心联合倾力打造的冬日高知名度英文数模挑战。题目偏欧美范，考查多目标连续约束优化及复杂经济、交通流拓扑分析。",
    requirements: "在校本专科生、研究生。3人一组，配出高纯度全英文建模报告。",
    valueAnalysis: "该比赛是冬季极具人气的高水准英文演练场。其获奖证书在中外各大保研系统与综合评价打分中拥有相当的知名度和加分评排资格，是在美赛(2月)前熟悉纯正英文数理表达的理想试锋刃。",
    pastPapersUrl: "http://www.nmmcm.org.cn/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "apmcm",
    name: "亚太地区大学生数学建模竞赛 (APMCM)",
    alias: "亚太杯",
    logoChar: "🌏",
    month: 11,
    timeline: {
      signup: "每年 9 月 - 11 月中旬",
      contest: "每年 11 月下旬 (4 天 96 小时)",
      results: "次年 3 月以前"
    },
    description: "由北京图象图形学学会主办的亚太大型数学建模英文赛事。是国内含金量较高、具有成熟组织架构的三大英文赛事之一。赛题在考察范式和风格理念上高度向美国数学建模（美赛）对齐。",
    requirements: "在校大专、本科生，甚至研究生. 3人组队，支持英文撰写。",
    valueAnalysis: "美赛前夕的终极前哨演练！能协助团队极高契合美赛四天百小时的拉练节奏。英文写作、排版及配图的全面校准，在学校的评奖评优加分方面占有一席之地。",
    pastPapersUrl: "http://www.apmcm.org/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "gda-finance",
    name: "大湾区港澳金融数学建模竞赛",
    alias: "大湾区金融赛",
    logoChar: "🏦",
    month: 11,
    timeline: {
      signup: "每年 9 月 - 11 月初",
      contest: "每年 11 月初 (3 天 72 小时)",
      results: "每年 12 月底"
    },
    description: "由粤港澳大湾区多家高新技术协会、精算与金融科技学会联合举办的垂直型建模大赛。深度考验数学在量化投资、资产收益预测、信用评级与金融衍生品对冲方案设计等实战商业场景的应用。",
    requirements: "大专、本科、研究生，3人一队，允许跨校组队。支持中、英文撰写。",
    valueAnalysis: "独具一格的量化实战模型赛。大湾区特批提供现金奖池（特等奖可获上万元现金及名企直通实习推荐绿卡）。对于以后立志向证券基金、量化私募、金融工程方向求职的同学，此证书极具说服力。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "全部人员",
    difficulty: 3
  },
  {
    id: "rz-china-intl",
    name: "「认证杯」数学建模国际赛（小美赛）",
    alias: "小美赛",
    logoChar: "🏷️",
    month: 12,
    timeline: {
      signup: "每年 10 月 - 11 月底",
      contest: "每年 12 月初 (4 天 96 小时)",
      results: "次年 2 月"
    },
    description: "由数学中国全力主推的仿真美赛。其出题方向、文献要求、评阅标准和图表排版挑剔程度全方位高仿美赛。主攻非定性随机大决策、深度高维连续控制等欧美流行赛题风格。",
    requirements: "在校生皆可。3人一队，必须采用全英文写作并提交。",
    valueAnalysis: "公认的美赛最终回演练首选。适合玩/练习，能获得真实的英文评审反馈。由于时空与美赛(2月)高契合，能完美检验并校正英文编译排版宏包及矢量作图代码规范，扫清最致命的美赛盲区。",
    pastPapersUrl: "http://www.tzmcm.cn/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "china-bigdata",
    name: "中国大数据建模年终总决赛",
    alias: "大数据总决赛",
    logoChar: "🏆",
    month: 12,
    timeline: {
      signup: "每年 10 月 - 11 月底通过多项前置周赛赛拔",
      contest: "每年 12 月底进行大尺度大数据终极PK (72小时封锁制)",
      results: "次年 1 月底"
    },
    description: "由中国工业与应用数学学会大数据专委会倾力打造的年终高奖金顶尖建模实战决赛。精算大型互联网流、分布式用户日志处理和高层级预测模型，竞争极端激烈。",
    requirements: "通常需要通过多轮资格周赛积点、才能受邀参与年终最后的顶级对抗。3人一队。",
    valueAnalysis: "代表了当年国内大数据建模最高的纯代码对抗水准！各大科技名企（腾讯、阿里、字节）对该奖项深度认可，提供极为丰厚的年末奖金池、且为获奖极客在冬招 and 春招中提供极速面试权。",
    pastPapersUrl: "https://www.tipdm.org/",
    targetAudience: "进阶",
    difficulty: 4
  }
];

export const PATHWAYS_DATA: LearningPath[] = [
  {
    id: "zero-base",
    name: "零基础入门路线 (一步步迈向国奖)",
    description: "专为完全没有数学建模基础、不会任何高阶编程及数理仿真推导的纯小白打造。以2个月为限带你完成工具启蒙、核心拼图和历年论文仿写。",
    steps: [
      {
        title: "第 1 阶段：工具链与系统工具库集成",
        duration: "第 1 - 2 周",
        description: "搭建数学建模的四大基石——Python (Jupyter toolkit), MATLAB 常用脚本模块，以及 LaTeX 快速高阶数学公式排版，学会跑通最基础的代码环境包。",
        skills: ["Python/Matlab 环境配置", "LaTeX 宏包公式书写", "利用 Markdown 做建模草稿自记"],
        recommendedModels: ["missing-value-treatment"]
      },
      {
        title: "第 2 阶段：核心预测、清洗与寻优模型奠基",
        duration: "第 3 - 5 周",
        description: "攻克预处理特征缩放，多维特征降维（PCA）以及综合优选评价（TOPSIS）。在拿到数据集能够准确处理缺失值与离群异常毛刺。",
        skills: ["IQR异常审查处理", "TOPSIS指标权重综合评价", "Z-score 极低量纲平滑"],
        recommendedModels: ["missing-value-treatment", "topsis", "pca-dimension-reduction"]
      },
      {
        title: "第 3 阶段：数据洗涤、时序外推与机理研究",
        duration: "第 6 - 8 周",
        description: "攻克时间序列 ARIMA 周期预测模型，以及基于少自变量趋势的灰色 GM 机制外推。进行科学的统计推导显著性测试。",
        skills: ["ADF单位根平稳测试", "灰色差分拟合外推", "SPSS/Statsmodels 方差分析"],
        recommendedModels: ["arima-sarima", "anova-correlation-test"]
      },
      {
        title: "第 4 阶段：历年国一的高阶论文仿制与摘要打磨",
        duration: "第 9 - 10 周",
        description: "抽取一至两篇特等奖论文为蓝本。在限定的 72 小时闭关时间内进行对照运行和矢量绘图，完美契合标准学术摘要书写结构。",
        skills: ["国赛排版样式套用", "黄金三段式大奖论文摘要", "高质量 Matplotlib 三维渲染"],
        recommendedModels: ["topsis", "heuristics-ga-sa-pso"]
      }
    ]
  },
  {
    id: "sprint",
    name: "比赛特奖冲刺路线 (美赛一等/国赛一等)",
    description: "面向已经有一到两次建模练习经验，对基础编程不陌生，希望在美赛（HMP/M prize）或国赛（特等奖提名）中真正拿到一流成果的竞逐奇兵。",
    steps: [
      {
        title: "第 1 阶段：多重共线偏回归与计量因果估计",
        duration: "第 1 - 3 周",
        description: "攻克工业级多重共自变量回归（Lasso、岭回归、偏最小二乘回归PLSR）和政策效应。学会评估干预因果系数（DID 和 PSM 多重匹配）。",
        skills: ["PLS相关权重分量提取", "DID 交互项显著性因果检验", "分位数分布回归自适应"],
        recommendedModels: ["regularized-regression", "econometric-causality"]
      },
      {
        title: "第 2 阶段：启发式群体自适应智能最优寻优",
        duration: "第 4 - 6 周",
        description: "熟练编写并操作主流启发式大目标最优搜索库（GA、PSO、模拟退火 SA）。自主在速度和交叉层添加控制算子以打破局部陷阱。",
        skills: ["GA染色体突变交叉调整", "PSO群体自愈飞行轨迹控制", "Metropolis 退火平衡约束"],
        recommendedModels: ["heuristics-ga-sa-pso"]
      },
      {
        title: "第 3 阶段：大数据树结构集成与神经网络炼丹",
        duration: "第 7 - 8 周",
        description: "熟练调用及搭建大规模高级回归算法体系（XGBoost, LightGBM）、BP 密集神经网络和支持向量SVR回归。对表格属性进行贡献度画图解析。",
        skills: ["泰勒损失展开二阶导控制", "BP多隐层误差梯度链反传", "SVR核映射不敏感管道设定"],
        recommendedModels: ["feature-selection-xgboost", "feature-selection-rf-importance"]
      },
      {
        title: "第 4 阶段：超级实战：4天100小时心跳极限测试",
        duration: "第 9 - 10 周",
        description: "模拟完全封锁一切学术咨询。三人成群进行美赛或国赛封闭式演练，极速攻破多目标、缺失大尺度信息、未知机制等疑难杂症，产产出顶配中英文双版论文。",
        skills: ["论文架构敏捷突击", "超高分辨率MATPLOT三维渲染 ", "多作者协作合并与公式完美对齐"],
        recommendedModels: ["heuristics-ga-sa-pso", "topsis"]
      }
    ]
  }
];

export interface AssessmentQuestion {
  id: number;
  question: string;
  options: string[];
  dimension: "abstraction" | "selection" | "foundation" | "algorithm" | "programming" | "expression";
  weight: number;
  correctOptionIndex: number; 
}

export const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: 1,
    question: "在面对一个复杂的城市交通流优化问题时，你的第一步通常是？",
    options: ["直接寻找现成的交通仿真软件进行模拟", "简化城市地图，将其抽象为节点与边的网络流模型", "查阅历年国赛优秀论文看别人怎么做", "手动计算每个红绿灯的配时方案"],
    dimension: "abstraction",
    weight: 1.0,
    correctOptionIndex: 1
  },
  {
    id: 2,
    question: "如果需要评估某项新政策对当地经济指标的「纯粹」影响（排除自然增长），你优先考虑？",
    options: ["普通的多元线性回归", "双重差分模型 (DID)", "灰色预测模型 (GM(1,1))", "层次分析法 (AHP)"],
    dimension: "selection",
    weight: 0.9,
    correctOptionIndex: 1
  },
  {
    id: 3,
    question: "对于一个典型的非凸、含有大量局部最优解的高维寻优问题，以下哪个算法表现更稳健？",
    options: ["单纯形法 (Simplex)", "牛顿下山法 (Newton)", "模拟退火算法 (SA)", "一元梯度下降 (SGD)"],
    dimension: "algorithm",
    weight: 0.8,
    correctOptionIndex: 2
  },
  {
    id: 4,
    question: "在 MATLAB 或 Python 中处理一个 1000x1000 的稀疏矩阵运算时，为保证效率你应该？",
    options: ["使用 for 循环遍历每一个非零元素", "将其转化为稠密数组 (Dense Array) 再运算", "使用专用的稀疏矩阵格式 (如 CSR/CSC)", "不用管，计算资源足够大"],
    dimension: "programming",
    weight: 1.0,
    correctOptionIndex: 2
  },
  {
    id: 5,
    question: "当你的回归模型残差项表现出明显的异方差性（Heteroscedasticity）时，数学上最直接的修正方式是？",
    options: ["增加样本量", "对因变量或自变量取对数变换", "更换为 BP 神经网络", "剔除一半的数据点"],
    dimension: "foundation",
    weight: 0.9,
    correctOptionIndex: 1
  },
  {
    id: 6,
    question: "在一篇优秀的数模论文中，「敏感性分析」部分的主要职责是？",
    options: ["展示模型的计算速度有多快", "证明模型在参数发生微小扰动时的稳定性", "增加页数让论文看起来更专业", "罗列所有的参考文献"],
    dimension: "expression",
    weight: 1.0,
    correctOptionIndex: 1
  },
  {
    id: 7,
    question: "如果要对 10 个不同的投资方案进行「定性叙述型指标」的量化排序，通常用？",
    options: ["变异系数法 (CV)", "模糊综合评价法 (FCE)", "主成分分析 (PCA)", "最小二乘法 (OLS)"],
    dimension: "selection",
    weight: 0.85,
    correctOptionIndex: 1
  },
  {
    id: 8,
    question: "在编程实现遗传算法 (GA) 时，处于什么目的我们要设置「变异概率」？",
    options: ["加快模型收敛速度", "让模型结果看起来更随机", "跳出局部最优解，增加种群多样性", "为了让代码看起来更复杂"],
    dimension: "programming",
    weight: 0.9,
    correctOptionIndex: 2
  },
  {
    id: 9,
    question: "如果一个线性规划问题的约束条件相互冲突导致无可行域，你应该？",
    options: ["强行运行求解器直到报错", "引入松弛变量将其转化为软约束或目标惩罚项", "认为此问题无法建模并放弃", "改变目标函数的系数"],
    dimension: "abstraction",
    weight: 0.95,
    correctOptionIndex: 1
  },
  {
    id: 10,
    question: "以下哪个数学概念是衡量两个随机变量「线性相关强度」的基石？",
    options: ["欧式距离", "协方差与相关系数", "海森矩阵 (Hessian)", "泰勒级数"],
    dimension: "foundation",
    weight: 1.0,
    correctOptionIndex: 1
  },
  {
    id: 11,
    question: "论文中「模型假设」撰写的核心准则是？",
    options: ["假设越多越好，显得严谨", "为了解题方便可以随意假设排除所有干扰项", "基于物理常识或已知背景进行必要的、合理的简化", "直接拷贝题目要求即可"],
    dimension: "expression",
    weight: 0.9,
    correctOptionIndex: 2
  },
  {
    id: 12,
    question: "当你在执行大规模分支定界法 (Branch & Bound) 求解整数规划时，主要的算力瓶颈通常出现在？",
    options: ["内存读取速度", "指数级增长的搜索树深度与节点数", "硬盘存储容量", "图形渲染显存"],
    dimension: "algorithm",
    weight: 0.9,
    correctOptionIndex: 1
  }
];
