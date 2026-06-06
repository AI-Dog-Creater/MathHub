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
      solution: "1. 先利用规则（明暗对比度阈值）预标 60% 数据；\n2. 剩余 40% 高疑似区交由人工精标；\n3. 结合主动学习，最终仅标注 3200 张即达 95% 模型精度。"
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
      solution: "1. 对高基数职业使用目标编码，以逾期率均值替代原始类别；\n2. 对低基数有序类别使用 Ordinal Encoding；\n3. 留一法交叉验证防止编码信息泄露。"
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
      solution: "1. 采用 IQR 剔除 5 组超界异常点，以局部中位数替代；\n2. 对剩余缺失点用随机森林回归插补；\n3. Z-score 二次扫描确认无残留离群点。"
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
      solution: "1. 基于用户 ID 与时间戳哈希精确去重，剔除 120 条；\n2. 剔除全选同一选项的 80 条无效问卷；\n3. 对缺失必填项的 35 条使用众数填充或删除；\n4. 最终保留 4765 条高质量样本。"
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
    principles: `1. 交叉特征：两特征相乘、相除、差值，如"客单价=销售额/购买次数"。
2. 多项式特征：x1², x1·x2, x2² 等，捕获非线性关系。
3. 分箱特征 (Binning)：将连续变量离散化，如年龄段、收入等级。
4. 时间特征：从日期提取年/月/日/星期/是否节假日等语义特征。
5. 聚合特征：groupby 后计算 count/mean/std/max 等统计量。`,
    scenarios: [
      "用户行为序列中提取"首次购买距注册天数"",
      "房价预测中加入"房间数×装修标准"交叉特征"
    ],
    limitations: [
      "过多特征会产生维度灾难，需配合特征筛选使用",
      "交叉特征可能导致过拟合，尤其是低频组合"
    ],
    caseStudy: {
      title: "电商用户购买行为特征构建",
      description: "原始数据仅有用户 ID、购买时间、商品类别，需预测用户下次购买时间间隔。",
      solution: "1. 从时间戳提取小时段（0-6/7-12/13-18/19-23）和是否周末；\n2. 计算用户历史购买频率、活跃天数跨度；\n3. 构造"商品类别交叉购买次数"矩阵；\n4. 最终特征从 5 维扩展到 38 维，模型 AUC 从 0.71 提升至 0.89。"
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
      solution: "1. 对人口和 GDP 增长率分别做 Z-score 标准化；\n2. 标准化后两特征均值为0、方差为1，消除量纲影响；\n3. 在等尺度下参与回归，两类指标贡献度均衡。"
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
3. 约束哑变量：引入"参照组"，其余类别相对于参照组的变化量即为哑变量系数。
4. 与连续变量交互：构造"类别×数值"交叉哑变量，捕获斜率差异。`,
    scenarios: [
      "回归模型中纳入"季节（春夏秋冬）"作为四个哑变量",
      "政策评估中"实验组/对照组"的二元哑变量"
    ],
    limitations: [
      "过多独热编码产生高维稀疏矩阵，增加计算成本",
      "完全编码会导致"虚拟变量陷阱"（完美多重共线性），需删除一列"
    ],
    caseStudy: {
      title: "产品质量等级与销售额关系回归",
      description: "产品等级为"优/良/合格/不合格"四个档次，回归中引入季节和区域哑变量。",
      solution: "1. 季节用四个哑变量，但删除"春季"作为参照组；\n2. 区域同样处理；\n3. 回归结果显示"等级优"相比"等级合格"平均销售额高 23.5 万元（p<0.01）。"
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
      solution: "1. 采用 MICE 方法，以其余 11 项指标预测缺失值；\n2. 迭代 20 轮后取均值；\n3. 填补后数据分布与原始完整数据分布的 KL 散度从 0.18 降至 0.04。"
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
      solution: "1. 采用 Daubechies-4 小波分解到第 6 层；\n2. 硬阈值滤除低能量高频小波系数；\n3. 重构后信噪比 (SNR) 从 12dB 提升至 28dB，位移曲线毛刺完全消失。"
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
      solution: "1. 对薪酬变量做 1% 双侧缩尾；\n2. 缩尾后 R² 从 0.12 提升至 0.67；\n3. 核心解释变量回归系数从 0.003 变为 0.31，方向一致但更稳定。"
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
      solution: "1. 将总体按城市等级（超一线/一线/二线/其他）和年龄段分层，共 16 层；\n2. 按各层实际人口比例分配样本量；\n3. 层内采用简单随机抽样，最终 5000 样本的均值估计精度达到 95% 置信区间 ±1.2%。"
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
      solution: "1. 按广告价值计算逆概率权重：高价广告给 5 倍权重；\n2. 同时引入时间衰减权重，近期样本权重为 1.2^天数差；\n3. 加权后模型的 CTR 预估偏差降低 18%，高价值广告曝光率提升 32%。"
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
      solution: "1. 对年收入取自然对数，分布接近正态；\n2. 对数变换后残差方差齐性检验通过 (White's test p=0.73)；\n3. 回归系数解释为"收入每增加 1%，因变量变化约 β%"。"
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
      solution: "1. 使用 SMOTE 将 487 条欺诈样本扩增到 25 万条（与正常样本 1:1 均衡）；\n2. 用 Borderline-SMOTE 对边界欺诈样本加强合成；\n3. 均衡后模型的欺诈召回率从 12% 提升至 89%，同时保持精度 78%。"
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
      solution: "1. ADASYN 发现恶性样本的边界区域（局部密度低）占比高达 60%；\n2. 在这些低密度区域合成约 200 个新恶性样本；\n3. 模型在恶性样本召回率从 45% 提升至 82%，同时假阳性率控制在 15%。"
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
      solution: "1. 随机下采样 800 张正常样本，与 800 张缺陷样本构成均衡训练集；\n2. 结合集成学习：重复随机下采样 10 次，每次训练一个 CNN 分类器；\n3. 10 个基分类器投票集成，缺陷召回率达 91%，误报率 <3%。"
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
      solution: "1. 对 499.5 万正常交易做 K-Means，聚成 5000 个簇；\n2. 用 5000 个簇质心替代全部正常交易；\n3. 与 5000 条异常样本构成 1:1 均衡训练集；\n4. 相比随机下采样，Cluster Centroids 保留了正常交易的分布特征，模型对新型欺诈识别率提升 27%。"
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
      solution: "1. 先对轻微瑕疵和严重缺陷做 SMOTE 插值扩增到各类 3000 件；\n2. 检测并删除 Tomek Links 中的 420 对边界样本；\n3. 均衡后 SVM 分类器边界清晰度提升，召回率从 62% 提升至 91%，同时误报率降低 40%。"
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
      solution: "1. SMOTE 将缺陷扩增到 8000 张；\n2. ENN 删除 3NN 中超过半数与自身类别不一致的样本，共剔除 1200 张；\n3. 清洗后模型对边界模糊缺陷的识别精确率从 61% 提升至 88%。"
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
      solution: "1. 对表达量做 Z-score 标准化；\n2. 设定方差阈值 0.1，剔除 7840 个低方差基因；\n3. 剩余 12160 个特征进入后续相关性筛选；\n4. 预筛选使后续模型训练速度提升 3.2 倍。"
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
      solution: "1. 训练 500 棵随机森林，用 Permutation Importance 评估；\n2. 选出前 5 关键指标：FEV1/FVC 比值、一秒用力呼气容积、残气量、DLCO、RV/TLC；\n3. 仅用这 5 项构建精简诊断模型，AUC=0.93，与全特征模型 AUC=0.95 相当，但检测成本降低 78%。"
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
      solution: "1. 训练 XGBoost 模型，提取 gain 重要度排序；\n2. 发现前 15 个特征已包含 92% 的信息量；\n3. 精简后模型 AUC 仅从 0.891 降至 0.883，推理速度提升 6 倍；\n4. 客户经理可直接解读 Top 15 特征的决策影响。"
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
      solution: "1. 计算 Pearson 相关系数矩阵，发现固投与 GDP 相关系数高达 0.97；\n2. 剔除相关系数 > 0.9 的一对冗余特征中信息量更少者；\n3. 从 8 个特征精简至 5 个独立指标，VIF 从平均 12.3 降至 2.8，OLS 回归不再发散。"
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
      solution: "1. 使用 sklearn 的 mutual_info_classif 计算各特征与 CTR 的互信息；\n2. 发现"时段×设备"交叉特征的 MI=0.38，远超其他单特征（最高 0.12）；\n3. 按 MI 排序取 Top 30 特征后，LR + 交叉特征模型的 AUC=0.82，超越 Pearson 筛选版本的 AUC=0.75。"
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
      description: "某电商平台收集了用户职业、年龄段、登录设备类型等 12 个类别特征，与"是否购买"标签做卡方检验筛选。",
      solution: "1. 对连续特征（年龄段、购买频次）做分箱离散化；\n2. 计算 12 个特征的卡方值；\n3. 发现"商品浏览深度"（χ²=142.3）和"历史购买次数"（χ²=98.7）的 p<0.001，高度相关；\n4. 保留 Top 8 特征后，朴素贝叶斯分类器精度从 61% 提升至 79%。"
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
      description: "某城市 8 个经济指标预测房价，其中"建筑面积""套内面积""公摊面积"三者高度共线性，OLS 回归系数符号出现倒号。",
      solution: "1. 计算所有特征 VIF，发现建筑面积=42.3、套内面积=38.1、公摊面积=19.7，均超过阈值 10；\n2. 剔除"建筑面积"（VIF 最高），保留套内和公摊；\n3. 重新回归后所有 VIF < 5，系数符号正常，AIC 从 342 降至 287。"
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
      solution: "1. 使用 SVM-RFE，从 100 个标志物出发递归删除；\n2. 10 折交叉验证显示，Top 12 标志物子集达最高 AUC=0.94；\n3. 这 12 个标志物中包括已知的 ER、PR、HER2 核心标志物，以及 4 个新发现的潜在标志物。"
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
      solution: "1. 通过 Z-score 规范各项数值；\n2. 降维发现前三个主分量累加贡献达 89.4%；\n3. 借此计算省市水质空间流形排名。"
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
      solution: "1. 使用 LDA 将 4 维降到 2 维（3 类→最多 2 维）；\n2. 前两个判别轴累加解释了 97.8% 的类间方差；\n3. 二维可视化下三个品种完全可分，分类准确率达 98%。"
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
      solution: "1. 在 784 维原始图像上计算 10-NN 图；\n2. Floyd-Warshall 计算所有点对测地距离；\n3. MDS 降维到 2 维；\n4. 二维可视化中数字 0 和 1 明显分离，数字 3/5/8 在边缘区域高度重叠。"
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
      solution: "1. 设 k=12，构造每个图像的 12 个最近邻图；\n2. 最小二乘计算局部重构系数；\n3. LLE 降到 3 维；\n4. 三个嵌入维度与真实光照和旋转角度高度相关（r>0.9）。"
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
      solution: "1. 采用 RBF 核（γ=0.05）执行 KPCA 降维到 3 维；\n2. 前 3 个核主分量解释了 85% 的非线性特征方差；\n3. 在 3 维核主分量空间中，不同故障模式形成明显分离的簇，AUC=0.97。"
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
4. t-SNE 用 t 分布替代高斯分布解决"拥挤问题"（高维距离差异在低维被压缩）。`,
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
      solution: "1. 先用 PCA 将 20000 维降到 50 维（加速 t-SNE）；\n2. 运行 t-SNE（perplexity=30）降到 2 维；\n3. 可视化中清晰可见 6 个细胞亚群，其中 2 个亚群在空间上邻近（提示分化连续性）。"
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
    id: "arima-sarima",
    name: "应用时间序列预测 (ARIMA & Seasonal ARIMA Series)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "针对时序平稳平滑趋势以及强回归季节特征序列的端到端单变量及协协变量预测武器。",
    principles: `1. ARIMA(p,d,q) 机制：
   具有 d 阶差分的自回归移动平均。
   φ(B)(1-B)^d * y_t = θ(B) * e_t (B为延迟算子，p/q为回归与移动平均截断阶数)
2. SarIMA(P,D,Q)s 复合季节时序：
   引入长滞后 s（如月度为12，季度为4）的二次季节阶层算子，能够强行平准多季节正弦或周期波谷冲击。`,
    scenarios: [
      "月中用电量、年均温度波动、景区日常客流量的周期内时序推演",
      "金融股票高阶均值中轴趋势外推估算"
    ],
    limitations: [
      "单时序模型极度抗拒高维非线性震荡或高频突降冲击（此时往往需额外增加协变量）",
      "历史跨度过小的数据无法完成稳定的 P/Q 估计"
    ],
    caseStudy: {
      title: "某冷链物流淡旺季销售预测",
      description: "收集了连续4年、48个月的乳制品库存数据，由于具有强一月旺、八月淡特性，传统回归偏小。",
      solution: "对48个月时序作一阶差分及12周期季节差分，经过 ADF 确认为平稳后，搭筑 SARIMAX(1,1,1)x(1,1,1)12 模型在最后几个月保持高鲁棒度预测。"
    },
    code: {
      python: `import statsmodels.api as sm
import numpy as np

np.random.seed(0)
t = np.arange(36)
y = 10 + 0.5*t + np.sin(2*np.pi*t/12)*5 + np.random.normal(0, 1, 36)
model = sm.tsa.statespace.SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
res = model.fit(disp=False)
forecast = res.forecast(steps=3)`,
      matlab: `% MATLAB 拟合一阶 ARIMA 并预测 3 步
y = randn(50, 1) + cumsum(0.1*ones(50, 1));
Mdl = arima(1,1,1);
EstMdl = estimate(Mdl, y);
[yF, yMSE] = forecast(EstMdl, 3, 'Y0', y);`
    },
    resources: [
      { title: "statsmodels 季节时序建模全讲解", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "grey-prediction",
    name: "灰色系统预测模型 (Grey System Models: GM(1,1))",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "在数学建模中享有盛名的"小样本预测奇兵"，专门应对样本量极度有限、缺乏物理先验的单调上升或下降体系。",
    principles: `1. 1-AGO 累加平滑：
   将原始噪声严重的波动序列 x0 依次累加，构造出增长单调平顺的序列 x1：x1_k = Σ x0_i。
2. 建立白化方程：
   dx1/dt + a*x1 = b (其中 a 为发展系数，b 为灰作用量)
3. 最小二乘求解：
   采用构建的数据矩阵 B 及常数向量 Y，求出 a,b 系数的最优估计。
4. 累减预测恢复：
   解白化微分方程代入原边界后，对相邻时刻作减法计算，即可恢复回 x0 的预测估计值。`,
    scenarios: [
      "新投产设备累计磨损厚度预测(通常仅有 5-6 批打磨样本)",
      "由于地质地貌变化缓慢导致的地区长期干旱程度估算"
    ],
    limitations: [
      "若序列中存在剧烈、频密的上下反复摆晃（增长非单调），灰色预测的精度会呈断崖式坍塌",
      "主要预测中短期走势，外推周期越长精确度越低"
    ],
    caseStudy: {
      title: "某航天高硬质阀门老化间隙预测",
      description: "在5次阶段性检测中发现阀门间距呈 0.15mm, 0.18mm, 0.22mm, 0.26mm, 0.31mm。面临样本极少瓶颈，求下一周期老化间距。",
      solution: "1. 检验级比检验在允许区间内，通过；\n2. 运用 GM(1,1) 一阶累加和白化拟合，外推第 6 周期得出高可靠度老损预测值。"
    },
    code: {
      python: `import numpy as np

x0 = np.array([1.5, 1.8, 2.2, 2.6, 3.1])
n = len(x0)
x1 = np.cumsum(x0)
B = np.zeros((n-1, 2))
for i in range(n-1):
    B[i, 0] = -0.5 * (x1[i] + x1[i+1])
    B[i, 1] = 1.0
Y = x0[1:].reshape(-1, 1)
a, b = np.linalg.lstsq(B, Y, rcond=None)[0].ravel()
pred = [(x0[0] - b/a) * np.exp(-a*k) + b/a for k in range(n+2)]
x0_pred = np.diff(pred)`,
      matlab: `% MATLAB 灰色一键累加回归拟合
x0 = [1.5 1.8 2.2 2.6 3.1]';
n = length(x0);
x1 = cumsum(x0);
B = [-0.5*(x1(1:end-1)+x1(2:end)), ones(n-1,1)];
u = B \\ x0(2:end);`
    },
    resources: [
      { title: "灰色系统理论体系全解读", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "econometric-causality",
    name: "双重差分DID与倾向评分匹配 (DID / PSM / Granger Causality)",
    category: "prediction",
    categoryName: "📈 预测模型",
    summary: "计量经济学及社会学科中评估某项外在政策/刺激带来真实净收益、净作用的核心评估系统。",
    principles: `1. 双重差分 DID (Difference-in-Differences)：
- 双重比较：实验组与控制组政策前后的两阶段二阶差变化，抵消随大趋势变化的天然增幅。
- 方程：Y = β0 + β1*Treat + β2*Post + β3*(Treat * Post) + ε (β3为核心因果效应)
2. 倾向得分匹配 PSM (Propensity Score Matching)：
- 利用 Logistic 估计每个对象投配实验组的倾向得分，从控制组中筛选最接近控制个体做"镜像对比"。
3. Granger 格兰杰因果检验：
- 通过检验时间序列 Y 在包含其过去项后是否其均方预测残差能显著降低，以辨明 X 是否是对 Y 的统计先兆。`,
    scenarios: [
      "评估某地区推出节能降碳政策对真实生产能耗削减的实证论证",
      "研究高新技术园区设立对周边生产总值拉动的准自然实验"
    ],
    limitations: [
      "DID 最牢不可破的红线基础必须通过"平行趋势检验"",
      "PSM 依赖强内生可塑因子，如果遗漏了至关重要混淆因子可能失效"
    ],
    caseStudy: {
      title: "自豪农业特许试点政策的因果拉力核定",
      description: "在50个县中的20个县试推行新式补给，若简单前后相减，会混淆气候、正常经济周期增量。",
      solution: "1. 采用 PSM 为这 20 个试点县寻找最完美的 20 个常规控制对照县；\n2. 将匹配后的数据套入 DID 双向固定回归，求解交叉主效应 β3 = 7.15%，表明政策的确促进了农业增效。"
    },
    code: {
      python: `import statsmodels.formula.api as smf
import pandas as pd

data = pd.DataFrame({
    'Yield': [10.2, 10.5, 9.8, 11.4, 12.1, 10.1, 10.2, 11.1],
    'Treated': [1, 1, 0, 0, 1, 1, 0, 0],
    'Post': [0, 1, 0, 1, 0, 1, 0, 1]
})
model = smf.ols('Yield ~ Treated * Post', data=data).fit()
print("DID 纯净因果效应 beta_3:", model.params['Treated:Post'])`,
      matlab: `% MATLAB 因果格兰杰断言
[h, p] = gctest(X, Y);  % 确认 Y 是否格兰杰因致 X`
    },
    resources: [
      { title: "statsmodels 计量多阶回归模块说明", url: "https://www.statsmodels.org", type: "link" }
    ]
  },
  {
    id: "fuzzy-eval",
    name: "模糊综合评定模型 (Fuzzy Comprehensive Evaluation)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "利用模糊数学隶属度理论，将边界模糊、定性不确定因素进行高精度定量化转化与客观抉择推导的模型。",
    principles: `1. 确立评价因素集 U 与等级论域集 V：
   U = {u1, u2, ... un} (各指标要素), V = {v1, v2, ... vm} (评价档次类别)。
2. 构建隶属度矩阵 R (n*m)：
   rij 表示对指标 ui 而言，其属于等级 vj 的隶属程度。
3. 模糊合成计算：
   B = A · R（通过 AHP 或熵权法计算指标权重向量 A，结合算子得到最终评价向量 B）。
4. 归一化判定等级归属。`,
    scenarios: [
      "重大工程、建筑物安全隐患及多等级风险评测",
      "水利自然环境毒性等级、水质优劣度复合综合判定"
    ],
    limitations: [
      "隶属函数取值存在一定定性主观因素",
      "当考量因素过多时，最后计算评分易产生"均等化"从而难以拉开明显梯次差距"
    ],
    caseStudy: {
      title: "某电网高压站防爆安全模糊评定",
      description: "受天气、温湿度变化及阀门老化程度影响，该站多项安全性指标多处模糊判定，希望能客观估计安全状态评分。",
      solution: "1. 拟定判定因素，采用高斯隶属函数构造隶属度 R；\n2. 算出各等级概率占比，结合 B = A · R 进行平滑计算；\n3. 高压站落在"完全无安全隐患"的分值为 0.725，处于合规状态。"
    },
    code: {
      python: `import numpy as np

R = np.array([
    [0.7, 0.2, 0.1],
    [0.5, 0.4, 0.1],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2]
])
A = np.array([0.4, 0.2, 0.3, 0.1])
B = np.dot(A, R)
B /= np.sum(B)
print("站安全等级模糊计算输出:", B)`,
      matlab: `% MATLAB 模糊模糊映射
A = [0.4, 0.2, 0.3, 0.1];
R = [0.7, 0.2, 0.1; 0.5, 0.4, 0.1; 0.8, 0.1, 0.1; 0.2, 0.6, 0.2];
B = A * R;`
    },
    resources: [
      { title: "模糊数学在工程综合评价及决策中核心应用", url: "https://www.bilibili.com", type: "video" }
    ]
  },
  {
    id: "coupling-coordination",
    name: "灰色关联与耦合协调度 (GRA & Coupling Coordination Model)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "用于量测多组时序因子曲线几何拓扑形态相似性，并判析两个或多个子系统间良性演化互动对齐深度的复合关联分析系统。",
    principles: `1. 灰色关联分析 (GRA)：
   - 提取参考序列与待对比多序列，消除量纲。
   - 灰色关联系数计算：ξ_i(k) = (minmin + ρ maxmax) / (|x0(k) - xi(k)| + ρ maxmax) (ρ 为分辨系数，一般取 0.5)
   - 对时间历程求均值得到关联度。
2. 耦合协调度模型：
   - 耦合度 C 表征子系统互动强烈势头：
     C = m × { [U1·U2·...·Um] / ( Π (Ui + Uj) )^m }^{1/m}
   - 最终协调等级 D：D = √(C × T) (D 越趋近 1 表明系统同频进步、实现共荣协调)。`,
    scenarios: [
      "评估某省"高新科创投入系统"与"自然生态平衡系统"互动协调趋势分析",
      "工业多道高热物理特征与材料热拉剪强度的几何形态主拉动因子寻找"
    ],
    limitations: [
      "若子系统自身的发展规模极其膨胀或存在虚高，耦合协调度 D 也可能产生"伪高"的良性假象"
    ],
    caseStudy: {
      title: "地区高层城建水平与工业减排耦合协调分析",
      description: "一组合计 10 年的时间面板。希望探寻高污染重工业转型中，经济指标上升与超低排放降噪是否齐头并进。",
      solution: "1. 归一化各自因子；\n2. 计算耦合度 C 与综合得分 T；\n3. D值由 2015 年 0.35 攀爬至 2024 年 0.82。表明由"极低失调拖累状态"演进至"高度良性耦合协调状态"。"
    },
    code: {
      python: `import numpy as np

U1 = np.array([0.3, 0.4, 0.6, 0.75])
U2 = np.array([0.2, 0.35, 0.5, 0.8])
C = 2 * np.sqrt(U1 * U2) / (U1 + U2)
T = 0.5 * U1 + 0.5 * U2
D = np.sqrt(C * T)
print("各时期的耦合协调发展得分 D:", D)`,
      matlab: `% MATLAB 计算二元耦合协调度
U1 = [0.3 0.4 0.6 0.75];
U2 = [0.2 0.35 0.5 0.8];
C = 2 * (U1.*U2).^0.5 ./ (U1 + U2);
T = 0.5*U1 + 0.5*U2;
D = sqrt(C .* T);`
    },
    resources: [
      { title: "GRA 曲线相似匹配理论论文", url: "https://journals.ametsoc.org", type: "paper" }
    ]
  },
  {
    id: "rsr-cv",
    name: "秩和比与变异系数综合评价 (Rank-Sum Ratio & Coefficient of Variation)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "融合统计学变异系数波动量测与秩次转换评价的高效、非参数位置综合排名与多区间归类诊断方法。",
    principles: `1. 变异系数排法 (CV)：
   利用指标波动较大则往往承载更高辨识度和决策信息量的特征进行极度客观权重分配：
   CV_j = σj/μj, 指标客观权重为 wj = CVj/Σ CVi。
2. 秩和比综合评价 (RSR)：
   - 将 n 个评估对象在 m 个指标下无量纲转变为对应的秩次排位值 Rij。
   - 秩和比平均分值 RSRi = Σ Rij / (n × m)。
   - 通过回归将 RSR 分数拟合至正态分布概率分位，进行最终的上、中、下不同档级分级判定。`,
    scenarios: [
      "临床医疗中心、全科门诊安全质量与住院流转效率综合业绩排查",
      "波动极高属性下多部门信用抵御等级客观大分流"
    ],
    limitations: [
      "RSR 全程对数据进行序位转换，使得信息中原始数值差距和倍数关系被完全丢弃，仅反映了序列前后顺序"
    ],
    caseStudy: {
      title: "某公共卫生科多疾病防救治大考排定",
      description: "评比 5 个社区中心。指标多变，希望得出公平无主观情绪且通过统计拟合检验的质量等级归类。",
      solution: "1. 采用变异系数赋权；\n2. 统一将各社区排秩，求 RSR 分数；\n3. 利用 Probit 分级回归，将其严密地划分为三强和两劣两个等级。"
    },
    code: {
      python: `import numpy as np
import scipy.stats as stats

X = np.array([[8, 90, 4.5], [6, 80, 5.0], [9, 95, 3.8], [7, 85, 4.2], [5, 75, 4.0]])
cv = np.std(X, axis=0) / np.mean(X, axis=0)
w = cv / np.sum(cv)
ranks = np.zeros_like(X)
for j in range(X.shape[1]):
    ranks[:, j] = stats.rankdata(X[:, j])
rsr = np.mean(ranks, axis=1) / X.shape[0]
print("RSR values:", rsr)`,
      matlab: `% MATLAB 计算秩和比 RSR
X = [8, 90, 4.5; 6, 80, 5.0; 9, 95, 3.8; 7, 85, 4.2; 5, 75, 4.0];
ranks = zeros(size(X));
for j = 1:size(X, 2)
    [~, ~, ranks(:, j)] = unique(X(:, j));
end
rsr = mean(ranks, 2) / size(X, 1);`
    },
    resources: [
      { title: "秩和比法 (RSR) 综合评价应用指南", url: "https://www.github.com", type: "link" }
    ]
  },
  {
    id: "topsis-entropy",
    name: "熵权 TOPSIS 综合评价 (TOPSIS with Entropy Weighting)",
    category: "evaluation",
    categoryName: "⭐ 综合评价",
    summary: "利用信息熵客观赋权，结合理想解距离的综合评价方法，是数学建模竞赛中使用频率最高的评价模型。",
    principles: `1. 数据标准化：消除量纲，对正向指标 xi' = (xi - min)/(max - min)，对负向指标反向。
2. 比重计算：pij = xij' / Σ xij'。
3. 熵值计算：ej = -Σ pij ln(pij) / ln(n)（j 列的熵值）。
4. 差异系数 dj = 1 - ej，权重 wj = dj / Σ dj。
5. 加权规范化矩阵：Zij = wj · xij'。
6. 计算到正理想解和负理想解的距离，选出综合贴近度 Ci 最大的方案。`,
    scenarios: [
      "多城市营商环境综合排名",
      "投资项目中多指标方案优劣排序"
    ],
    limitations: [
      "熵权法仅基于数据离散程度赋权，与指标实际重要性可能不一致",
      "TOPSIS 依赖极端最优最劣解，若数据存在异常值影响较大"
    ],
    caseStudy: {
      title: "某高新区企业创新能力综合排名",
      description: "5 家候选企业，需从研发投入强度（负向）、科技成果转化率（正向）、人才密度（正向）三个维度进行综合排名。",
      solution: "1. 对研发强度做负向标准化，其余正向标准化；\n2. 计算三指标熵值：ej = [0.91, 0.84, 0.88]；\n3. 权重 wj = [0.31, 0.38, 0.31]；\n4. 计算贴近度：C = [0.72, 0.65, 0.81, 0.58, 0.69]；\n5. 企业综合排名：A > C > E > B > D。"
    },
    code: {
      python: `import numpy as np

# 标准化矩阵（正向）
X = np.array([[80, 65, 0.5], [70, 80, 0.3], [90, 75, 0.6]])
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
# 熵值法权重
P = X_norm / X_norm.sum(axis=0)
ej = -np.sum(P * np.log(P + 1e-10), axis=0) / np.log(len(P))
dj = 1 - ej
wj = dj / dj.sum()
# 加权
Z = wj * X_norm
Z_pos = Z.max(axis=0); Z_neg = Z.min(axis=0)
D_pos = np.sqrt(((Z - Z_pos)**2).sum(axis=1))
D_neg = np.sqrt(((Z - Z_neg)**2).sum(axis=1))
C = D_neg / (D_pos + D_neg)
print("贴近度 C:", C)`,
      matlab: `% MATLAB TOPSIS 综合评价
X = [80 65 0.5; 70 80 0.3; 90 75 0.6];
X_norm = (X - min(X)) ./ (max(X) - min(X) + eps);
P = X_norm ./ sum(X_norm);
ej = -sum(P .* log(P + eps)) / log(size(P,1));
wj = (1 - ej) / sum(1 - ej);
Z = X_norm .* wj;`
    },
    resources: [
      { title: "TOPSIS 熵权法综合评价实战教程", url: "https://www.bilibili.com", type: "video" }
    ]
  },

  // ==================== 6. 统计分析 (stats-analysis) ====================
  {
    id: "anova-correlation-test",
    name: "相关性检验与方差显著性 (Correlation & ANOVA)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "探循多类目以及连续自变量之间是否存在具有高置信度拉动的经典推断模型指南。",
    principles: `1. Pearson 与 Spearman 等级相关：
   - Pearson：r = Cov(X,Y)/(σX·σY)，要求数据正态线性。
   - Spearman：利用元素各值的大小"秩位序次"求 Pearson，容许偏态且契合单调弯曲。
2. 双尾双样本 T 检验：
   - 比较两个正态分布两组均值是否具有显著差别：
     t = (X̄1 - X̄2) / √(Sp² (1/n1 + 1/n2))。
3. 方差分析 (ANOVA)：
   - 衡量单因子或多因子对结果响应引起的波动。
   - 计算组间均方 MSD 与组内均方 MSE 之比 F = MSD / MSE。
   - 若 F > F_crit (即 p-value < 0.05) 则推翻零假设，表明此控制因子起了显著作用。`,
    scenarios: [
      "验证不同配方试剂或不同温度梯度下农作物真实的产量差异是否显著",
      "回归建模前探测诸特征要素与目的标签的相关相关性"
    ],
    limitations: [
      "Pearson 与 T 检验严重挑剔"正态、独立和方差齐性"，有一项破产均会导致极高的第一类错误发生"
    ],
    caseStudy: {
      title: "某电车不同驾驶模式制动距离方差分析",
      description: "在"舒适、节能、运动"三驾驶姿态下各测试15辆车，希望确定动力调节是否影响刹车极限。",
      solution: "1. 运行 Shapiro-Wilk 测试制动正态分布；\n2. 开展单因素 ANOVA，解得 F = 15.2，p-value = 0.00012，确信模式制动差异极显著；\n3. 运用 Tukey HSD 对各对各配方做检验，挑出真正的差异对偶。"
    },
    code: {
      python: `import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.DataFrame({
    'Yield': [80, 85, 78, 92, 95, 91, 70, 75, 72],
    'Brand': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
})
mod = ols('Yield ~ C(Brand)', data=df).fit()
table = sm.stats.anova_lm(mod, typ=2)
print("方差分析结果:\n", table)`,
      matlab: `% MATLAB 单因素与多因方差分析一键执行
y = [80 85 78 92 95 91 70 75 72]';
group = {'A','A','A','B','B','B','C','C','C'}';
[p, tbl, stats] = anova1(y, group);`
    },
    resources: [
      { title: "scipy 统计学与方差库官方接口说明", url: "https://docs.scipy.org/doc/scipy/reference/stats.html", type: "link" }
    ]
  },
  {
    id: "non-parametric",
    name: "卡方与非参数推断检验 (Chi-Square & Non-Parametric Inference)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "当正态及方差齐性假定发生灾难性破产，或面临具有分类、离散特征的定性因果检验指南。",
    principles: `1. 卡方独立性检验 (χ² Independence Test)：
   - 构建多维交叉列联表。
   - 卡方公式：χ² = Σ (Oi - Ei)² / Ei。
   - 对比临界阻断判定属性因子在统计上是否具有绝对耦合关联。
2. Mann-Whitney U 检验：
   非正态下，等比例或多阶替代双样本 T 检验，客观比较两组无量纲秩位大势中轴。
3. Kolmogorov-Smirnov (KS) 测试：
   量测实测分布 Fn(x) 是否服从某一给定的先验理论（如严格服从正态、均匀等偏离边界）。`,
    scenarios: [
      "验证社交平台上性别与偏爱手机品牌之间是否统计学上完全独立相关",
      "不服从正态的特殊噪声信号下两组平均寿命对比"
    ],
    limitations: [
      "卡方列联表中，如果有超 20% 以上的单网格其理论期望频度 < 5，其估计偏差会呈现爆发式扭曲失实"
    ],
    caseStudy: {
      title: "某产品购买意向与地域广告投放关联度推断",
      description: "收集了上千份问卷。两个定性要素："居住城市（南北）"和"偏好型号（大中小）"，要求寻找相关连带关系。",
      solution: "1. 拟出 2×3 关系联表格；\n2. 套用卡方独立检验得出 χ² = 32.1，p = 0.00001；\n3. 否定独立假设，证明地域购买习惯确有关联拉动。"
    },
    code: {
      python: `import scipy.stats as stats
import numpy as np

observed = np.array([[30, 10], [15, 45]])
chi2, p_val, dof, expected = stats.chi2_contingency(observed)
d_stat, ks_p = stats.kstest(np.random.randn(100), 'norm')
print("卡方 p 值:", p_val, "正态两度分布检验:", ks_p)`,
      matlab: `% MATLAB 列联表卡方独立性检验
obs = [30, 10; 15, 45];
[h, p, stats] = crosstab([1*ones(40,1); 2*ones(60,1)], [ones(30,1);2*ones(10,1);ones(15,1);2*ones(45,1)]);`
    },
    resources: [
      { title: "非参数统计检验及卡方交叉学通论", url: "https://github.com", type: "link" }
    ]
  },
  {
    id: "regularized-regression",
    name: "多元正则化回归与偏最小二乘 (Lasso / Ridge / PLSR)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "降服回归分析中的致命阻击者——共多重线性的强力武器，通过引入惩罚项和主成分映射解决病态矩阵。",
    principles: `1. 岭回归 (Ridge Regression)：
   L = ||y - Xw||² + α||w||² (对权重的 L2 正则加入偏置矩形，解决 X'X 奇异不可逆局面)。
2. Lasso 回归 (Lasso Regression)：
   L = ||y - Xw||² + α||w||1 (对参数进行 L1 形式的绝对极值限制，可诱导不相关参数系数快速归 0 以实现高精度自动特征筛选)。
3. 偏最小二乘回归 (PLSR)：
   在自变量和因变量上共同开采主成分空间交互投影得分，专门解决特征远远冗多于样本、极其严重的共线拟合。`,
    scenarios: [
      "高维度、各因子互为孪生、共线性爆炸的综合回归场景",
      "含特征超大且大部分无意义高维金融表格下的鲁棒提取归因预测"
    ],
    limitations: [
      "PLSR 多主元结合后的复合式物理属性不如原始一元 OLS 显露易解释，造成对非专家成员的阅读困难"
    ],
    caseStudy: {
      title: "某地区生态足迹对区域经济多元共线因子解耦",
      description: "在收集的生态与总用气用水指数里，各指标间几乎具有相同的变化走势，多元 OLS 模型系数符号完全反常。",
      solution: "1. 选拔 Lasso 进行自适应筛选收缩；\n2. 在 α=0.15 下把其余8个共线性极高但虚高的杂音因子权重压缩至 0，仅拔高 2 项最强健的核心解释轴，彻底拯救退役模型。"
    },
    code: {
      python: `from sklearn.linear_model import Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
import numpy as np

X = np.random.randn(80, 10)
y = X[:, 0]*2.5 + X[:, 1]*-1.2 + np.random.normal(0, 0.1, 80)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
pls = PLSRegression(n_components=2)
pls.fit(X, y)`,
      matlab: `% MATLAB 执行 PLS 和 Lasso 回归
X = randn(80, 10); y = X(:, 1)*3 + randn(80, 1);
[XL, YL, XS, YS, beta] = plsregress(X, y, 2);
[B, FitInfo] = lasso(X, y);`
    },
    resources: [
      { title: "偏最小二乘 (PLSR) 共线性克服深入白皮书", url: "https://www.github.com", type: "link" }
    ]
  },
  {
    id: "unsupervised-clustering",
    name: "无监督聚类 (K-Means & Hierarchical Clustering)",
    category: "stats-analysis",
    categoryName: "📐 统计分析",
    summary: "缺乏明确类别判定指南下，通过距离、密度或拓扑质心，对多特征样本池进行自主聚合的无监督代表算法。",
    principles: `1. K-Means 质心聚类算法：
   - 随机任定 K 个簇质心 μ1, ... μk。
   - 迭代：给每个样本分发到到它最近的测点 μc。根据均位置重心更新质心 μc = (1/|Sc|) Σ Xi 直到不发生位移。
2. 自底向上分层凝聚聚类 (Hierarchical Clustering)：
   - 将每个体当做独一无二是独立的簇。
   - 提取全两簇中几何间最短的间距。
   - 依次从小到大逐层吸收编排，组筑完整树状谱系关系，由下至上不断连成树族。`,
    scenarios: [
      "市场消费群体或电网大能耗客体自主精细化画像特征归聚",
      "地理遥感图像块及斑点自动多层聚簇聚类"
    ],
    limitations: [
      "K-Means 是横向凸空间几何切割模型，面临流线月牙形、双核非凸群聚时表现极度糟糕",
      "K 值必须带有高度经验给定"
    ],
    caseStudy: {
      title: "某数字产业园入驻大公司用能非参刻画",
      description: "拥有一年的高频用能曲线，并不能确定具体是哪些用电大户、低层平稳户分类。",
      solution: "1. 规范能耗参数。使用手肘法（Elbow-method）定位最优 K = 3；\n2. 开展 K-Means，把 300 家工厂干净地隔离到"白班消耗主力组"、"稳态夜间运营组"和"极低休眠状态组"三大特征阵营中。"
    },
    code: {
      python: `from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

X = np.random.randn(100, 2)
km = KMeans(n_clusters=3, random_state=42)
km.fit(X)
labels = km.labels_
Z = linkage(X, 'ward')`,
      matlab: `% MATLAB K-Means 与分层聚类树绘制
X = rand(100, 2);
[idx, C] = kmeans(X, 3);
Z = linkage(X, 'ward');
dendrogram(Z);`
    },
    resources: [
      { title: "手肘法与聚类轮廓系数判定标准讲座", url: "https://www.youtube.com", type: "video" }
    ]
  },

  // ==================== 7. 规划求解 (programming-solvers) ====================
  {
    id: "heuristics-ga-sa-pso",
    name: "基因进化与群体启发算法 (AGs / PSO / Simulated Annealing)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "应对高阶自变量、不可导、含有繁复分段断层大目标的启发式全局随机搜索套件。",
    principles: `1. 遗传算法 (Genetic Algorithm)：
   - 基因池进行锦标赛选择，交叉互换 (pc)，随机变异 (pm)。依靠繁衍更替实现对多维非凸域大覆盖概率筛选。
2. 粒子群寻优 (PSO)：
   - 模拟集群飞鸟前突探路。速度及轨迹公式：
     vi^{new} = w vi + c1 r1 (pbest_i - xi) + c2 r2 (gbest - xi)
3. 模拟退火 (SA)：
   - 依照 P = exp(-δE/T)，允许按温度概率接受暂时劣质解，以撞破局部极值围堵，寻找真正的物理全局终点谷底。`,
    scenarios: [
      "多地点、大体量车辆最佳配送路线安排(TSP及车辆规划VRP)",
      "多目标不规则工件在不平整切件面中的最紧密排料和下料优化"
    ],
    limitations: [
      "作为随机数渐近探索算法，求解具有高阶精确边界约束的纯线性目标算力极其滞缓"
    ],
    caseStudy: {
      title: "某复合冷链调配多点 TSP 求解",
      description: "在 30 个零售网点间寻找一个总里程数最低、且要避开温控极限约束的最优闭环闭廊路程。",
      solution: "1. 选拔具有极佳位置交叉性质的遗传算法 (GA)；\n2. 设定 200 染色体群落，运行 300 轮遗传突变选择；\n3. 高效从具有 10^32 宇宙般夸张组合数的解空间中，淘洗出了 540km 唯一的低里程最优解。"
    },
    code: {
      python: `import numpy as np

np.random.seed(0)
n_particles = 15
p = np.random.uniform(-5, 5, (n_particles, 2))
v = np.zeros_like(p)
pbest = p.copy()
gbest = p[0]
for gen in range(10):
    v = 0.5*v + 1.5*np.random.rand()*(pbest - p)
    p += v`,
      matlab: `% MATLAB 模拟退火算法优化寻找极值
optim_fun = @(x) x^2 + 4*sin(x);
opts = optimoptions('simulannealbnd','PlotFcns',{@saplotbestf});
[x_opt, fval] = simulannealbnd(optim_fun, 3.0, [-10], [10], opts);`
    },
    resources: [
      { title: "scikit-opt: 启发式进化粒子群求解器一览", url: "https://github.com/guofei9987/scikit-opt", type: "link" }
    ]
  },
  {
    id: "simplex-solvers",
    name: "线性规划单纯形求解器 (Simplex & Boundary Simplex Methods)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "运筹学历久弥新的核心代表。专门应对标准的线性方程与多维度平面约束下的高精准顶点走势锁定器。",
    principles: `1. 计算逻辑：
   - 任何可行解域 Ax ≤ b 必然是一个凸多面体。它的全局最优值必须落在多面体的某个顶点上。
2. 递进寻优：
   - 从可行域中的某个基础可行点(顶点)起航。
   - 依次检查相邻顶点处的检验数 σj。
   - 沿着目标函数值改进最快的"多面体棱边"从一个顶点优雅地滑移到下一个顶点，直到完成唯一精解定案。
3. 修正单纯形 (Bland 规则偏置)：
   防范在面临重合松弛点时反复"死锁并无限循环陷入退化局"，保证有限步内优雅终止解。`,
    scenarios: [
      "化工厂中用固定种类的原材料加工不同市场价格成品时的最大化收益配比",
      "高能物流运输调度中最低成本规划分配"
    ],
    limitations: [
      "在最不幸的病态特定案例（如 Klee-Minty 扭曲多面体）中其时间复杂度会直接恶化至指数级 O(2^n)"
    ],
    caseStudy: {
      title: "某建筑建材公司最优货款配平",
      description: "包含 4 种原石等级，5 级工序约束，要在符合各地区生产规范不等限制的前提下获得最大收益。",
      solution: "1. 拟定标准线性规划矩阵，包含20多项硬性不等式约束；\n2. 搭载 Simplex 单纯形求解，仅用了12步空间多角滑落，就百分百输出了 12.5% 利润暴增的精确定值，非近似启发性所得。"
    },
    code: {
      python: `from scipy.optimize import linprog

c = [-2, -3]
A = [[1, 2], [3, 1]]
b = [8, 12]
res = linprog(c, A_ub=A, b_ub=b, method='simplex')
print("最优解 [x, y]:", res.x, "极大收益:", -res.fun)`,
      matlab: `% MATLAB 求解大尺度线性单纯形法
c = [-2; -3]; A = [1 2; 3 1]; b = [8; 12];
options = optimoptions('linprog','Algorithm','simplex');
[x, fval] = linprog(c, A, b, [], [], zeros(2,1), [], options);`
    },
    resources: [
      { title: "凸多面积单纯拟合推演教材", url: "https://www.youtube.com", type: "link" }
    ]
  },
  {
    id: "interior-point-bfgs",
    name: "内点法与高阶拟牛顿梯度 (Interior Point & BFGS Quasi-Newton)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "极其强大的大型二次及非线性极值逼近器。通过在约束边界内部安置"对数阻尼势垒"引导极值降落，并完美依靠二阶 Hessian 拟合。",
    principles: `1. 内点障碍法 (Interior Point Barrier)：
   - 将原边界约束 xj ≥ 0 转为对数势垒添加在目标后面构成：
     min f(x) - μ Σ ln(xj)。
   - 当参数 μ → 0 时，从可行域的内部顺滑下降至正确的边界答案点，免除多维高阶交点阻碍。
2. 拟牛顿法 (BFGS 式梯度)：
   - 避免巨难解的 Hessian 二阶微分逆 H⁻¹ 阻碍。
   - 采用递增外积更新：Bk+1 = Bk + (yk yk^T)/(yk^T sk) - (Bk sk sk^T Bk)/(sk^T Bk sk) 迭代拟合超面曲率，每步都保持二次二次收敛效率。`,
    scenarios: [
      "千万级别变量且伴随繁复非线性物理阻碍的流体能效最优规划",
      "二次曲面超大投资风险资产马科维茨边界边界逼近"
    ],
    limitations: [
      "若非线性方程极为恶劣、属于非凹凸结构，内点下降必将死机在某一隐蔽的马面鞍点"
    ],
    caseStudy: {
      title: "某不透水大坝混凝土浇筑配料高阶规划",
      description: "含有非线性的材料受力本构关系（带有弹塑性抛物线），要在符合防剪流限制下极小化自重负荷。",
      solution: "1. 将其配平为以弹塑应力为约束障碍的非线性规划；\n2. 引入对数障碍内点策略，利用 BFGS 收敛，不需计算任何导数就以 2.5 秒极优耗时，解得了精准配配重方案。"
    },
    code: {
      python: `from scipy.optimize import minimize
import numpy as np

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 2})
res = minimize(objective, [0.0, 0.0], method='SLSQP', constraints=cons)
print("高度收敛结果 x:", res.x)`,
      matlab: `% MATLAB 搭载内点法一键大二次非线性优化
fun = @(x) (x(1)-1)^2 + (x(2)-2.5)^2;
options = optimoptions('fmincon','Algorithm','interior-point');
[x, fval] = fmincon(fun, [0; 0], [1 -2], [2], [], [], [], [], [], options);`
    },
    resources: [
      { title: "内点拉格朗日势垒下降前沿著作", url: "https://www.springer.com", type: "book" }
    ]
  },
  {
    id: "nelder-mead-monte-carlo",
    name: "无梯度下山单纯形与蒙特卡洛 (Nelder-Mead & Monte Carlo Method)",
    category: "programming-solvers",
    categoryName: "🎯 规划求解",
    summary: "极端场景下的万能解，针对彻底不可微、带有频繁条件分段，乃至只能通过高维随机模拟产生统计大数的极端求解系统。",
    principles: `1. Nelder-Mead (下山单纯形法)：
   - 构建含 n+1 个空间自研顶点的单纯形（如二维空间是个多变三角形，三维是四面体）。
   - 通过：反射、扩张、收缩、缩减。四项渐次搜索操作在曲折面内滚动爬行并收拢、自塌陷定位出不可导的谷底。
2. 蒙特卡洛积分与模拟估计 (Monte Carlo)：
   - 基于大数定律和几何概率，通过在多维边界内播下成千上万均匀随机点，根据统计命中概率逼近超越解析积分公式和极值限界。`,
    scenarios: [
      "物理装置中有分段逻辑判断、不可导不可微的恶劣损失寻优",
      "高维多孔径复杂核废、金融衍生品多路径高维空间随机积分逼近"
    ],
    limitations: [
      "Nelder-Mead 只能寻找局部的极值定位，在多峰严重阻隔下容易因三角形坍缩早死",
      "蒙特卡洛以 1/√N 的低斜速率收敛，需要耗费庞大极其冗长计算点才能压缩最后几位不确定误差"
    ],
    caseStudy: {
      title: "某齿轮加工由于油膜分段间断导致的非连续非解析调优",
      description: "在摩擦磨损模型中，润滑油膜随着工况突变发生高位开关突跳，无法使用任何梯度或牛顿法。",
      solution: "1. 搭建 Nelder-Mead 下山单纯形在三维度参数内探路移动；\n2. 同时调用 1000 万次蒙特卡洛进行热冷损不规则概率积分；\n3. 二者咬合极优找出兼避开关冲击的超高稳态低阻调配方案。"
    },
    code: {
      python: `from scipy.optimize import minimize
import numpy as np

def non_differentiable_objective(x):
    cond = 5.0 if x[0] > 1.5 else 1.0
    return (x[0] - 2)**2 + np.abs(x[1] - 3) * cond

res = minimize(non_differentiable_objective, [0.0, 0.0], method='Nelder-Mead')
print("下山单纯形无需微分寻优结果 x:", res.x)`,
      matlab: `% MATLAB 搭载无梯度 fminsearch (Nelder-Mead)
fun = @(x) (x(1)-2)^2 + abs(x(2)-3)*(1 + 4*(x(1)>1.5));
[x, fval] = fminsearch(fun, [0, 0]);`
    },
    resources: [
      { title: "Nelder-Mead 拓扑形变几何说明视频", url: "https://www.youtube.com", type: "video" }
    ]
  }
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
    name: ""科创杯"大学生数学建模挑战赛",
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
    name: ""泰迪杯"大学生数据挖掘挑战赛",
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
    name: ""MathorCup高校杯"数学建模挑战赛",
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
    valueAnalysis: "国赛/美赛春季"黄金大热身赛"。规模巨大，每年有上万支甚至数万支队伍参赛，是进行正统格式、高强度拉练的一流平台，适合玩/练习，实战性极强。",
    pastPapersUrl: "http://www.mathorcup.org/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "huadong-cup",
    name: ""华东杯"大学生数学建模邀请赛",
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
    name: ""华中杯"大学生数学建模挑战赛",
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
    name: ""认证杯"中国数学建模网络挑战赛",
    alias: "认证杯",
    logoChar: "🏷️",
    month: 4,
    timeline: {
      signup: "每年 2 月 - 4 月中旬",
      contest: "每年 4 月中旬 (一阶段 3 天) & 5 月下旬 (二阶段 3 天)",
      results: "每年 6 月 - 7 月"
    },
    description: "由内蒙古自治区数学学会主办，数学中国网全力承办的多阶段持续性建模大赛。赛题倡导"回归物理、契合机制、重在实操"的宗旨，帮助数模组经历两阶段磨炼脱胎换骨。",
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
    valueAnalysis: "被称作"民间国赛二号"。在全国覆盖范围和品牌影响力极其深远，适合玩/练习。其高容量的参评队伍加上极佳的严谨度，甚至其获奖概率和得分模型就是国赛评选概率的最佳对应指标。",
    pastPapersUrl: "https://mcm.cumt.edu.cn/",
    targetAudience: "新手",
    difficulty: 3
  },
  {
    id: "zq-cup",
    name: ""中青杯"全国大学生数学建模竞赛",
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
    valueAnalysis: "数模圈知名的"硬核工科杯"。对于电气工程、动力机械、自动化控制等专业的同学，其含金量和行业认可几乎可直接对齐全国大奖，是应聘国家电网和南方电网的一大保研求职杀手锏。",
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
    name: ""深圳杯"数学建模挑战赛",
    alias: "深圳杯",
    logoChar: "🌴",
    month: 7,
    timeline: {
      signup: "每年 4 月 - 6 月初",
      contest: "每年 7 月初发放预赛论文 (初赛30天) -> 8 月赴深圳进行总决赛终极线下答辩",
      results: "每年 8 月中旬"
    },
    description: "由中国工业与应用数学学会联合深圳市相关部门主办，是数模届中最令人神往的"公费极客游"高难度大赛。赛题彻底源发于深圳市交通管理、高新技术规划及公共卫生真实的顶层大难题。",
    requirements: "不限学历，博士、硕士、本科无障碍竞争，3~4人一队。免报名费，最终专家组评估初赛排前 20 的队伍，将获全免差旅费赴深圳参与夏令营落地答辩！",
    valueAnalysis: "含金量极大，含金量极高！能入围赴深答辩的唯有顶级极客团队。不仅在各大顶尖 985 高校保研中被直接核定为国家级顶尖奖项，更是荣登国内顶尖团队答辩、直接展示自身数理造诣的无上秀场。",
    pastPapersUrl: "http://www.mcm.edu.cn/",
    targetAudience: "进阶",
    difficulty: 4
  },
  {
    id: "huashu-cup",
    name: ""华数杯"全国大学生数学建模竞赛",
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
    name: ""高教社杯"全国大学生数学建模竞赛 (国赛)",
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
    name: ""华为杯"中国研究生数学建模竞赛",
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
    name: ""数维杯"国际大学生数学建模挑战赛",
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
    name: ""认证杯"数学建模国际赛（小美赛）",
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
        recommendedModels: ["missing-value-treatment", "topsis-entropy", "pca-dimension-reduction"]
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
        recommendedModels: ["topsis-entropy", "heuristics-ga-sa-pso"]
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
        recommendedModels: ["heuristics-ga-sa-pso", "topsis-entropy"]
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
    question: "如果需要评估某项新政策对当地经济指标的'纯粹'影响（排除自然增长），你优先考虑？",
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
    question: "在一篇优秀的数模论文中，'敏感性分析'部分的主要职责是？",
    options: ["展示模型的计算速度有多快", "证明模型在参数发生微小扰动时的稳定性", "增加页数让论文看起来更专业", "罗列所有的参考文献"],
    dimension: "expression",
    weight: 1.0,
    correctOptionIndex: 1
  },
  {
    id: 7,
    question: "如果要对 10 个不同的投资方案进行'定性叙述型指标'的量化排序，通常用？",
    options: ["变异系数法 (CV)", "模糊综合评价法 (FCE)", "主成分分析 (PCA)", "最小二乘法 (OLS)"],
    dimension: "selection",
    weight: 0.85,
    correctOptionIndex: 1
  },
  {
    id: 8,
    question: "在编程实现遗传算法 (GA) 时，处于什么目的我们要设置'变异概率'？",
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
    question: "以下哪个数学概念是衡量两个随机变量'线性相关强度'的基石？",
    options: ["欧式距离", "协方差与相关系数", "海森矩阵 (Hessian)", "泰勒级数"],
    dimension: "foundation",
    weight: 1.0,
    correctOptionIndex: 1
  },
  {
    id: 11,
    question: "论文中'模型假设'撰写的核心准则是？",
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
