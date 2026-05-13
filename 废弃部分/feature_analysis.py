import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# ========================== MySQL配置（请修改）==========================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "111111",       # 修改为你的 MySQL 密码
    "database": "bishe_nanxie_db",     # 修改为你的数据库名
    "port": 3306
}
TABLE_NAME = "clean_shoe_train_data"

# ========================== 读取数据 ==========================
def load_all_data():
    conn = pymysql.connect(**DB_CONFIG)
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"读取到 {len(df)} 行，{len(df.columns)} 列")
    return df

df = load_all_data()

# ========================== 预处理：字段类型转换 ==========================
# 1. 将 '折扣率' 可能为百分数字符串转为小数，例如 "80%" -> 0.80
if df['折扣率'].dtype == object:
    df['折扣率'] = df['折扣率'].str.replace('%', '').astype(float) / 100.0

# 2. is_return, is_plus 如果是中文“是/否”转为0/1
if df['is_plus'].dtype == object:
    df['is_plus'] = df['is_plus'].map({'是':1, '否':0}).astype(float)
if df['is_return'].dtype == object:
    df['is_return'] = df['is_return'].map({'是':1, '否':0}).astype(float)

# 3. 缺失值简单处理（填充均值/众数）
#    数值型：填充中位数；类别型：填充"未知"
num_cols = ['price', 'original_price', '折扣率', 'comment_count', 'good_rate', 'user_age', 'is_plus', 'is_return']
cat_cols = ['brand', 'upper_material', 'sole_material', 'close_style', '风格', 'season', 'user_gender', 'province', 'order_hour', 'order_time_slot']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df[col].fillna('未知', inplace=True)

# ========================== 方法1：相关性热图（数值字段） ==========================
plt.figure(figsize=(12, 8))
# 选择数值型字段（排除month_sale自身）
num_features = ['price', 'original_price', '折扣率', 'comment_count', 'good_rate', 'user_age', 'is_plus', 'is_return']
num_features = [c for c in num_features if c in df.columns]
corr_matrix = df[num_features + ['month_sale']].corr()
sns.heatmap(corr_matrix[['month_sale']].sort_values(by='month_sale', ascending=False), annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('数值字段与月销量的相关系数')
plt.tight_layout()
plt.show()

# ========================== 方法2：类别字段箱线图（选前几个最相关的） ==========================
# 计算每个类别字段的 ANOVA F值或直接看中位数差异
from sklearn.feature_selection import f_classif
# 先将类别字段编码为数值
cat_encoded = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        cat_encoded[col] = le.fit_transform(df[col])

# 计算F值（用month_sale作为连续目标）
X_cat = pd.DataFrame(cat_encoded)
y = df['month_sale']
f_values, p_values = f_classif(X_cat, y)
f_series = pd.Series(f_values, index=X_cat.columns).sort_values(ascending=False)
print("类别字段的F值（越大表示组间差异越明显）：")
print(f_series)

# 对F值最高的2~3个字段画箱线图
top_cats = f_series.head(3).index
for col in top_cats:
    plt.figure(figsize=(10, 6))
    # 限制显示常见类别，避免图太乱
    top_groups = df[col].value_counts().head(10).index
    sub_df = df[df[col].isin(top_groups)]
    sns.boxplot(x=col, y='month_sale', data=sub_df)
    plt.xticks(rotation=30)
    plt.title(f'不同 {col} 的月销量分布')
    plt.tight_layout()
    plt.show()

# ========================== 方法3：决策树特征重要性（全部特征） ==========================
# 将类别特征进行标签编码（简单，树模型对顺序不敏感）
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 准备X
feature_cols = num_features + cat_cols
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].copy()
y = df['month_sale']

# 对类别列用OneHot编码（决策树也可以直接标签编码，但为了公平，我们用OneHot）
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [c for c in num_features if c in df.columns]),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), [c for c in cat_cols if c in df.columns])
    ])
X_encoded = preprocessor.fit_transform(X)

# 训练浅层决策树（防止过拟合，同时特征重要性可靠）
dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)
dt.fit(X_encoded, y)

# 获取特征名字
cat_ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out([c for c in cat_cols if c in df.columns])
all_feature_names = [c for c in num_features if c in df.columns] + list(cat_ohe_cols)

importance = pd.Series(dt.feature_importances_, index=all_feature_names).sort_values(ascending=False)
print("\n决策树特征重要性（Top 20）：")
print(importance.head(20))

# 画图
plt.figure(figsize=(10, 8))
importance.head(15).plot.barh()
plt.title('决策树特征重要性 (Top 15)')
plt.tight_layout()
plt.show()