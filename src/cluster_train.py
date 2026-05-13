import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score
import joblib
import os
import pymysql
from scipy.sparse import issparse

# 全局定义：核心特征对应回归筛选结果
numeric_features = ['comment_count', 'user_age', 'original_price', 'discount_rate', 'price', 'good_rate', 'is_return']
categorical_features = ['close_style', 'sole_material', 'style', 'brand', 'is_plus']

# MySQL连接配置
from config import (
    MYSQL_HOST,
    MYSQL_USER,
    MYSQL_PWD,
    MYSQL_DB
)

DB_CONFIG = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "password": MYSQL_PWD,
    "database": MYSQL_DB,
    "port": 3306
}
TABLE_NAME = "clean_shoe_train_data"
KMEANS_MODEL_PATH = "../models/kmeans_cluster.pkl"
PREPROCESSOR_PATH = "../models/preprocessor.pkl"
ELBOW_PNG = "../output/images/K-Means肘部法则图.png"
BOXPLOT_PNG = "../output/images/各聚类簇月销量箱线图.png"
SCATTER_PNG = "../output/images/原价-月销量聚类散点图.png"
CLUSTER_COMPARE_CSV = "../output/csv/聚类模型对比结果.csv"
CLUSTER_DATA_CSV = "../output/csv/聚类后商品数据.csv"
CLUSTER_STAT_CSV = "../output/csv/聚类簇特征均值.csv"

# 创建输出目录
os.makedirs(os.path.dirname(KMEANS_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ELBOW_PNG), exist_ok=True)
os.makedirs(os.path.dirname(CLUSTER_COMPARE_CSV), exist_ok=True)

def load_data_from_mysql():
    """从MySQL读取数据，加载回归筛选后的核心特征"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = f"""
            SELECT comment_count, close_style, user_age, original_price, discount_rate, 
                   price, good_rate, sole_material, style, brand, is_plus, is_return, month_sale
            FROM {TABLE_NAME}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  从 MySQL 读取到 {len(df)} 条数据，使用回归筛选的11个核心特征")

        # 数据类型清洗
        if df['discount_rate'].dtype == object:
            df['discount_rate'] = df['discount_rate'].str.replace('%', '').astype(float) / 100.0
        if df['is_plus'].dtype == object:
            df['is_plus'] = df['is_plus'].map({'是': 1, '否': 0}).astype(int)
        if df['is_return'].dtype == object:
            df['is_return'] = df['is_return'].map({'是': 1, '否': 0}).astype(int)

        # 缺失值填充
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_features:
            df[col] = df[col].astype(str)
            df[col].fillna('未知', inplace=True)

        df.rename(columns={'month_sale': 'sales'}, inplace=True)
        return df

    except Exception as e:
        print(f"❌ 连接 MySQL 失败: {e}")
        raise


def train_cluster_model():
    """多模型对比聚类训练，输出聚类结果与评估指标"""
    # 清理旧文件
    for path in [KMEANS_MODEL_PATH, PREPROCESSOR_PATH, ELBOW_PNG, BOXPLOT_PNG, SCATTER_PNG]:
        if os.path.exists(path):
            print(f"  发现旧文件 {path}，删除后重新训练...")
            os.remove(path)

    print("  开始K-Means聚类分析训练（使用回归筛选核心特征）...")
    data = load_data_from_mysql()

    # 特征预处理
    print(f"\n使用回归筛选的核心特征：")
    print(f"   数值特征：{numeric_features}")
    print(f"   类别特征：{categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
    X = data[numeric_features + categorical_features]
    X_encoded = preprocessor.fit_transform(X)
    print(f"\n  特征预处理完成，共 {X_encoded.shape[1]} 维原始特征")

    # 转换为稠密矩阵
    X_dense = X_encoded.toarray() if issparse(X_encoded) else X_encoded

    # 肘部法则确定最佳聚类数
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    sse = []
    k_range = range(2, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_dense)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, sse, 'o-', linewidth=2, markersize=10)
    plt.xlabel('聚类数K', fontsize=14)
    plt.ylabel('SSE（簇内平方和）', fontsize=14)
    plt.title('肘部法则确定最佳聚类数', fontsize=18, pad=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ELBOW_PNG, dpi=300, bbox_inches='tight')
    plt.show()

    # 选择K=4执行最终聚类
    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_dense)
    print(f"\n  完成K={best_k}聚类，各簇样本量：")
    print(data['cluster'].value_counts().sort_index())

    # 统计各簇核心特征
    print("\n" + "=" * 60)
    print(f"各簇核心特征统计（数值特征取平均值）")
    print("=" * 60)
    cluster_num_stats = data.groupby('cluster')[numeric_features + ['sales']].mean().round(2)
    print(cluster_num_stats)

    print("\n" + "各簇类别特征（展示每个簇占比最高的类别）：")
    for col in categorical_features:
        print(f"\n【{col}】：")
        top_cat = data.groupby('cluster')[col].value_counts().groupby(level=0).head(1)
        print(top_cat)

    # 可视化：不同聚类簇的月销量分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='sales', data=data, palette='Set2')
    plt.xlabel('聚类簇编号', fontsize=14)
    plt.ylabel('月销量', fontsize=14)
    plt.title('不同聚类簇的月销量分布对比', fontsize=18, pad=20)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=300, bbox_inches='tight')
    plt.show()

    # 可视化：原价-月销量空间分布
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='original_price', y='sales', hue='cluster', data=data, palette='Set2', s=50, edgecolor='white')
    plt.xlabel('商品原价', fontsize=14)
    plt.ylabel('月销量', fontsize=14)
    plt.title('不同聚类簇在「原价-月销量」空间的分布', fontsize=18, pad=20)
    plt.tick_params(labelsize=12)
    plt.legend(title='cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=300, bbox_inches='tight')
    plt.show()

    # 多聚类模型效果对比
    print("\n" + "=" * 60)
    print("不同聚类模型效果对比（Calinski-Harabasz指数，越高越好）")
    print("=" * 60)
    kmeans_score = calinski_harabasz_score(X_dense, data['cluster'])
    print(f"K-Means (K=4) → CH指数 = {kmeans_score:.2f}")

    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X_dense)
    agg_score = calinski_harabasz_score(X_dense, agg_labels)
    print(f"层次聚类(K=4) → CH指数 = {agg_score:.2f}")

    dbscan = DBSCAN(eps=3, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_dense)
    valid_mask = dbscan_labels != -1
    if len(np.unique(dbscan_labels[valid_mask])) >= 2:
        dbscan_score = calinski_harabasz_score(X_dense[valid_mask], dbscan_labels[valid_mask])
        print(f"DBSCAN → CH指数 = {dbscan_score:.2f}，自动识别簇数 = {len(np.unique(dbscan_labels[valid_mask]))}，异常点个数 = {np.sum(~valid_mask)}")
    else:
        dbscan_score = 0
        print(f"DBSCAN → 无法识别有效簇，不参与对比")

    compare_df = pd.DataFrame([
        {"模型": "K-Means(K=4)", "聚类数": 4, "CH指数": round(kmeans_score, 2)},
        {"模型": "层次聚类(K=4)", "聚类数": 4, "CH指数": round(agg_score, 2)},
        {"模型": "DBSCAN", "聚类数": len(np.unique(dbscan_labels[valid_mask])) if dbscan_score>0 else 0, "CH指数": round(dbscan_score, 2) if dbscan_score>0 else 0}
    ])
    compare_df.to_csv(CLUSTER_COMPARE_CSV, index=False, encoding='utf_8_sig')
    print("=" * 60)
    print(f"  模型对比完成，结果已保存到 {CLUSTER_COMPARE_CSV}")

    # 保存结果与模型
    data.to_csv(CLUSTER_DATA_CSV, index=False, encoding='utf_8_sig')
    cluster_num_stats.to_csv(CLUSTER_STAT_CSV, encoding='utf_8_sig')
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("\n  聚类训练完成，所有结果、图片、模型已保存")
    print(f"   - 模型文件：{KMEANS_MODEL_PATH}, {PREPROCESSOR_PATH}")
    print(f"   - 结果表格：{CLUSTER_COMPARE_CSV}, {CLUSTER_DATA_CSV}, {CLUSTER_STAT_CSV}")
    print(f"   - 可视化图片：{ELBOW_PNG}, {BOXPLOT_PNG}, {SCATTER_PNG}")


if __name__ == '__main__':
    train_cluster_model()