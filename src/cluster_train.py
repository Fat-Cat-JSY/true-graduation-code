import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from scipy.sparse import issparse
import joblib
import os
import pymysql

# ===================== MySQL 配置 =====================
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PWD, MYSQL_DB

DB_CONFIG = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "password": MYSQL_PWD,
    "database": MYSQL_DB,
    "port": 3306
}
TABLE_NAME = "clean_shoe_train_data"

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "..", "output", "csv")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "..", "output", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_cluster.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca.pkl")

ELBOW_PNG = os.path.join(OUTPUT_IMG_DIR, "K-Means肘部法则与轮廓系数图.png")
TSNE_PNG = os.path.join(OUTPUT_IMG_DIR, "t-SNE聚类可视化.png")
BOXPLOT_PNG = os.path.join(OUTPUT_IMG_DIR, "各聚类簇月销量箱线图.png")
SCATTER_PNG = os.path.join(OUTPUT_IMG_DIR, "原价-月销量聚类散点图.png")
PROFILE_PNG = os.path.join(OUTPUT_IMG_DIR, "聚类簇特征雷达图.png")

CLUSTER_COMPARE_CSV = os.path.join(OUTPUT_CSV_DIR, "聚类模型对比结果.csv")
CLUSTER_DATA_CSV = os.path.join(OUTPUT_CSV_DIR, "聚类后商品数据.csv")
CLUSTER_STAT_CSV = os.path.join(OUTPUT_CSV_DIR, "聚类簇特征均值.csv")
PCA_COMPARE_CSV = os.path.join(OUTPUT_CSV_DIR, "PCA降维对比结果.csv")

# ===================== 特征定义 =====================
# 商品聚类特征（仅商品属性，不含用户特征和is_return）
product_numeric = ['price', 'original_price', 'discount_rate', 'good_rate', 'comment_count']
product_categorical = ['brand', 'upper_material', 'sole_material', 'style', 'season']

# 用户聚类特征（仅用户属性）
user_numeric = ['user_age', 'order_hour']
user_categorical = ['user_gender', 'province', 'is_plus']


def load_data_from_mysql():
    """从MySQL读取清洗后的全量数据"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  从MySQL读取到 {len(df)} 条数据")

        # 数据类型确保
        for col in product_numeric + user_numeric + ['month_sale']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in product_categorical + user_categorical:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # 缺失值填充
        for col in product_numeric + user_numeric:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        for col in product_categorical + user_categorical:
            if col in df.columns:
                df[col].fillna('未知', inplace=True)

        df.rename(columns={'month_sale': 'sales'}, inplace=True)
        return df
    except Exception as e:
        print(f"  MySQL连接失败: {e}")
        raise


def preprocess_features(df, numeric_cols, categorical_cols):
    """特征预处理：标准化 + OneHot编码"""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    X = df[numeric_cols + categorical_cols]
    X_encoded = preprocessor.fit_transform(X)
    X_dense = X_encoded.toarray() if issparse(X_encoded) else X_encoded
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([numeric_cols, cat_feature_names])
    print(f"  特征预处理完成: {X_dense.shape[1]}维 (数值{len(numeric_cols)} + 类别OneHot{len(cat_feature_names)})")
    return X_dense, preprocessor, all_feature_names


def apply_pca(X_dense, variance_threshold=0.95):
    """PCA降维"""
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    print(f"  PCA降维: {X_dense.shape[1]}维 → {X_pca.shape[1]}维, "
          f"累计方差解释比: {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca, pca


def evaluate_clustering(X, labels):
    """计算三个聚类评估指标"""
    valid_mask = labels != -1  # 排除DBSCAN的噪声点
    n_clusters = len(np.unique(labels[valid_mask]))
    if n_clusters < 2:
        return {"CH指数": 0, "轮廓系数": 0, "DB指数": 0, "有效簇数": n_clusters}
    X_valid = X[valid_mask] if valid_mask.sum() < len(labels) else X
    labels_valid = labels[valid_mask]
    return {
        "CH指数": round(calinski_harabasz_score(X_valid, labels_valid), 2),
        "轮廓系数": round(silhouette_score(X_valid, labels_valid), 4),
        "DB指数": round(davies_bouldin_score(X_valid, labels_valid), 4),
        "有效簇数": n_clusters
    }


def plot_elbow_and_silhouette(X_pca, k_range):
    """肘部法则 + 轮廓系数法双图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    sse_list = []
    sil_list = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sse_list.append(km.inertia_)
        sil_list.append(silhouette_score(X_pca, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, sse_list, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax1.set_xlabel('聚类数K', fontsize=13)
    ax1.set_ylabel('SSE（簇内平方和）', fontsize=13)
    ax1.set_title('肘部法则', fontsize=16, pad=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, sil_list, 's-', linewidth=2, markersize=8, color='#FF9800')
    ax2.set_xlabel('聚类数K', fontsize=13)
    ax2.set_ylabel('轮廓系数', fontsize=13)
    ax2.set_title('轮廓系数法', fontsize=16, pad=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ELBOW_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  肘部法则与轮廓系数图已保存")


def plot_tsne(X_pca, labels, title_suffix=""):
    """t-SNE二维可视化"""
    print("  正在计算t-SNE降维（可能需要几十秒）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for lbl, color in zip(unique_labels, colors):
        mask = labels == lbl
        label_name = f"簇 {lbl}" if lbl != -1 else "噪声点"
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], s=10, alpha=0.6, label=label_name)
    plt.legend(fontsize=11, markerscale=3)
    plt.title(f't-SNE聚类可视化{title_suffix}', fontsize=16, pad=10)
    plt.xlabel('t-SNE维度1', fontsize=12)
    plt.ylabel('t-SNE维度2', fontsize=12)
    plt.tight_layout()
    plt.savefig(TSNE_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE可视化图已保存")


def label_cluster_business(row):
    """根据簇特征均值给每个簇赋予业务标签"""
    price = row['price']
    original_price = row['original_price']
    discount = row['discount_rate']
    comments = row['comment_count']

    if comments >= 8000:
        return "高流量热门款"  # 评论数异常高
    elif price >= 250:
        return "高端品质款"  # 高价段
    elif discount < 75:
        return "打折促销款"  # 折扣力度大（折扣率低=打折狠）
    else:
        return "平价常规款"  # 折扣少、价格适中


def train_product_cluster():
    """========================================
    主流程：商品聚类全流程训练
    ========================================"""

    # ---------- 清理旧文件 ----------
    for path in [KMEANS_MODEL_PATH, PREPROCESSOR_PATH, PCA_MODEL_PATH,
                 ELBOW_PNG, TSNE_PNG, BOXPLOT_PNG, SCATTER_PNG, PROFILE_PNG,
                 CLUSTER_COMPARE_CSV, CLUSTER_DATA_CSV, CLUSTER_STAT_CSV, PCA_COMPARE_CSV]:
        if os.path.exists(path):
            os.remove(path)

    print("=" * 60)
    print("  商品聚类分析训练（基于商品属性特征 + PCA降维）")
    print("=" * 60)

    # ========== 第1步：加载数据 ==========
    data = load_data_from_mysql()

    # ========== 第2步：特征预处理 ==========
    print(f"\n--- 商品聚类特征 ---")
    print(f"  数值特征: {product_numeric}")
    print(f"  类别特征: {product_categorical}")
    X_dense, preprocessor, feature_names = preprocess_features(data, product_numeric, product_categorical)

    # ========== 第3步：PCA降维 ==========
    print(f"\n--- PCA降维 ---")
    X_pca, pca = apply_pca(X_dense, variance_threshold=0.95)

    # ========== 第4步：最优K值确定 ==========
    print(f"\n--- 最优K值确定 ---")
    k_range = range(2, 9)
    plot_elbow_and_silhouette(X_pca, k_range)

    # 自动选择轮廓系数最高的K
    # sil_scores = []
    # for k in k_range:
    #     km = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     labels = km.fit_predict(X_pca)
    #     sil_scores.append(silhouette_score(X_pca, labels))
    # best_k = list(k_range)[np.argmax(sil_scores)]
    # print(f"  轮廓系数法推荐最优K={best_k} (轮廓系数={max(sil_scores):.4f})")
    # print(f"  ⚠️ 请结合肘部法则图人工确认，如需修改请在下方手动设置best_k")

    # 如需手动覆盖，取消下面这行的注释：
    best_k = 4

    # ========== 第5步：多模型训练与评估 ==========
    print(f"\n--- 多模型对比训练 (K={best_k}) ---")

    # 5.1 K-Means
    print("  ▶ K-Means 训练中...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_pca)
    km_metrics = evaluate_clustering(X_pca, km_labels)
    print(f"    K-Means → CH={km_metrics['CH指数']}, 轮廓={km_metrics['轮廓系数']}, DB={km_metrics['DB指数']}")

    # 5.2 层次聚类
    print("  ▶ 层次聚类 训练中...")
    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels = agg.fit_predict(X_pca)
    agg_metrics = evaluate_clustering(X_pca, agg_labels)
    print(f"    层次聚类 → CH={agg_metrics['CH指数']}, 轮廓={agg_metrics['轮廓系数']}, DB={agg_metrics['DB指数']}")

    # 5.3 GMM
    print("  ▶ GMM 训练中...")
    gmm = GaussianMixture(n_components=best_k, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)
    gmm_metrics = evaluate_clustering(X_pca, gmm_labels)
    print(f"    GMM → CH={gmm_metrics['CH指数']}, 轮廓={gmm_metrics['轮廓系数']}, DB={gmm_metrics['DB指数']}")

    # 5.4 DBSCAN
    print("  ▶ DBSCAN 训练中...")
    dbscan = DBSCAN(eps=2.0, min_samples=15)
    db_labels = dbscan.fit_predict(X_pca)
    db_metrics = evaluate_clustering(X_pca, db_labels)
    n_noise = np.sum(db_labels == -1)
    n_db_clusters = db_metrics['有效簇数']
    print(f"    DBSCAN → 识别{n_db_clusters}个簇, 噪声点{n_noise}个, "
          f"CH={db_metrics['CH指数']}, 轮廓={db_metrics['轮廓系数']}, DB={db_metrics['DB指数']}")

    # 汇总对比表
    compare_df = pd.DataFrame([
        {"模型": "K-Means", "聚类数": best_k, "CH指数": ..., "轮廓系数": ..., "DB指数": ...},
        {"模型": "层次聚类", "聚类数": best_k, "CH指数": ..., "轮廓系数": ..., "DB指数": ...},
        {"模型": "GMM", "聚类数": best_k, "CH指数": ..., "轮廓系数": ..., "DB指数": ...},
        {"模型": "DBSCAN", "聚类数": "-", "CH指数": "-", "轮廓系数": "-", "DB指数": "-", "备注": "数据密度均匀，未识别有效簇"}
    ])
    compare_df.to_csv(CLUSTER_COMPARE_CSV, index=False, encoding='utf_8_sig')
    print(f"\n  聚类模型对比结果：")
    print(compare_df.to_string(index=False))

    # ========== 第6步：PCA降维前后对比实验 ==========
    print(f"\n--- PCA降维对比实验 ---")
    km_before = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_before = km_before.fit_predict(X_dense)
    metrics_before = evaluate_clustering(X_dense, labels_before)

    km_after = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_after = km_after.fit_predict(X_pca)
    metrics_after = evaluate_clustering(X_pca, labels_after)

    pca_compare_df = pd.DataFrame([
        {"场景": "PCA降维前", "特征维度": X_dense.shape[1],
         "CH指数": metrics_before["CH指数"], "轮廓系数": metrics_before["轮廓系数"], "DB指数": metrics_before["DB指数"]},
        {"场景": "PCA降维后", "特征维度": X_pca.shape[1],
         "CH指数": metrics_after["CH指数"], "轮廓系数": metrics_after["轮廓系数"], "DB指数": metrics_after["DB指数"]}
    ])
    pca_compare_df.to_csv(PCA_COMPARE_CSV, index=False, encoding='utf_8_sig')
    print(f"  PCA对比结果：")
    print(pca_compare_df.to_string(index=False))

    # ========== 第7步：最终聚类结果（选用K-Means on PCA） ==========
    print(f"\n--- 最终聚类结果（K-Means, K={best_k}, PCA降维空间）---")
    data['cluster'] = km_labels
    print(f"  各簇样本量：")
    print(data['cluster'].value_counts().sort_index())

    # ========== 第8步：聚类簇画像 ==========
    print(f"\n--- 聚类簇画像分析 ---")
    profile_numeric = product_numeric + ['sales']
    cluster_stats = data.groupby('cluster')[profile_numeric].mean().round(2)
    cluster_stats['业务标签'] = cluster_stats.apply(label_cluster_business, axis=1)
    cluster_stats.to_csv(CLUSTER_STAT_CSV, encoding='utf_8_sig')

    print(cluster_stats.to_string())

    # 各簇类别特征分布
    print("\n各簇类别特征（占比最高的类别）：")
    for col in product_categorical:
        print(f"\n【{col}】")
        top_cat = data.groupby('cluster')[col].value_counts().groupby(level=0).head(2)
        print(top_cat)

    # ========== 第9步：可视化 ==========
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 9.1 t-SNE可视化
    plot_tsne(X_pca, km_labels)

    # 9.2 各簇月销量箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='sales', data=data, palette='Set2')
    plt.xlabel('聚类簇编号', fontsize=13)
    plt.ylabel('月销量', fontsize=13)
    plt.title('不同聚类簇的月销量分布对比', fontsize=16, pad=10)
    plt.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    # 9.3 原价-月销量散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='original_price', y='sales', hue='cluster', data=data,
                    palette='Set2', s=30, alpha=0.5, edgecolor='white')
    plt.xlabel('商品原价（元）', fontsize=13)
    plt.ylabel('月销量', fontsize=13)
    plt.title('不同聚类簇在「原价-月销量」空间的分布', fontsize=16, pad=10)
    plt.tick_params(labelsize=11)
    plt.legend(title='簇', fontsize=10)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    # 9.4 聚类簇特征雷达图
    radar_cols = ['price', 'discount_rate', 'good_rate', 'comment_count', 'sales']
    radar_stats = data.groupby('cluster')[radar_cols].mean()
    # 归一化到0-1
    radar_norm = (radar_stats - radar_stats.min()) / (radar_stats.max() - radar_stats.min())
    labels_radar = ['价格', '折扣率', '好评率', '评价数', '月销量']
    angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = plt.cm.Set2(np.linspace(0, 1, best_k))
    for i, row in radar_norm.iterrows():
        values = row.tolist()
        values += values[:1]
        cluster_label = cluster_stats.loc[i, '业务标签']
        ax.plot(angles, values, 'o-', linewidth=2, label=f"簇{i}({cluster_label})", color=colors_radar[i])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=12)
    ax.set_title('各聚类簇特征雷达图', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(PROFILE_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 第10步：保存模型与结果 ==========
    data.to_csv(CLUSTER_DATA_CSV, index=False, encoding='utf_8_sig')
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(pca, PCA_MODEL_PATH)

    print("\n" + "=" * 60)
    print("  商品聚类训练完成！")
    print(f"  模型: {KMEANS_MODEL_PATH}, {PREPROCESSOR_PATH}, {PCA_MODEL_PATH}")
    print(f"  结果: {CLUSTER_COMPARE_CSV}, {CLUSTER_DATA_CSV}, {CLUSTER_STAT_CSV}, {PCA_COMPARE_CSV}")
    print(f"  图表: {ELBOW_PNG}, {TSNE_PNG}, {BOXPLOT_PNG}, {SCATTER_PNG}, {PROFILE_PNG}")
    print("=" * 60)

    return data


def train_user_cluster():
    """========================================
    附加实验：用户聚类分析
    ========================================"""""

    print("\n" + "=" * 60)
    print("  用户聚类分析训练（基于用户属性特征）")
    print("=" * 60)

    data = load_data_from_mysql()

    print(f"\n--- 用户聚类特征 ---")
    print(f"  数值特征: {user_numeric}")
    print(f"  类别特征: {user_categorical}")

    X_dense, preprocessor, feature_names = preprocess_features(data, user_numeric, user_categorical)
    X_pca, pca = apply_pca(X_dense, variance_threshold=0.95)

    # K值搜索
    k_range = range(2, 7)
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"  用户聚类最优K={best_k} (轮廓系数={max(sil_scores):.4f})")

    # 最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data['user_cluster'] = kmeans.fit_predict(X_pca)

    # 画像
    user_profile_numeric = user_numeric + ['sales']
    user_cluster_stats = data.groupby('user_cluster')[user_profile_numeric].mean().round(2)
    print(f"\n  用户聚类各簇特征均值：")
    print(user_cluster_stats.to_string())

    # 各簇类别特征
    for col in user_categorical:
        print(f"\n【{col}】")
        top_cat = data.groupby('user_cluster')[col].value_counts().groupby(level=0).head(2)
        print(top_cat)

    # 保存
    user_cluster_csv = os.path.join(OUTPUT_CSV_DIR, "用户聚类结果.csv")
    data.to_csv(user_cluster_csv, index=False, encoding='utf_8_sig')
    print(f"\n  用户聚类结果已保存到 {user_cluster_csv}")

    return data


if __name__ == '__main__':
    # 商品聚类
    train_product_cluster()

    # 用户聚类（不要了先）
    # train_user_cluster()