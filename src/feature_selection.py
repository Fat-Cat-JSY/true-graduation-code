import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pymysql
import os

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
FEATURE_IMPORTANCE_PATH = "../output/csv/特征重要性对比结果.csv"
MODEL_COMPARE_PATH = "../output/csv/回归模型评估对比.csv"
FINAL_IMPORTANCE_PATH = "../output/csv/原始特征总重要性排序.csv"
FEATURE_IMPORTANCE_PNG = "../output/images/三种回归特征重要性对比图.png"

# 创建输出目录
os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PNG), exist_ok=True)

def load_data():
    """从MySQL加载全量原始数据，用于回归特征筛选"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = f"""
            SELECT brand, original_price, price, discount_rate, comment_count, good_rate, 
                   user_age, is_plus, sole_material, close_style, style, is_return, month_sale
            FROM {TABLE_NAME}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  从MySQL读取到 {len(df)} 条原始数据用于特征筛选")

        # 数据类型清洗
        if df['discount_rate'].dtype == object:
            df['discount_rate'] = df['discount_rate'].str.replace('%', '').astype(float) / 100.0
        if df['is_plus'].dtype == object:
            df['is_plus'] = df['is_plus'].map({'是': 1, '否': 0}).astype(int)
        if df['is_return'].dtype == object:
            df['is_return'] = df['is_return'].map({'是': 1, '否': 0}).astype(int)

        # 缺失值填充
        num_cols = ['original_price', 'price', 'discount_rate', 'comment_count', 'good_rate', 'user_age', 'is_return']
        cat_cols = ['brand', 'is_plus', 'sole_material', 'close_style', 'style']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        for col in cat_cols:
            df[col] = df[col].astype(str)
            df[col].fillna('未知', inplace=True)

        df.rename(columns={'month_sale': 'sales'}, inplace=True)
        return df, num_cols, cat_cols
    except Exception as e:
        print(f"  读取数据失败: {e}")
        raise


def train_three_regression():
    """训练三种回归模型，对比评估并提取特征重要性"""
    # 清理旧输出文件
    for path in [FEATURE_IMPORTANCE_PATH, MODEL_COMPARE_PATH, FINAL_IMPORTANCE_PATH, FEATURE_IMPORTANCE_PNG]:
        if os.path.exists(path):
            os.remove(path)

    print("  开始三种回归模型训练，对比提取特征重要性...")
    data, num_cols, cat_cols = load_data()
    X = data[num_cols + cat_cols]
    y = data['sales']

    # 划分训练集测试集 8:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  数据集划分完成，训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条")

    # 统一特征预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = np.concatenate([num_cols, cat_features])
    print(f"  特征预处理完成，共 {X_train_encoded.shape[1]} 维特征")

    # 初始化三种回归模型
    models = [
        ("线性回归", LinearRegression()),
        ("随机森林", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
    ]

    # 存储对比结果
    compare_results = []
    importance_all = []

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 遍历训练每个模型
    for idx, (name, model) in enumerate(models):
        print(f"\n▶ 正在训练 {name} ...")
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        compare_results.append({
            "模型名称": name,
            "R²决定系数": round(r2, 4),
            "MAE平均绝对误差": round(mae, 2),
            "RMSE均方根误差": round(rmse, 2)
        })
        print(f"   {name} 评估完成：R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

        # 提取特征重要性
        if name == "线性回归":
            importance = np.abs(model.coef_)
        else:
            importance = model.feature_importances_

        # 归一化重要性，适配跨模型对比
        importance_norm = importance / np.sum(importance)
        for fname, imp in zip(all_feature_names, importance_norm):
            importance_all.append({
                "模型名称": name,
                "特征名称": fname,
                "归一化特征重要性": imp
            })

        # 绘制Top10特征重要性
        temp_df = pd.DataFrame({"特征名称": all_feature_names, "重要性": importance_norm}) \
            .sort_values("重要性", ascending=False).head(10)
        sns.barplot(x="重要性", y="特征名称", data=temp_df, palette="Blues_r", ax=axes[idx])
        axes[idx].set_title(f"{name} Top10特征重要性", fontsize=14, pad=10)
        axes[idx].set_xlabel("归一化重要性", fontsize=10)
        axes[idx].set_ylabel("特征名称", fontsize=10)

    # 保存模型评估对比结果
    compare_df = pd.DataFrame(compare_results)
    compare_df.to_csv(MODEL_COMPARE_PATH, index=False, encoding='utf_8_sig')
    print("\n" + "=" * 60)
    print("三种回归模型评估对比结果：")
    print(compare_df.to_string(index=False))
    print("=" * 60)
    print(f"  模型对比结果已保存到 {MODEL_COMPARE_PATH}")

    # 保存全量特征重要性
    importance_df = pd.DataFrame(importance_all)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding='utf_8_sig')

    # 统计原始特征平均总重要性，合并OneHot拆分的类别特征
    raw_importance = {}
    all_raw_features = num_cols + cat_cols
    for col in all_raw_features:
        raw_importance[col] = 0
    for col in all_raw_features:
        if col in num_cols:
            col_imp = importance_df[(importance_df['特征名称'] == col)]['归一化特征重要性'].mean()
            raw_importance[col] = col_imp
        else:
            mask = importance_df['特征名称'].str.startswith(col + '_')
            col_imp = importance_df[mask]['归一化特征重要性'].mean()
            raw_importance[col] = col_imp
    raw_importance_df = pd.DataFrame([
        {"原始特征名称": k, "平均特征重要性": round(v, 4)} for k, v in raw_importance.items()
    ]).sort_values(by="平均特征重要性", ascending=False).reset_index(drop=True)
    raw_importance_df.to_csv(FINAL_IMPORTANCE_PATH, index=False, encoding='utf_8_sig')
    print(f"\n  原始特征平均总重要性排序已保存到 {FINAL_IMPORTANCE_PATH}")
    print("\n  原始特征重要性排序（降序）：")
    print(raw_importance_df.to_string(index=False))

    # 保存特征重要性对比图
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=300, bbox_inches='tight')
    print(f"\n  三种回归特征重要性对比图已保存到 {FEATURE_IMPORTANCE_PNG}")
    plt.show()

    print("\n  三种回归特征筛选全部完成，可根据排序结果选择核心特征用于后续聚类")


if __name__ == '__main__':
    train_three_regression()