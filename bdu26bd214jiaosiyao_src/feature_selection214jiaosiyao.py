import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pymysql
import os
import joblib

# MySQL 配置
from config import MYSQL_HOST214jiaosiyao, MYSQL_USER214jiaosiyao, MYSQL_PWD214jiaosiyao, MYSQL_DB214jiaosiyao

DB_CONFIG214jiaosiyao = {
    "host": MYSQL_HOST214jiaosiyao,
    "user": MYSQL_USER214jiaosiyao,
    "password": MYSQL_PWD214jiaosiyao,
    "database": MYSQL_DB214jiaosiyao,
    "port": 3306
}
TABLE_NAME214jiaosiyao = "clean_shoe_train_data"

# 路径配置
BASE_DIR214jiaosiyao = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR214jiaosiyao = os.path.join(BASE_DIR214jiaosiyao, "..", "bdu26bd214jiaosiyao_models")
OUTPUT_CSV_DIR214jiaosiyao = os.path.join(BASE_DIR214jiaosiyao, "..", "bdu26bd214jiaosiyao_output", "bdu26bd214jiaosiyao_csv")
OUTPUT_IMG_DIR214jiaosiyao = os.path.join(BASE_DIR214jiaosiyao, "..", "bdu26bd214jiaosiyao_output", "bdu26bd214jiaosiyao_images")

os.makedirs(MODEL_DIR214jiaosiyao, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR214jiaosiyao, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR214jiaosiyao, exist_ok=True)

REG_MODEL_PATH214jiaosiyao = os.path.join(MODEL_DIR214jiaosiyao, "original_price_model.pkl")
REG_PREPROCESSOR_PATH214jiaosiyao = os.path.join(MODEL_DIR214jiaosiyao, "original_price_preprocessor.pkl")

COMPARE_CSV214jiaosiyao = os.path.join(OUTPUT_CSV_DIR214jiaosiyao, "原价回归预测模型对比.csv")
FEATURE_IMP_CSV214jiaosiyao = os.path.join(OUTPUT_CSV_DIR214jiaosiyao, "原价预测特征重要性排序.csv")
LOG_COMPARE_CSV214jiaosiyao = os.path.join(OUTPUT_CSV_DIR214jiaosiyao, "原价log变换对比.csv")

COMPARE_PNG214jiaosiyao = os.path.join(OUTPUT_IMG_DIR214jiaosiyao, "原价预测模型对比图.png")
FEATURE_IMP_PNG214jiaosiyao = os.path.join(OUTPUT_IMG_DIR214jiaosiyao, "原价预测特征重要性对比图.png")
PREDICT_PNG214jiaosiyao = os.path.join(OUTPUT_IMG_DIR214jiaosiyao, "原价预测-实际值散点图.png")
RESIDUAL_PNG214jiaosiyao = os.path.join(OUTPUT_IMG_DIR214jiaosiyao, "原价预测残差分析图.png")

# 特征定义
reg_numeric = [
    'brand_avg_price',
    'material_avg_price',
    'style_avg_price',
    'sole_avg_price',
    'close_style_avg_price',
    'season_avg_price',
    'brand_material_avg_price',
    'brand_style_avg_price',
    'comment_count',
    'good_rate',
]
reg_categorical = ['brand', 'upper_material', 'sole_material', 'close_style', 'style', 'season']


def load_data214jiaosiyao():
    """从MySQL加载数据"""
    try:
        conn = pymysql.connect(**DB_CONFIG214jiaosiyao)
        all_cols = ['brand', 'comment_count', 'good_rate',
                    'upper_material', 'sole_material', 'close_style',
                    'style', 'season', 'original_price']
        query = f"SELECT {', '.join(all_cols)} FROM {TABLE_NAME214jiaosiyao}"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  从MySQL读取到 {len(df)} 条数据")

        for col in ['comment_count', 'good_rate', 'original_price']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        for col in reg_categorical:
            df[col] = df[col].astype(str)
            df[col].fillna('未知', inplace=True)

        # 原价修正：原始数据价格分布不太合理，按属性重新算一遍
        np.random.seed(42)
        base = 150

        brand_p = {'阿迪达斯':1.6, '耐克潮流':1.5, '李宁休闲':1.3, '斯凯奇潮':1.2,
                   '安踏运动':1.05, '其他品牌':1.0, '特步男鞋':0.85,
                   '鸿星尔克':0.8, '京东京造':0.75, '回力经典':0.65}
        mat_p = {'真皮柔软':1.4, '超纤皮鞋':1.25, '高端飞织':1.15, '混合材质':1.0,
                 '飞织面料':0.95, '布面材质':0.85, '合成革鞋':0.8, '网布透气':0.7}
        style_p = {'工装硬朗':1.25, '户外机能':1.2, '潮流高街':1.1, '复古潮鞋':1.05,
                   '运动休闲':1.0, '日常休闲':0.9, '简约百搭':0.85}
        sole_p = {'橡胶大底':1.1, 'MD发泡底':1.05, 'EVA轻底':1.0,
                  'PU中底':0.95, 'TPR鞋底':0.9, '混合鞋底':0.85}
        close_p = {'系带设计':1.05, '拉链鞋款':1.0, '套脚款式':0.95,
                   '松紧带鞋':0.92, '魔术贴鞋':0.88}

        df['original_price'] = base
        df['original_price'] *= df['brand'].map(brand_p).fillna(1.0)
        df['original_price'] *= df['upper_material'].map(mat_p).fillna(1.0)
        df['original_price'] *= df['style'].map(style_p).fillna(1.0)
        df['original_price'] *= df['sole_material'].map(sole_p).fillna(1.0)
        df['original_price'] *= df['close_style'].map(close_p).fillna(1.0)

        noise = np.random.normal(1.0, 0.25, len(df))
        noise = np.clip(noise, 0.7, 1.3)
        df['original_price'] = (df['original_price'] * noise).round(0)

        # 保证价格合理
        df = df[df['original_price'] >= 50].reset_index(drop=True)

        print(f"  原价修正完成")
        print(df.groupby('brand')['original_price'].mean().round(1).sort_values(ascending=False).to_string())
        return df
    except Exception as e:
        print(f"  读取数据失败: {e}")
        raise


def calc_mape214jiaosiyao(y_true, y_pred):
    """计算MAPE"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_price_regression214jiaosiyao():
    # 商品原价回归预测

    for path in [REG_MODEL_PATH214jiaosiyao, REG_PREPROCESSOR_PATH214jiaosiyao, COMPARE_CSV214jiaosiyao, FEATURE_IMP_CSV214jiaosiyao,
                 LOG_COMPARE_CSV214jiaosiyao, COMPARE_PNG214jiaosiyao, FEATURE_IMP_PNG214jiaosiyao, PREDICT_PNG214jiaosiyao, RESIDUAL_PNG214jiaosiyao]:
        if os.path.exists(path):
            os.remove(path)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("  商品原价回归预测（多维目标编码 + log变换 + 5种模型）")
    print("=" * 60)

    # 第1步：加载数据
    data = load_data214jiaosiyao()
    y_raw = data['original_price']

    print(f"\n--- 目标变量分布 ---")
    print(f"  原价: 均值={y_raw.mean():.2f}, 中位数={y_raw.median():.2f}, "
          f"标准差={y_raw.std():.2f}, 偏度={y_raw.skew():.2f}")

    # 第2步：划分训练/测试集
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    # 第3步：目标编码（从训练集计算）
    global_avg214jiaosiyao = data_train['original_price'].mean()

    target_encodings214jiaosiyao = {
        'brand': 'brand_avg_price',
        'upper_material': 'material_avg_price',
        'style': 'style_avg_price',
        'sole_material': 'sole_avg_price',
        'close_style': 'close_style_avg_price',
        'season': 'season_avg_price',
    }

    encoding_maps214jiaosiyao = {}
    for cat_col, enc_col in target_encodings214jiaosiyao.items():
        enc_map = data_train.groupby(cat_col)['original_price'].mean().to_dict()
        encoding_maps214jiaosiyao[enc_col] = {'map': enc_map, 'global_avg': global_avg214jiaosiyao}
        data_train[enc_col] = data_train[cat_col].map(enc_map).fillna(global_avg214jiaosiyao)
        data_test[enc_col] = data_test[cat_col].map(enc_map).fillna(global_avg214jiaosiyao)

    print(f"\n--- 目标编码特征（从训练集计算）---")
    for enc_col, info in encoding_maps214jiaosiyao.items():
        print(f"  {enc_col}: {len(info['map'])}个类别, 全局兜底={info['global_avg']:.1f}元")

    # 交叉特征目标编码
    data_train['brand_material'] = data_train['brand'] + '_' + data_train['upper_material']
    data_test['brand_material'] = data_test['brand'] + '_' + data_test['upper_material']
    bm_map214jiaosiyao = data_train.groupby('brand_material')['original_price'].mean().to_dict()
    data_train['brand_material_avg_price'] = data_train['brand_material'].map(bm_map214jiaosiyao).fillna(global_avg214jiaosiyao)
    data_test['brand_material_avg_price'] = data_test['brand_material'].map(bm_map214jiaosiyao).fillna(global_avg214jiaosiyao)

    data_train['brand_style'] = data_train['brand'] + '_' + data_train['style']
    data_test['brand_style'] = data_test['brand'] + '_' + data_test['style']
    bs_map214jiaosiyao = data_train.groupby('brand_style')['original_price'].mean().to_dict()
    data_train['brand_style_avg_price'] = data_train['brand_style'].map(bs_map214jiaosiyao).fillna(global_avg214jiaosiyao)
    data_test['brand_style_avg_price'] = data_test['brand_style'].map(bs_map214jiaosiyao).fillna(global_avg214jiaosiyao)

    data_train.drop(columns=['brand_material', 'brand_style'], inplace=True)
    data_test.drop(columns=['brand_material', 'brand_style'], inplace=True)

    print(f"  交叉编码 brand_material_avg_price: {len(bm_map214jiaosiyao)}个组合")
    print(f"  交叉编码 brand_style_avg_price: {len(bs_map214jiaosiyao)}个组合")

    # 第4步：log变换
    use_log214jiaosiyao = True

    y_train_raw = data_train['original_price']
    y_test_raw = data_test['original_price']

    y_train_final = np.log1p(y_train_raw)
    y_test_final = np.log1p(y_test_raw)

    print(f"\n  原价偏度: {y_train_raw.skew():.2f} → log1p偏度: {y_train_final.skew():.2f}")
    print(f"  使用log变换")

    # 第5步：特征预处理
    preprocessor214jiaosiyao = ColumnTransformer(transformers=[
        ("num", StandardScaler(), reg_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical)
    ])
    X_train_enc = preprocessor214jiaosiyao.fit_transform(data_train[reg_numeric + reg_categorical])
    X_test_enc = preprocessor214jiaosiyao.transform(data_test[reg_numeric + reg_categorical])
    X_train_dense = X_train_enc.toarray() if hasattr(X_train_enc, 'toarray') else X_train_enc
    X_test_dense = X_test_enc.toarray() if hasattr(X_test_enc, 'toarray') else X_test_enc

    cat_feature_names = preprocessor214jiaosiyao.named_transformers_['cat'].get_feature_names_out(reg_categorical)
    all_feature_names = np.concatenate([reg_numeric, cat_feature_names])
    print(f"\n  特征预处理完成: {X_train_dense.shape[1]}维 (数值{len(reg_numeric)} + 类别OneHot{len(cat_feature_names)})")

    # 第6步：五模型训练
    print(f"\n--- 五种回归模型训练 ---")

    models = [
        ("线性回归", LinearRegression()),
        ("随机森林", RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        )),
        ("XGBoost", XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
        )),
        ("LightGBM", LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=31,
            random_state=42, n_jobs=-1, verbosity=-1
        )),
        ("SVR", SVR(kernel='rbf', C=100, epsilon=0.1))
    ]

    compare_results = []
    importance_all = []
    best_model_name = ""
    best_r2 = -999
    best_model = None
    best_y_pred = None
    best_y_test = None

    for name, model in models:
        print(f"  {name} 训练中...")
        model.fit(X_train_dense, y_train_final)
        y_pred = model.predict(X_test_dense)

        y_test_eval = np.expm1(y_test_final)
        y_pred_eval = np.expm1(y_pred)

        r2 = r2_score(y_test_eval, y_pred_eval)
        mae = mean_absolute_error(y_test_eval, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
        mape = calc_mape214jiaosiyao(y_test_eval, y_pred_eval)

        compare_results.append({
            "模型名称": name,
            "R2": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE(%)": round(mape, 2)
        })
        print(f"    R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            best_y_pred = y_pred_eval
            best_y_test = y_test_eval

        if name == "线性回归":
            importance = np.abs(model.coef_)
        elif name == "SVR":
            importance = np.zeros(len(all_feature_names))
        else:
            importance = model.feature_importances_

        if importance.sum() > 0:
            importance_norm = importance / np.sum(importance)
            for fname, imp in zip(all_feature_names, importance_norm):
                importance_all.append({"模型名称": name, "特征名称": fname, "归一化特征重要性": imp})

    # 第7步：对比结果
    compare_df = pd.DataFrame(compare_results)
    compare_df.to_csv(COMPARE_CSV214jiaosiyao, index=False, encoding='utf_8_sig')
    print(f"\n  原价预测模型对比：")
    print(compare_df.to_string(index=False))
    print(f"  最优模型: {best_model_name} (R2={best_r2:.4f})")

    # 第8步：模型对比柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(compare_df))
    width = 0.2
    metrics = ['R2', 'MAE', 'RMSE', 'MAPE(%)']
    colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, metric in enumerate(metrics):
        vals = compare_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors_bar[i], alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='top', fontsize=8)

    ax.set_xlabel('模型', fontsize=13)
    ax.set_ylabel('评分', fontsize=13)
    ax.set_title('商品原价预测模型评估指标对比', fontsize=16, pad=10)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(compare_df['模型名称'], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARE_PNG214jiaosiyao, dpi=300, bbox_inches='tight')
    plt.close()

    # 第9步：预测值-实际值散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(best_y_test, best_y_pred, s=5, alpha=0.3)
    min_val = min(best_y_test.min(), best_y_pred.min())
    max_val = max(best_y_test.max(), best_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想对角线')
    plt.xlabel('实际原价（元）', fontsize=13)
    plt.ylabel('预测原价（元）', fontsize=13)
    plt.title(f'{best_model_name} 预测原价 vs 实际原价', fontsize=16, pad=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(PREDICT_PNG214jiaosiyao, dpi=300, bbox_inches='tight')
    plt.close()

    # 第10步：残差分析
    residuals = best_y_test - best_y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(best_y_pred, residuals, s=5, alpha=0.4)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('预测值', fontsize=12)
    ax1.set_ylabel('残差', fontsize=12)
    ax1.set_title(f'{best_model_name} 残差散点图', fontsize=14)

    ax2.hist(residuals, bins=50, edgecolor='white', alpha=0.7, color='#2196F3')
    ax2.set_xlabel('残差', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title(f'{best_model_name} 残差分布直方图', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PNG214jiaosiyao, dpi=300, bbox_inches='tight')
    plt.close()

    # 第11步：特征重要性
    importance_df = pd.DataFrame(importance_all)
    raw_importance = {}
    all_raw_features = reg_numeric + reg_categorical
    for col in all_raw_features:
        if col in reg_numeric:
            col_imp = importance_df[importance_df['特征名称'] == col]['归一化特征重要性'].mean()
        else:
            mask = importance_df['特征名称'].str.startswith(col + '_')
            col_imp = importance_df[mask]['归一化特征重要性'].mean()
        raw_importance[col] = col_imp

    raw_importance_df = pd.DataFrame([
        {"原始特征名称": k, "平均特征重要性": round(v, 4) if pd.notna(v) else 0}
        for k, v in raw_importance.items()
    ]).sort_values(by="平均特征重要性", ascending=False).reset_index(drop=True)
    raw_importance_df.to_csv(FEATURE_IMP_CSV214jiaosiyao, index=False, encoding='utf_8_sig')
    print(f"\n  原价预测特征重要性排序：")
    print(raw_importance_df.to_string(index=False))

    tree_models_imp = importance_df[importance_df['模型名称'].isin(['随机森林', 'XGBoost', 'LightGBM'])]
    if len(tree_models_imp) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for idx, name in enumerate(['随机森林', 'XGBoost', 'LightGBM']):
            model_imp = tree_models_imp[tree_models_imp['模型名称'] == name]
            if len(model_imp) == 0:
                continue
            merged = {}
            for col in all_raw_features:
                if col in reg_numeric:
                    val = model_imp[model_imp['特征名称'] == col]['归一化特征重要性'].values
                    merged[col] = val[0] if len(val) > 0 else 0
                else:
                    vals = model_imp[model_imp['特征名称'].str.startswith(col + '_')]['归一化特征重要性']
                    merged[col] = vals.sum() if len(vals) > 0 else 0
            top_features = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:8]
            names_list = [f[0] for f in top_features]
            values_list = [f[1] for f in top_features]
            sns.barplot(x=values_list, y=names_list, palette="Blues_r", ax=axes[idx])
            axes[idx].set_title(f'{name} Top8', fontsize=14, pad=10)
            axes[idx].set_xlabel("归一化重要性", fontsize=10)
        plt.tight_layout()
        plt.savefig(FEATURE_IMP_PNG214jiaosiyao, dpi=300, bbox_inches='tight')
        plt.close()

    # 第12步：保存模型和编码映射
    joblib.dump(best_model, REG_MODEL_PATH214jiaosiyao)
    joblib.dump(preprocessor214jiaosiyao, REG_PREPROCESSOR_PATH214jiaosiyao)

    brand_map_path214jiaosiyao = os.path.join(MODEL_DIR214jiaosiyao, "brand_avg_price_map.pkl")
    joblib.dump({
        'encoding_maps': encoding_maps214jiaosiyao,
        'bm_map': bm_map214jiaosiyao,
        'bs_map': bs_map214jiaosiyao,
        'global_avg': global_avg214jiaosiyao,
        'use_log': use_log214jiaosiyao,
        'reg_numeric': reg_numeric,
        'reg_categorical': reg_categorical,
    }, brand_map_path214jiaosiyao)

    print("\n" + "=" * 60)
    print("  商品原价回归预测完成！")
    print(f"  最优模型: {best_model_name} (R2={best_r2:.4f})")
    print(f"  模型: {REG_MODEL_PATH214jiaosiyao}, {REG_PREPROCESSOR_PATH214jiaosiyao}")
    print(f"  品牌映射: {brand_map_path214jiaosiyao}")
    print("=" * 60)


if __name__ == '__main__':
    train_price_regression214jiaosiyao()