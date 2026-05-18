from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pymysql
from config import *
import numpy as np

app = Flask(__name__, template_folder='templates')
app.config['JSON_AS_ASCII'] = False

# ===================== 聚类模型特征定义 =====================
cluster_numeric = ['price', 'original_price', 'discount_rate', 'good_rate', 'comment_count']
cluster_categorical = ['brand', 'upper_material', 'sole_material', 'style', 'season']

# ===================== 加载聚类模型 =====================
BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
KMEANS_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "kmeans_cluster.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_MODEL_DIR, "preprocessor.pkl")
PCA_PATH = os.path.join(BASE_MODEL_DIR, "pca.pkl")

kmeans_model = joblib.load(KMEANS_MODEL_PATH)
preprocessor_global = joblib.load(PREPROCESSOR_PATH)
pca_model = joblib.load(PCA_PATH)
print("✅ 聚类模型加载完成（preprocessor → PCA → KMeans）")

# ===================== 加载原价回归模型 =====================
REG_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "original_price_model.pkl")
REG_PREPROCESSOR_PATH = os.path.join(BASE_MODEL_DIR, "original_price_preprocessor.pkl")
BRAND_MAP_PATH = os.path.join(BASE_MODEL_DIR, "brand_avg_price_map.pkl")

price_model = joblib.load(REG_MODEL_PATH)
price_preprocessor = joblib.load(REG_PREPROCESSOR_PATH)
brand_map_data = joblib.load(BRAND_MAP_PATH)
print("✅ 原价回归模型加载完成")

# ===================== 数据库配置 =====================
ALL_TABLES_CONFIG = {
    "train_data": {"table_name": "clean_shoe_train_data", "table_desc": "清洗后的训练主数据"},
    "brand_analysis": {"table_name": "brand_analysis", "table_desc": "品牌分析结果表"},
    "material_analysis": {"table_name": "material_analysis", "table_desc": "鞋面材质分析结果表"},
    "user_analysis": {"table_name": "user_analysis", "table_desc": "用户群体分析结果表"},
    "time_analysis": {"table_name": "time_analysis", "table_desc": "下单时段分析结果表"},
    "province_top10": {"table_name": "province_top10", "table_desc": "省份订单Top10结果表"},
    "price_bin_analysis": {"table_name": "price_bin_analysis", "table_desc": "价格区间分析结果表"}
}


def get_mysql_conn():
    return pymysql.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PWD,
        database=MYSQL_DB, port=3306, charset=MYSQL_CHARSET
    )


def get_table_data(table_key):
    if table_key not in ALL_TABLES_CONFIG:
        return None
    table_name = ALL_TABLES_CONFIG[table_key]["table_name"]
    conn = get_mysql_conn()
    try:
        sql = f"SELECT * FROM `{table_name}`;"
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        print(f"获取{table_name}失败：{str(e)}")
        return None
    finally:
        conn.close()


# ===================== 数据接口 =====================
@app.route('/api/train_scatter', methods=['GET'])
def get_train_scatter():
    df = get_table_data("price_bin_analysis")
    if df is None or len(df) == 0:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    bin_to_x = {"0-100元": 50, "100-200元": 150, "200-300元": 250, "300元以上": 350}
    res = []
    for _, row in df.iterrows():
        x_val = bin_to_x.get(row['price_bin'], 0)
        res.append({
            "x": float(x_val), "y": float(row['goods_count']),
            "z": float(row['return_rate']) * 100, "price_bin": row['price_bin'],
            "total_month_sale": float(row['total_month_sale']),
            "goods_count": float(row['goods_count']),
            "return_rate": float(row['return_rate'])
        })
    return jsonify({"code": 200, "data": res})


@app.route('/api/dashboard/brand', methods=['GET'])
def get_brand_data():
    df = get_table_data("brand_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app.route('/api/dashboard/province', methods=['GET'])
def get_province_data():
    df = get_table_data("province_top10")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app.route('/api/dashboard/price_bin', methods=['GET'])
def get_price_data():
    df = get_table_data("price_bin_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app.route('/api/dashboard/user', methods=['GET'])
def get_user_data():
    df = get_table_data("user_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app.route('/api/dashboard/material', methods=['GET'])
def get_material_data():
    df = get_table_data("material_analysis")
    if df is None or len(df) == 0: return jsonify({"code": 200, "data": []})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app.route('/api/dashboard/time', methods=['GET'])
def get_time_data():
    df = get_table_data("time_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


# ===================== 聚类预测接口 =====================
@app.route('/api/cluster_predict', methods=['POST'])
def cluster_predict():
    try:
        req = request.get_json()

        input_df = pd.DataFrame([{
            "price": float(req["price"]),
            "original_price": float(req["original_price"]),
            "discount_rate": float(req["discount_rate"]),
            "good_rate": float(req["good_rate"]),
            "comment_count": int(req["comment_count"]),
            "brand": str(req["brand"]).strip(),
            "upper_material": str(req["upper_material"]).strip(),
            "sole_material": str(req["sole_material"]).strip(),
            "style": str(req["style"]).strip(),
            "season": str(req["season"]).strip()
        }])

        # 预处理 → PCA降维 → KMeans预测
        X_input = preprocessor_global.transform(input_df)
        if hasattr(X_input, 'toarray'):
            X_input = X_input.toarray()
        X_pca = pca_model.transform(X_input)
        cluster_id = int(kmeans_model.predict(X_pca)[0])

        cluster_desc = [
            "【簇0 - 打折促销款】该商品属于打折促销群体：折扣力度大，价格偏低，以网布/飞织材质为主，风格偏简约百搭与日常休闲，适合追求性价比的消费者。",
            "【簇1 - 平价常规款】该商品属于平价常规群体：价格适中，折扣温和，材质以飞织/合成革为主，风格覆盖运动休闲与复古潮鞋，是平台的主流走量款。",
            "【簇2 - 高端品质款】该商品属于高端品质群体：原价较高，以真皮/超纤皮等高端材质为主，风格偏工装硬朗与潮流高街，适合注重品质的消费者。",
            "【簇3 - 高流量热门款】该商品属于高流量热门群体：评论数远高于其他群体，价格适中偏上，材质以混合材质为主，风格偏潮流高街，属于平台明星爆款。"
        ][cluster_id] if cluster_id < 4 else "该商品特征超出模型训练范围，属于未知类别"

        return jsonify({"code": 200, "cluster_id": cluster_id, "cluster_desc": cluster_desc})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 400, "msg": f"参数错误：{str(e)}"})


# ===================== 原价回归预测接口 =====================
@app.route('/api/price_predict', methods=['POST'])
def price_predict():
    try:
        req = request.get_json()
        brand = str(req["brand"]).strip()

        # 从保存的映射表计算brand_avg_price
        brand_price_map = brand_map_data['map']
        global_avg = brand_map_data['global_avg']
        brand_avg_price = brand_price_map.get(brand, global_avg)

        input_df = pd.DataFrame([{
            "brand_avg_price": brand_avg_price,
            "comment_count": int(req["comment_count"]),
            "good_rate": float(req["good_rate"]),
            "brand": brand,
            "upper_material": str(req["upper_material"]).strip(),
            "sole_material": str(req["sole_material"]).strip(),
            "close_style": str(req["close_style"]).strip(),
            "style": str(req["style"]).strip(),
            "season": str(req["season"]).strip()
        }])

        X_input = price_preprocessor.transform(input_df)
        if hasattr(X_input, 'toarray'):
            X_input = X_input.toarray()

        y_pred = price_model.predict(X_input)[0]

        # 如果训练时用了log1p变换，需要反变换
        if brand_map_data.get('use_log', False):
            predicted_price = float(np.expm1(y_pred))
        else:
            predicted_price = float(y_pred)

        predicted_price = round(max(predicted_price, 0), 2)

        return jsonify({
            "code": 200,
            "predicted_price": predicted_price,
            "brand_avg_price": round(brand_avg_price, 1)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 400, "msg": f"参数错误：{str(e)}"})

# ===================== 页面路由 =====================
@app.route('/')
def dashboard():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/price_predict')
def price_predict_page():
    return render_template('price_predict.html')

if __name__ == '__main__':
    print(" * 可视化大屏：http://127.0.0.1:5000")
    print(" * 聚类预测界面：http://127.0.0.1:5000/predict")
    print(" * 回归预测界面：http://127.0.0.1:5000/price_predict")

    app.run(debug=False, host='127.0.0.1', port=5000)