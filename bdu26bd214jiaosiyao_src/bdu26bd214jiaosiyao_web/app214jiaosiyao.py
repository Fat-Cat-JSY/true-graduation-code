from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pymysql
from config import *
import numpy as np
import os

app214jiaosiyao = Flask(__name__, template_folder='bdu26bd214jiaosiyao_templates')
app214jiaosiyao.config['JSON_AS_ASCII'] = False

# 聚类模型特征定义
cluster_numeric = ['price', 'original_price', 'discount_rate', 'good_rate', 'comment_count']
cluster_categorical = ['brand', 'upper_material', 'sole_material', 'style', 'season']

# 加载聚类模型
BASE_MODEL_DIR214jiaosiyao = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bdu26bd214jiaosiyao_models')
KMEANS_MODEL_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "kmeans_cluster.pkl")
PREPROCESSOR_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "preprocessor.pkl")
PCA_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "pca.pkl")

kmeans_model214jiaosiyao = joblib.load(KMEANS_MODEL_PATH214jiaosiyao)
preprocessor_global214jiaosiyao = joblib.load(PREPROCESSOR_PATH214jiaosiyao)
pca_model214jiaosiyao = joblib.load(PCA_PATH214jiaosiyao)
print("聚类模型加载完成")

# 加载原价回归模型
REG_MODEL_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "original_price_model.pkl")
REG_PREPROCESSOR_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "original_price_preprocessor.pkl")
BRAND_MAP_PATH214jiaosiyao = os.path.join(BASE_MODEL_DIR214jiaosiyao, "brand_avg_price_map.pkl")

price_model214jiaosiyao = joblib.load(REG_MODEL_PATH214jiaosiyao)
price_preprocessor214jiaosiyao = joblib.load(REG_PREPROCESSOR_PATH214jiaosiyao)
brand_map_data214jiaosiyao = joblib.load(BRAND_MAP_PATH214jiaosiyao)
print("原价回归模型加载完成")

# 数据库配置
ALL_TABLES_CONFIG214jiaosiyao = {
    "train_data": {"table_name": "clean_shoe_train_data"},
    "brand_analysis": {"table_name": "brand_analysis"},
    "material_analysis": {"table_name": "material_analysis"},
    "user_analysis": {"table_name": "user_analysis"},
    "time_analysis": {"table_name": "time_analysis"},
    "province_top10": {"table_name": "province_top10"},
    "price_bin_analysis": {"table_name": "price_bin_analysis"}
}


def get_mysql_conn214jiaosiyao():
    return pymysql.connect(
        host=MYSQL_HOST214jiaosiyao, user=MYSQL_USER214jiaosiyao, password=MYSQL_PWD214jiaosiyao,
        database=MYSQL_DB214jiaosiyao, port=3306, charset=MYSQL_CHARSET214jiaosiyao
    )


def get_table_data214jiaosiyao(table_key):
    if table_key not in ALL_TABLES_CONFIG214jiaosiyao:
        return None
    table_name = ALL_TABLES_CONFIG214jiaosiyao[table_key]["table_name"]
    conn = get_mysql_conn214jiaosiyao()
    try:
        sql = f"SELECT * FROM `{table_name}`;"
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        print(f"获取{table_name}失败：{str(e)}")
        return None
    finally:
        conn.close()


# 数据接口
@app214jiaosiyao.route('/api/train_scatter', methods=['GET'])
def get_train_scatter214jiaosiyao():
    df = get_table_data214jiaosiyao("price_bin_analysis")
    if df is None or len(df) == 0:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    res = []
    for _, row in df.iterrows():
        res.append({
            "price_bin": row['price_bin'],
            "goods_count": int(row['goods_count']),
            "totalMonthSale": int(row['total_month_sale']),
            "returnRate": round(float(row['return_rate']) * 100, 2)
        })
    return jsonify({"code": 200, "data": res})


@app214jiaosiyao.route('/api/dashboard/brand', methods=['GET'])
def get_brand_data214jiaosiyao():
    df = get_table_data214jiaosiyao("brand_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app214jiaosiyao.route('/api/dashboard/province', methods=['GET'])
def get_province_data214jiaosiyao():
    df = get_table_data214jiaosiyao("province_top10")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app214jiaosiyao.route('/api/dashboard/price_bin', methods=['GET'])
def get_price_data214jiaosiyao():
    df = get_table_data214jiaosiyao("price_bin_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app214jiaosiyao.route('/api/dashboard/user', methods=['GET'])
def get_user_data214jiaosiyao():
    df = get_table_data214jiaosiyao("user_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app214jiaosiyao.route('/api/dashboard/material', methods=['GET'])
def get_material_data214jiaosiyao():
    df = get_table_data214jiaosiyao("material_analysis")
    if df is None or len(df) == 0: return jsonify({"code": 200, "data": []})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


@app214jiaosiyao.route('/api/dashboard/time', methods=['GET'])
def get_time_data214jiaosiyao():
    df = get_table_data214jiaosiyao("time_analysis")
    if df is None: return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({"code": 200, "data": df.to_dict(orient='records')})


# 聚类预测接口
@app214jiaosiyao.route('/api/cluster_predict', methods=['POST'])
def cluster_predict214jiaosiyao():
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

        X_input = preprocessor_global214jiaosiyao.transform(input_df)
        if hasattr(X_input, 'toarray'):
            X_input = X_input.toarray()
        X_pca = pca_model214jiaosiyao.transform(X_input)
        cluster_id = int(kmeans_model214jiaosiyao.predict(X_pca)[0])

        cluster_desc = [
            "簇0-打折促销款：折扣力度大、价格低，网布飞织为主，偏简约休闲风格。",
            "簇1-平价常规款：价格适中，折扣温和，飞织合成革为主，运动休闲和复古潮鞋为主。",
            "簇2-高端品质款：原价较高，真皮超纤皮为主，偏工装硬朗和潮流高街。",
            "簇3-高流量热门款：评论数远高于其他群体，价格适中偏上，混合材质为主，偏潮流高街。"
        ][cluster_id] if cluster_id < 4 else "特征超出模型训练范围，属于未知类别"

        return jsonify({"code": 200, "cluster_id": cluster_id, "cluster_desc": cluster_desc})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 400, "msg": f"参数错误：{str(e)}"})


# 原价回归预测接口（适配新版模型）
@app214jiaosiyao.route('/api/price_predict', methods=['POST'])
def price_predict214jiaosiyao():
    try:
        req = request.get_json()
        brand = str(req["brand"]).strip()

        # 从保存的映射表读取各维度目标编码
        encoding_maps = brand_map_data214jiaosiyao['encoding_maps']
        bm_map = brand_map_data214jiaosiyao['bm_map']
        bs_map = brand_map_data214jiaosiyao['bs_map']
        global_avg = brand_map_data214jiaosiyao['global_avg']

        # 计算各维度均价特征
        def get_enc_val(enc_col, cat_value):
            info = encoding_maps.get(enc_col, {})
            return info.get('map', {}).get(cat_value, info.get('global_avg', global_avg))

        brand_avg = get_enc_val('brand_avg_price', brand)
        material_avg = get_enc_val('material_avg_price', str(req["upper_material"]).strip())
        style_avg = get_enc_val('style_avg_price', str(req["style"]).strip())
        sole_avg = get_enc_val('sole_avg_price', str(req["sole_material"]).strip())
        close_avg = get_enc_val('close_style_avg_price', str(req["close_style"]).strip())
        season_avg = get_enc_val('season_avg_price', str(req["season"]).strip())

        brand_material_key = brand + '_' + str(req["upper_material"]).strip()
        brand_style_key = brand + '_' + str(req["style"]).strip()
        brand_material_avg = bm_map.get(brand_material_key, global_avg)
        brand_style_avg = bs_map.get(brand_style_key, global_avg)

        input_df = pd.DataFrame([{
            "brand_avg_price": brand_avg,
            "material_avg_price": material_avg,
            "style_avg_price": style_avg,
            "sole_avg_price": sole_avg,
            "close_style_avg_price": close_avg,
            "season_avg_price": season_avg,
            "brand_material_avg_price": brand_material_avg,
            "brand_style_avg_price": brand_style_avg,
            "comment_count": int(req["comment_count"]),
            "good_rate": float(req["good_rate"]),
            "brand": brand,
            "upper_material": str(req["upper_material"]).strip(),
            "sole_material": str(req["sole_material"]).strip(),
            "close_style": str(req["close_style"]).strip(),
            "style": str(req["style"]).strip(),
            "season": str(req["season"]).strip()
        }])

        X_input = price_preprocessor214jiaosiyao.transform(input_df)
        if hasattr(X_input, 'toarray'):
            X_input = X_input.toarray()

        y_pred = price_model214jiaosiyao.predict(X_input)[0]

        if brand_map_data214jiaosiyao.get('use_log', False):
            predicted_price = float(np.expm1(y_pred))
        else:
            predicted_price = float(y_pred)

        predicted_price = round(max(predicted_price, 0), 2)

        return jsonify({
            "code": 200,
            "predicted_price": predicted_price,
            "brand_avg_price": round(brand_avg, 1)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 400, "msg": f"参数错误：{str(e)}"})


# 页面路由
@app214jiaosiyao.route('/')
def dashboard214jiaosiyao():
    return render_template('index214jiaosiyao.html')


@app214jiaosiyao.route('/predict')
def predict_page214jiaosiyao():
    return render_template('predict214jiaosiyao.html')


@app214jiaosiyao.route('/price_predict')
def price_predict_page214jiaosiyao():
    return render_template('price_predict214jiaosiyao.html')

if __name__ == '__main__':
    print(" * 可视化大屏：http://127.0.0.1:5000")
    print(" * 聚类预测界面：http://127.0.0.1:5000/predict")
    print(" * 回归预测界面：http://127.0.0.1:5000/price_predict")
    app214jiaosiyao.run(debug=False, host='127.0.0.1', port=5000)