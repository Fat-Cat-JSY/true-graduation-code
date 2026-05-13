from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pymysql
from config import *

app = Flask(__name__, template_folder='templates')
app.config['JSON_AS_ASCII'] = False

# 全局特征定义
numeric_features = ['comment_count', 'user_age', 'original_price', 'discount_rate', 'price', 'good_rate', 'is_return']
categorical_features = ['close_style', 'sole_material', 'style', 'brand', 'is_plus']

# 加载聚类模型
KMEANS_MODEL_PATH = "../../models/kmeans_cluster.pkl"
PREPROCESSOR_PATH = "../../models/preprocessor.pkl"
kmeans_model = joblib.load(KMEANS_MODEL_PATH)
preprocessor_global = joblib.load(PREPROCESSOR_PATH)
print("✅ 聚类模型加载完成")

# 数据库表配置，JSON格式存储所有表信息
ALL_TABLES_CONFIG = {
    "train_data": {
        "table_name": "clean_shoe_train_data",
        "table_desc": "清洗后的训练主数据"
    },
    "brand_analysis": {
        "table_name": "brand_analysis",
        "table_desc": "品牌分析结果表"
    },
    "material_analysis": {
        "table_name": "material_analysis",
        "table_desc": "鞋面材质分析结果表"
    },
    "user_analysis": {
        "table_name": "user_analysis",
        "table_desc": "用户群体分析结果表"
    },
    "time_analysis": {
        "table_name": "time_analysis",
        "table_desc": "下单时段分析结果表"
    },
    "province_top10": {
        "table_name": "province_top10",
        "table_desc": "省份订单Top10结果表"
    },
    "price_bin_analysis": {
        "table_name": "price_bin_analysis",
        "table_desc": "价格区间分析结果表"
    }
}

# 创建MySQL连接
def get_mysql_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PWD,
        database=MYSQL_DB,
        port=3306,
        charset=MYSQL_CHARSET
    )

# 通用方法：从MySQL获取指定表全量数据
def get_table_data(table_key):
    if table_key not in ALL_TABLES_CONFIG:
        print(f"未找到表配置：{table_key}")
        return None
    table_name = ALL_TABLES_CONFIG[table_key]["table_name"]
    conn = get_mysql_conn()
    try:
        sql = f"SELECT * FROM `{table_name}`;"
        df = pd.read_sql(sql, conn)
        print(f"成功获取{ALL_TABLES_CONFIG[table_key]['table_desc']}，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"获取{table_name}失败：{str(e)}")
        return None
    finally:
        conn.close()

# 价格区间分析数据接口
@app.route('/api/train_scatter', methods=['GET'])
def get_train_scatter():
    df = get_table_data("price_bin_analysis")
    if df is None or len(df) == 0:
        return jsonify({"code": 500, "msg": "获取数据失败"})

    # 把价格区间转成数字x，方便散点定位
    bin_to_x = {
        "0-100元": 50,
        "100-200元": 150,
        "200-300元": 250,
        "300元以上": 350
    }
    # 转成前端需要的完整格式
    res = []
    for _, row in df.iterrows():
        x_val = bin_to_x.get(row['price_bin'], 0)
        res.append({
            "x": float(x_val),
            "y": float(row['goods_count']),  # 改成商品数量
            "z": float(row['return_rate']) * 100,  # 退货率（原先是return_rate，已*100）
            "price_bin": row['price_bin'],
            "total_month_sale": float(row['total_month_sale']),
            "goods_count": float(row['goods_count']),
            "return_rate": float(row['return_rate'])
        })
    print(f"价格区间散点图返回数据: {res}")
    return jsonify({
        "code": 200,
        "data": res
    })

# 品牌分析数据接口
@app.route('/api/dashboard/brand', methods=['GET'])
def get_brand_data():
    df = get_table_data("brand_analysis")
    if df is None:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 省份Top10数据接口
@app.route('/api/dashboard/province', methods=['GET'])
def get_province_data():
    df = get_table_data("province_top10")
    if df is None:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 价格区间分析数据接口
@app.route('/api/dashboard/price_bin', methods=['GET'])
def get_price_data():
    df = get_table_data("price_bin_analysis")
    if df is None:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 用户分析数据接口
@app.route('/api/dashboard/user', methods=['GET'])
def get_user_data():
    df = get_table_data("user_analysis")
    if df is None:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 材质分析数据接口
@app.route('/api/dashboard/material', methods=['GET'])
def get_material_data():
    df = get_table_data("material_analysis")
    # print(f"材质分析表拿到数据行数: {len(df) if df is not None else 0}")
    # print(f"材质分析表内容: \n{df}")
    if df is None or len(df) == 0:
        return jsonify({"code": 200, "data": []})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 下单时段分析数据接口
@app.route('/api/dashboard/time', methods=['GET'])
def get_time_data():
    df = get_table_data("time_analysis")
    if df is None:
        return jsonify({"code": 500, "msg": "获取数据失败"})
    return jsonify({
        "code": 200,
        "data": df.to_dict(orient='records')
    })

# 聚类预测接口
@app.route('/api/cluster_predict', methods=['POST'])
def cluster_predict():
    try:
        req_data = request.get_json()

        input_df = pd.DataFrame([{
            "comment_count": int(req_data["comment_count"]),
            "user_age": int(req_data["user_age"]),
            "original_price": float(req_data["original_price"]),
            "discount_rate": float(req_data["discount_rate"]),
            "price": float(req_data["price"]),
            "good_rate": float(req_data["good_rate"]),
            "is_return": str(int(req_data["is_return"])),
            "close_style": str(req_data["close_style"]).strip(),
            "sole_material": str(req_data["sole_material"]).strip(),
            "style": str(req_data["style"]).strip(),
            "brand": str(req_data["brand"]).strip(),
            "is_plus": str(int(req_data["is_plus"]))
        }])

        X_input = preprocessor_global.transform(input_df)
        if hasattr(X_input, 'toarray'):
            X_input = X_input.toarray()
        cluster_id = int(kmeans_model.predict(X_input)[0])

        cluster_desc = [
            "该商品属于【平价高评论 会员偏好 简约百搭款】群体：占总样本42%，评论数正常，原价偏低，会员购买占比高，主打简约百搭风格，不支持7天无理由退货，平均月销量约1018件，属于典型的平价走量款，构成平台销量基本盘。",
            "该商品属于【超高流量 非会员 日常休闲款】群体：占总样本3%，评论数是其他群体的10倍，原价略高于平价群体，非会员购买占比最高，主打日常休闲风格，平均月销量约973件，属于高流量低转化的小众网红款。",
            "该商品属于【高端高价 会员偏好 工装品牌款】群体：占总样本12%，平均原价约487元（是平价群体的2.7倍），会员购买占比高，主打工装硬朗风格，品牌以一线运动品牌为主，平均月销量约1059件，是四个群体中销量最高的品类，利润空间最大，属于核心高价值群体。",
            "该商品属于【平价高评论 会员偏好 复古潮鞋款】群体：占总样本43%，原价179元，会员购买占比高，全部支持7天无理由退货，主打复古潮鞋风格，平均月销量约1016件，和简约百搭款共同构成平台平价市场的销量基本盘。"
        ][cluster_id] if cluster_id < 4 else "该商品特征超出模型训练范围，属于未知类别"
        return jsonify({
            "code": 200,
            "cluster_id": cluster_id,
            "cluster_desc": cluster_desc
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 400, "msg": f"参数错误：{str(e)}"})

# 👇 页面路由全部放在全局，保证跳转正常
# 根路由 = 可视化大屏（你的index就是大屏）
@app.route('/')
def dashboard():
    return render_template('index.html')

# 预测页面路由
@app.route('/predict')
def predict_page():
    return render_template('predict.html')

if __name__ == '__main__':
    print(" * 可视化大屏：http://127.0.0.1:5000")
    print(" * 销量聚类预测界面：http://127.0.0.1:5000/predict")
    app.run(debug=False, host='127.0.0.1', port=5000)