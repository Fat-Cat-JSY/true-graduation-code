import os
import pandas as pd
import subprocess
from sqlalchemy import create_engine
from config import (MYSQL_HOST214jiaosiyao, MYSQL_USER214jiaosiyao, MYSQL_PWD214jiaosiyao,
                    MYSQL_DB214jiaosiyao, MYSQL_CHARSET214jiaosiyao, HDFS_NAMENODE214jiaosiyao, SPARK_MASTER214jiaosiyao)

# 配置参数
RAW_DATA_PATH214jiaosiyao = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bdu26bd214jiaosiyao_data", "bdu26bd214jiaosiyao_raw", "noClean_data.csv")
CLEAN_CSV_PATH214jiaosiyao = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bdu26bd214jiaosiyao_data", "bdu26bd214jiaosiyao_clean", "clean_data.csv")
HDFS_TARGET_DIR214jiaosiyao = "/user/hive/warehouse/clean_shoe_data/"
MYSQL_TABLE_NAME214jiaosiyao = "clean_shoe_train_data"

# 存储模式：0=仅CSV, 1=CSV+MySQL, 2=CSV+MySQL+HDFS
STORAGE_MODE214jiaosiyao = 2

# 数据读取与预处理
df = pd.read_csv(RAW_DATA_PATH214jiaosiyao, encoding="utf-8-sig")

# 重命名列名
df.rename(columns={
    "商品编号": "goods_id",
    "商品名称": "goods_name",
    "品牌": "brand",
    "价格（元）": "price",
    "原价（元）": "original_price",
    "折扣率": "discount_rate",
    "月销量": "month_sale",
    "累计评价数": "comment_count",
    "好评率": "good_rate",
    "鞋面材质": "upper_material",
    "鞋底材质": "sole_material",
    "闭合方式": "close_style",
    "风格": "style",
    "适用季节": "season",
    "用户年龄": "user_age",
    "用户性别": "user_gender",
    "用户所在省份": "province",
    "是否PLUS会员": "is_plus",
    "下单时间（小时）": "order_hour",
    "是否退货": "is_return"
}, inplace=True)

print(f"原始数据共 {len(df)} 行，开始预处理...")

# 填充缺失值
missing_report = df.isnull().sum()
missing_report = missing_report[missing_report > 0]
if len(missing_report) > 0:
    print("缺失值报告：")
    for col, cnt in missing_report.items():
        print(f"  {col}: {cnt}条缺失 ({cnt / len(df) * 100:.2f}%)")
else:
    print("无缺失值")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna("未知")
    else:
        df[col] = df[col].fillna(df[col].median())

# 是否会员转0/1
df['is_plus'] = df['is_plus'].map(lambda x: 1 if str(x).strip() == "是" else 0)
if "is_return" in df.columns:
    df["is_return"] = df["is_return"].map(lambda x: 1 if str(x).strip() == "是" else 0)

# 下单时段划分
def get_time_slot214jiaosiyao(hour):
    hour = int(hour)
    if 0 <= hour < 7:
        return "深夜(0-6点)"
    elif 7 <= hour < 13:
        return "上午(7-12点)"
    elif 13 <= hour < 19:
        return "下午(13-18点)"
    else:
        return "晚上(19-23点)"
df["order_time_slot"] = df["order_hour"].apply(get_time_slot214jiaosiyao)

# 异常值过滤
before_count = len(df)
df = df[
    (df["original_price"] > 0) &
    (df["month_sale"] >= 0) &
    (df["discount_rate"] >= 0) & (df["discount_rate"] <= 100) &
    (df["good_rate"] >= 0) & (df["good_rate"] <= 100) &
    (df["price"] <= df["original_price"]) &
    (df["user_age"] >= 18) & (df["user_age"] <= 80)
].reset_index(drop=True)
print(f"异常值过滤：移除 {before_count - len(df)} 条，剩余 {len(df)} 条")
print(f"预处理完成，剩余 {len(df)} 行干净数据")

# 保存本地CSV
df.to_csv(CLEAN_CSV_PATH214jiaosiyao, index=False, encoding="utf-8")
print(f"干净CSV已保存到: {CLEAN_CSV_PATH214jiaosiyao}")

# 存储到MySQL+HDFS
if STORAGE_MODE214jiaosiyao == "2":
    engine = create_engine(
        f'mysql+pymysql://{MYSQL_USER214jiaosiyao}:{MYSQL_PWD214jiaosiyao}@{MYSQL_HOST214jiaosiyao}:3306/{MYSQL_DB214jiaosiyao}?charset=utf8mb4&use_unicode=1',
        echo=False
    )
    with engine.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {MYSQL_TABLE_NAME214jiaosiyao};")
    df.to_sql(
        name=MYSQL_TABLE_NAME214jiaosiyao, con=engine, if_exists="replace",
        index=False, chunksize=1000
    )
    print(f"  干净数据已存入MySQL表: clean_shoe_train_data")

    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", HDFS_TARGET_DIR214jiaosiyao], capture_output=True)
    result = subprocess.run(
        ["hdfs", "dfs", "-put", "-f", CLEAN_CSV_PATH214jiaosiyao, HDFS_TARGET_DIR214jiaosiyao],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"干净CSV已上传至HDFS: {HDFS_TARGET_DIR214jiaosiyao}")
    else:
        print(f"HDFS上传失败: {result.stderr}")

print("  预处理完成！")