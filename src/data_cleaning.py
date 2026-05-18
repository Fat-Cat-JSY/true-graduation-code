import pandas as pd
import subprocess
from sqlalchemy import create_engine
from config import (MYSQL_HOST, MYSQL_USER, MYSQL_PWD, MYSQL_DB, MYSQL_CHARSET,
                    HDFS_NAMENODE, SPARK_MASTER)   # 从config导入HDFS_NAMENODE

# ================= 配置参数 =================
RAW_DATA_PATH = "D:/毕设代码/nanXieDaPing/data/raw/noClean_data.csv"
CLEAN_CSV_PATH = "D:/毕设代码/nanXieDaPing/data/clean/clean_data.csv"
HDFS_TARGET_DIR = "/user/hive/warehouse/clean_shoe_data/"
MYSQL_TABLE_NAME = "clean_shoe_train_data"

# 存储模式：0=仅CSV, 1=CSV+MySQL, 2=CSV+MySQL+HDFS
STORAGE_MODE = 2   # 正式运行设为2

# ================= 数据读取与预处理 =================
df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")

# 重命名列名，统一字段格式
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


# 通用预处理
print(f"原始数据共 {len(df)} 行，开始预处理...")

# 缺失值填充：分类特征填充未知，数值特征填充均值
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

# 处理是否会员is_plus
# 原始是"是/否"中文，转成0/1
df['is_plus'] = df['is_plus'].map(lambda x: 1 if str(x).strip() == "是" else 0)
# 二值字段转换：是否退货转换为0/1
if "is_return" in df.columns:
    df["is_return"] = df["is_return"].map(lambda x: 1 if str(x).strip() == "是" else 0)

# 下单时段划分：按小时分为四个时段
def get_time_slot(hour):
    hour = int(hour)
    if 0 <= hour < 7:
        return "深夜(0-6点)"
    elif 7 <= hour < 13:
        return "上午(7-12点)"
    elif 13 <= hour < 19:
        return "下午(13-18点)"
    else:
        return "晚上(19-23点)"
df["order_time_slot"] = df["order_hour"].apply(get_time_slot)

# 异常值过滤：去除年龄、价格、销量异常的脏数据
before_count = len(df)
df = df[
    (df["original_price"] > 0) &
    (df["month_sale"] >= 0) &
    (df["discount_rate"] >= 0) & (df["discount_rate"] <= 100) &
    (df["good_rate"] >= 0) & (df["good_rate"] <= 100) &
    (df["price"] <= df["original_price"]) &  # 折扣价不应高于原价
    (df["user_age"] >= 18) & (df["user_age"] <= 80)  # 合理年龄范围
].reset_index(drop=True)
print(f"异常值过滤：移除 {before_count - len(df)} 条，剩余 {len(df)} 条")
print(f"预处理完成，剩余 {len(df)} 行干净数据")

# 存储预处理结果
# 保存本地CSV
df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8")
print(f"干净CSV已保存到: {CLEAN_CSV_PATH}")

# 存储到MySQL+HDFS
if STORAGE_MODE == "2":
    engine = create_engine(
        f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:3306/{MYSQL_DB}?charset=utf8mb4&use_unicode=1',
        echo=False
    )
    with engine.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {MYSQL_TABLE_NAME};")
    df.to_sql(
        name=MYSQL_TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=1000
    )
    print(f"  干净数据已存入MySQL表: clean_shoe_train_data")

    hdfs_full_path = f"{HDFS_NAMENODE}{HDFS_TARGET_DIR}"
    # 确保目录存在（若不存在则创建）
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", HDFS_TARGET_DIR], capture_output=True)
    # 上传文件（覆盖）
    result = subprocess.run(
        ["hdfs", "dfs", "-put", "-f", CLEAN_CSV_PATH, HDFS_TARGET_DIR],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"干净CSV已上传至HDFS: {HDFS_TARGET_DIR}")
    else:
        print(f"HDFS上传失败: {result.stderr}")

print("  预处理全流程完成！")