# -*- coding: utf-8 -*-
import os
import sys
os.environ['SPARK_HOME'] = '/opt/apps/spark'
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python/lib/py4j-0.10.9.7-src.zip'))

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from config import *

# 配置参数
hive_table_name = "nanxie_clean"
result_save_path = "/bishe/nanxie/analysis_result_hive"

# 初始化连接Hive的Spark会话
spark = SparkSession.builder \
    .appName("nanxie_hive_mysql") \
    .master("local[*]") \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .config("spark.driver.extraClassPath", "/opt/apps/spark/jars/mysql-connector-j-8.0.33.jar") \
    .enableHiveSupport() \
    .getOrCreate()

print("  Spark初始化完成，连接Hive...")
df = spark.sql(f"SELECT * FROM {hive_table_name} WHERE brand IS NOT NULL AND brand != 'brand'")
print(f"  从Hive读取成功！总数据量：{df.count()} 行")
df.createOrReplaceTempView("clean_shoes")

# 分析1：品牌维度统计
print("\n===== 【分析1：不同品牌核心指标】=====")
brand_analysis = spark.sql("""
SELECT 
    brand,
    COUNT(*) AS goods_count,
    SUM(month_sale) AS total_month_sale,
    ROUND(AVG(price), 2) AS avg_price,
    ROUND(AVG(is_return), 4) AS return_rate,
    ROUND(AVG(good_rate), 2) AS avg_good_rate
FROM clean_shoes
GROUP BY brand
ORDER BY total_month_sale DESC
""")
brand_analysis.show(truncate=False)
brand_analysis.coalesce(1).write.csv(
    path="/home/analysis/brand_analysis",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

# 分析2：材质维度统计
print("\n===== 【分析2：不同鞋面材质核心指标】=====")
material_analysis = spark.sql("""
SELECT 
    upper_material,
    COUNT(*) AS goods_count,
    SUM(month_sale) AS total_month_sale,
    ROUND(AVG(price), 2) AS avg_price
FROM clean_shoes
GROUP BY upper_material
ORDER BY total_month_sale DESC
""")
material_analysis.show(truncate=False)
material_analysis.coalesce(1).write.csv(
    path="/home/analysis/material_analysis",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

# 分析3：用户维度统计
print("\n===== 【分析3：不同用户群体核心指标】=====")
user_analysis = spark.sql("""
SELECT 
    user_gender,
    is_plus,
    COUNT(*) AS order_count,
    ROUND(AVG(is_return), 4) AS return_rate,
    ROUND(AVG(user_age), 1) AS avg_user_age
FROM clean_shoes
GROUP BY user_gender, is_plus
ORDER BY order_count DESC
""")
user_analysis.show(truncate=False)
user_analysis.coalesce(1).write.csv(
    path="/home/analysis/user_analysis",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

# 分析4：时段维度统计
print("\n===== 【分析4：不同下单时段核心指标】=====")
time_analysis = spark.sql("""
SELECT 
    order_time_slot,
    COUNT(*) AS order_count,
    ROUND(AVG(is_return), 4) AS return_rate
FROM clean_shoes
GROUP BY order_time_slot
ORDER BY 
    CASE order_time_slot 
        WHEN '深夜(0-6点)' THEN 1
        WHEN '上午(7-12点)' THEN 2
        WHEN '下午(13-18点)' THEN 3
        WHEN '晚上(19-23点)' THEN 4
    END
""")
time_analysis.show(truncate=False)
time_analysis.coalesce(1).write.csv(
    path="/home/analysis/time_analysis",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

# 分析5：省份维度统计
print("\n===== 【分析5：下单量Top10省份】=====")
province_analysis = spark.sql("""
SELECT 
    province,
    COUNT(*) AS order_count,
    ROUND(SUM(month_sale), 0) AS total_sale
FROM clean_shoes
GROUP BY province
ORDER BY order_count DESC
LIMIT 10
""")
province_analysis.show(truncate=False)
province_analysis.coalesce(1).write.csv(
    path="/home/analysis/province_top10",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

# 分析6：价格区间维度统计
print("\n===== 【分析6：不同价格区间销量统计】=====")
price_bin_analysis = df.withColumn(
    "price_bin",
    when(col("price") <= 100, "0-100元")
   .when((col("price") > 100) & (col("price") <= 200), "100-200元")
   .when((col("price") > 200) & (col("price") <= 300), "200-300元")
   .otherwise("300元以上")
).groupBy("price_bin").agg(
    count("*").alias("goods_count"),
    sum("month_sale").alias("total_month_sale"),
    round(avg("is_return"), 4).alias("return_rate")
).orderBy(
    when(col("price_bin") == "0-100元", 1)
   .when(col("price_bin") == "100-200元", 2)
   .when(col("price_bin") == "200-300元", 3)
   .when(col("price_bin") == "300元以上", 4)
)

price_bin_analysis.show(truncate=False)
price_bin_analysis.coalesce(1).write.csv(
    path="/home/analysis/price_bin_analysis",
    header=True,
    encoding="UTF-8",
    mode="overwrite"
)

print(f"\n  全量分析完成！所有结果都已经导出到 /home/analysis/ 目录下")
spark.stop()