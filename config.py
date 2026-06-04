# 路径配置
import os
BASE_DIR214jiaosiyao = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH214jiaosiyao = os.path.join(BASE_DIR214jiaosiyao, "bdu26bd214jiaosiyao_data", "bdu26bd214jiaosiyao_raw","noClean_data.csv")
CLEAN_DATA_PATH214jiaosiyao = os.path.join(BASE_DIR214jiaosiyao, "bdu26bd214jiaosiyao_data","bdu26bd214jiaosiyao_clean", "clean_data.csv")

# MySQL配置
MYSQL_HOST214jiaosiyao = "127.0.0.1"
MYSQL_USER214jiaosiyao = "root"
MYSQL_PWD214jiaosiyao = "111111"
MYSQL_DB214jiaosiyao = "bishe_nanxie_db"
MYSQL_CHARSET214jiaosiyao = "utf8mb4"

# Spark和集群配置
SPARK_MASTER214jiaosiyao = "spark://192.168.64.101:7077"
SPARK_HOME214jiaosiyao = "/opt/apps/spark"
PYTHON_HOME214jiaosiyao = "/usr/bin/python3"
HDFS_NAMENODE214jiaosiyao = "hdfs://192.168.64.101:9000"
HIVE_METASTORE_URIS214jiaosiyao = "thrift://192.168.64.101:9083"
HADOOP_CONF_DIR214jiaosiyao = "/opt/apps/hadoop/etc/hadoop"