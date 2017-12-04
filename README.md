# Recommendation-System

## 包含的文件

*CF.py*：基于物品的协同过滤模型的实现

*ICCF.py*：在协同过滤模型的基础上引入标签数据聚类的算法，通过随机梯度下降优化两者的参数比例

*SVD.py*：矩阵分解算法

*data_manager.py*：与数据相关的处理类

*multiProcess.py*：使用多进程离线计算*item-item*之间由*Ratings*计算皮尔逊相关系数的相似度和由*Tag Genomes*计算欧式距离得到的相似度

## 用法说明

预先执行*multiProcess.py*文件计算出相似度并存储到本地，然后可以采用*CF*、*ICCF*、*SVD*等算法构造推荐模型。