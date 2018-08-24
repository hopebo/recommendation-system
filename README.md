# Recommendation-System

## Overview

*CF.py*：Item-based collaborative filtering algorithm.

*ICCF.py*：Introduce item clustering into collaborative filtering algorithm.

*SVD.py*：Singular Value Decomposition algorithm.

*data_manager.py*：Data process related class.

*multiProcess.py*: Concurrently calculate Pearson Correlation Coefficient by *Ratings* and Euclidean Distance by *Tag Genomes*.

## Utilizing the Project

Previously run *multiProcess.py* to calculate correlation coefficients and save them locally. Then use *CF.py*, *ICCF.py* or *SVD.py* to train different recommendation models.

## Benchmark

### RMSE comparison of different algorithms

|    Method     |  RMSE  |
| :-----------: | :----: |
|   MovieAvg    | 1.1162 |
| Item-based CF | 0.9770 |
|      SVD      | 0.9256 |
|     ICCF      | 0.9113 |

### Precision and Recall comparison between ICCF and SVD

<img src="https://github.com/hopebo/Recommendation-System/blob/master/images/Recall%26Precision.png" width="150" height="200" alt="Precision and Recall"/>

### F-score comparsion between ICCF and SVD

![F-score](https://github.com/hopebo/Recommendation-System/blob/master/images/F-score.png){:height="70%" width="70%"}