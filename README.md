# BangDreamDifficultyPrediction
一个能够预测Bangdream游戏谱面难度的模型

# 文件结构

├── charts/  bestdori谱面json文件  
├── dataset/ 数据集保存目录  
├── model/   模型保存目录  
├── all_predict.py     对charts/所有谱面进行难度预测并输出cs表格  
├── build_dataset.py   对charts/所有谱面构建数据集  
├── data_utils.py      数据操作方法  
├── get_charts.py      下载谱面json文件，需要bestdori-api模块  
├── point_predict.py   预测相对难度并生成折线图  
├── predict.py         预测指定谱面难度  
├── segment_predict.py 预测累加难度并生成折线图和动画  
├── train.py           模型训练  
└── README.md          仓库文档  
