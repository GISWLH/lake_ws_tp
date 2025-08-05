# 需求文档

## 介绍

本项目是一个湖泊水量模拟归因模型

是通过tabpfn模型实现的，tabpfn是一个预训练的表格树模型，主要是通过from tabpfn import TabPFNRegressor实现的，和随机森林类似

# 定义特征和目标变量
feature_columns = ['ET', 'GPP', 'LST_DAY', 'NPP', 'PRECIPITATION', 'Soil_Moisture', 'SRAD', 'Temperature', \
    'Snow_Cover', 'Wind_Speed', 'Vap', 'NDVI', 'Nighttime', 'CO2', 'Cropland_Area', 'Forest_Area', 'Steppe_Area',\
         'Non-Vegetated/Artificial Land_Area', 'Wetland_Area', 'Snow/Ice_Area', 'Population_Density', 'Human Influence Index', 'CH4', 'N2O', 'SF6']

y = df['Water_Volumn_Change']

## 重要

运行代码所需的环境在conda activate tab，因此运行py代码前先激活环境
timeout xx其中xx应该大于5分钟，因为代码运行较慢，可以多给一些时间
我不喜欢定义太过复杂的函数，并运行main函数，我是深度jupyter notebook用户，我喜欢直接的代码，简单的函数定义是可以接受的
使用matplotlib可视化，绘图使用Arial字体(在linux中手动增加我们的arial字体），绘图中的图片标记都用英文

## 组织

代码在code下

数据在data下

## 变量

Water_Volumn_Change为因变量，湖泊蓄水量变化

以下变量都为自变量

| ET   | GPP  | LST_DAY | NPP  | PRECIPITATION | Soil_Moisture | SRAD | Temperature | Snow_Cover | Wind_Speed | Vap  | NDVI | Nighttime | CO2  | Cropland_Area | Forest_Area | Steppe_Area | Non-Vegetated/Artificial Land_Area | Wetland_Area | Snow/Ice_Area | Population_Density | Human Influence Index | CH4  | N2O  | SF6  |
| ---- | ---- | ------- | ---- | ------------- | ------------- | ---- | ----------- | ---------- | ---------- | ---- | ---- | --------- | ---- | ------------- | ----------- | ----------- | ---------------------------------- | ------------ | ------------- | ------------------ | --------------------- | ---- | ---- | ---- |
|      |      |         |      |               |               |      |             |            |            |      |      |           |      |               |             |             |                                    |              |               |                    |                       |      |      |      |