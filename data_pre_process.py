#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本
湖泊水量模拟归因模型数据预处理

本脚本用于处理湖泊水量变化数据，删除包含空值的行，并将处理后的数据保存为CSV格式。

项目介绍：
本项目是一个湖泊水量模拟归因模型，通过TabPFN模型实现。TabPFN是一个预训练的表格树模型，
主要通过from tabpfn import TabPFNRegressor实现，和随机森林类似。

变量说明：
- Water_Volumn_Change: 因变量，湖泊蓄水量变化
- 自变量包括：ET, GPP, LST_DAY, NPP, PRECIPITATION, Soil_Moisture, SRAD, 
  Temperature, Snow_Cover, Wind_Speed, Vap, NDVI, Nighttime, CO2, 
  Cropland_Area, Forest_Area, Steppe_Area, Non-Vegetated/Artificial Land_Area, 
  Wetland_Area, Snow/Ice_Area, Population_Density, Human Influence Index, 
  CH4, N2O, SF6

作者：湖泊水量模拟项目组
日期：2025年
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_data(file_path):
    """
    加载数据文件
    
    Args:
        file_path (str): 数据文件路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            print(f"成功加载Excel文件: {file_path}")
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(f"成功加载CSV文件: {file_path}")
        else:
            raise ValueError("不支持的文件格式，请使用Excel或CSV文件")
        
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        return df
    
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def check_missing_values(df):
    """
    检查数据中的缺失值
    
    Args:
        df (pd.DataFrame): 输入数据框
        
    Returns:
        pd.Series: 每列的缺失值数量
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    print("\n=== 缺失值统计 ===")
    print(f"总行数: {len(df)}")
    
    for col in df.columns:
        if missing_values[col] > 0:
            print(f"{col}: {missing_values[col]} 个缺失值 ({missing_percentage[col]:.2f}%)")
    
    # 统计包含任何缺失值的行数
    rows_with_missing = df.isnull().any(axis=1).sum()
    print(f"\n包含缺失值的行数: {rows_with_missing} ({(rows_with_missing/len(df)*100):.2f}%)")
    
    return missing_values

def remove_rows_with_missing_values(df):
    """
    删除包含任何缺失值的行
    
    Args:
        df (pd.DataFrame): 输入数据框
        
    Returns:
        pd.DataFrame: 删除缺失值后的数据框
    """
    original_shape = df.shape
    
    # 删除包含任何NaN值的行
    df_cleaned = df.dropna()
    
    new_shape = df_cleaned.shape
    removed_rows = original_shape[0] - new_shape[0]
    
    print(f"\n=== 数据清理结果 ===")
    print(f"原始数据形状: {original_shape}")
    print(f"清理后数据形状: {new_shape}")
    print(f"删除的行数: {removed_rows}")
    print(f"保留的数据比例: {(new_shape[0]/original_shape[0]*100):.2f}%")
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    保存清理后的数据为CSV格式
    
    Args:
        df (pd.DataFrame): 清理后的数据框
        output_path (str): 输出文件路径
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存为CSV文件
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n数据已成功保存到: {output_path}")
        print(f"保存的数据形状: {df.shape}")
        
    except Exception as e:
        print(f"保存数据时出错: {e}")

def main():
    """
    主函数：执行数据预处理流程
    """
    print("=== 湖泊水量模拟归因模型 - 数据预处理 ===")
    print("删除包含空值的数据行，转换为CSV格式保存")
    
    # 定义输入和输出路径
    input_files = [
        "total_1.csv",      # CSV文件（优先使用）
        "data/total.xlsx"   # Excel文件
    ]
    
    # 检查哪个文件存在
    input_file = None
    for file_path in input_files:
        if os.path.exists(file_path):
            input_file = file_path
            break
    
    if input_file is None:
        print("错误：找不到输入数据文件")
        print("请确保以下文件之一存在：")
        for file_path in input_files:
            print(f"  - {file_path}")
        return
    
    # 定义输出路径
    output_file = "data/cleaned_data.csv"
    
    print(f"\n使用输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 1. 加载数据
    df = load_data(input_file)
    if df is None:
        return
    
    # 2. 检查缺失值
    missing_values = check_missing_values(df)
    
    # 3. 删除包含缺失值的行
    df_cleaned = remove_rows_with_missing_values(df)
    
    # 4. 保存清理后的数据
    save_cleaned_data(df_cleaned, output_file)
    
    print("\n=== 数据预处理完成 ===")

# 直接执行代码（适合Jupyter notebook用户）
if __name__ == "__main__":
    main()
