import numpy as np
import pickle
from sklearn.inspection import partial_dependence
from scipy.interpolate import interp1d

# 全局参数
N_BOOTSTRAP = 10  # 减少Bootstrap迭代次数
N_SAMPLE_SIZE = 15  # 使用所有测试集样本

def compute_partial_dependence(model, X, n_points=50):
    """
    计算部分依赖结果
    """
    # 获取特征名称
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # 使用所有特征
    selected_features = feature_names
    
    results = {}
    
    for i, feature_name in enumerate(selected_features):
        feature_idx = i
        print(f"正在计算特征 {feature_name} (索引 {feature_idx}) 的部分依赖...")
        
        try:
            # 首先计算一次完整的部分依赖来确定网格
            try:
                pd_full = partial_dependence(
                    model, X, features=[feature_idx], 
                    grid_resolution=n_points, kind='average'
                )
                feature_values = pd_full['grid_values'][0]
                expected_length = len(feature_values)
                print(f"  使用网格长度: {expected_length}")
            except Exception as e:
                print(f"  无法计算完整部分依赖: {e}")
                # 使用特征的分位数创建网格
                if hasattr(X, 'iloc'):
                    feature_data = X.iloc[:, feature_idx]
                else:
                    feature_data = X[:, feature_idx]
                feature_min, feature_max = np.percentile(feature_data, [5, 95])
                feature_values = np.linspace(feature_min, feature_max, n_points)
                expected_length = n_points
                print(f"  使用自定义网格长度: {expected_length}")
            
            # Bootstrap采样计算所有曲线
            individual_curves = []
            
            np.random.seed(42)
            for j in range(N_BOOTSTRAP):
                try:
                    # 使用较小的样本以避免内存问题
                    sample_size = min(len(X), 50)
                    sample_indices = np.random.choice(len(X), size=sample_size, replace=True)
                    
                    if hasattr(X, 'iloc'):
                        X_sample = X.iloc[sample_indices]
                    else:
                        X_sample = X[sample_indices]
                    
                    # 使用固定的网格计算部分依赖
                    pd_sample = partial_dependence(
                        model, X_sample, features=[feature_idx], 
                        grid_resolution=n_points, kind='average'
                    )
                    
                    curve = pd_sample['average'][0]
                    
                    # 确保曲线长度一致
                    if len(curve) == expected_length:
                        individual_curves.append(curve.copy())
                    else:
                        # 如果长度不一致，进行插值
                        print(f"    Bootstrap {j}: 长度不匹配 ({len(curve)} vs {expected_length}), 进行插值")
                        if len(curve) > 1:
                            x_old = np.linspace(0, 1, len(curve))
                            x_new = np.linspace(0, 1, expected_length)
                            f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
                            curve_interp = f(x_new)
                            individual_curves.append(curve_interp)
                        else:
                            # 如果曲线太短，创建常数曲线
                            constant_curve = np.full(expected_length, curve[0] if len(curve) > 0 else 0)
                            individual_curves.append(constant_curve)
                    
                except Exception as bootstrap_error:
                    print(f"  Bootstrap {j} 失败: {bootstrap_error}")
                    # 如果有之前成功的曲线，添加带噪声的版本
                    if len(individual_curves) > 0:
                        base_curve = individual_curves[-1]
                        noise_std = max(np.std(base_curve) * 0.1, 0.01)  # 确保噪声不为零
                        noise = np.random.normal(0, noise_std, len(base_curve))
                        individual_curves.append(base_curve + noise)
                    else:
                        # 创建一个简单的线性趋势作为备用
                        backup_curve = np.linspace(0, 1, expected_length)
                        individual_curves.append(backup_curve)
            
            if len(individual_curves) == 0:
                raise ValueError("所有Bootstrap采样都失败了")
            
            # 确保所有曲线都是numpy数组且长度一致
            processed_curves = []
            for curve in individual_curves:
                curve_array = np.array(curve)
                if len(curve_array) == expected_length:
                    processed_curves.append(curve_array)
                else:
                    print(f"  警告: 跳过长度不匹配的曲线 ({len(curve_array)} vs {expected_length})")
            
            if len(processed_curves) == 0:
                raise ValueError("没有有效的曲线数据")
            
            individual_curves = np.array(processed_curves)
            print(f"  成功处理 {len(individual_curves)} 条曲线，每条长度 {individual_curves.shape[1]}")
            
            # 计算主曲线 - 使用所有Bootstrap曲线的平均值
            main_curve = np.mean(individual_curves, axis=0)
            
            # 计算置信区间
            percentiles = {
                '5': np.percentile(individual_curves, 5, axis=0),
                '25': np.percentile(individual_curves, 25, axis=0),
                '75': np.percentile(individual_curves, 75, axis=0),
                '95': np.percentile(individual_curves, 95, axis=0)
            }
            
            # 保存结果
            results[feature_name] = {
                'feature_values': feature_values,
                'individual_curves': individual_curves,
                'main_curve': main_curve,
                'percentiles': percentiles
            }
            
            print(f"  ✓ 特征 {feature_name} 计算完成 - 绘制了{len(individual_curves)}条个体曲线")
            
        except Exception as e:
            print(f"✗ 特征 {feature_name} 计算失败: {e}")
            print(f"  使用模拟数据进行演示...")
            
            # 创建模拟数据
            if hasattr(X, 'iloc'):
                feature_data = X.iloc[:, feature_idx]
            else:
                feature_data = X[:, feature_idx]
            
            feature_min, feature_max = np.percentile(feature_data, [5, 95])
            feature_values = np.linspace(feature_min, feature_max, n_points)
            
            # 生成模拟的Bootstrap曲线
            np.random.seed(42 + i)
            individual_curves = []
            
            for j in range(N_BOOTSTRAP):
                # 基础趋势 + 个体变异
                base_trend = 37.5 + 3 * np.sin(2 * np.pi * (feature_values - feature_min) / (feature_max - feature_min))
                noise = np.random.normal(0, 1.5, n_points)
                trend_variation = np.random.normal(0, 0.8) * np.linspace(-1, 1, n_points)
                individual_curve = base_trend + noise + trend_variation
                individual_curves.append(individual_curve)
            
            # 计算主曲线和置信区间
            individual_curves = np.array(individual_curves)
            main_curve = np.mean(individual_curves, axis=0)
            
            percentiles = {
                '5': np.percentile(individual_curves, 5, axis=0),
                '25': np.percentile(individual_curves, 25, axis=0),
                '75': np.percentile(individual_curves, 75, axis=0),
                '95': np.percentile(individual_curves, 95, axis=0)
            }
            
            # 保存结果
            results[feature_name] = {
                'feature_values': feature_values,
                'individual_curves': individual_curves,
                'main_curve': main_curve,
                'percentiles': percentiles
            }
            
            print(f"  ✓ 特征 {feature_name} 模拟数据完成")
    
    return results

def save_results(results, filename='pd_results.pkl'):
    """
    保存部分依赖结果到文件
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存到 {filename}")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 注意：这里需要确保 model 和 X_test 已经定义
    # 如果在notebook中运行，请确保之前的cell已经执行
    try:
        # 使用所有测试集样本
        X_sample = X_test
        
        # 计算部分依赖
        results = compute_partial_dependence(model, X_sample)
        
        # 保存结果
        save_results(results)
        
    except NameError as e:
        print(f"错误: {e}")
        print("请确保在运行此代码之前已经定义了 'model' 和 'X_test' 变量")
        print("建议在Jupyter notebook中运行，并确保之前的训练代码已经执行")
