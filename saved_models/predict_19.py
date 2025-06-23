import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

def load_preprocessors(model_dir):
    """加载预处理器"""
    with open(os.path.join(model_dir, 'preprocessors.pkl'), 'rb') as f:
        preprocessors = pickle.load(f)
    # 只返回 scaler，因为我们不需要 selector
    return preprocessors['scaler']

def predict_load_shedding(input_file='input_template_19.xlsx', 
                         model_dir='upgrade_19',
                         output_file='predictions_19.csv',
                         n_features=5):  # 添加参数控制特征数量
    """
    预测负荷削减潜力
    """
    # 加载数据
    try:
        df = pd.read_excel(input_file)
        # 只选择前n_features列
        df = df.iloc[:, :n_features]
        print(f"使用的特征: {list(df.columns)}")
    except Exception as e:
        raise Exception(f"读取Excel文件失败: {str(e)}")
    
    # 加载模型和预处理器
    model = load_model(os.path.join(model_dir, 'nn_model.keras'))
    scaler = load_preprocessors(model_dir)
    
    # 预处理数据
    X = df.values
    # 只对选择的特征进行缩放
    scaler_mean = scaler.mean_[:n_features]
    scaler_scale = scaler.scale_[:n_features]
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    X_scaled = scaler.transform(X)
    
    # 预测
    predictions = model.predict(X_scaled)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'building_id': df.index + 1,  # 从1开始编号
        'predicted_load_shedding_kw': predictions.flatten()
    })
    
    # 保存结果
    results.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        predict_load_shedding()
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")