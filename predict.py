import os
import numpy as np
import pandas as pd
import joblib
import json
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def load_models_and_resources():
    """加载训练好的模型和相关资源"""
    # 加载随机森林模型和相关文件
    rf_model = joblib.load('RF_Model/rf_model.pkl')
    encoder = joblib.load('RF_Model/feature_encoder.pkl')
    with open('RF_Model/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # 加载ARIMA模型
    arima_model = SARIMAXResults.load('ARIMA_Model/arima_model.pkl')

    return rf_model, encoder, feature_columns, arima_model


def prepare_rf_features(detector_id, direction, lane_id, hour, minute, day_of_week,
                        speed, occupancy, std_dev, encoder, feature_columns):
    """准备随机森林模型的特征"""
    # 创建基础特征DataFrame
    base_features = {
        'detector_id': [detector_id],
        'direction': [direction],
        'lane_id': [lane_id],
        'hour': [hour],
        'minute': [minute],
        'day_of_week': [day_of_week],
        'is_weekend': [1 if day_of_week >= 5 else 0],
        'speed': [speed],
        'occupancy': [occupancy],
        'std_dev': [std_dev]
    }

    # 添加滞后特征 (假设使用3个滞后窗口)
    for col in ['speed', 'occupancy', 'std_dev']:
        for i in range(1, 4):
            base_features[f'{col}_lag_{i}'] = [base_features[col][0] * (1 - i * 0.1)]  # 模拟滞后特征

    # 转换为DataFrame
    features_df = pd.DataFrame(base_features)

    # 类别特征编码
    categorical_cols = ['detector_id', 'direction', 'lane_id']
    encoded_features = encoder.transform(features_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # 合并所有特征
    numeric_cols = ['hour', 'minute', 'day_of_week', 'is_weekend', 'speed', 'occupancy', 'std_dev',
                    'speed_lag_1', 'speed_lag_2', 'speed_lag_3',
                    'occupancy_lag_1', 'occupancy_lag_2', 'occupancy_lag_3',
                    'std_dev_lag_1', 'std_dev_lag_2', 'std_dev_lag_3']

    final_features = pd.concat([
        features_df[numeric_cols],
        encoded_df
    ], axis=1)

    # 确保特征顺序与训练时一致
    final_features = final_features.reindex(columns=feature_columns, fill_value=0)

    return final_features


def predict_volume(rf_model, arima_model, features, last_arima_obs=None):
    """使用两个模型进行预测"""
    # 随机森林预测
    rf_pred = rf_model.predict(features)[0]

    # ARIMA预测 (需要提供历史观测值)
    if last_arima_obs is not None:
        # 在实际应用中，应该更新模型状态而不是每次都重新拟合
        arima_pred = arima_model.get_forecast(steps=1, exog=None).predicted_mean[0]
    else:
        # 如果没有历史数据，使用随机森林预测作为备用
        arima_pred = rf_pred

    return {
        'random_forest_prediction': rf_pred,
        'arima_prediction': arima_pred,
        'ensemble_prediction': (rf_pred + arima_pred) / 2  # 简单平均集成
    }


def main():
    # 加载模型和资源
    rf_model, encoder, feature_columns, arima_model = load_models_and_resources()

    # 示例特征值 (这些值应该来自实际应用或用户输入)
    sample_features = {
        'detector_id': 'D1001',
        'direction': 'NB',  # 北向
        'lane_id': 'L1',
        'hour': 17,  # 下午5点
        'minute': 30,  # 30分
        'day_of_week': 2,  # 周二 (0=周一, 6=周日)
        'speed': 65.5,  # 平均速度 65.5 km/h
        'occupancy': 0.25,  # 占有率 25%
        'std_dev': 8.2  # 标准差 8.2
    }

    # 准备随机森林特征
    features_df = prepare_rf_features(
        detector_id=sample_features['detector_id'],
        direction=sample_features['direction'],
        lane_id=sample_features['lane_id'],
        hour=sample_features['hour'],
        minute=sample_features['minute'],
        day_of_week=sample_features['day_of_week'],
        speed=sample_features['speed'],
        occupancy=sample_features['occupancy'],
        std_dev=sample_features['std_dev'],
        encoder=encoder,
        feature_columns=feature_columns
    )

    # 模拟最后一个ARIMA观测值 (实际应用中应该从历史数据获取)
    last_arima_obs = 120  # 假设上一个时间点的车流量是120

    # 进行预测
    predictions = predict_volume(rf_model, arima_model, features_df, last_arima_obs)

    # 打印结果
    print("\n预测结果:")
    print(f"- 随机森林预测车流量: {predictions['random_forest_prediction']:.2f}")
    print(f"- ARIMA预测车流量: {predictions['arima_prediction']:.2f}")
    print(f"- 集成预测车流量: {predictions['ensemble_prediction']:.2f}")


if __name__ == "__main__":
    main()