import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import json
from matplotlib import rcParams
import warnings

# 设置中文字体和全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
rcParams.update({
    'font.size': 12,
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

# 1. 创建保存结果的文件夹
model_folders = ['RF_Model', 'ARIMA_Model', 'Visualizations']
for folder in model_folders:
    os.makedirs(folder, exist_ok=True)

# 2. 加载数据
print("正在加载数据...")
try:
    data = pd.read_csv('output1.csv', nrows=20000)
except FileNotFoundError:
    raise FileNotFoundError("请确保output1.csv文件存在于当前目录")

# 3. 数据预处理
print("数据预处理中...")

# 3.1 处理日期时间列
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['period_from'])
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['day_of_week'] = data['datetime'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 3.2 处理空值和异常值
numeric_cols = ['speed', 'occupancy', 'std_dev']
data[numeric_cols] = data[numeric_cols].replace(0, np.nan)

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].rolling(5, min_periods=1).mean())

# 3.3 特征编码
categorical_cols = ['detector_id', 'direction', 'lane_id']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# 3.4 合并特征
features = pd.concat([
    data[['hour', 'minute', 'day_of_week', 'is_weekend'] + numeric_cols],
    encoded_df
], axis=1)

# 3.5 标签
target = data['volume']

# 4. 特征工程
print("特征工程处理中...")

def add_time_features(df, window_size=3):
    """添加时间滞后特征"""
    for col in numeric_cols:
        for i in range(1, window_size + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    return df

features = add_time_features(features)
features = features.iloc[3:].reset_index(drop=True)
target = target.iloc[3:].reset_index(drop=True)

# 5. 处理缺失值
print("处理缺失值...")
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# 6. 数据可视化
print("生成可视化图表...")

def save_plot(fig, filename, title):
    """保存图表的标准函数"""
    fig.suptitle(title, fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 6.1 相关性热力图
plt.figure(figsize=(16, 12))
corr_matrix = features.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
heatmap = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
heatmap.set_title('特征相关性热力图', fontdict={'fontsize':16}, pad=12)
save_plot(plt, 'Visualizations/feature_correlation_heatmap.png', '特征相关性热力图')

# 6.2 时间序列趋势图
plt.figure(figsize=(14, 6))
sample_data = data.iloc[:500].copy()
sample_data['time'] = sample_data['datetime'].dt.strftime('%H:%M')
lineplot = sns.lineplot(data=sample_data, x='time', y='volume', hue='direction', ci=None)
lineplot.set_title('不同方向车流量随时间变化趋势', fontsize=16)
lineplot.set_xlabel('时间', fontsize=12)
lineplot.set_ylabel('车流量', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='方向', title_fontsize='12', fontsize='10')
save_plot(plt, 'Visualizations/time_series_trend.png', '不同方向车流量随时间变化趋势')

# 7. 划分数据集
print("划分训练集和测试集...")
split_idx = int(len(features) * 0.8)
X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

# 8. 随机森林模型
print("训练随机森林模型...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 8.1 保存RF模型和特征
joblib.dump(rf_model, 'RF_Model/rf_model.pkl')
rf_feature_importance = pd.DataFrame({
    '特征': X_train.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)
rf_feature_importance.to_csv('RF_Model/rf_feature_importance.csv', index=False)

# 8.2 RF特征重要性可视化
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='重要性', y='特征', data=rf_feature_importance.head(20))
barplot.set_title('随机森林特征重要性TOP20', fontsize=16)
barplot.set_xlabel('特征重要性', fontsize=12)
barplot.set_ylabel('特征名称', fontsize=12)
save_plot(plt, 'RF_Model/rf_feature_importance.png', '随机森林特征重要性TOP20')

# 9. ARIMA模型
print("训练ARIMA模型...")

# 9.1 准备ARIMA数据
arima_data = data[['datetime', 'volume']].copy()
arima_data.set_index('datetime', inplace=True)
arima_data = arima_data.resample('5T').mean().fillna(method='ffill')

# 9.2 自动选择ARIMA参数
auto_model = auto_arima(
    arima_data['volume'],
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='aic'
)

print(f"自动选择的ARIMA参数: {auto_model.order}")

# 9.3 训练ARIMA模型
arima_model = SARIMAX(
    arima_data['volume'],
    order=auto_model.order,
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=True
)
arima_model_fit = arima_model.fit(disp=False)

# 9.4 保存模型
arima_model_fit.save('ARIMA_Model/arima_model.pkl')

# 10. 模型评估
def evaluate_model(y_true, y_pred, model_name):
    """评估模型并生成可视化"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

    # 保存指标
    with open(f'{model_name}_Model/metrics.txt', 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 预测对比图
    plt.figure(figsize=(14, 6))
    plt.plot(y_true.values, label='真实值', color='blue', alpha=0.7)
    plt.plot(y_pred, label='预测值', color='orange', alpha=0.7)

    # 标注指标
    text_str = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    plt.text(0.75*len(y_true), 0.85*max(y_true), text_str,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontproperties='SimHei')

    plt.title(f'{model_name}模型预测对比', fontproperties='SimHei')
    plt.xlabel('时间点', fontproperties='SimHei')
    plt.ylabel('车流量', fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei'})
    save_plot(plt, f'{model_name}_Model/prediction_comparison.png', f'{model_name}模型预测对比')

    return metrics

# 10.1 评估随机森林
rf_metrics = evaluate_model(y_test, rf_model.predict(X_test), 'Random_Forest')

# 10.2 评估ARIMA
arima_pred = arima_model_fit.get_forecast(steps=len(y_test)).predicted_mean
arima_metrics = evaluate_model(y_test, arima_pred, 'ARIMA')

# 11. 模型对比可视化
plt.figure(figsize=(12, 6))
metrics_df = pd.DataFrame({'随机森林': rf_metrics, 'ARIMA': arima_metrics})
ax = metrics_df.plot(kind='bar', rot=0)
ax.set_title('模型性能指标对比', fontproperties='SimHei')
ax.set_ylabel('分数值', fontproperties='SimHei')

for i, col in enumerate(metrics_df.columns):
    for j, val in enumerate(metrics_df[col]):
        ax.text(i - 0.18 + j * 0.36, val + 0.01, f'{val:.3f}', ha='center')

save_plot(plt, 'Visualizations/model_comparison.png', '模型性能指标对比')

# 12. 保存编码器和配置
joblib.dump(encoder, 'RF_Model/feature_encoder.pkl')
with open('RF_Model/feature_columns.json', 'w') as f:
    json.dump(list(features.columns), f, ensure_ascii=False)

print("""
所有处理完成！
结果保存位置：
- 随机森林模型: RF_Model/
- ARIMA模型: ARIMA_Model/
- 可视化结果: Visualizations/
""")