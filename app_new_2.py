import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import joblib
import json
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


# 数据库初始化
def init_db():
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)

    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL,
                  approved INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()


# 添加管理员用户（如果不存在）
def add_admin_user():
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)

    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed_password = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, password, role, approved) VALUES (?, ?, ?, ?)",
                  ('admin', hashed_password, 'admin', 1))
        conn.commit()
    conn.close()


# 加载模型和资源
def load_models_and_resources():
    """加载训练好的模型和相关资源"""
    # 确保模型目录存在
    os.makedirs('models/RF_Model', exist_ok=True)
    os.makedirs('models/ARIMA_Model', exist_ok=True)

    # 加载随机森林模型和相关文件
    rf_model = joblib.load('models/RF_Model/rf_model.pkl')
    encoder = joblib.load('models/RF_Model/feature_encoder.pkl')
    with open('models/RF_Model/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # 加载ARIMA模型
    arima_model = SARIMAXResults.load('models/ARIMA_Model/arima_model.pkl')

    return rf_model, encoder, feature_columns, arima_model


# 初始化
init_db()
add_admin_user()
rf_model, encoder, feature_columns, arima_model = load_models_and_resources()
# 在已有导入部分添加
import seaborn as sns
from dateutil import parser
# 在文件顶部添加以下代码设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 修改 load_traffic_data 函数
def load_traffic_data():
    data_path = 'output1.csv'
    speed_volume_path = 'traffic_speed_volume_occ_info-slp.csv'

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['period_from'])

        # 加载检测器与道路名称映射
        if os.path.exists(speed_volume_path):
            road_info = pd.read_csv(speed_volume_path)
            road_mapping = road_info[['AID_ID_Number', 'Road_SC']].drop_duplicates()
            road_dict = dict(zip(road_mapping['AID_ID_Number'], road_mapping['Road_SC']))
            df['road_name'] = df['detector_id'].map(road_dict)
        else:
            df['road_name'] = '未知道路'

        return df
    return None


# 初始化交通数据
traffic_df = load_traffic_data()


# 修改 traffic_history 路由
@app.route('/traffic_history', methods=['GET', 'POST'])
def traffic_history():
    if 'username' not in session:
        flash('请先登录', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        start_hour = int(request.form.get('start_hour', 0))
        start_minute = int(request.form.get('start_minute', 0))
        end_hour = int(request.form.get('end_hour', 23))
        end_minute = int(request.form.get('end_minute', 59))
        detector_id = request.form.get('detector_id', '')
        lane_type = request.form.get('lane_type', 'all')

        # 创建完整的时间戳
        start_datetime = pd.to_datetime(f"{start_date} {start_hour}:{start_minute}")
        end_datetime = pd.to_datetime(f"{end_date} {end_hour}:{end_minute}")

        # 过滤数据
        filtered_df = traffic_df.copy()
        filtered_df = filtered_df[(filtered_df['datetime'] >= start_datetime) &
                                (filtered_df['datetime'] <= end_datetime)]

        if detector_id:
            filtered_df = filtered_df[filtered_df['detector_id'] == detector_id]
            # 获取对应的道路名称
            road_name = filtered_df.iloc[0]['road_name'] if not filtered_df.empty else ''
        else:
            road_name = ''

        if lane_type != 'all':
            filtered_df = filtered_df[filtered_df['lane_id'].str.contains(lane_type)]

        # 生成图表
        img_base64 = generate_history_plot(filtered_df)

        return render_template('traffic_history.html',
                             plot_data=img_base64,
                             form_data=request.form,
                             detector_ids=traffic_df['detector_id'].unique(),
                             lane_types=['Fast Lane', 'Slow Lane', 'Middle Lane'],
                             road_name=road_name)

    # 默认显示最近一天的数据
    default_end = traffic_df['datetime'].max()
    default_start = default_end - timedelta(days=1)

    default_data = {
        'start_date': default_start.strftime('%Y-%m-%d'),
        'end_date': default_end.strftime('%Y-%m-%d'),
        'start_hour': 0,
        'start_minute': 0,
        'end_hour': 23,
        'end_minute': 59,
        'detector_id': '',
        'lane_type': 'all'
    }

    return render_template('traffic_history.html',
                         form_data=default_data,
                         detector_ids=traffic_df['detector_id'].unique(),
                         lane_types=['Fast Lane', 'Slow Lane', 'Middle Lane'],
                         road_name='')


# 添加新的可视化图片展示路由
@app.route('/visualizations')
def visualizations():
    if 'username' not in session:
        flash('请先登录', 'danger')
        return redirect(url_for('login'))

    # 确保路径指向 static/Visualizations
    vis_dir = os.path.join(app.static_folder, 'Visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)  # 自动创建目录（可选）
        return render_template('visualizations.html', images=[])

    # 只加载图片文件，并确保路径正确
    images = [f for f in os.listdir(vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template('visualizations.html', images=images)


# 添加获取道路名称的路由
@app.route('/get_road_name')
def get_road_name():
    detector_id = request.args.get('detector_id')
    if not detector_id or traffic_df is None:
        return jsonify({'road_name': ''})

    # 从数据中查找道路名称
    road_name = traffic_df[traffic_df['detector_id'] == detector_id]['road_name'].iloc[0] \
        if not traffic_df[traffic_df['detector_id'] == detector_id].empty else '未知道路'

    return jsonify({'road_name': road_name})
def generate_history_plot(df, granularity='hour'):
    if df.empty:
        return None

    plt.figure(figsize=(12, 6))

    # Group data based on selected granularity
    if granularity == 'hour':
        freq = 'H'
        time_format = '%m-%d %H:%M'
    elif granularity == '30min':
        freq = '30T'
        time_format = '%m-%d %H:%M'
    else:  # daily
        freq = 'D'
        time_format = '%m-%d'

    # Group data
    plot_df = df.groupby([pd.Grouper(key='datetime', freq=freq), 'lane_id']).agg({
        'volume': 'sum',
        'speed': 'mean'
    }).reset_index()

    # Create subplots (now only 2 instead of 3)
    plt.subplot(2, 1, 1)
    for lane in plot_df['lane_id'].unique():
        lane_data = plot_df[plot_df['lane_id'] == lane]
        plt.plot(lane_data['datetime'], lane_data['volume'], label=f'{lane} Volume')
    plt.title('Historical Traffic Data')
    plt.ylabel('Volume (vehicles/hour)')  # Added unit
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for lane in plot_df['lane_id'].unique():
        lane_data = plot_df[plot_df['lane_id'] == lane]
        plt.plot(lane_data['datetime'], lane_data['speed'], label=f'{lane} Speed')
    plt.ylabel('Speed (km/h)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save chart to memory
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode('utf-8')

# 路由定义
@app.route('/')
def home():
    if 'username' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('predict'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            if user[4] == 1:  # 检查用户是否已批准
                session['username'] = user[1]
                session['role'] = user[3]
                flash('登录成功!', 'success')
                return redirect(url_for('home'))
            else:
                flash('您的账号尚未获得管理员批准，请等待', 'warning')
        else:
            flash('用户名或密码错误', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('您已成功登出', 'success')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return redirect(url_for('register'))

        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()

        try:
            hashed_password = generate_password_hash(password)
            c.execute("INSERT INTO users (username, password, role, approved) VALUES (?, ?, ?, ?)",
                      (username, hashed_password, role, 1 if role == 'admin' else 0))
            conn.commit()
            flash('注册成功! ' + ('您现在是管理员' if role == 'admin' else '请等待管理员批准'), 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('用户名已存在', 'danger')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/admin')
def admin():
    if 'username' not in session or session['role'] != 'admin':
        flash('请以管理员身份登录', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, role, approved FROM users WHERE role='user'")
    pending_users = c.fetchall()
    conn.close()

    return render_template('admin.html', pending_users=pending_users)


@app.route('/approve_user/<int:user_id>')
def approve_user(user_id):
    if 'username' not in session or session['role'] != 'admin':
        flash('请以管理员身份登录', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    flash('用户已批准', 'success')
    return redirect(url_for('admin'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('请先登录', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # 获取表单数据
        detector_id = request.form.get('detector_id', 'D1001')
        direction = request.form.get('direction', 'NB')
        lane_id = request.form.get('lane_id', 'L1')
        hour = int(request.form.get('hour', 17))
        minute = int(request.form.get('minute', 30))
        day_of_week = int(request.form.get('day_of_week', 2))
        speed = float(request.form.get('speed', 65.5))
        occupancy = float(request.form.get('occupancy', 0.25))
        std_dev = float(request.form.get('std_dev', 8.2))

        # 准备随机森林特征
        features_df = prepare_rf_features(
            detector_id=detector_id,
            direction=direction,
            lane_id=lane_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            speed=speed,
            occupancy=occupancy,
            std_dev=std_dev,
            encoder=encoder,
            feature_columns=feature_columns
        )

        # 模拟最后一个ARIMA观测值
        last_arima_obs = 120

        # 进行预测
        predictions = predict_volume(rf_model, arima_model, features_df, last_arima_obs)

        # 保存预测结果到session以便在图表页面使用
        session['last_prediction'] = {
            'features': {
                'detector_id': detector_id,
                'direction': direction,
                'lane_id': lane_id,
                'hour': hour,
                'minute': minute,
                'day_of_week': day_of_week,
                'speed': speed,
                'occupancy': occupancy,
                'std_dev': std_dev
            },
            'predictions': predictions
        }

        return render_template('predict.html',
                               predictions=predictions,
                               form_data=request.form)

    # 默认值
    default_data = {
        'detector_id': 'D1001',
        'direction': 'NB',
        'lane_id': 'L1',
        'hour': '17',
        'minute': '30',
        'day_of_week': '2',
        'speed': '65.5',
        'occupancy': '0.25',
        'std_dev': '8.2'
    }

    return render_template('predict.html', form_data=default_data)


@app.route('/chart')
def chart():
    if 'username' not in session:
        flash('请先登录', 'danger')
        return redirect(url_for('login'))

    if 'last_prediction' not in session:
        flash('请先进行一次预测', 'warning')
        return redirect(url_for('predict'))

    # 获取时间范围和粒度参数
    time_range = request.args.get('time_range', 'day')
    granularity = request.args.get('granularity', 'hour')

    # 生成模拟时间序列数据
    timestamps, rf_values, arima_values, ensemble_values = generate_time_series_data(
        session['last_prediction']['predictions'],
        time_range,
        granularity
    )

    # 准备图表数据
    chart_data = {
        'timestamps': timestamps,
        'rf_values': rf_values,
        'arima_values': arima_values,
        'ensemble_values': ensemble_values
    }

    return render_template('chart.html',
                           chart_data=chart_data,
                           time_range=time_range,
                           granularity=granularity,
                           form_data=session['last_prediction']['features'])


@app.route('/get_chart_image')
def get_chart_image():
    if 'last_prediction' not in session:
        return jsonify({'error': 'No prediction data available'})

    time_range = request.args.get('time_range', 'day')
    granularity = request.args.get('granularity', 'hour')

    # 生成时间序列数据
    timestamps, rf_values, arima_values, ensemble_values = generate_time_series_data(
        session['last_prediction']['predictions'],
        time_range,
        granularity
    )

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, rf_values, label='随机森林预测')
    plt.plot(timestamps, arima_values, label='ARIMA预测')
    plt.plot(timestamps, ensemble_values, label='集成预测', linestyle='--', linewidth=2)

    plt.title('交通流量预测结果')
    plt.xlabel('时间')
    plt.ylabel('车流量')
    plt.legend()
    plt.grid(True)

    # 旋转x轴标签
    plt.xticks(rotation=45)

    # 保存图表到内存
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    # 返回Base64编码的图像
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return jsonify({'image': img_base64})


# 辅助函数
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


def generate_time_series_data(predictions, time_range='day', granularity='hour'):
    """生成时间序列数据用于图表展示"""
    # 根据时间范围和粒度确定数据点数量
    if time_range == 'day':
        n_points = 24 if granularity == 'hour' else 48
    elif time_range == 'week':
        n_points = 7 if granularity == 'day' else 14
    else:  # month
        n_points = 30 if granularity == 'day' else 60

    # 生成时间戳
    now = datetime.now()
    if time_range == 'day':
        timestamps = [now + timedelta(hours=i) if granularity == 'hour'
                      else now + timedelta(minutes=30 * i)
                      for i in range(n_points)]
    elif time_range == 'week':
        timestamps = [now + timedelta(days=i) if granularity == 'day'
                      else now + timedelta(hours=12 * i)
                      for i in range(n_points)]
    else:  # month
        timestamps = [now + timedelta(days=i) if granularity == 'day'
                      else now + timedelta(hours=12 * i)
                      for i in range(n_points)]

    # 格式化时间戳
    if granularity == 'hour':
        timestamp_str = [ts.strftime('%m-%d %H:%M') for ts in timestamps]
    elif granularity == 'day':
        timestamp_str = [ts.strftime('%m-%d') for ts in timestamps]
    else:  # 30分钟粒度
        timestamp_str = [ts.strftime('%m-%d %H:%M') for ts in timestamps]

    # 基于预测结果生成模拟数据
    base_rf = predictions['random_forest_prediction']
    base_arima = predictions['arima_prediction']
    base_ensemble = predictions['ensemble_prediction']

    # 添加一些随机波动
    rf_values = [base_rf * (1 + 0.1 * np.sin(i / 2) + 0.05 * np.random.randn()) for i in range(n_points)]
    arima_values = [base_arima * (1 + 0.08 * np.cos(i / 3) + 0.03 * np.random.randn()) for i in range(n_points)]
    ensemble_values = [(rf + arima) / 2 for rf, arima in zip(rf_values, arima_values)]

    return timestamp_str, rf_values, arima_values, ensemble_values


if __name__ == '__main__':
    app.run(debug=True)