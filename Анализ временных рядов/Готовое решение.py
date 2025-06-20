import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
'''
import warnings — импортирует модуль warnings, который управляет предупреждениями в Python.
warnings.filterwarnings('ignore') — устанавливает фильтр, который игнорирует все предупреждения.
Используется для подавления предупреждений в процессе выполнения программы. Это особенно полезно, когда вы хотите избежать вывода предупреждений, 
которые могут мешать чтению результатов или логам, например, при использовании устаревших функций или библиотек.
'''

# Загрузка данных
file_path = "C:\PYTHON\ApowerREC\Прикладной анализ данных\Анализ временных рядов\Анализ временных рядов_Курсовой\ittensive.time.series.04.csv"
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', decimal=',',encoding='cp1251')
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    df.set_index('Дата', inplace=True)
    df.sort_index(inplace=True)

    return df

# Визуализация данных
def plot_data(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Закрытие'], label='Цена закрытия')
    plt.title('Динамика курса акций Сбербанка')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid()
    plt.show()

# Проверка стационарности
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

# SARIMA модель
def sarima_model(train, test):
    # Автоподбор параметров SARIMA
    model = auto_arima(train, seasonal=True, m=12,
                      suppress_warnings=True,
                      stepwise=True, trace=True)
    
    print(model.summary())
    
    # Обучение модели
    sarima = SARIMAX(train, 
                    order=model.order, 
                    seasonal_order=model.seasonal_order)
    sarima_fit = sarima.fit(disp=False)
    
    # Прогнозирование
    sarima_pred = sarima_fit.predict(start=len(train), 
                                   end=len(train)+len(test)-1,
                                   dynamic=False)
    
    # Оценка качества
    mae = mean_absolute_error(test, sarima_pred)
    rmse = np.sqrt(mean_squared_error(test, sarima_pred))
    print(f'SARIMA MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    
    return sarima_fit, sarima_pred, mae, rmse

# Подготовка данных для LSTM
def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# LSTM модель
def lstm_model(train, test, n_steps=30):
    # Масштабирование данных
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    
    # Подготовка данных
    X_train, y_train = prepare_lstm_data(train_scaled, n_steps)
    X_test, y_test = prepare_lstm_data(test_scaled, n_steps)
    
    # Создание модели
    model = Sequential()
    model.add(LSTM(50, activation='relu', 
                 input_shape=(n_steps, 1), 
                 return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Обучение
    history = model.fit(X_train, y_train, 
                       epochs=50, batch_size=32,
                       validation_data=(X_test, y_test),
                       verbose=1, shuffle=False)
    
    # Прогнозирование
    lstm_pred = model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
    
    # Оценка качества
    mae = mean_absolute_error(test[n_steps:], lstm_pred)
    rmse = np.sqrt(mean_squared_error(test[n_steps:], lstm_pred))
    print(f'LSTM MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    
    return model, lstm_pred, mae, rmse

# Создание фичей для XGBoost
def create_features(df, target, lags=5):
    df = df.copy()
    
    # Лаги
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # Скользящие статистики
    df['rolling_mean'] = df[target].rolling(window=7).mean()
    df['rolling_std'] = df[target].rolling(window=7).std()
    
    # День недели и месяц
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df.dropna(inplace=True)
    return df

# XGBoost модель
def xgboost_model(train, test, target='Закрытие'):
    # Создание фичей
    train_feat = create_features(train, target)
    test_feat = create_features(test, target)
    
    X_train = train_feat.drop(target, axis=1)
    y_train = train_feat[target]
    X_test = test_feat.drop(target, axis=1)
    y_test = test_feat[target]
    
    # Поиск оптимальных параметров
    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    model = XGBRegressor(objective='reg:squarederror')
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_absolute_error')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f'Best parameters: {grid.best_params_}')
    
    # Прогнозирование
    xgb_pred = best_model.predict(X_test)
    
    # Оценка качества
    mae = mean_absolute_error(y_test, xgb_pred)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print(f'XGBoost MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    
    return best_model, xgb_pred, mae, rmse

# Ансамбль моделей
def ensemble_predictions(sarima_pred, lstm_pred, xgb_pred, weights):
    # Выравниваем длины прогнозов
    min_len = min(len(sarima_pred), len(lstm_pred), len(xgb_pred))
    sarima_pred = sarima_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    xgb_pred = xgb_pred[:min_len]

    sarima_pred = np.array(sarima_pred).ravel()
    lstm_pred = np.array(lstm_pred).ravel()
    xgb_pred = np.array(xgb_pred).ravel()

    # Взвешенное усреднение
    ensemble_pred = (weights[0] * sarima_pred +
                    weights[1] * lstm_pred +
                    weights[2] * xgb_pred)

    return ensemble_pred

# Визуализация результатов
def plot_results(test, sarima_pred, lstm_pred, xgb_pred, ensemble_pred):
    plt.figure(figsize=(14, 7))
    
    # Для выравнивания графиков берем общую длину
    min_len = min(len(test), len(sarima_pred), len(lstm_pred), len(xgb_pred))
    
    plt.plot(test.index[-min_len:], test.values[-min_len:], 
             label='Фактические значения', color='black')
    plt.plot(test.index[-min_len:], sarima_pred[-min_len:], 
             label='SARIMA', linestyle='--')
    plt.plot(test.index[-min_len:], lstm_pred[-min_len:], 
             label='LSTM', linestyle='--')
    plt.plot(test.index[-min_len:], xgb_pred[-min_len:], 
             label='XGBoost', linestyle='--')
    plt.plot(test.index[-min_len:], ensemble_pred[-min_len:], 
             label='Ансамбль', linewidth=2, color='red')
    
    plt.title('Сравнение моделей прогнозирования')
    plt.xlabel('Дата')
    plt.ylabel('Цена закрытия')
    plt.legend()
    plt.grid()
    plt.show()

# Прогнозирование на будущее
def forecast_future(models, last_data, n_steps=30, target='Закрытие'):
    sarima_model, lstm_model, xgboost_model = models
    scaler = MinMaxScaler()
    
    # Подготовка последних данных
    last_values = last_data[target].values[-n_steps:]
    last_scaled = scaler.fit_transform(last_values.reshape(-1, 1))
    
    # SARIMA прогноз
    sarima_forecast = sarima_model.get_forecast(steps=n_steps)
    sarima_pred = sarima_forecast.predicted_mean
    
    # LSTM прогноз
    lstm_input = last_scaled.reshape(1, n_steps, 1)
    lstm_pred_scaled = []
    current_batch = lstm_input
    
    for i in range(n_steps):
        current_pred = lstm_model.predict(current_batch)[0]
        lstm_pred_scaled.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], 
                                [current_pred]).reshape(1, n_steps, 1)
    
    print("lstm_pred_scaled       ", lstm_pred_scaled)

    lstm_pred = scaler.inverse_transform(np.array(lstm_pred_scaled).reshape(-1, 1))

    # XGBoost прогноз
    xgb_pred = []
    current_features = create_features(last_data, target).iloc[-1].drop(target)
    # print('проверка',last_data)
    print(last_data)
    print("Количество признаков:", current_features.shape[0])
    print("Имена признаков:", current_features.index.tolist())

    for i in range(n_steps):
        pred = xgboost_model.predict(current_features.values.reshape(1, -1))[0]
        xgb_pred.append(pred)
        
        # Обновляем фичи для следующего шага
        new_features = current_features.copy()
        for lag in range(1, 6):
            if lag == 1:
                new_features[f'lag_{lag}'] = pred
            else:
                new_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        
        current_features = new_features
    
    # Ансамбль
    weights = [0.4, 0.3, 0.3]  # Веса моделей

    # Преобразуем в одномерные массивы
    sarima_pred = np.array(sarima_pred).flatten()
    lstm_pred = np.array(lstm_pred).flatten()
    xgb_pred = np.array(xgb_pred).flatten()
    ensemble_forecast = ensemble_predictions(sarima_pred, lstm_pred, xgb_pred, weights)
    
    # Создаем даты для прогноза
    last_date = last_data.index[-1]
    print('Создаем даты для прогноза',last_date)
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)
    print('Создаем даты для прогноза', forecast_dates)
    # Результаты
    forecast_df = pd.DataFrame({
        'SARIMA': sarima_pred,
        'LSTM': lstm_pred,
        'XGBoost': xgb_pred,
        'Ensemble': ensemble_forecast
    }, index=forecast_dates)
    
    return forecast_df

#
#
# ------------------------------------Основной код
if __name__ == "__main__":
    # Загрузка данных
    df = load_data('ittensive.time.series.04.csv')
    
    # Визуализация
    plot_data(df)
    
    # Проверка стационарности
    print("Проверка стационарности:")
    check_stationarity(df['Закрытие'])
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # SARIMA модель
    print("\nSARIMA модель:")
    sarima_fit, sarima_pred, sarima_mae, sarima_rmse = sarima_model(
        train['Закрытие'], test['Закрытие'])
    
    # LSTM модель
    print("\nLSTM модель:")
    lstm_model, lstm_pred, lstm_mae, lstm_rmse = lstm_model(
        train['Закрытие'], test['Закрытие'])
    
    # XGBoost модель
    print("\nXGBoost модель:")
    # xgboost_model, xgb_pred, xgb_mae, xgb_rmse = xgboost_model(train, test)
    xgboost_model, xgb_pred, xgb_mae, xgb_rmse = xgboost_model(train, test)
    
    # Ансамбль
    weights = [0.4, 0.3, 0.3]  # Веса для SARIMA, LSTM, XGBoost
    ensemble_pred = ensemble_predictions(sarima_pred, lstm_pred, xgb_pred, weights)

    # Оценка ансамбля
    min_len = min(len(test), len(ensemble_pred))
    ensemble_mae = mean_absolute_error(test['Закрытие'].values[-min_len:],
                                     ensemble_pred[-min_len:])
    ensemble_rmse = np.sqrt(mean_squared_error(test['Закрытие'].values[-min_len:],
                                             ensemble_pred[-min_len:]))
    print(f'\nАнсамбль MAE: {ensemble_mae:.2f}, RMSE: {ensemble_rmse:.2f}')

    # Визуализация результатов
    plot_results(test['Закрытие'], sarima_pred, lstm_pred, xgb_pred, ensemble_pred)

    # Прогноз на январь 2022 (30 дней)
    print("\nПрогноз на январь 2022:")
    models = (sarima_fit, lstm_model, xgboost_model)
    forecast = forecast_future(models, df, n_steps=30)

    print(forecast)

    # Визуализация прогноза
    plt.figure(figsize=(14, 7))
    plt.plot(df['Закрытие'].iloc[-60:], label='Исторические данные')
    plt.plot(forecast.index, forecast['Ensemble'],
             label='Прогноз ансамбля', color='red')
    plt.fill_between(forecast.index,
                    forecast['Ensemble'] - 10,
                    forecast['Ensemble'] + 10,
                    color='red', alpha=0.1)
    plt.title('Прогноз курса акций Сбербанка на январь 2022')
    plt.xlabel('Дата')
    plt.ylabel('Цена закрытия')
    plt.legend()
    plt.grid()
    plt.show()
