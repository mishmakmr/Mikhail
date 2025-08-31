"""
Инструкция
Загрузите данные и посчитайте модели линейной регрессии для 50 зданий по ансамблю регрессионных моделей: в первой модели весь оптимальный набор метеорологических данных,
во второй - дни недели и праздники, в третьей - недели года, в четвертой - месяцы. Финальное значение показателя рассчитайте как взвешенное арифметическое среднее показателей
всех моделей, взяв веса для первой и второй модели как 3/8, а для третьей и четвертой - как 1/8.
Загрузите данные решения, посчитайте значение энергопотребления для требуемых дат для тех зданий, которые посчитаны в модели, и выгрузите результат в виде CSV-файла (submission.csv)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_log_error, make_scorer
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# ==================== 1. КЛАССЫ И ФУНКЦИИ ====================
class NonNegativeLinearRegression(LinearRegression, RegressorMixin):
    """Линейная регрессия с неотрицательными предсказаниями"""
    def predict(self, X):
        return np.maximum(super().predict(X), 0)

def safe_rmsle(y_true, y_pred):
    """Защищенная версия RMSLE"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(np.maximum(y_true, 0) + 1e-10, 
                                        y_pred + 1e-10))

def add_time_features(df):
    """Добавление временных признаков"""
    df['hour'] = df['timestamp'].dt.hour.astype('int8')
    df['weekday'] = df['timestamp'].dt.weekday.astype('int8')
    df['week'] = df['timestamp'].dt.isocalendar().week.astype('int8')
    df['month'] = df['timestamp'].dt.month.astype('int8')
    
    # Праздничные дни
    cal = calendar()
    holidays = cal.holidays(start=df['timestamp'].min(), end=df['timestamp'].max())
    df['is_holiday'] = df['timestamp'].dt.date.isin([h.date() for h in holidays]).astype('int8')
    
    if 'meter_reading' in df.columns:
        df['meter_reading_log'] = np.log1p(df['meter_reading'])
    
    return df

def evaluate_models(models, X, y, cv=5):
    """Оценка моделей с кросс-валидацией"""
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X[name], y, cv=cv, 
                               scoring=make_scorer(safe_rmsle, greater_is_better=False))
        results[name] = {
            'mean_rmsle': -scores.mean(),
            'std_rmsle': scores.std()
        }
        print(f"{name:8} | RMSLE: {-scores.mean():.4f} (±{scores.std():.4f})")
    return results

# ==================== 2. ЗАГРУЗКА ДАННЫХ ====================
print("1. Загрузка данных...")
buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather_train = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
train = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz")
test = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/test.csv.gz")
weather_test = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_test.csv.gz")

# ==================== 3. ПРЕДОБРАБОТКА ====================
print("\n2. Предобработка данных...")

# Фильтрация и объединение
train = train[(train['building_id'] < 50) & (train['meter'] == 0)].copy()
test = test[(test['building_id'] < 50) & (test['meter'] == 0)].copy()

train = pd.merge(train, buildings, on='building_id', how='left')
train = pd.merge(train, weather_train, on=['site_id', 'timestamp'], how='left')
test = pd.merge(test, buildings, on='building_id', how='left')
test = pd.merge(test, weather_test, on=['site_id', 'timestamp'], how='left')

# Преобразование дат и добавление признаков
for df in [train, test]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    add_time_features(df)
    for col in ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed']:
        df[col] = df[col].interpolate(limit_direction='both')

# Создание dummy-переменных
for i in range(7):
    train[f'weekday_{i}'] = (train['weekday'] == i).astype('int8')
    test[f'weekday_{i}'] = (test['weekday'] == i).astype('int8')

for i in range(1, 53):
    train[f'week_{i}'] = (train['week'] == i).astype('int8')
    test[f'week_{i}'] = (test['week'] == i).astype('int8')

for i in range(1, 13):
    train[f'month_{i}'] = (train['month'] == i).astype('int8')
    test[f'month_{i}'] = (test['month'] == i).astype('int8')

# Разделение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    train.drop('meter_reading_log', axis=1),
    train['meter_reading_log'],
    test_size=0.2,
    random_state=42
)
X_val = add_time_features(X_val.copy())

# ==================== 4. ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ ====================
print("\n3. Определение моделей...")
models = {
    'weather': NonNegativeLinearRegression(),
    'days': NonNegativeLinearRegression(),
    'weeks': NonNegativeLinearRegression(),
    'months': NonNegativeLinearRegression()
}

features = {
    'weather': ['hour', 'building_id', 'air_temperature', 'dew_temperature',
               'sea_level_pressure', 'wind_speed'],
    'days': ['hour', 'building_id', 'is_holiday'] + [f'weekday_{i}' for i in range(7)],
    'weeks': ['hour', 'building_id'] + [f'week_{i}' for i in range(1, 53)],
    'months': ['hour', 'building_id'] + [f'month_{i}' for i in range(1, 13)]
}

# Подготовка данных для каждой модели
X_train_dict = {}
X_val_dict = {}
X_test_dict = {}

for name in models.keys():
    X_train_dict[name] = X_train[features[name]].drop(['hour', 'building_id'], axis=1)
    X_val_dict[name] = X_val[features[name]].drop(['hour', 'building_id'], axis=1)
    X_test_dict[name] = test[features[name]].drop(['hour', 'building_id'], axis=1)

# ==================== 5. ОБУЧЕНИЕ И ОЦЕНКА ====================
print("\n4. Обучение и оценка моделей...")

# Кросс-валидация
print("\nКросс-валидация на обучающей выборке:")
cv_results = evaluate_models(models, X_train_dict, y_train)

# Обучение на полных данных
for name, model in models.items():
    model.fit(X_train_dict[name], y_train)

# Оценка на валидационной выборке
print("\nОценка на валидационной выборке:")
val_predictions = {}
for name, model in models.items():
    val_pred = model.predict(X_val_dict[name])
    val_predictions[name] = val_pred
    print(f"{name:8} | RMSLE: {safe_rmsle(y_val, val_pred):.4f}")

# ==================== 6. АНСАМБЛЬ ====================
print("\n5. Формирование ансамбля...")
weights = [3/8, 3/8, 1/8, 1/8]
final_val_pred = np.zeros_like(y_val)

for (name, pred), weight in zip(val_predictions.items(), weights):
    final_val_pred += pred * weight

ensemble_rmsle = safe_rmsle(y_val, final_val_pred)
print(f"\nОшибка ансамбля на валидации: {ensemble_rmsle:.4f}")

# Визуализация
model_names = ['Погодная', 'Дни недели', 'Недели года', 'Месяцы']
errors = [safe_rmsle(y_val, pred) for pred in val_predictions.values()]

plt.figure(figsize=(12, 6))
plt.bar(model_names + ['Ансамбль'], errors + [ensemble_rmsle],
       color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Сравнение качества моделей (RMSLE)')
plt.ylabel('Ошибка (чем меньше, тем лучше)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==================== 7. ФИНАЛЬНОЕ ПРЕДСКАЗАНИЕ ====================
print("\n6. Формирование финальных предсказаний...")
test_predictions = []
for name, model in models.items():
    test_predictions.append(model.predict(X_test_dict[name]))

final_test_pred = np.zeros_like(test_predictions[0])
for pred, weight in zip(test_predictions, weights):
    final_test_pred += pred * weight

test['meter_reading'] = np.expm1(final_test_pred)
test.loc[test['meter_reading'] < 0, 'meter_reading'] = 0

# ==================== 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
print("\n7. Сохранение результатов...")
# ==================== 8.1 Вариант 1_только 50 зданий (без нулей) ==================
submission = test[['row_id', 'meter_reading']].copy() # только 50 зданий (без нулей)
submission.to_csv('submission.csv', index=False)
print("\nГотово! Результаты по 50 зданиям сохранены в submission.csv")

# Подсчет ненулевых значений
non_zero_count = (submission['meter_reading'] > 0).sum()
print(f"Количество ненулевых значений в submission.csv: {non_zero_count}")

# ==================== 8.2 Вариант 2_все здания, но meter_reading = 0 для тех, что не входят в 50 зданий ==============
submission_1 = pd.merge(
    pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/test.csv.gz", usecols=['row_id']),
    test[['row_id', 'meter_reading']],
    on='row_id',
    how='left'
)
submission_1['meter_reading'] = submission_1['meter_reading'].fillna(0)
submission_1.to_csv('submission_1.csv', index=False)
print("\nГотово! Результаты по всем зданиям сохранены в submission_1.csv")

# Подсчет ненулевых значений
non_zero_count_1 = (submission_1['meter_reading'] > 0).sum()
print(f"Количество ненулевых значений в submission_1.csv: {non_zero_count_1}")

# Вывод
# Кросс-валидация на обучающей выборке:
# weather  | RMSLE: 0.8328 (±0.0016)
# days     | RMSLE: 0.9684 (±0.0019)
# weeks    | RMSLE: 0.2693 (±0.0016)
# months   | RMSLE: 0.3638 (±0.0015)
#
# Оценка на валидационной выборке:
# weather  | RMSLE: 0.8333
# days     | RMSLE: 0.9690
# weeks    | RMSLE: 0.2716
# months   | RMSLE: 0.3651

# Ошибка ансамбля на валидации: 0.7636