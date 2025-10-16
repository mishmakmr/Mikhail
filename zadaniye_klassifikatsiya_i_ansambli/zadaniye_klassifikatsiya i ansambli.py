'''
Инструкция
Загрузите данные, приведите их к числовым, заполните пропуски, нормализуйте данные и оптимизируйте память.

Сформируйте параллельный ансамбль из CatBoost, градиентного бустинга, XGBoost и LightGBM. Используйте лучшие гиперпараметры, подобранные ранее, или найдите их через перекрестную проверку. Итоговое решение рассчитайте на основании самого точного предсказания класса у определенной модели ансамбля: выберите для каждого класса модель, которая предсказывает его лучше всего.

Проведите расчеты и выгрузите результат в виде submission.csv

Данные:
* video.ittensive.com/machine-learning/prudential/train.csv.gz
* video.ittensive.com/machine-learning/prudential/test.csv.gz
* video.ittensive.com/machine-learning/prudential/sample_submission.csv.gz
'''

# Подключение библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

# Создаем каппа-скор для оценки
def quadratic_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

kappa_scorer = make_scorer(quadratic_kappa, greater_is_better=True)

# Загрузка данных
print("Загрузка данных...")
data = pd.read_csv("https://video.ittensive.com/machine-learning/prudential/train.csv.gz")
print(data.info())

# Предобработка данных
def data_preprocess(df):
    df["Product_Info_2_1"] = df["Product_Info_2"].str.slice(0, 1)
    df["Product_Info_2_2"] = pd.to_numeric(df["Product_Info_2"].str.slice(1, 2))
    df.drop(labels=["Product_Info_2"], axis=1, inplace=True)
    for l in df["Product_Info_2_1"].unique():
        df["Product_Info_2_1" + l] = df["Product_Info_2_1"].isin([l]).astype("int8")
    df.drop(labels=["Product_Info_2_1"], axis=1, inplace=True)
    df.fillna(value=-1, inplace=True)
    return df

data = data_preprocess(data)

# Набор столбцов для расчета
columns_groups = ["Insurance_History", "InsuredInfo", "Medical_Keyword",
                  "Family_Hist", "Medical_History", "Product_Info"]
columns = ["Wt", "Ht", "Ins_Age", "BMI"]
for cg in columns_groups:
    columns.extend(data.columns[data.columns.str.startswith(cg)])
print(f"Количество признаков: {len(columns)}")

# Нормализация данных
scaler = preprocessing.StandardScaler()
data_transformed = pd.DataFrame(scaler.fit_transform(pd.DataFrame(data, columns=columns)))
columns_transformed = data_transformed.columns
data_transformed["Response"] = data["Response"] - 1  # классы 0-7

# Оптимизация памяти
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2), 'Мб (минус',
          round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df

data_transformed = reduce_mem_usage(data_transformed)
print(data_transformed.info())

# Подготовка данных для обучения
x = pd.DataFrame(data_transformed, columns=columns_transformed)
y = data_transformed["Response"]

# Разделение на train/validation для быстрого подбора параметров
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=17, stratify=y)

print("Начинаем подбор гиперпараметров...")

# 1. Настройка XGBoost
print("Подбор параметров для XGBoost...")
param_grid_xgb = {
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

model_xgb = XGBClassifier(random_state=17, eval_metric='mlogloss')
grid_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=3, scoring=kappa_scorer, n_jobs=-1, verbose=1)
grid_xgb.fit(x_train, y_train)

print(f"Лучшие параметры XGBoost: {grid_xgb.best_params_}")
print(f"Лучший Kappa XGBoost: {grid_xgb.best_score_:.4f}")

# Вычисление точности на валидационной выборке
y_pred_xgb = grid_xgb.best_estimator_.predict(x_val)
# Преобразуем в 1D массив если нужно
y_pred_xgb = np.ravel(y_pred_xgb)
accuracy_xgb = (y_pred_xgb == y_val).mean()
kappa_xgb = quadratic_kappa(y_val, y_pred_xgb)
print(f"Точность XGBoost на валидационной выборке: {accuracy_xgb:.4f}")
print(f"Kappa XGBoost на валидационной выборке: {kappa_xgb:.4f}")
print()

# 2. Настройка CatBoost
print("Подбор параметров для CatBoost...")
param_grid_cb = {
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'iterations': [500, 1000]
}

model_cb = CatBoostClassifier(
    random_state=17, 
    verbose=False, 
    loss_function='MultiClass',
    classes_count=8,
    auto_class_weights='Balanced'
)
grid_cb = GridSearchCV(model_cb, param_grid_cb, cv=3, scoring=kappa_scorer, n_jobs=-1, verbose=1)
grid_cb.fit(x_train, y_train)

print(f"Лучшие параметры CatBoost: {grid_cb.best_params_}")
print(f"Лучший Kappa CatBoost: {grid_cb.best_score_:.4f}")

# Вычисление точности на валидационной выборке
y_pred_cb = grid_cb.best_estimator_.predict(x_val)
# Преобразуем в 1D массив - это исправляет ошибку
y_pred_cb = np.ravel(y_pred_cb)
accuracy_cb = (y_pred_cb == y_val).mean()
kappa_cb = quadratic_kappa(y_val, y_pred_cb)
print(f"Точность CatBoost на валидационной выборке: {accuracy_cb:.4f}")
print(f"Kappa CatBoost на валидационной выборке: {kappa_cb:.4f}")
print()

# 3. Настройка GradientBoosting
print("Подбор параметров для GradientBoosting...")
param_grid_gbc = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

model_gbc = GradientBoostingClassifier(random_state=17)
grid_gbc = GridSearchCV(model_gbc, param_grid_gbc, cv=3, scoring=kappa_scorer, n_jobs=-1, verbose=1)
grid_gbc.fit(x_train, y_train)

print(f"Лучшие параметры GradientBoosting: {grid_gbc.best_params_}")
print(f"Лучший Kappa GradientBoosting: {grid_gbc.best_score_:.4f}")

# Вычисление точности на валидационной выборке
y_pred_gbc = grid_gbc.best_estimator_.predict(x_val)
y_pred_gbc = np.ravel(y_pred_gbc)
accuracy_gbc = (y_pred_gbc == y_val).mean()
kappa_gbc = quadratic_kappa(y_val, y_pred_gbc)
print(f"Точность GradientBoosting на валидационной выборке: {accuracy_gbc:.4f}")
print(f"Kappa GradientBoosting на валидационной выборке: {kappa_gbc:.4f}")
print()

# 4. Настройка LightGBM
print("Подбор параметров для LightGBM...")
param_grid_lgb = {
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

model_lgb = lgb.LGBMClassifier(random_state=17, objective='multiclass', num_class=8)
grid_lgb = GridSearchCV(model_lgb, param_grid_lgb, cv=3, scoring=kappa_scorer, n_jobs=-1, verbose=1)
grid_lgb.fit(x_train, y_train)

print(f"Лучшие параметры LightGBM: {grid_lgb.best_params_}")
print(f"Лучший Kappa LightGBM: {grid_lgb.best_score_:.4f}")

# Вычисление точности на валидационной выборке
y_pred_lgb = grid_lgb.best_estimator_.predict(x_val)
y_pred_lgb = np.ravel(y_pred_lgb)
accuracy_lgb = (y_pred_lgb == y_val).mean()
kappa_lgb = quadratic_kappa(y_val, y_pred_lgb)
print(f"Точность LightGBM на валидационной выборке: {accuracy_lgb:.4f}")
print(f"Kappa LightGBM на валидационной выборке: {kappa_lgb:.4f}")
print()

# Обучение финальных моделей на всех данных с лучшими параметрами
print("Обучение финальных моделей на всех данных...")

# XGBoost с лучшими параметрами
model_xgb_final = XGBClassifier(**grid_xgb.best_params_, random_state=17, eval_metric='mlogloss')
model_xgb_final.fit(x, y)

# CatBoost с лучшими параметрами
model_cb_final = CatBoostClassifier(
    **grid_cb.best_params_, 
    random_state=17, 
    verbose=False, 
    loss_function='MultiClass',
    classes_count=8,
    auto_class_weights='Balanced'
)
model_cb_final.fit(x, y)

# GradientBoosting с лучшими параметрами
model_gbc_final = GradientBoostingClassifier(**grid_gbc.best_params_, random_state=17)
model_gbc_final.fit(x, y)

# LightGBM с лучшими параметрами
model_lgb_final = lgb.LGBMClassifier(**grid_lgb.best_params_, random_state=17, objective='multiclass', num_class=8)
model_lgb_final.fit(x, y)

# СОЗДАНИЕ СТЕКИНГ-АНСАМБЛЯ
print("=" * 50)
print("СОЗДАНИЕ СТЕКИНГ-АНСАМБЛЯ")
print("=" * 50)

# Определяем базовые модели для стекинга
estimators = [
    ('xgb', model_xgb_final),
    ('catboost', model_cb_final),
    ('gbc', model_gbc_final),
    ('lgb', model_lgb_final)
]

# Создаем стекинг-классификатор с LogisticRegression в качестве мета-модели
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        multi_class='multinomial',
        random_state=17,
        max_iter=1000,
        C=0.1
    ),
    cv=5,
    passthrough=False,  # Используем только предсказания базовых моделей
    n_jobs=-1
)

# Обучаем стекинг-модель
print("Обучение стекинг-ансамбля...")
stacking_model.fit(x, y)

# Оценка качества стекинга на валидационной выборке
print("Оценка качества стекинг-ансамбля...")
y_pred_stacking_val = stacking_model.predict(x_val)
y_pred_stacking_val = np.ravel(y_pred_stacking_val)  # Преобразуем в 1D
accuracy_stacking = (y_pred_stacking_val == y_val).mean()
kappa_stacking = quadratic_kappa(y_val, y_pred_stacking_val)

print(f"Качество стекинг-ансамбля на валидационной выборке:")
print(f"Точность: {accuracy_stacking:.4f}")
print(f"Kappa: {kappa_stacking:.4f}")

# Загрузка тестовых данных
print("Загрузка тестовых данных...")
data_test = pd.read_csv("https://video.ittensive.com/machine-learning/prudential/test.csv.gz")
data_test = data_preprocess(data_test)
data_test = reduce_mem_usage(data_test)
data_test_transformed = pd.DataFrame(scaler.transform(pd.DataFrame(data_test, columns=columns)))
print(data_test_transformed.info())

# Предсказания на тестовых данных с помощью стекинг-ансамбля
print("Расчет предсказаний с помощью стекинг-ансамбля...")
test_predictions = stacking_model.predict(data_test_transformed)
test_predictions = np.ravel(test_predictions)  # Преобразуем в 1D
data_test["Response"] = test_predictions + 1  # возвращаем классы 1-8

# Создание submission файла
submission = pd.read_csv("https://video.ittensive.com/machine-learning/prudential/sample_submission.csv.gz")
submission["Response"] = data_test["Response"].astype("int8")
submission.to_csv("submission_stacking.csv", index=False)
print(f"Файл submission_stacking.csv создан! Размер: {len(submission)} строк")

# Проверка качества стекинга на обучающих данных
print("Проверка качества стекинг-ансамбля на обучающих данных...")
train_predictions = stacking_model.predict(x)
train_predictions = np.ravel(train_predictions)  # Преобразуем в 1D
final_kappa = quadratic_kappa(y, train_predictions)
final_accuracy = (train_predictions == y).mean()

print(f"Финальное качество стекинг-ансамбля на обучающих данных:")
print(f"Quadratic Kappa: {final_kappa:.4f}")
print(f"Accuracy: {final_accuracy:.4f}")
print("Матрица ошибок:")
print(confusion_matrix(y, train_predictions))

# Сводка по лучшим параметрам и качеству моделей
print("\n" + "="*60)
print("СВОДКА ПО ЛУЧШИМ ПАРАМЕТРАМ И КАЧЕСТВУ МОДЕЛЕЙ")
print("="*60)
print(f"XGBoost: {grid_xgb.best_params_}")
print(f"  Точность: {accuracy_xgb:.4f}, Kappa: {kappa_xgb:.4f}")
print(f"CatBoost: {grid_cb.best_params_}")
print(f"  Точность: {accuracy_cb:.4f}, Kappa: {kappa_cb:.4f}")
print(f"GradientBoosting: {grid_gbc.best_params_}")
print(f"  Точность: {accuracy_gbc:.4f}, Kappa: {kappa_gbc:.4f}")
print(f"LightGBM: {grid_lgb.best_params_}")
print(f"  Точность: {accuracy_lgb:.4f}, Kappa: {kappa_lgb:.4f}")
print(f"СТЕКИНГ-АНСАМБЛЬ:")
print(f"  Точность: {accuracy_stacking:.4f}, Kappa: {kappa_stacking:.4f}")
print(f"  Финальная точность на полных данных: {final_accuracy:.4f}")
print(f"  Финальный Kappa на полных данных: {final_kappa:.4f}")
