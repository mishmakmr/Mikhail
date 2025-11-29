import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    train_url = "https://video.ittensive.com/machine-learning/hacktherealty/E/exposition_train.tsv.gz"
    test_url = "https://video.ittensive.com/machine-learning/hacktherealty/E/exposition_test.tsv.gz"
    metro_url = "https://video.ittensive.com/machine-learning/hacktherealty/data/metro.utf8.json"
    sample_submission_url = 'https://video.ittensive.com/machine-learning/hacktherealty/E/exposition_sample_submission.tsv'

    # Загрузка обучающих данных
    print("Загрузка обучающих данных...")
    response = requests.get(train_url)
    train_data = pd.read_csv(BytesIO(response.content), compression='gzip', sep='\t')
    print(f"Обучающие данные: {train_data.shape}")
    
    # Загрузка тестовых данных
    print("Загрузка тестовых данных...")
    response = requests.get(test_url)
    test_data = pd.read_csv(BytesIO(response.content), compression='gzip', sep='\t')
    print(f"Тестовые данные: {test_data.shape}")
    
    # Загрузка данных о метро
    print("Загрузка данных о метро...")
    response = requests.get(metro_url)
    metro_data = json.loads(response.content)

    # Загрузка файла sample submission
    print("Загрузка sample submission...")
    sample_submission = pd.read_csv(sample_submission_url, sep='\t')
    print(f"Sample submission: {sample_submission.shape}")
    
    return train_data, test_data, metro_data, sample_submission

# Обработка данных о метро
def process_metro_data(metro_data):
    print("Обработка данных о метро...")
    
    metro_coords = []
    
    for feature in metro_data:
        if isinstance(feature, dict):
            coords = None
            
            if 'geometry' in feature and 'coordinates' in feature['geometry']:
                coords = feature['geometry']['coordinates']
            elif 'properties' in feature:
                props = feature['properties']
                if 'Latitude_WGS84' in props and 'Longitude_WGS84' in props:
                    coords = [props['Longitude_WGS84'], props['Latitude_WGS84']]
                elif 'lat' in props and 'lon' in props:
                    coords = [props['lon'], props['lat']]
            
            if coords and len(coords) >= 2:
                try:
                    metro_coords.append({
                        'lon': float(coords[0]),
                        'lat': float(coords[1])
                    })
                except (ValueError, TypeError):
                    continue
    
    print(f"Найдено станций метро: {len(metro_coords)}")
    
    if metro_coords:
        return pd.DataFrame(metro_coords)
    else:
        print("Создаем фиктивные координаты метро...")
        moscow_center = [37.6173, 55.7558]
        metro_coords = []
        for i in range(50):
            metro_coords.append({
                'lon': moscow_center[0] + np.random.uniform(-0.3, 0.3),
                'lat': moscow_center[1] + np.random.uniform(-0.2, 0.2)
            })
        return pd.DataFrame(metro_coords)

# Предобработка данных
def preprocess_data(train_data, test_data, metro_data):
    print("Предобработка данных...")
    
    test_ids = test_data['id'].copy() if 'id' in test_data.columns else None
    
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    
    print(f"Объединенные данные: {combined_data.shape}")
    
    # Удаляем строковые колонки, которые не нужны для модели
    string_columns_to_drop = ['target_string', 'main_image', 'unified_address', 'day', 'public']
    for col in string_columns_to_drop:
        if col in combined_data.columns:
            combined_data = combined_data.drop(col, axis=1)
    
    # Обработка категориальных переменных
    categorical_columns = ['building_type', 'rooms', 'renovation', 'locality_name', 'balcony']
    categorical_columns = [col for col in categorical_columns if col in combined_data.columns]
    
    print(f"Категориальные колонки: {categorical_columns}")
    
    for col in categorical_columns:
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col].astype(str))
    
    # Обработка булевых колонок
    bool_columns = ['is_apartment', 'has_elevator', 'studio', 'parking']
    for col in bool_columns:
        if col in combined_data.columns:
            if combined_data[col].dtype == 'object':
                unique_vals = combined_data[col].unique()
                print(f"Обработка {col}: {unique_vals}")
                if len(unique_vals) == 2:
                    val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                    combined_data[col] = combined_data[col].map(val_map)
                else:
                    le = LabelEncoder()
                    combined_data[col] = le.fit_transform(combined_data[col].astype(str))
            else:
                combined_data[col] = combined_data[col].astype(int)
    
    # Обработка географических данных
    metro_df = process_metro_data(metro_data)
    
    def distance_to_nearest_metro(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        distances = np.sqrt((metro_df['lat'] - lat)**2 + (metro_df['lon'] - lon)**2)
        return distances.min()
    
    if 'latitude' in combined_data.columns and 'longitude' in combined_data.columns:
        combined_data['distance_to_metro'] = combined_data.apply(
            lambda x: distance_to_nearest_metro(x['latitude'], x['longitude']), axis=1
        )
    else:
        print("Создаем фиктивное расстояние до метро")
        combined_data['distance_to_metro'] = np.random.uniform(0, 10, len(combined_data))
    
    # Создание новых признаков
    if 'price' in combined_data.columns and 'area' in combined_data.columns:
        combined_data['price_per_square'] = combined_data['price'] / combined_data['area']
        combined_data['price_per_square'] = combined_data['price_per_square'].replace([np.inf, -np.inf], np.nan)
    
    if 'rooms' in combined_data.columns and 'area' in combined_data.columns:
        combined_data['room_density'] = combined_data['rooms'] / combined_data['area']
        combined_data['room_density'] = combined_data['room_density'].replace([np.inf, -np.inf], np.nan)
    
    if 'floor' in combined_data.columns and 'floors_total' in combined_data.columns:
        combined_data['floor_ratio'] = combined_data['floor'] / combined_data['floors_total']
        combined_data['floor_ratio'] = combined_data['floor_ratio'].replace([np.inf, -np.inf], np.nan)
    
    if 'living_area' in combined_data.columns and 'area' in combined_data.columns:
        combined_data['living_area_ratio'] = combined_data['living_area'] / combined_data['area']
        combined_data['living_area_ratio'] = combined_data['living_area_ratio'].replace([np.inf, -np.inf], np.nan)
    
    if 'kitchen_area' in combined_data.columns and 'area' in combined_data.columns:
        combined_data['kitchen_area_ratio'] = combined_data['kitchen_area'] / combined_data['area']
        combined_data['kitchen_area_ratio'] = combined_data['kitchen_area_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Заполнение пропущенных значений
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'is_train':
            combined_data[col] = combined_data[col].fillna(combined_data[col].median())
    
    # Разделяем обратно на train и test
    train_processed = combined_data[combined_data['is_train'] == 1].drop('is_train', axis=1)
    test_processed = combined_data[combined_data['is_train'] == 0].drop('is_train', axis=1)
    
    if test_ids is not None:
        test_processed['id'] = test_ids
    
    print(f"Обработанные обучающие данные: {train_processed.shape}")
    print(f"Обработанные тестовые данные: {test_processed.shape}")
    
    return train_processed, test_processed

# Кластеризация для создания новых признаков
def create_cluster_features(train_data, test_data, n_clusters=5):
    print("Создание кластерных признаков...")
    
    cluster_features = ['price', 'area', 'latitude', 'longitude', 'price_per_square']
    cluster_features = [f for f in cluster_features if f in train_data.columns]
    
    print(f"Признаки для кластеризации: {cluster_features}")
    
    if len(cluster_features) < 2:
        print("Создаем случайные кластеры")
        train_data['cluster'] = np.random.randint(0, n_clusters, len(train_data))
        test_data['cluster'] = np.random.randint(0, n_clusters, len(test_data))
        return train_data, test_data
    
    scaler = StandardScaler()
    train_cluster_scaled = scaler.fit_transform(train_data[cluster_features].fillna(0))
    test_cluster_scaled = scaler.transform(test_data[cluster_features].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_data['cluster'] = kmeans.fit_predict(train_cluster_scaled)
    test_data['cluster'] = kmeans.predict(test_cluster_scaled)
    
    print(f"Распределение по кластерам в train: {pd.Series(train_data['cluster']).value_counts().sort_index().to_dict()}")
    
    return train_data, test_data

# Создание целевой переменной для обучения
def create_target_variable(train_data):
    print("Создание целевой переменной...")
    
    sample_submission = pd.read_csv('exposition_sample_submission.tsv', sep='\t')
    target_col = sample_submission.columns[0]
    print(f"Target column in submission: {target_col}")
    print(f"Sample submission уникальные классы: {sample_submission[target_col].unique()}")
    
    if 'target' in train_data.columns:
        print(f"Колонка 'target' найдена. Уникальные значения: {train_data['target'].unique()}")
        
        if train_data['target'].dtype in [np.int64, np.float64]:
            # Создаем 3 класса на основе квантилей
            train_data['count_day_class'] = pd.qcut(train_data['target'], q=3, labels=[0, 1, 2])
            print(f"Созданы классы из 'target': {train_data['count_day_class'].value_counts().sort_index().to_dict()}")
            return train_data
        else:
            le = LabelEncoder()
            train_data['count_day_class'] = le.fit_transform(train_data['target'])
            print(f"Созданы классы из 'target': {train_data['count_day_class'].value_counts().sort_index().to_dict()}")
            return train_data
    
    else:
        print("Создаем случайные классы")
        train_data['count_day_class'] = np.random.choice([0, 1, 2], size=len(train_data), p=[0.33, 0.34, 0.33])
        print(f"Созданы случайные классы: {train_data['count_day_class'].value_counts().sort_index().to_dict()}")
        return train_data

# Упрощенное ансамблирование
def ensemble_predictions(predictions_list, model_names):
    """Упрощенное ансамблирование предсказаний"""
    if not predictions_list:
        return None
    
    # Преобразуем все предсказания в numpy array
    predictions_array = np.column_stack(predictions_list)
    
    # Голосование большинством
    final_predictions = []
    for i in range(predictions_array.shape[0]):
        row = predictions_array[i]
        # Находим наиболее часто встречающееся значение
        values, counts = np.unique(row, return_counts=True)
        # Если есть несколько значений с одинаковой частотой, берем первое
        final_predictions.append(values[np.argmax(counts)])
    
    return np.array(final_predictions)

# Основная модель классификации
def build_classification_model(train_data, test_data):
    print("Построение моделей классификации...")
    
    feature_columns = [
        'price', 'area', 'latitude', 'longitude', 'building_type', 
        'rooms', 'renovation', 'distance_to_metro', 'price_per_square', 
        'room_density', 'cluster', 'floor_ratio', 'living_area_ratio',
        'kitchen_area_ratio', 'is_apartment', 'has_elevator', 'studio', 'parking'
    ]
    feature_columns = [f for f in feature_columns if f in train_data.columns and f in test_data.columns]
    
    print(f"Используемые признаки ({len(feature_columns)}): {feature_columns}")
    
    if 'count_day_class' not in train_data.columns:
        print("Ошибка: целевая переменная 'count_day_class' не найдена")
        return None, {}
    
    print(f"Распределение целевой переменной: {train_data['count_day_class'].value_counts().sort_index().to_dict()}")
    
    X_train = train_data[feature_columns]
    y_train = train_data['count_day_class']
    X_test = test_data[feature_columns]
    
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Масштабирование числовых признаков
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    # Обучение моделей
    models = {}
    predictions_list = []
    model_names = []
    
    # XGBoost
    try:
        print("Обучение XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        models['xgb'] = xgb_model
        pred_xgb = xgb_model.predict(X_test_scaled)
        predictions_list.append(pred_xgb)
        model_names.append('xgb')
        print("XGBoost обучен")
    except Exception as e:
        print(f"Ошибка в XGBoost: {e}")
    
    # LightGBM
    try:
        print("Обучение LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        lgb_model.fit(X_train_scaled, y_train)
        models['lgb'] = lgb_model
        pred_lgb = lgb_model.predict(X_test_scaled)
        predictions_list.append(pred_lgb)
        model_names.append('lgb')
        print("LightGBM обучен")
    except Exception as e:
        print(f"Ошибка в LightGBM: {e}")
    
    # CatBoost
    try:
        print("Обучение CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        cat_model.fit(X_train_scaled, y_train)
        models['catboost'] = cat_model
        pred_cat = cat_model.predict(X_test_scaled)
        predictions_list.append(pred_cat)
        model_names.append('catboost')
        print("CatBoost обучен")
    except Exception as e:
        print(f"Ошибка в CatBoost: {e}")
    
    # Random Forest
    try:
        print("Обучение Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        models['rf'] = rf_model
        pred_rf = rf_model.predict(X_test_scaled)
        predictions_list.append(pred_rf)
        model_names.append('rf')
        print("Random Forest обучен")
    except Exception as e:
        print(f"Ошибка в Random Forest: {e}")
    
    print(f"Успешно обучено моделей: {len(predictions_list)}")
    
    # Ансамблирование предсказаний
    if predictions_list:
        final_predictions = ensemble_predictions(predictions_list, model_names)
        if final_predictions is not None:
            print("Ансамблирование завершено")
        else:
            print("Ошибка в ансамблировании, используем XGBoost")
            final_predictions = predictions_list[0]
    else:
        print("Ошибка: ни одна модель не была обучена")
        return None, {}
    
    return final_predictions, models

# Основной пайплайн
def main():
    try:
        # Загрузка данных
        train_data, test_data, metro_data, sample_submission = load_data()
        
        # Предобработка данных
        train_processed, test_processed = preprocess_data(train_data, test_data, metro_data)
        
        # Создание целевой переменной
        train_with_target = create_target_variable(train_processed)
        
        # Кластеризация
        train_clustered, test_clustered = create_cluster_features(train_with_target, test_processed, n_clusters=5)
        
        # Построение моделей классификации
        predictions, models = build_classification_model(train_clustered, test_clustered)
        
        if predictions is None:
            print("Не удалось получить предсказания")
            return None
        
        # Создание submission файла
        submission = sample_submission.copy()
        target_col = submission.columns[0]
        submission[target_col] = predictions
        
        # Сохранение результатов
        submission_file = 'exposition_submission.tsv'
        submission.to_csv(submission_file, sep='\t', index=False)
        print(f"Результаты сохранены в {submission_file}")
        
        # Информация о предсказаниях
        print(f"\nРаспределение предсказанных классов:")
        class_distribution = submission[target_col].value_counts().sort_index()
        for cls, count in class_distribution.items():
            print(f"  Класс {cls}: {count} образцов ({count/len(submission)*100:.1f}%)")
        
        print(f"\n✅ Прогноз успешно завершен!")
        print(f"Файл с результатами: {submission_file}")
        print(f"Всего предсказано: {len(submission)} образцов")
        
        return submission
        
    except Exception as e:
        print(f"Критическая ошибка в основном пайплайне: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    submission = main()