# life_expectancy_prediction.py
"""
Прогнозирование продолжительности жизни с использованием ансамблевых методов
Life Expectancy Prediction using Ensemble Methods

Автор: [Makarin Mikhail
Date: 2025
"""

# Подключение библиотек
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
from sklearn import manifold
import umap
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import urllib.request
import gzip
import shutil
import tarfile
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import re
import warnings
warnings.filterwarnings('ignore')

class LifeExpectancyPredictor:
    """
    Класс для прогнозирования продолжительности жизни с использованием ансамблевых методов
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.models = {}
        self.feature_sets = {}
        self.predictions = {}
        
    def setup_environment(self):
        """Настройка окружения и стилей графиков"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        os.makedirs('graphs', exist_ok=True)
        print("✅ Окружение настроено")

    def save_plot(self, fig, filename, dpi=300):
        """Сохраняет график в папку graphs"""
        path = os.path.join('graphs', filename)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 График сохранен: {path}")

    def download_file(self, url, filename):
        """Скачивает файл если он отсутствует"""
        if not os.path.exists(filename):
            print(f"Скачивание {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✅ {filename} загружен")
        else:
            print(f"✅ {filename} уже существует")

    def check_required_files(self):
        """Проверяет наличие всех необходимых файлов"""
        required_files = {
            'rosstat.csv': 'https://video.ittensive.com/machine-learning/sc-tatar2020/rosstat/rosstat.csv'
        }
        
        print("=" * 50)
        print("ПРОВЕРКА И ЗАГРУЗКА ДАННЫХ")
        print("=" * 50)
        
        for file_key, url in required_files.items():
            self.download_file(url, file_key)
        
        all_exists = True
        for file_key in required_files.keys():
            if not os.path.exists(file_key):
                print(f"❌ Отсутствует: {file_key}")
                all_exists = False
            else:
                file_size = os.path.getsize(file_key) / (1024 * 1024)
                print(f"✅ {file_key}: {file_size:.1f} Мб")
        
        if all_exists:
            print("🎉 Все файлы готовы к работе!")
        else:
            print("⚠️ Некоторые файлы отсутствуют!")
        
        return all_exists

    def linear_extrapolation(self, x):
        """Линейная экстраполяция для заполнения пропусков"""
        X = np.array(x.dropna().index.astype(int)).reshape(-1, 1)
        Y = np.array(x.dropna().values).reshape(-1, 1)
        if X.shape[0] > 0:
            f = LinearRegression().fit(X, Y)
            for i in x.index:
                v = x.loc[i]
                if v != v:
                    v = f.predict([[int(i)]])[0][0]
                    if v < 0:
                        v = 0
                    x.loc[i] = v
        return x

    def clean_feature_names(self, feature_names):
        """Очищает названия признаков от специальных символов"""
        cleaned_names = []
        for name in feature_names:
            cleaned = re.sub(r'[^\w]', '_', str(name))
            cleaned = re.sub(r'_+', '_', cleaned)
            cleaned = cleaned.strip('_')
            if not cleaned:
                cleaned = 'feature'
            cleaned_names.append(cleaned)
        return cleaned_names

    def safe_corr(self, x, y):
        """Безопасное вычисление корреляции с обработкой NaN"""
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0
        return np.corrcoef(x[mask], y[mask])[0, 1]

    def load_and_preprocess_data(self):
        """Загрузка и предобработка данных"""
        print("\n" + "="*50)
        print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*50)

        data = pd.read_csv("rosstat.csv", na_values=["-", " - ","...","…"," -"])
        self.raw_data = data.copy()

        features = data["feature"]
        data.drop(labels=["feature"], inplace=True, axis=1)
        data.interpolate(method="linear", axis=1, inplace=True)
        data = data.apply(self.linear_extrapolation, axis=1, result_type="expand")
        data["feature"] = features
        data.dropna(inplace=True)
        features = data["feature"]

        data = data.T[:len(data.columns)-1].astype("float")
        data.drop(labels=["2019"], inplace=True)

        data = pd.DataFrame(self.scaler.fit_transform(data))
        data.columns = features
        
        self.features_array = np.array(features)
        self.data_scaled = data
        
        # Целевая переменная
        self.y_column = "ОЖИДАЕМАЯ ПРОДОЛЖИТЕЛЬНОСТЬ ЖИЗНИ ПРИ РОЖДЕНИИ 1.16.1. Все население (число лет)"
        self.y = data[self.y_column]
        
        # Признаки
        y_columns = [
            "ОЖИДАЕМАЯ ПРОДОЛЖИТЕЛЬНОСТЬ ЖИЗНИ ПРИ РОЖДЕНИИ 1.16.2. Мужчины (число лет)",
            "ОЖИДАЕМАЯ ПРОДОЛЖИТЕЛЬНОСТЬ ЖИЗНИ ПРИ РОЖДЕНИИ 1.16.3. Женщины (число лет)"
        ]
        
        columns_to_drop = [self.y_column]
        for col in y_columns:
            if col in data.columns:
                columns_to_drop.append(col)

        self.x = data.drop(labels=columns_to_drop, axis=1)
        
        print("✅ Данные загружены и подготовлены")
        print(f"📊 Размерность данных: {self.data_scaled.shape}")
        print(f"🎯 Целевая переменная: {self.y_column}")
        print(f"🔢 Количество признаков: {self.x.shape[1]}")

    def create_feature_sets(self):
        """Создание 3 наборов признаков разными методами"""
        print("\n" + "="*50)
        print("СОЗДАНИЕ 3 НАБОРОВ ПРИЗНАКОВ")
        print("="*50)

        # Набор 1: Матричные методы
        print("Набор 1: Матричные методы...")
        ensemble_matrix = self._matrix_methods_ensemble()
        self.top5_set1_indices = np.argsort(ensemble_matrix)[::-1][:5]
        self.top5_set1_features = [self.features_array[i] for i in self.top5_set1_indices]
        
        # Набор 2: Статистические методы
        print("\nНабор 2: Статистические методы...")
        ensemble_stats = self._statistical_methods_ensemble()
        self.top5_set2_indices = np.argsort(ensemble_stats)[::-1][:5]
        self.top5_set2_features = [self.features_array[i] for i in self.top5_set2_indices]
        
        # Набор 3: Комбинированный ансамбль
        print("\nНабор 3: Комбинированный ансамбль...")
        ensemble_combined = self._combined_ensemble(ensemble_matrix, ensemble_stats)
        self.top5_set3_indices = np.argsort(ensemble_combined)[::-1][:5]
        self.top5_set3_features = [self.features_array[i] for i in self.top5_set3_indices]
        
        self._visualize_feature_sets(ensemble_matrix, ensemble_stats, ensemble_combined)

    def _matrix_methods_ensemble(self):
        """Матричные методы для выделения признаков"""
        pca = PCA(n_components=10, random_state=self.random_state).fit(self.x)
        svd = TruncatedSVD(n_components=10, random_state=self.random_state).fit(self.x)
        ica = FastICA(n_components=10, random_state=self.random_state).fit(self.x)
        nmf = NMF(n_components=10, random_state=self.random_state, max_iter=1000).fit(self.x)
        umap_model = umap.UMAP(n_components=2, random_state=self.random_state, n_jobs=1).fit(self.x)
        mds = manifold.MDS(n_components=2, random_state=self.random_state, n_init=1).fit(self.x)

        ensemble_matrix = np.zeros(len(self.x.columns))
        models_matrix = [pca, svd, ica, nmf]
        
        for model in models_matrix:
            if hasattr(model, 'components_'):
                components = np.abs(model.components_)
                for comp in components:
                    importance = MinMaxScaler().fit_transform(comp.reshape(-1, 1)).flatten()
                    importance = np.nan_to_num(importance, nan=0.0)
                    ensemble_matrix += importance

        for model in [umap_model, mds]:
            if hasattr(model, 'embedding_'):
                embedding = model.embedding_
                for comp in range(embedding.shape[1]):
                    comp_importance = np.zeros(self.x.shape[1])
                    for feat in range(self.x.shape[1]):
                        corr = self.safe_corr(embedding[:, comp], self.x.iloc[:, feat])
                        comp_importance[feat] = abs(corr)
                    comp_importance = MinMaxScaler().fit_transform(comp_importance.reshape(-1, 1)).flatten()
                    comp_importance = np.nan_to_num(comp_importance, nan=0.0)
                    ensemble_matrix += comp_importance

        return np.nan_to_num(ensemble_matrix, nan=0.0)

    def _statistical_methods_ensemble(self):
        """Статистические методы для выделения признаков"""
        # Корреляция
        correlations = np.array([abs(self.safe_corr(self.x[col], self.y)) for col in self.x.columns])
        correlations = MinMaxScaler().fit_transform(correlations.reshape(-1, 1)).flatten()
        correlations = np.nan_to_num(correlations, nan=0.0)

        # Взаимная информация
        mi = mutual_info_regression(self.x, self.y, random_state=self.random_state)
        mi = MinMaxScaler().fit_transform(mi.reshape(-1, 1)).flatten()
        mi = np.nan_to_num(mi, nan=0.0)

        # Важность из деревьев
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state).fit(self.x, self.y)
        rf_importance = MinMaxScaler().fit_transform(rf.feature_importances_.reshape(-1, 1)).flatten()
        rf_importance = np.nan_to_num(rf_importance, nan=0.0)

        # Линейные модели
        lasso = Lasso(alpha=0.1, random_state=self.random_state, max_iter=1000).fit(self.x, self.y)
        lasso_importance = MinMaxScaler().fit_transform(np.abs(lasso.coef_).reshape(-1, 1)).flatten()
        lasso_importance = np.nan_to_num(lasso_importance, nan=0.0)

        ridge = Ridge(alpha=0.1).fit(self.x, self.y)
        ridge_importance = MinMaxScaler().fit_transform(np.abs(ridge.coef_).reshape(-1, 1)).flatten()
        ridge_importance = np.nan_to_num(ridge_importance, nan=0.0)

        return correlations + mi + rf_importance + lasso_importance + ridge_importance

    def _combined_ensemble(self, ensemble_matrix, ensemble_stats):
        """Комбинированный ансамбль методов"""
        et = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state).fit(self.x, self.y)
        et_importance = MinMaxScaler().fit_transform(et.feature_importances_.reshape(-1, 1)).flatten()
        et_importance = np.nan_to_num(et_importance, nan=0.0)

        en = ElasticNet(alpha=0.1, random_state=self.random_state, max_iter=1000).fit(self.x, self.y)
        en_importance = MinMaxScaler().fit_transform(np.abs(en.coef_).reshape(-1, 1)).flatten()
        en_importance = np.nan_to_num(en_importance, nan=0.0)

        return (ensemble_matrix + ensemble_stats + et_importance + en_importance) / 4

    def _visualize_feature_sets(self, ensemble_matrix, ensemble_stats, ensemble_combined):
        """Визуализация наборов признаков"""
        fig1 = plt.figure(figsize=(15, 10))

        # Топ-5 признаков из каждого набора
        plt.subplot(2, 2, 1)
        sets_features = (self.top5_set1_features + self.top5_set2_features + self.top5_set3_features)
        sets_scores = (list(ensemble_matrix[self.top5_set1_indices]) + 
                      list(ensemble_stats[self.top5_set2_indices]) + 
                      list(ensemble_combined[self.top5_set3_indices]))
        
        colors = ['blue'] * 5 + ['green'] * 5 + ['red'] * 5
        plt.barh(range(len(sets_features)), sets_scores, color=colors, alpha=0.7)
        plt.yticks(range(len(sets_features)), 
                  [f'{feat[:40]}...' if len(feat) > 40 else feat for feat in sets_features])
        plt.xlabel('Важность')
        plt.title('Топ-5 признаков из 3 наборов')
        plt.grid(True, alpha=0.3)

        # Пересечение наборов
        plt.subplot(2, 2, 2)
        set1_set = set(self.top5_set1_indices)
        set2_set = set(self.top5_set2_indices)
        set3_set = set(self.top5_set3_indices)

        categories = ['Только набор 1', 'Только набор 2', 'Только набор 3', 
                     '1 и 2', '1 и 3', '2 и 3', 'Все три']
        values = [
            len(set1_set - set2_set - set3_set),
            len(set2_set - set1_set - set3_set), 
            len(set3_set - set1_set - set2_set),
            len((set1_set & set2_set) - set3_set),
            len((set1_set & set3_set) - set2_set),
            len((set2_set & set3_set) - set1_set),
            len(set1_set & set2_set & set3_set)
        ]

        plt.bar(categories, values, color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Количество признаков')
        plt.title('Пересечение признаков между наборами')
        plt.grid(True, alpha=0.3)

        # Сравнение методов
        plt.subplot(2, 2, 3)
        methods = ['Матричные', 'Статистические', 'Комбинированные']
        scores = [
            np.sum(ensemble_matrix[self.top5_set1_indices]),
            np.sum(ensemble_stats[self.top5_set2_indices]),
            np.sum(ensemble_combined[self.top5_set3_indices])
        ]

        plt.bar(methods, scores, color=['blue', 'green', 'red'], alpha=0.7)
        plt.ylabel('Суммарная важность топ-5 признаков')
        plt.title('Сравнение методов отбора')
        plt.grid(True, alpha=0.3)

        for i, score in enumerate(scores):
            plt.text(i, score + max(scores)*0.01, f'{score:.2f}', ha='center', va='bottom')

        # Распределение важности
        plt.subplot(2, 2, 4)
        matrix_norm = ensemble_matrix / (np.max(ensemble_matrix) + 1e-10)
        stats_norm = ensemble_stats / (np.max(ensemble_stats) + 1e-10)
        combined_norm = ensemble_combined / (np.max(ensemble_combined) + 1e-10)

        plt.plot(np.sort(matrix_norm)[::-1][:20], label='Матричные', marker='o')
        plt.plot(np.sort(stats_norm)[::-1][:20], label='Статистические', marker='s')
        plt.plot(np.sort(combined_norm)[::-1][:20], label='Комбинированные', marker='^')
        plt.xlabel('Ранг признака')
        plt.ylabel('Нормализованная важность')
        plt.title('Распределение важности признаков')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig1, '01_feature_sets_comparison.png')
        plt.show()

    def create_stacking_ensemble(self, x_train, y_train, x_test, ensemble_name):
        """Создает ансамбль стекинга и возвращает предсказания"""
        print(f"Создание ансамбля стекинга: {ensemble_name}")
        
        # Очищаем названия признаков
        x_train_clean = x_train.copy()
        x_test_clean = x_test.copy()
        clean_columns = self.clean_feature_names(x_train.columns)
        x_train_clean.columns = clean_columns
        x_test_clean.columns = clean_columns
        
        # Базовые модели
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=self.random_state),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=50, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=self.random_state),
            'XGBoost': XGBRegressor(n_estimators=50, random_state=self.random_state),
            'LightGBM': LGBMRegressor(n_estimators=50, random_state=self.random_state, verbose=-1),
            'CatBoost': CatBoostRegressor(iterations=50, verbose=False, random_state=self.random_state),
            'SVR': SVR(kernel='rbf', C=1.0),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state, max_iter=1000),
            'Ridge': Ridge(alpha=0.1, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, random_state=self.random_state, max_iter=1000)
        }
        
        # Предсказания базовых моделей
        base_predictions = []
        model_names = []
        
        for name, model in models.items():
            try:
                if name == 'LightGBM':
                    model.fit(x_train_clean, y_train)
                    pred = model.predict(x_test_clean)
                else:
                    model.fit(x_train, y_train)
                    pred = model.predict(x_test)
                
                base_predictions.append(pred)
                model_names.append(name)
                print(f"  ✅ {name} обучена")
            except Exception as e:
                print(f"  ❌ {name} ошибка: {e}")
        
        # Мета-модель
        if len(base_predictions) > 0:
            base_pred_matrix = np.column_stack(base_predictions)
            meta_model = LinearRegression()
            
            # Out-of-fold предсказания для обучения мета-модели
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            meta_features = []
            meta_target = []
            
            for train_idx, val_idx in kf.split(x_train):
                x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                x_tr_clean = x_tr.copy()
                x_val_clean = x_val.copy()
                x_tr_clean.columns = self.clean_feature_names(x_tr.columns)
                x_val_clean.columns = self.clean_feature_names(x_val.columns)
                
                fold_predictions = []
                for name, model_config in models.items():
                    try:
                        if name == 'RandomForest':
                            model_clone = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                        elif name == 'ExtraTrees':
                            model_clone = ExtraTreesRegressor(n_estimators=50, random_state=self.random_state)
                        elif name == 'GradientBoosting':
                            model_clone = GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
                        elif name == 'XGBoost':
                            model_clone = XGBRegressor(n_estimators=50, random_state=self.random_state)
                        elif name == 'LightGBM':
                            model_clone = LGBMRegressor(n_estimators=50, random_state=self.random_state, verbose=-1)
                            model_clone.fit(x_tr_clean, y_tr)
                            pred = model_clone.predict(x_val_clean)
                        elif name == 'CatBoost':
                            model_clone = CatBoostRegressor(iterations=50, verbose=False, random_state=self.random_state)
                        elif name == 'SVR':
                            model_clone = SVR(kernel='rbf', C=1.0)
                        elif name == 'KNN':
                            model_clone = KNeighborsRegressor(n_neighbors=5)
                        elif name == 'Lasso':
                            model_clone = Lasso(alpha=0.1, random_state=self.random_state, max_iter=1000)
                        elif name == 'Ridge':
                            model_clone = Ridge(alpha=0.1, random_state=self.random_state)
                        elif name == 'ElasticNet':
                            model_clone = ElasticNet(alpha=0.1, random_state=self.random_state, max_iter=1000)
                        
                        if name != 'LightGBM':
                            model_clone.fit(x_tr, y_tr)
                            pred = model_clone.predict(x_val)
                        
                        fold_predictions.append(pred)
                    except Exception as e:
                        continue
                
                if fold_predictions:
                    meta_features.append(np.column_stack(fold_predictions))
                    meta_target.extend(y_val)
            
            if meta_features and meta_target:
                meta_features = np.vstack(meta_features)
                meta_model.fit(meta_features, meta_target)
                
                # Финальное предсказание
                final_prediction = meta_model.predict(base_pred_matrix)
                print(f"  🎯 Ансамбль {ensemble_name} создан ({len(model_names)} моделей)")
                return final_prediction[0], model_names
        
        print(f"  ⚠️ Ансамбль {ensemble_name} не создан")
        return None, []

    def prepare_training_data(self):
        """Подготовка данных для обучения"""
        print("Подготовка данных для обучения ансамблей...")
        
        # Создаем наборы данных для каждого набора признаков
        self.x_set1 = self.data_scaled[[self.features_array[i] for i in self.top5_set1_indices]]
        self.x_set2 = self.data_scaled[[self.features_array[i] for i in self.top5_set2_indices]]
        self.x_set3 = self.data_scaled[[self.features_array[i] for i in self.top5_set3_indices]]
        
        # Сдвигаем данные для прогнозирования (предсказываем следующий год)
        self.x_train_set1 = self.x_set1[:-1]
        self.y_train_set1 = self.y[1:]
        
        self.x_train_set2 = self.x_set2[:-1]
        self.y_train_set2 = self.y[1:]
        
        self.x_train_set3 = self.x_set3[:-1]
        self.y_train_set3 = self.y[1:]
        
        # Данные для прогноза на 2019 год
        self.x_pred_set1 = self.x_set1[-1:]
        self.x_pred_set2 = self.x_set2[-1:]
        self.x_pred_set3 = self.x_set3[-1:]
        
        print("✅ Данные подготовлены для обучения")

    def run_prediction(self):
        """Запуск процесса прогнозирования"""
        print("\n" + "="*50)
        print("СОЗДАНИЕ 3 АНСАМБЛЕЙ СТЕКИНГА")
        print("="*50)
        
        self.prepare_training_data()
        
        # Создаем 3 ансамбля стекинга
        predictions_2019 = []
        self.ensemble_info = []
        
        # Ансамбль 1
        pred1, models1 = self.create_stacking_ensemble(
            self.x_train_set1, self.y_train_set1, self.x_pred_set1, 
            "Ансамбль 1 (Матричные методы)"
        )
        if pred1 is not None:
            predictions_2019.append(pred1)
            self.ensemble_info.append(("Ансамбль 1", len(models1), models1))
        
        # Ансамбль 2
        pred2, models2 = self.create_stacking_ensemble(
            self.x_train_set2, self.y_train_set2, self.x_pred_set2, 
            "Ансамбль 2 (Статистические методы)"
        )
        if pred2 is not None:
            predictions_2019.append(pred2)
            self.ensemble_info.append(("Ансамбль 2", len(models2), models2))
        
        # Ансамбль 3
        pred3, models3 = self.create_stacking_ensemble(
            self.x_train_set3, self.y_train_set3, self.x_pred_set3, 
            "Ансамбль 3 (Комбинированный)"
        )
        if pred3 is not None:
            predictions_2019.append(pred3)
            self.ensemble_info.append(("Ансамбль 3", len(models3), models3))
        
        if predictions_2019:
            self.final_prediction_2019 = np.mean(predictions_2019)
            self._process_results(predictions_2019)
            self._predict_2020()
        else:
            print("❌ Не удалось создать ансамбли для прогнозирования")

    def _process_results(self, predictions_2019):
        """Обработка и визуализация результатов"""
        print("\n" + "="*50)
        print("ПРОГНОЗИРОВАНИЕ НА 2019 ГОД")
        print("="*50)
        
        # Преобразуем обратно в исходную шкалу
        data_temp = self.data_scaled.copy()
        data_temp[self.y_column] = np.ones(len(data_temp)) * self.final_prediction_2019
        data_original = self.scaler.inverse_transform(data_temp)
        self.predicted_life_2019 = data_original[0][np.where(self.features_array == self.y_column)[0][0]]
        
        print("📊 РЕЗУЛЬТАТЫ:")
        print(f"🎯 Прогноз продолжительности жизни на 2019 год: {self.predicted_life_2019:.2f} лет")
        print(f"🔢 Использовано ансамблей: {len(predictions_2019)}")
        
        print("\n🤖 ИНФОРМАЦИЯ ОБ АНСАМБЛЯХ:")
        for name, count, models in self.ensemble_info:
            print(f"  {name}: {count} моделей")
            print(f"    Модели: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        
        self._visualize_results(predictions_2019)

    def _visualize_results(self, predictions_2019):
        """Визуализация результатов прогнозирования"""
        fig2 = plt.figure(figsize=(15, 10))
        
        # Фактическая динамика и прогноз
        plt.subplot(2, 2, 1)
        years = self.data_scaled.index.astype(str)
        actual_life = self.scaler.inverse_transform(self.data_scaled)[:, np.where(self.features_array == self.y_column)[0][0]]
        
        plt.plot(years, actual_life, marker='o', linewidth=2, label='Фактические данные', color='blue')
        plt.axhline(y=self.predicted_life_2019, color='red', linestyle='--', linewidth=2, 
                   label=f'Прогноз на 2019: {self.predicted_life_2019:.2f} лет')
        plt.xlabel('Год')
        plt.ylabel('Продолжительность жизни (лет)')
        plt.title('Динамика и прогноз продолжительности жизни')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Сравнение предсказаний ансамблей
        plt.subplot(2, 2, 2)
        ensemble_names = [info[0] for info in self.ensemble_info]
        
        # Преобразуем предсказания в исходную шкалу
        original_preds = []
        for pred in predictions_2019:
            data_temp = self.data_scaled.copy()
            data_temp[self.y_column] = np.ones(len(data_temp)) * pred
            data_original = self.scaler.inverse_transform(data_temp)
            original_pred = data_original[0][np.where(self.features_array == self.y_column)[0][0]]
            original_preds.append(original_pred)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = plt.bar(ensemble_names, original_preds, color=colors[:len(ensemble_names)], alpha=0.7)
        plt.ylabel('Продолжительность жизни (лет)')
        plt.title('Предсказания отдельных ансамблей')
        plt.grid(True, alpha=0.3)
        
        for bar, pred in zip(bars, original_preds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{pred:.2f}', ha='center', va='bottom')
        
        # Состав ансамблей
        plt.subplot(2, 2, 3)
        sizes = [info[1] for info in self.ensemble_info]
        plt.pie(sizes, labels=ensemble_names, autopct='%1.1f%%', colors=colors[:len(ensemble_names)])
        plt.title('Количество моделей в ансамблях')
        
        # Качество наборов признаков
        plt.subplot(2, 2, 4)
        feature_set_quality = []
        for i, ensemble_name in enumerate(ensemble_names):
            if i < len(original_preds):
                deviation = abs(original_preds[i] - self.predicted_life_2019)
                quality_score = 1.0 / (1.0 + deviation)
                feature_set_quality.append(quality_score)
            else:
                feature_set_quality.append(0.5)
        
        while len(feature_set_quality) < 3:
            feature_set_quality.append(0.5)
        
        set_names = ['Набор 1', 'Набор 2', 'Набор 3'][:len(feature_set_quality)]
        
        plt.bar(set_names, feature_set_quality, color=['blue', 'green', 'red'], alpha=0.7)
        plt.ylabel('Качество набора признаков')
        plt.title('Качество наборов признаков\n(на основе точности прогноза)')
        plt.grid(True, alpha=0.3)
        
        for i, quality in enumerate(feature_set_quality):
            plt.text(i, quality + 0.01, f'{quality:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.save_plot(fig2, '02_prediction_results_2019.png')
        plt.show()

    def _predict_2020(self):
        """Прогноз на 2020 год с экстраполяцией трендов"""
        print("\n" + "="*50)
        print("ДОПОЛНИТЕЛЬНО: ПРОГНОЗ НА 2020 ГОД")
        print("="*50)
        
        print("⚠️  Для точного прогноза на 2020 год необходимы фактические значения факторов за 2019 год.")
        print("📈 Используем экстраполяцию тренда для оценки...")
        
        def extrapolate_trend(series, future_years=1):
            """Экстраполирует тренд временного ряда"""
            X = np.arange(len(series)).reshape(-1, 1)
            model = LinearRegression().fit(X, series)
            future_X = np.arange(len(series), len(series) + future_years).reshape(-1, 1)
            return model.predict(future_X)[0]
        
        # Прогноз факторов на 2019 год для каждого набора
        x_2020_set1 = []
        for feature_idx in self.top5_set1_indices:
            feature_series = self.data_scaled[self.features_array[feature_idx]]
            predicted_value = extrapolate_trend(feature_series.values)
            x_2020_set1.append(predicted_value)
        
        x_2020_set2 = []
        for feature_idx in self.top5_set2_indices:
            feature_series = self.data_scaled[self.features_array[feature_idx]]
            predicted_value = extrapolate_trend(feature_series.values)
            x_2020_set2.append(predicted_value)
        
        x_2020_set3 = []
        for feature_idx in self.top5_set3_indices:
            feature_series = self.data_scaled[self.features_array[feature_idx]]
            predicted_value = extrapolate_trend(feature_series.values)
            x_2020_set3.append(predicted_value)
        
        # Прогноз на 2020 год
        predictions_2020 = []
        
        pred1, models1 = self.create_stacking_ensemble(
            self.x_train_set1, self.y_train_set1, 
            pd.DataFrame([x_2020_set1], columns=self.x_train_set1.columns), 
            "Ансамбль 1 для 2020"
        )
        if pred1 is not None:
            predictions_2020.append(pred1)
        
        pred2, models2 = self.create_stacking_ensemble(
            self.x_train_set2, self.y_train_set2, 
            pd.DataFrame([x_2020_set2], columns=self.x_train_set2.columns), 
            "Ансамбль 2 для 2020"
        )
        if pred2 is not None:
            predictions_2020.append(pred2)
        
        pred3, models3 = self.create_stacking_ensemble(
            self.x_train_set3, self.y_train_set3, 
            pd.DataFrame([x_2020_set3], columns=self.x_train_set3.columns), 
            "Ансамбль 3 для 2020"
        )
        if pred3 is not None:
            predictions_2020.append(pred3)
        
        if predictions_2020:
            final_prediction_2020 = np.mean(predictions_2020)
            
            # Преобразуем в исходную шкалу
            data_temp = self.data_scaled.copy()
            data_temp[self.y_column] = np.ones(len(data_temp)) * final_prediction_2020
            data_original = self.scaler.inverse_transform(data_temp)
            self.predicted_life_2020 = data_original[0][np.where(self.features_array == self.y_column)[0][0]]
            
            print(f"🎯 Прогноз продолжительности жизни на 2020 год: {self.predicted_life_2020:.2f} лет")
            print(f"📈 Прирост по сравнению с 2019 годом: {self.predicted_life_2020 - self.predicted_life_2019:+.2f} лет")
            
            self._visualize_2020_prediction()
        else:
            print("❌ Не удалось построить прогноз на 2020 год")

    def _visualize_2020_prediction(self):
        """Визуализация прогноза на 2020 год"""
        fig3 = plt.figure(figsize=(12, 6))

        years = self.data_scaled.index.astype(str)
        actual_life = self.scaler.inverse_transform(self.data_scaled)[:, np.where(self.features_array == self.y_column)[0][0]]
        
        years_extended = list(years) + ['2019', '2020']
        life_extended = list(actual_life) + [self.predicted_life_2019, self.predicted_life_2020]

        plt.plot(years_extended[:-2], life_extended[:-2], marker='o', linewidth=2,
                label='Фактические данные', color='blue')
        plt.plot(['2018', '2019', '2020'], [actual_life[-1], self.predicted_life_2019, self.predicted_life_2020],
                marker='s', linewidth=2, label='Прогноз', color='red', linestyle='--')
        plt.xlabel('Год')
        plt.ylabel('Продолжительность жизни (лет)')
        plt.title('Прогноз продолжительности жизни на 2019-2020 годы')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        self.save_plot(fig3, '03_2020_prediction.png')
        plt.show()

    def run_complete_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ПРОДОЛЖИТЕЛЬНОСТИ ЖИЗНИ")
        print("="*60)
        
        # Настройка окружения
        self.setup_environment()
        
        # Проверка и загрузка данных
        if not self.check_required_files():
            print("❌ Не все файлы загружены. Прерывание выполнения.")
            return
        
        # Загрузка и предобработка данных
        self.load_and_preprocess_data()
        
        # Создание наборов признаков
        self.create_feature_sets()
        
        # Прогнозирование
        self.run_prediction()
        
        print("\n" + "="*50)
        print("ВЫПОЛНЕНИЕ ЗАДАНИЯ ЗАВЕРШЕНО")
        print("="*50)
        print("✅ Создано 3 набора по 5 наиболее важных признаков")
        print("✅ Создано 3 ансамбля стекинга с разнородными моделями")
        print("✅ Построен прогноз продолжительности жизни на 2019 год")
        print("✅ Построен прогноз продолжительности жизни на 2020 год (с экстраполяцией)")
        print("📊 Все графики сохранены в папке 'graphs'")
        print("="*50)


# Запуск анализа
if __name__ == "__main__":
    predictor = LifeExpectancyPredictor(random_state=42)
    predictor.run_complete_analysis()