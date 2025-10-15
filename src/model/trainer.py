"""
Обучение модели предсказания с учётом спринтов
"""
import pickle
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import MODEL_CONFIG
from src.data.loader import F1DataLoader
from src.data.cleaner import DataCleaner

class ModelTrainer:
    def __init__(self):
        """Инициализация тренера модели"""
        self.loader = F1DataLoader()
        self.cleaner = DataCleaner()
        self.model = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def train(self, years):
        """Обучение модели на данных за указанные года"""
        print("\n🔄 Загрузка данных...")

        # Загружаем данные за все года
        all_data = []
        total_races = 0
        total_sprints = 0

        for year in years:
            try:
                print(f"  📅 Загрузка данных за {year}...")
                year_data = self.loader.load_season_data(year)
                all_data.append(year_data)

                # Подсчитываем количество гонок и спринтов
                if 'session_type' in year_data.columns:
                    races = len(year_data[year_data['session_type'] == 'race']) // 20
                    sprints = len(year_data[year_data['session_type'] == 'sprint']) // 20
                    total_races += races
                    total_sprints += sprints
                    print(f"     ✓ {races} гонок, {sprints} спринтов")

            except Exception as e:
                print(f"  ⚠️  Ошибка загрузки {year}: {e}")
                continue

        if not all_data:
            raise ValueError("Не удалось загрузить данные ни за один год!")

        # Объединяем все данные
        import pandas as pd
        df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Всего записей: {len(df)}")
        print(f"   🏁 Гонок: ~{total_races}")
        print(f"   ⚡ Спринтов: ~{total_sprints}")

        # Подготовка данных
        print("\n🧹 Подготовка данных...")
        x, y, processed_df = self.cleaner.prepare_training_data(df)

        # Разделение на train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=MODEL_CONFIG.get('test_size', 0.15),
            random_state=MODEL_CONFIG.get('random_state', 42),
        )

        print(f"\n🎯 Обучающая выборка: {len(x_train)} записей")
        print(f"🧪 Тестовая выборка: {len(x_test)} записей")

        # Обучение модели
        print("\n🤖 Обучение модели Gradient Boosting...")
        self.model = GradientBoostingRegressor(
            n_estimators=MODEL_CONFIG.get('n_estimators', 300),
            max_depth=MODEL_CONFIG.get('max_depth', 12),
            random_state=MODEL_CONFIG.get('random_state', 42),
            learning_rate=MODEL_CONFIG.get('learning_rate', 0.1),
            subsample=MODEL_CONFIG.get('subsample', 0.8),
            min_samples_split=MODEL_CONFIG.get('min_samples_split', 5),
            min_samples_leaf=MODEL_CONFIG.get('min_samples_leaf', 2),
            max_features=MODEL_CONFIG.get('max_features', 'sqrt'),
            validation_fraction=MODEL_CONFIG.get('validation_fraction', 0.1),
            n_iter_no_change=MODEL_CONFIG.get('n_iter_no_change', 10),
            verbose=MODEL_CONFIG.get('verbose', 1)
        )

        self.model.fit(x_train, y_train)

        # Оценка модели
        print("\n📈 Оценка качества модели...")
        y_pred = self.model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Accuracy для топ-10
        y_pred_rounded = np.round(y_pred).astype(int)
        y_pred_rounded = np.clip(y_pred_rounded, 1, 20)

        top10_accuracy = np.mean((y_test <= 10) == (y_pred_rounded <= 10))

        # Дополнительные метрики
        top3_accuracy = np.mean((y_test <= 3) == (y_pred_rounded <= 3))

        # Feature importance (топ-5 признаков)
        feature_importance = dict(zip(
            self.cleaner.feature_columns,
            self.model.feature_importances_
        ))
        top_features = sorted(feature_importance.items(), key=lambda lambda_x: lambda_x[1], reverse=True)[:5]

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'accuracy': top10_accuracy,
            'top3_accuracy': top3_accuracy,
            'years': years,
            'train_size': len(x_train),
            'test_size': len(x_test),
            'total_races': total_races,
            'total_sprints': total_sprints,
            'top_features': [{'name': name, 'importance': float(imp)} for name, imp in top_features]
        }

        # Сохранение модели
        model_name = f"model_{'-'.join(map(str, years))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.pkl"

        model_data = {
            'model': self.model,
            'cleaner': self.cleaner,
            'metrics': metrics,
            'config': MODEL_CONFIG,
            'feature_columns': self.cleaner.feature_columns
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Сохраняем метаданные
        meta_path = self.models_dir / f"{model_name}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'name': model_name,
                'years': years,
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in metrics.items()},
                'created_at': datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)

        metrics['model_path'] = str(model_path)

        return metrics

    def list_models(self):
        """Получение списка доступных моделей"""
        models = []

        for meta_file in self.models_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                model_file = self.models_dir / f"{meta['name']}.pkl"
                if model_file.exists():
                    meta['path'] = str(model_file)
                    models.append(meta)
            except Exception:
                continue

        # Сортируем по дате создания
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return models

    @staticmethod
    def load_model(model_path):
        """Загрузка сохраненной модели"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data