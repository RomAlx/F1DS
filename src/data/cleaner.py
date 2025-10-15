"""
Очистка и подготовка данных с учётом спринтов
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataCleaner:
    def __init__(self):
        """Инициализация обработчика данных"""
        self.driver_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.event_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.feature_columns = None
        self.fitted = False

    def prepare_training_data(self, df):
        """Подготовка данных для обучения"""
        df = df.copy()

        # Удаляем записи без результатов
        df = df[df['position'].notna()].copy()

        # Добавляем признак типа сессии
        if 'session_type' in df.columns:
            df['is_sprint'] = (df['session_type'] == 'sprint').astype(int)
        else:
            df['is_sprint'] = 0

        # Энкодинг категориальных признаков
        df['driver_encoded'] = self.driver_encoder.fit_transform(df['driver'])
        df['team_encoded'] = self.team_encoder.fit_transform(df['team'])
        df['event_encoded'] = self.event_encoder.fit_transform(df['event'])

        # Заполнение пропусков
        df = self._fill_missing_values(df)

        # Создание дополнительных признаков
        df = self._create_features(df)

        # Нормализация числовых признаков
        numeric_features = [
            'grid_position', 'quali_position',
            'q1_time', 'q2_time', 'q3_time',
            'fastest_lap', 'avg_lap_time', 'pit_stops',
            'avg_speed', 'air_temp', 'track_temp', 'humidity',
            'driver_encoded', 'team_encoded', 'event_encoded'
        ]

        # Фильтруем только существующие колонки
        numeric_features = [col for col in numeric_features if col in df.columns]

        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])

        # Добавляем is_sprint в список признаков
        self.feature_columns = numeric_features + ['dnf', 'dsq', 'rainfall', 'is_sprint']
        self.fitted = True

        # Целевая переменная
        x = df[self.feature_columns]
        y = df['position']

        return x, y, df

    def prepare_prediction_data(self, df):
        """Подготовка данных для предсказания"""
        if not self.fitted:
            raise ValueError("Сначала нужно обучить cleaner на training данных!")

        df = df.copy()

        # Проверяем наличие необходимых колонок
        required_columns = ['driver', 'team', 'event']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"❌ Отсутствуют обязательные колонки: {missing_columns}\n"
                f"   Доступные колонки: {list(df.columns)}"
            )

        # Обрабатываем признак спринта
        if 'session_type' in df.columns:
            df['is_sprint'] = (df['session_type'] == 'sprint').astype(int)
        elif 'is_sprint' not in df.columns:
            df['is_sprint'] = 0

        # Энкодинг с использованием уже обученных энкодеров
        df['driver_encoded'] = self._safe_transform(self.driver_encoder, df['driver'])
        df['team_encoded'] = self._safe_transform(self.team_encoder, df['team'])
        df['event_encoded'] = self._safe_transform(self.event_encoder, df['event'])

        # Заполнение пропусков
        df = self._fill_missing_values(df)

        # Создание признаков
        df = self._create_features(df)

        # Нормализация
        numeric_features = [col for col in self.feature_columns
                          if col not in ['dnf', 'dsq', 'rainfall', 'is_sprint']]
        numeric_features = [col for col in numeric_features if col in df.columns]

        df[numeric_features] = self.scaler.transform(df[numeric_features])

        # Проверяем наличие всех необходимых колонок
        for col in self.feature_columns:
            if col not in df.columns:
                print(f"⚠️  Добавляем отсутствующую колонку: {col}")
                df[col] = 0

        x = df[self.feature_columns]

        return x, df

    @staticmethod
    def _safe_transform(encoder, values):
        """Безопасное преобразование с обработкой неизвестных значений"""
        result = []
        unknown_values = []

        for val in values:
            if val in encoder.classes_:
                result.append(encoder.transform([val])[0])
            else:
                # Для неизвестных значений используем средний класс
                result.append(len(encoder.classes_) // 2)
                if val not in unknown_values:
                    unknown_values.append(val)

        if unknown_values:
            print(f"⚠️  Неизвестные значения (используем средний класс): {unknown_values[:3]}...")

        return result

    @staticmethod
    def _fill_missing_values(df):
        """Заполнение пропущенных значений"""
        # Временные признаки - медианой
        time_features = ['q1_time', 'q2_time', 'q3_time', 'fastest_lap', 'avg_lap_time']
        for col in time_features:
            if col in df.columns:
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 75.0
                df[col] = df[col].fillna(median_value)

        # Скорость - медианой
        if 'avg_speed' in df.columns:
            median_value = df['avg_speed'].median()
            if pd.isna(median_value):
                median_value = 150.0
            df['avg_speed'] = df['avg_speed'].fillna(median_value)

        # Погода - средними значениями
        weather_features = ['air_temp', 'track_temp', 'humidity']
        defaults = {'air_temp': 25.0, 'track_temp': 35.0, 'humidity': 50.0}
        for col in weather_features:
            if col in df.columns:
                mean_value = df[col].mean()
                if pd.isna(mean_value):
                    mean_value = defaults[col]
                df[col] = df[col].fillna(mean_value)

        # Пит-стопы - нулем
        if 'pit_stops' in df.columns:
            df['pit_stops'] = df['pit_stops'].fillna(0)

        # Rainfall - нулем
        if 'rainfall' in df.columns:
            df['rainfall'] = df['rainfall'].fillna(0)

        # DNF и DSQ
        if 'dnf' in df.columns:
            df['dnf'] = df['dnf'].fillna(0)
        if 'dsq' in df.columns:
            df['dsq'] = df['dsq'].fillna(0)

        # is_sprint
        if 'is_sprint' in df.columns:
            df['is_sprint'] = df['is_sprint'].fillna(0)

        return df

    @staticmethod
    def _create_features(df):
        """Создание дополнительных признаков"""
        # Разница между позицией на старте и квалификацией
        if 'grid_position' in df.columns and 'quali_position' in df.columns:
            df['grid_quali_diff'] = df['grid_position'] - df['quali_position']

        # Качество квалификации (есть ли время в Q3)
        if 'q3_time' in df.columns:
            df['made_q3'] = (~df['q3_time'].isna()).astype(int)

        return df

    @staticmethod
    def aggregate_season_results(df):
        """Агрегация результатов по сезону для гонщиков и команд"""
        # Результаты по гонщикам
        driver_results = df.groupby('driver').agg({
            'points': 'sum',
            'position': 'mean',
            'dnf': 'sum',
            'team': 'first'
        }).reset_index()

        driver_results.columns = ['driver', 'total_points', 'avg_position', 'dnf_count', 'team']
        driver_results = driver_results.sort_values('total_points', ascending=False)

        # Результаты по командам
        team_results = df.groupby('team').agg({
            'points': 'sum',
            'position': 'mean',
        }).reset_index()

        team_results.columns = ['team', 'total_points', 'avg_position']
        team_results = team_results.sort_values('total_points', ascending=False)

        return driver_results, team_results