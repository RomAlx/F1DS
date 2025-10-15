"""
Интерактивное меню приложения
"""
import os
from datetime import datetime
from config import DEFAULT_TRAIN_YEARS, PREDICT_YEAR, CACHE_DIR
from src.data.loader import F1DataLoader
from src.model.trainer import ModelTrainer
from src.model.predictor import F1Predictor

class MainMenu:
    def __init__(self):
        self.loader = F1DataLoader()
        self.trainer = ModelTrainer()
        self.predictor = F1Predictor()
        self.current_year = datetime.now().year

    @staticmethod
    def clear_screen():
        """Очистка экрана"""
        os.system('clear' if os.name == 'posix' else 'cls')

    @staticmethod
    def print_header():
        """Вывод заголовка"""
        print("=" * 60)
        print("🏎️  F1 PREDICTOR - Система предсказания результатов F1")
        print("=" * 60)
        print()

    @staticmethod
    def print_menu():
        """Вывод главного меню"""
        print("📋 Доступные действия:")
        print()
        print("  1. 💾 Кэширование данных за год (включая спринты)")
        print("  2. 🎓 Обучить модель")
        print("  3. 🧪 Проверить модель на тестовом году")
        print("  4. 🔮 Сделать предсказание на следующий год")
        print("  5. 📊 Посмотреть доступные модели")
        print("  0. 🚪 Выход")
        print()

    def cache_data(self):
        """Кэширование данных за год"""
        print("\n" + "="*60)
        print("💾 КЭШИРОВАНИЕ ДАННЫХ")
        print("="*60)

        default_year = self.current_year - 1
        year_input = input(f"\n📅 Введите год для кэширования (по умолчанию {default_year}): ").strip()

        year = int(year_input) if year_input else default_year

        print(f"\n⏳ Загрузка данных за {year} год...")
        print("   Это может занять несколько минут...")
        print("   📦 Будут закэшированы гонки, квалификации и спринты")

        try:
            self.loader.cache_season(year)
            print(f"\n✅ Данные за {year} год успешно закэшированы!")
            print(f"📁 Расположение: {CACHE_DIR}")
        except Exception as e:
            print(f"\n❌ Ошибка при кэшировании: {e}")

        input("\n⏎ Нажмите Enter для продолжения...")

    def train_model(self):
        """Обучение модели"""
        print("\n" + "="*60)
        print("🎓 ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)

        print(f"\n📌 Года по умолчанию: {', '.join(map(str, DEFAULT_TRAIN_YEARS))}")
        print("\n1. Использовать года из конфига")
        print("2. Ввести года вручную")

        choice = input("\nВаш выбор (1/2): ").strip()

        if choice == "2":
            years_input = input("\n📅 Введите года через запятую (например: 2019,2021,2023): ").strip()
            try:
                years = [int(y.strip()) for y in years_input.split(",")]
            except ValueError:
                print("\n❌ Неверный формат! Используются года из конфига.")
                years = DEFAULT_TRAIN_YEARS
        else:
            years = DEFAULT_TRAIN_YEARS

        print(f"\n⏳ Начинаю обучение на данных: {', '.join(map(str, years))}")
        print("   Это может занять несколько минут...")
        print("   📊 Модель будет учитывать как гонки, так и спринты")

        try:
            metrics = self.trainer.train(years)

            print("\n" + "="*60)
            print("✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
            print("="*60)
            print(f"\n📊 Метрики качества:")
            print(f"   • MAE (средняя абсолютная ошибка): {metrics['mae']:.3f}")
            print(f"   • RMSE (корень из среднеквадратичной ошибки): {metrics['rmse']:.3f}")
            print(f"   • Accuracy (точность топ-10): {metrics['accuracy']:.2%}")
            print(f"\n💾 Модель сохранена: {metrics['model_path']}")

        except Exception as e:
            print(f"\n❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()

        input("\n⏎ Нажмите Enter для продолжения...")

    def test_model(self):
        """Проверка модели на тестовом году"""
        print("\n" + "="*60)
        print("🧪 ПРОВЕРКА МОДЕЛИ")
        print("="*60)

        # Показываем доступные модели
        available_models = self.trainer.list_models()

        if not available_models:
            print("\n❌ Нет обученных моделей! Сначала обучите модель.")
            input("\n⏎ Нажмите Enter для продолжения...")
            return

        print("\n📋 Доступные модели:")
        for idx, model_info in enumerate(available_models, 1):
            print(f"   {idx}. {model_info['name']} - обучена на {len(model_info['years'])} годах")

        model_choice = input(f"\nВыберите модель (1-{len(available_models)}): ").strip()

        try:
            model_idx = int(model_choice) - 1
            model_path = available_models[model_idx]['path']
        except (ValueError, IndexError):
            print("\n❌ Неверный выбор! Используется последняя модель.")
            model_path = available_models[-1]['path']

        test_year = input("\n📅 Введите год для тестирования: ").strip()

        try:
            test_year = int(test_year)
            print(f"\n⏳ Тестирование модели на данных {test_year} года...")
            print("   📊 Учитываются все сессии (гонки и спринты)")

            results = self.predictor.test_model(model_path, test_year)

            print("\n" + "="*60)
            print("✅ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
            print("="*60)
            print(f"\n📊 Метрики:")
            print(f"   • MAE: {results['metrics']['mae']:.3f}")
            print(f"   • RMSE: {results['metrics']['rmse']:.3f}")
            print(f"   • Accuracy (топ-10): {results['metrics']['accuracy']:.2%}")
            print(f"\n📁 Результаты сохранены в: {results['output_dir']}")

        except Exception as e:
            print(f"\n❌ Ошибка при тестировании: {e}")
            import traceback
            traceback.print_exc()

        input("\n⏎ Нажмите Enter для продолжения...")

    def make_prediction(self):
        """Предсказание на следующий год"""
        print("\n" + "="*60)
        print("🔮 ПРЕДСКАЗАНИЕ НА СЛЕДУЮЩИЙ ГОД")
        print("="*60)

        # Показываем доступные модели
        available_models = self.trainer.list_models()

        if not available_models:
            print("\n❌ Нет обученных моделей! Сначала обучите модель.")
            input("\n⏎ Нажмите Enter для продолжения...")
            return

        print("\n📋 Доступные модели:")
        for idx, model_info in enumerate(available_models, 1):
            print(f"   {idx}. {model_info['name']} - обучена на {len(model_info['years'])} годах")

        model_choice = input(f"\nВыберите модель (1-{len(available_models)}): ").strip()

        try:
            model_idx = int(model_choice) - 1
            model_path = available_models[model_idx]['path']
        except (ValueError, IndexError):
            print("\n❌ Неверный выбор! Используется последняя модель.")
            model_path = available_models[-1]['path']

        predict_year = input(f"\n📅 Год для предсказания (по умолчанию {PREDICT_YEAR}): ").strip()
        predict_year = int(predict_year) if predict_year else PREDICT_YEAR

        print(f"\n⏳ Генерация предсказаний на {predict_year} год...")
        print("   🏎️  Симулируются все гонки и спринты")
        print("   ⚡ Используется актуальная система начисления очков F1")

        try:
            results = self.predictor.predict_season(model_path, predict_year)

            print("\n" + "="*60)
            print("✅ ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО")
            print("="*60)

            print("\n🏆 ТОП-3 КОМАНДЫ:")
            for idx, team in enumerate(results['top_teams'][:3], 1):
                points = team.get('total_points', team.get('predicted_points', 0))
                wins = team.get('wins', 0)
                podiums = team.get('podiums', 0)
                print(f"   {idx}. {team['team']} - {points:.1f} очков (🏆{wins} 🏅{podiums})")

            print("\n🏁 ТОП-10 ГОНЩИКОВ:")
            for idx, driver in enumerate(results['top_drivers'][:10], 1):
                points = driver.get('total_points', driver.get('predicted_points', 0))
                wins = driver.get('wins', 0)
                podiums = driver.get('podiums', 0)
                sprint_wins = driver.get('sprint_wins', 0)

                # Формируем строку статистики
                stats = f"🏆{wins}"
                if sprint_wins > 0:
                    stats += f" (S:{sprint_wins})"
                stats += f" 🏅{podiums}"

                print(f"   {idx}. {driver['driver']} ({driver['team']}) - {points:.1f} очков {stats}")

            print(f"\n📁 Все результаты сохранены в: {results['output_dir']}")
            print("   • CSV файлы с предсказаниями")
            print("   • JSON с полными данными (включая спринты)")
            print("   • Графики и визуализации")
            print("   • Детализация гонка-за-гонкой")

        except Exception as e:
            print(f"\n❌ Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()

        input("\n⏎ Нажмите Enter для продолжения...")

    def list_models(self):
        """Просмотр доступных моделей"""
        print("\n" + "="*60)
        print("📊 ДОСТУПНЫЕ МОДЕЛИ")
        print("="*60)

        available_models = self.trainer.list_models()

        if not available_models:
            print("\n❌ Нет обученных моделей!")
        else:
            print()
            for idx, model_info in enumerate(available_models, 1):
                print(f"{idx}. 📦 {model_info['name']}")
                print(f"   📅 Обучена на годах: {', '.join(map(str, model_info['years']))}")
                print(f"   📁 Путь: {model_info['path']}")
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    print(f"   📊 MAE: {metrics.get('mae', 'N/A'):.3f}, Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
                print()

        input("\n⏎ Нажмите Enter для продолжения...")

    def run(self):
        """Главный цикл меню"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_menu()

            choice = input("➤ Выберите действие: ").strip()

            if choice == "1":
                self.cache_data()
            elif choice == "2":
                self.train_model()
            elif choice == "3":
                self.test_model()
            elif choice == "4":
                self.make_prediction()
            elif choice == "5":
                self.list_models()
            elif choice == "0":
                print("\n👋 До встречи!")
                break
            else:
                print("\n❌ Неверный выбор! Попробуйте снова.")
                input("\n⏎ Нажмите Enter для продолжения...")