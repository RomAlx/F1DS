"""
Главный файл запуска проекта F1 Predictor
"""
import sys
from src.core.menu import MainMenu

def main():
    """Точка входа в приложение"""
    try:
        menu = MainMenu()
        menu.run()
    except KeyboardInterrupt:
        print("\n\n👋 Выход из программы. До встречи!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()