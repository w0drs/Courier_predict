# для теста ранера ci
def test_import():
    """Тест импорта основных библиотек."""
    try:
        import mlflow
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import yaml

        # Если импорт прошел успешно
        assert True

    except ImportError as e:
        print(f"Import error: {e}")
        assert False


if __name__ == "__main__":
    test_import()
    print("Tests passed")
