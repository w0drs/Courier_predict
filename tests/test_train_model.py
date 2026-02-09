import pandas as pd
import tempfile
import os
import yaml
import joblib
from click.testing import CliRunner
from src import train_lgbm_model


class TestTrainModel:
    """Тесты для обучения модели."""

    def setup_method(self):
        self.runner = CliRunner()

        # Тестовые данные
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'cat_feature': ['A', 'B', 'A', 'B', 'A'],
            'target': [10, 20, 30, 40, 50]
        })

        # Конфигурация
        self.config = {
            'model_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5
            },
            'target_col': 'target',
            'categorical_features': ['cat_feature']
        }

        # Временные файлы
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, 'data.csv')
        self.model_path = os.path.join(self.temp_dir, 'model.joblib')
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')

        # Сохраняем
        self.df.to_csv(self.data_path, index=False)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_train_model_cli(self):
        """Тест CLI команды обучения."""
        result = self.runner.invoke(
            train_lgbm_model,
            [self.data_path, self.model_path, self.config_path]
        )

        assert result.exit_code == 0
        assert os.path.exists(self.model_path)

        # Проверяем что модель загружается
        model = joblib.load(self.model_path)
        assert hasattr(model, 'predict')

        # Проверяем info файл
        info_path = self.model_path.replace('.joblib', '.info.yaml')
        assert os.path.exists(info_path)