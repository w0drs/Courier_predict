import pandas as pd
import tempfile
import os
import yaml
from click.testing import CliRunner
import shutil
from src import drop_features


class TestDropFeatures:
    """Тесты для функции drop_features."""

    def setup_method(self):
        """Настройка тестовых данных."""
        self.runner = CliRunner()

        # Создаем тестовые данные
        self.train_data = pd.DataFrame({
            'store_id': [1, 2, 3],
            'col_to_keep': [10, 20, 30],
            'col_to_drop': ['a', 'b', 'c'],
            'another_to_drop': [100, 200, 300]
        })

        self.test_data = pd.DataFrame({
            'store_id': [4, 5],
            'col_to_keep': [40, 50],
            'col_to_drop': ['d', 'e'],
            'another_to_drop': [400, 500]
        })

        # Создаем временные файлы
        self.temp_dir = tempfile.mkdtemp()
        self.train_path = os.path.join(self.temp_dir, 'train.csv')
        self.test_path = os.path.join(self.temp_dir, 'test.csv')
        self.train_output = os.path.join(self.temp_dir, 'train_output.csv')
        self.test_output = os.path.join(self.temp_dir, 'test_output.csv')
        self.params_path = os.path.join(self.temp_dir, 'params.yaml')

        # Сохраняем данные
        self.train_data.to_csv(self.train_path, index=False)
        self.test_data.to_csv(self.test_path, index=False)

        # Создаем params.yaml
        params = {
            'drop_features': {
                'cols': ['col_to_drop', 'another_to_drop']
            }
        }
        with open(self.params_path, 'w', encoding='utf-8') as f:
            yaml.dump(params, f)

    def teardown_method(self):
        """Очистка временных файлов."""
        shutil.rmtree(self.temp_dir)

    def test_drop_features_cli(self):
        """Тест CLI команды."""
        result = self.runner.invoke(
            drop_features,
            [
                self.train_path,
                self.test_path,
                self.train_output,
                self.test_output,
                self.params_path
            ]
        )

        assert result.exit_code == 0

        # Проверяем выходные файлы
        train_result = pd.read_csv(self.train_output)
        test_result = pd.read_csv(self.test_output)

        # Колонки должны быть удалены
        assert 'col_to_drop' not in train_result.columns
        assert 'another_to_drop' not in train_result.columns
        assert 'col_to_keep' in train_result.columns

        # Данные должны сохраниться
        assert len(train_result) == 3
        assert len(test_result) == 2

    def test_only_train_data(self):
        """Тест когда передается только train data."""
        result = self.runner.invoke(
            drop_features,
            [
                self.train_path,
                "",  # Пустой test path
                self.train_output,
                "",  # Пустой test output
                self.params_path
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(self.train_output)

    def test_missing_params_file(self):
        """Тест когда params файл не существует."""
        result = self.runner.invoke(
            drop_features,
            [
                self.train_path,
                self.test_path,
                self.train_output,
                self.test_output,
                'non_existent.yaml'
            ]
        )

        assert result.exit_code != 0  # Должен упасть