import pandas as pd
import tempfile
import os
from click.testing import CliRunner
from src import merge_data


class TestMergeData:
    """Тесты для функции merge_data."""

    def setup_method(self):
        """Настройка тестовых данных."""
        self.runner = CliRunner()

        # Создаем тестовые данные
        self.facts_data = pd.DataFrame({
            'store_id': [1, 2, 1, 2],
            'calendar_dt': [
                '2025-11-17', '2025-11-17',  # Предыдущая неделя для train
                '2025-11-24', '2025-11-24'  # Для test
            ],
            'facts_value': [100, 200, 300, 400]
        })

        self.shifts_data = pd.DataFrame({
            'store_id': [1, 2, 1, 2],
            'calendar_dt': [
                '2025-11-24', '2025-11-24',  # Текущая неделя для train
                '2025-12-01', '2025-12-01'  # Для test (если нужно)
            ],
            'shifts_value': [50, 60, 70, 80]
        })

        self.train_target = pd.DataFrame({
            'store_id': [1, 2],
            'calendar_dt': ['2025-11-24', '2025-11-24'],
            'target': [10, 20]
        })

        self.test_target = pd.DataFrame({
            'store_id': [1, 2],
            'store_type': ['A', 'B']  # Тестовые данные без даты
        })

        # Временные файлы
        self.temp_dir = tempfile.mkdtemp()
        self.facts_path = os.path.join(self.temp_dir, 'facts.csv')
        self.shifts_path = os.path.join(self.temp_dir, 'shifts.csv')
        self.train_path = os.path.join(self.temp_dir, 'train.csv')
        self.test_path = os.path.join(self.temp_dir, 'test.csv')
        self.train_output = os.path.join(self.temp_dir, 'train_merged.csv')
        self.test_output = os.path.join(self.temp_dir, 'test_merged.csv')

        # Сохраняем
        self.facts_data.to_csv(self.facts_path, index=False)
        self.shifts_data.to_csv(self.shifts_path, index=False)
        self.train_target.to_csv(self.train_path, index=False)
        self.test_target.to_csv(self.test_path, index=False)

    def teardown_method(self):
        """Очистка."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_merge_data_cli(self):
        """Тест CLI команды."""
        result = self.runner.invoke(
            merge_data,
            [
                self.facts_path,
                self.shifts_path,
                self.train_path,
                self.test_path,
                self.train_output,
                self.test_output
            ]
        )

        assert result.exit_code == 0

        # Проверяем train результат
        train_result = pd.read_csv(self.train_output)
        assert 'facts_value' in train_result.columns
        assert 'shifts_value' in train_result.columns
        assert 'target' in train_result.columns

        # Проверяем test результат
        test_result = pd.read_csv(self.test_output)
        assert 'facts_value' in test_result.columns
        assert 'shifts_value' in test_result.columns
        assert 'calendar_dt' in test_result.columns
        # Дата должна быть 2025-11-24
        assert (test_result['calendar_dt'] == '2025-11-24').all()