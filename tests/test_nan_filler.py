import pandas as pd
import numpy as np
from src import NanFiller


class TestNanFiller:
    """Тесты для класса NanFiller."""

    def setup_method(self):
        """Настройка тестовых данных."""
        self.df = pd.DataFrame({
            'store_lifetime_in_days': [3, 10, 15, 2, 30],  # Новые: 3, 2 дня
            'numeric_col': [1.0, 2.0, np.nan, np.nan, 5.0],
            'flag_col': [1, 0, np.nan, np.nan, 1],
            'cat_col': ['A', 'B', np.nan, np.nan, 'C']
        })

        self.nan_cols = ['numeric_col']
        self.flag_cols = ['flag_col']
        self.cat_cols = ['cat_col']

        self.filler = NanFiller()

    def test_fit(self):
        """Тест обучения filler."""
        self.filler.fit(
            self.df,
            nan_cols=self.nan_cols,
            flag_cols=self.flag_cols,
            cat_cols=self.cat_cols
        )

        # Проверяем что медианы вычислены
        assert 'numeric_col' in self.filler.medians
        assert self.filler.medians['numeric_col'] == 2.0  # Медиана из [1.0, 2.0, 5.0]

        # Проверяем моды
        assert 'flag_col' in self.filler.modes
        assert self.filler.modes['flag_col'] == 1  # Мода [1, 0, 1] = 1

        # Проверяем категории
        assert 'cat_col' in self.filler.cat_fill
        assert self.filler.cat_fill['cat_col'] == 'пусто'

    def test_transform_new_stores(self):
        """Тест заполнения для новых магазинов (<7 дней)."""
        self.filler.fit(
            self.df,
            nan_cols=self.nan_cols,
            flag_cols=self.flag_cols,
            cat_cols=self.cat_cols
        )

        result = self.filler.transform(self.df)

        # Проверяем новые магазины (дни 3 и 2)
        new_mask = result['store_lifetime_in_days'] < 7

        print("\nНовые магазины:")
        print(result.loc[new_mask, ['store_lifetime_in_days', 'numeric_col', 'flag_col', 'cat_col']])

        # Проверяем что NaN заполнились 0
        # Строка с индексом 3 (новый магазин с NaN)
        assert result.loc[3, 'numeric_col'] == 0
        assert result.loc[3, 'flag_col'] == 0
        assert result.loc[3, 'cat_col'] == 'пусто'

        # Строка с индексом 0 (новый магазин с НЕ-NaN значениями)
        assert result.loc[0, 'numeric_col'] == 1.0
        assert result.loc[0, 'flag_col'] == 1
        assert result.loc[0, 'cat_col'] == 'A'

    def test_transform_old_stores(self):
        """Тест заполнения для старых магазинов (>=7 дней)."""
        self.filler.fit(
            self.df,
            nan_cols=self.nan_cols,
            flag_cols=self.flag_cols,
            cat_cols=self.cat_cols
        )

        result = self.filler.transform(self.df)

        # Старые магазины
        old_mask = result['store_lifetime_in_days'] >= 7

        # Числовые колонки должны быть медианой
        nan_idx = 2  # индекс строки с nan
        assert result.loc[nan_idx, 'numeric_col'] == 2.0

        # Флаги должны быть модой
        assert result.loc[nan_idx, 'flag_col'] == 1

    def test_fit_transform(self):
        """Тест fit_transform."""
        result = self.filler.fit_transform(
            self.df,
            nan_cols=self.nan_cols,
            flag_cols=self.flag_cols,
            cat_cols=self.cat_cols
        )

        # Проверяем что нет NaN
        assert not result['numeric_col'].isna().any()
        assert not result['flag_col'].isna().any()
        assert not result['cat_col'].isna().any()

    def test_with_empty_dataframe(self):
        """Тест с пустым DataFrame."""
        empty_df = pd.DataFrame(columns=['store_lifetime_in_days', 'numeric_col'])
        filler = NanFiller()

        result = filler.fit_transform(empty_df, ['numeric_col'], [], [])
        assert result.empty

    def test_column_not_in_dataframe(self):
        """Тест когда указаны колонки которых нет в данных."""
        filler = NanFiller()
        result = filler.fit_transform(
            self.df,
            nan_cols=['non_existent_col'],
            flag_cols=[],
            cat_cols=[]
        )

        # Не должно упасть, просто игнорирует
        assert 'non_existent_col' not in filler.medians