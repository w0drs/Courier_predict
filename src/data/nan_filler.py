import pandas as pd


class NanFiller:
    """
    Класс для заполнения пропущенных значений в данных о курьерах.

    Особенности:
    - Для новых магазинов (<7 дней) пропуски в числовых колонках заполняются 0
    - Для старых магазинов используются медианы из обучающей выборки
    - Флаги (бинарные признаки) заполняются модой или 0 для новых магазинов
    - Категориальные признаки заполняются константным значением

    Attributes:
        medians (dict): Медианы числовых колонок, вычисленные при fit()
        modes (dict): Моды бинарных колонок (0 или 1)
        cat_fill (dict): Значения для заполнения категориальных колонок
        lifetime_col (str): Название колонки с временем жизни магазина
    """
    def __init__(self, lifetime_col='store_lifetime_in_days'):
        """
        Инициализация NanFiller.

        Args:
            lifetime_col: Колонка для определения новых магазинов (<7 дней)
        """
        self.medians = {}  # Сохранение медиан для числовых колонок
        self.modes = {}  # для флагов
        self.cat_fill = {}  # Для категорий
        self.lifetime_col = lifetime_col

    def fit(self, df, nan_cols=None, flag_cols=None, cat_cols=None):
        """
        Обучает filler на данных: вычисляет медианы, моды и заполнители.

        Args:
            df: DataFrame с обучающими данными
            nan_cols: Список числовых колонок для заполнения медианой
            flag_cols: Список бинарных колонок (0/1) для заполнения модой
            cat_cols: Список категориальных колонок для заполнения 'пусто'

        Returns:
            self: Обученный объект NanFiller
        """

        if nan_cols is None: nan_cols = []
        if flag_cols is None: flag_cols = []
        if cat_cols is None: cat_cols = []

        # Числовые (не флаги)
        for col in nan_cols:
            if col in df.columns:
                self.medians[col] = df[col].median()

        # Флаги (бинарные)
        for col in flag_cols:
            if col in df.columns:
                # Мода для флагов (0 или 1)
                mode_val = df[col].mode()
                self.modes[col] = mode_val[0] if not mode_val.empty else 0

        # Категориальные
        for col in cat_cols:
            if col in df.columns:
                self.cat_fill[col] = 'пусто'

        return self

    def transform(self, df) -> pd.DataFrame:
        """
        Заполняет пропуски в данных согласно обученной логике.

        Args:
            df: DataFrame для заполнения пропусков

        Returns:
            DataFrame с заполненными пропусками

        Принцип:
            - Новые магазины (<7 дней): числовые → 0, флаги → 0
            - Старые магазины: числовые → медиана, флаги → мода
            - Категориальные → 'пусто' для всех
        """
        df_transformed = df.copy()
        is_new = df_transformed[self.lifetime_col].fillna(0) < 7

        # Числовые колонки
        for col, median in self.medians.items():
            if col in df_transformed.columns:
                df_transformed.loc[is_new & df_transformed[col].isna(), col] = 0
                df_transformed.loc[~is_new & df_transformed[col].isna(), col] = median

        # Флаги
        for col, mode_val in self.modes.items():
            if col in df_transformed.columns:
                # Новые магазины: флаг = 0 (не было высокой нагрузки)
                df_transformed.loc[is_new & df_transformed[col].isna(), col] = 0
                # Старые магазины: флаг = мода (чаще всего было 0 или 1)
                df_transformed.loc[~is_new & df_transformed[col].isna(), col] = mode_val

        # Категориальные
        for col, fill_val in self.cat_fill.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(fill_val).astype('category')

        return df_transformed

    def fit_transform(self, df, nan_cols, flag_cols, cat_cols):
        """
        Комбинация fit() и transform() в одном вызове.

        Args:
            df: DataFrame для обучения и трансформации
            nan_cols: Числовые колонки
            flag_cols: Бинарные колонки
            cat_cols: Категориальные колонки

        Returns:
            DataFrame с заполненными пропусками
        """

        self.fit(df, nan_cols, flag_cols, cat_cols)
        return self.transform(df)
