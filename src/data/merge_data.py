import click
import pandas as pd

@click.command()
@click.argument('facts_filepath', type=click.Path(exists=True))
@click.argument('shifts_filepath', type=click.Path(exists=True))
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path())
@click.argument('train_output', type=click.Path())
@click.argument('test_output', type=click.Path())
def merge_data(facts_filepath,
               shifts_filepath,
               train_filepath,
               test_filepath=None,
               train_output=None,
               test_output=None) -> None:
    """
    Merge datasets for courier deficit prediction.
    Args:
        facts_filepath: real data for the PREVIOUS week
        shifts_filepath: forecast for the CURRENT week
        train_filepath: target - shortage of couriers for the CURRENT week
        test_filepath: We need to forecast for a week 2025-11-24
        train_output: path where we will save the processed file with train data
        test_output: way to save test processed data
    The final train and test data will be saved in csv files.
    """
    train_df = pd.read_csv(train_filepath)
    facts_df = pd.read_csv(facts_filepath)
    shifts_df = pd.read_csv(shifts_filepath)

    for df in [train_df, facts_df, shifts_df]:
        df['calendar_dt'] = pd.to_datetime(df['calendar_dt'])

    # Создаем колонку с датой неделю назад
    train_data = train_df.copy()
    train_data['prev_week'] = train_data['calendar_dt'] - pd.Timedelta(days=7)

    # Соединение с facts по предыдущей неделе
    train_merged = train_data.merge(
        facts_df,
        left_on=['store_id', 'prev_week'],
        right_on=['store_id', 'calendar_dt'],
        how='left',
        suffixes=('', '_facts')
    )

    # Соединение с shifts по текущей неделе
    train_merged = train_merged.merge(
        shifts_df,
        on=['store_id', 'calendar_dt'],
        how='left',
        suffixes=('', '_shifts')
    )

    train_merged = train_merged.drop(columns=['prev_week'])

    if train_output:
        train_merged.to_csv(train_output, index=False)

    # Делаем merge для тестовых данных
    if test_filepath:
        test_df = pd.read_csv(test_filepath)

        test_date = pd.to_datetime('2025-11-24')
        prev_week = test_date - pd.Timedelta(days=7)

        test_data = test_df.copy()
        test_data['calendar_dt'] = test_date
        test_data['prev_week'] = prev_week

        test_merged = test_data.merge(
            facts_df[facts_df['calendar_dt'] == prev_week],
            left_on=['store_id', 'prev_week'],
            right_on=['store_id', 'calendar_dt'],
            how='left',
            suffixes=('', '_facts')
        )

        test_merged = test_merged.merge(
            shifts_df[shifts_df['calendar_dt'] == test_date],
            left_on=['store_id', 'calendar_dt'],
            right_on=['store_id', 'calendar_dt'],
            how='left',
            suffixes=('', '_shifts')
        )

        test_merged = test_merged.drop(columns=['prev_week'])
        if test_output:
            test_merged.to_csv(test_output, index=False)

if __name__ == "__main__":
    merge_data()




