import pandas as pd
import click


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path())
@click.argument('train_output_path', type=click.Path())
@click.argument('test_output_path', type=click.Path())
def build_features(train_filepath,
                   test_filepath,
                   train_output_path,
                   test_output_path) -> None:
    """
    Feature generation for predicting courier shortages.

    Creates new features:
    1. staff_prediction_gap - difference between a forecast and a personnel fact
    2. order_prediction_gap - difference between forecast and fact of orders
    3. past_productivity - last week's performance (orders/courier)
    4. predicted_productivity - forecast performance for current week

    Args:
        train_filepath: path to the training data
        test_filepath: Path to the test data (optional)
        train_output_path: path to save the enriched training data
        test_output_path: path to save the enriched test data
    """
    train_df = pd.read_csv(train_filepath)
    # Разница между прогнозом и реальностью прошлой недели
    train_df['staff_prediction_gap'] = train_df['predicted_staff_value'] - train_df['fact_staff_value_lag_1']
    train_df['orders_prediction_gap'] = train_df['predicted_num_orders'] - train_df['fact_num_orders_lag_1']

    # Сколько заказов на одного курьера в прошлой неделе
    train_df['past_productivity'] = (train_df['fact_num_orders_lag_1'] /
                                     train_df['fact_staff_value_lag_1'].replace(0, 1))
    # Прогнозная производительность на эту неделю
    train_df['predicted_productivity'] = (train_df['predicted_num_orders'] /
                                          train_df['predicted_staff_value'].replace(0, 1))
    if train_output_path:
        train_df.to_csv(train_output_path, index=False)

    if test_filepath:
        test_df = pd.read_csv(test_filepath)
        test_df['staff_prediction_gap'] = test_df['predicted_staff_value'] - test_df['fact_staff_value_lag_1']
        test_df['orders_prediction_gap'] = test_df['predicted_num_orders'] - test_df['fact_num_orders_lag_1']

        test_df['past_productivity'] = (test_df['fact_num_orders_lag_1'] /
                                         test_df['fact_staff_value_lag_1'].replace(0, 1))
        test_df['predicted_productivity'] = (test_df['predicted_num_orders'] /
                                              test_df['predicted_staff_value'].replace(0, 1))

        if test_output_path:
            test_df.to_csv(test_output_path, index=False)


if __name__ == "__main__":
    build_features()
