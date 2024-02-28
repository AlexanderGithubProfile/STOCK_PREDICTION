import pandas as pd

def data_loader(TICKER):
    """
    Функция для загрузки данных из различных источников.

    Parameters:
    TICKER (str): Код тикера компании.

    Returns:
    tuple: Кортеж с загруженными данными:
        - historical_candles: данные о ценах акций компании;
        - forecast: данные о прогнозе цен акций;
        - income: данные о доходах компании;
        - forcast_income: данные о прогнозе доходов компании;
        - dividends: данные о дивидендах компании;
        - correlation_table: таблица корреляции.
    """
    # Загрузка данных о ценах акций
    historical_candles = pd.read_csv(f'data\{TICKER.lower()}_h_2015-2024.csv', 
                        index_col='Unnamed: 0', 
                        parse_dates=['begin', 'end'])
    
    # Загрузка прогноза цен акций
    forecast = pd.read_csv(f'export\streamlit\{TICKER}_forecast.csv', 
                            index_col='Unnamed: 0', 
                            parse_dates=['ds'])
    
    # Загрузка данных о доходах компании
    income = pd.read_csv(f'export\streamlit\{TICKER.lower()}_income.csv', 
                            index_col='Unnamed: 0', 
                            parse_dates=['ds'])
    
    # Загрузка прогноза доходов компании
    forcast_income = pd.read_csv(f'export\streamlit\{TICKER.lower()}_forcast_income.csv', 
                            index_col='Unnamed: 0', 
                            parse_dates=['ds'])
    
    # Загрузка данных о дивидендах компании
    dividends = pd.read_csv(f'export\streamlit\{TICKER.lower()}_dividend_table.csv', 
                            index_col='Unnamed: 0')
    
    # Загрузка таблицы корреляции
    correlation_table = pd.read_csv(f'export\streamlit\{TICKER.lower()}_correlation_table.csv', 
                            index_col='Unnamed: 0')
    correlation_table.index = correlation_table.index.str.rstrip(', млрд. руб.')
    correlation_table = correlation_table.iloc[:,:2] \
                                    .rename(columns={'normalized_rmse':'RMSE_norm', 
                                                     'Корреляция':'Чистая прибыль'}) \
                                    .rename({'Операционный денежный поток':'Опер.ден.поток'})
    
    return historical_candles, forecast, income, forcast_income, dividends, correlation_table

def get_stock_list():
    """
    Функция для получения списка доступных тикеров.

    Returns:
    list: Список доступных тикеров.
    """
    sums_data = pd.read_csv('export/streamlit/00_sums_data.csv')
    stock_list = sums_data.name.apply(lambda x: str(x).upper()).to_list()
    return stock_list, sums_data
