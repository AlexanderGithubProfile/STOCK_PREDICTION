import streamlit as st
import pandas as pd
from utils.data_loader import data_loader, get_stock_list
from utils.plot_loader import plot_loader
from utils.income_loader import income_loader
from utils.utils import create_summary_table_and_correlation_heatmap

#Установка конфигурации страницы
st.set_page_config(layout="wide")

def main():

    # Изображение на боковой панели
    st.sidebar.image(r'img\1.png', use_column_width=True)
    st.sidebar.image(r'img\11.png', use_column_width=True)

    # Получение списка доступных тикеров
    stock_list, sums_table = get_stock_list()

    # Выбор тикера из списка
    TICKER = st.sidebar.selectbox("Выберите строку для отображения графика:", stock_list)

    # Загрузка данных
    historical_candles, forecast, income, forcast_income, dividends, correlation_table = data_loader(TICKER)

    # Отображение основного графика
    st.subheader('ПРОГНОЗИРОВАНИЕ СТОИМОСТИ') # Заголовок
    plot_loader(TICKER, historical_candles, forecast, income)

    # # Выбор горизонта планирования
    # n_years = st.sidebar.slider('Горизонт планирования', 1, 365)

    # Отображение таблицы с фундаментальными метриками
    st.subheader('НАИБОЛЕЕ ЗНАЧИМЫЕ ФУНДАМЕНТАЛЬНЫЕ МЕТРИКИ')
    st.write('ПАРАМЕТРЫ РАСПОЛОЖЕНЫ ПО УБЫВАНИЮ ИХ КОРРЕЛЯЦИЯ С КАПИТАЛИЗАЦИЕЙ КОМПАНИИ И ИХ ПРОГНОЗИРУЕМОСТЬЮ')
    st.markdown('<hr>', unsafe_allow_html=True)
    create_summary_table_and_correlation_heatmap(sums_table, correlation_table)

    # Вывод истории дивидендов на боковой панели
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.header('ИСТОРИЯ ДИВИДЕНДОВ')
    dividends.iloc[:,:2] = dividends.iloc[:,:2].applymap(lambda x: pd.to_datetime(x).date())
    st.sidebar.dataframe(dividends.set_index('Объяв. див.') \
                    .head(10).style.format({'Дивиденд': '{:^10.2f}'}),
                    width=280, height=None)
    
    # Отображение дополнительных метрик P/E рядом с графиком прогноза дохода
    income_loader(forecast, income, forcast_income)
    
if __name__ == "__main__":
    main()
