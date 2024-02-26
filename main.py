import streamlit as st
from streamlit_echarts import st_echarts
import plotly.express as px
import pandas as pd
import datetime as dt
import numpy as np
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
DATA_PATH = 'export/'

# Загрузка данных из CSV-файла
data = pd.read_csv('export/streamlit/00_sums_data.csv', index_col='name')
st.set_page_config(layout="wide")
# Заголовок
st.subheader('ПРОГНОЗИРОВАНИЕ СТОИМОСТИ')
st.sidebar.image(r'data\1.png', use_column_width=True)
st.sidebar.image(r'data\11.png', use_column_width=True)


def data_loader(TICKER):
    data = pd.read_csv(f'data\{TICKER.lower()}_h_2015-2024.csv', 
                        index_col='Unnamed: 0', 
                        parse_dates=['begin', 'end'])
    forecast = pd.read_csv(f'export\streamlit\{TICKER}_forecast.csv', 
                        index_col='Unnamed: 0', 
                        parse_dates=['ds'])
    income = pd.read_csv(f'export\streamlit\{TICKER.lower()}_income.csv', 
                        index_col='Unnamed: 0', 
                        parse_dates=['ds'])
    forcast_income = pd.read_csv(f'export\streamlit\{TICKER.lower()}_forcast_income.csv', 
                        index_col='Unnamed: 0', 
                        parse_dates=['ds'])
    dividends = pd.read_csv(f'export\streamlit\{TICKER.lower()}_dividend_table.csv', 
                        index_col='Unnamed: 0')
    correlation_table = pd.read_csv(f'export\streamlit\{TICKER.lower()}_correlation_table.csv', 
                        index_col='Unnamed: 0')
    return data, forecast, income, forcast_income, dividends, correlation_table

def plot_loader(TICKER, data):
    left, right = st.columns([3, 2])
    with right:
        data_ = pd.read_csv('export/streamlit/00_sums_data.csv')
        data_.name = data_.name.apply(str.upper)
        data_.set_index('name', inplace=True)
        
    # Заменяем значение 43 на data.loc[TICKER]['%_change']
        st.markdown(f"<div style='font-size: 20px; padding-left: 420px; text-align: left;'><br><br><br>прогнозная доходность <span style='font-size: 100px;'>{int(data_.loc[TICKER]['%_change'].round()):+d}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 20px; padding-left: 420px; text-align: left;'>при оценке границы риска <span style='font-size: 100px;padding-left: 20px;'>{int(data_.loc[TICKER]['potential_risk'].round()):+d}%</span></div>", unsafe_allow_html=True)

    with left:
        forecast_filtered = forecast[(forecast['ds'] >= '2022-02-20') & (forecast['ds'] <= forecast['ds'].max())]
        fig3 = go.Figure(data=[
                # Предсказанные значения
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat'], 
                            mode='lines',
                            name='Прогноз',  
                            line=dict(color='orange', width=4.5)),
                # График цен акций
                go.Scatter(x=data['begin'], 
                            y=data['open'], 
                            mode='lines', 
                            name='Факт. знач.',
                            line=dict(color='rgba(82, 126, 233, 0.8)', width=1)),
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat_upper'], 
                            mode='lines',
                            showlegend=False,
                            fill='tonexty',
                            fillcolor='rgba(186, 191, 204, 0.05)', 
                            line=dict(color='rgba(186, 191, 204, 0.1)')),
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat_lower'],
                            name='Гран.прогн.', 
                            mode='lines', 
                            fill='tonexty',
                            fillcolor='rgba(186, 191, 204, 0.05)',  
                            line=dict(color='rgba(186, 191, 204, 0.1)'))
        ])
        fig3.update_layout(title=dict(
                            text=TICKER,
                            font=dict(size=60)),
                            xaxis_title='Дата',
                            yaxis_title='Цена',
                            height=600,
                            width=1150,
                            xaxis_rangeslider_visible=True
                            #legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="right", x=1)
                            ) 
        fig3.update_xaxes(rangeslider=dict(visible=True), range=['2021-05-20', forecast['ds'].max()])
        fig3.update_yaxes(range=[forecast['yhat'].min()-50, forecast['yhat'].max()+50])
        fig3.update_xaxes(range=['2021-05-01', '2025-05-01'])
        last_predicted_price = forecast_filtered['yhat'].iloc[-1]
        last_predicted_date = forecast_filtered['ds'].max()

        # Добавление горизонтальной линии к последней предсказанной цене акции
        fig3.add_shape(type="line",
                            x0=data['begin'].min(), y0=last_predicted_price,
                            x1=forecast_filtered['ds'].max(), y1=last_predicted_price,
                            opacity=0.5,  # Прозрачность
                            line=dict(color="grey", dash="dashdot"),
                            name='Последняя предсказанная цена')

        fig3.add_shape(type="line",
                            x0=last_predicted_date, y0=0,
                            x1=last_predicted_date, y1=last_predicted_price,
                            line=dict(color="grey", dash="dashdot"),
                            opacity=0.5,  # Прозрачность
                            name='Последняя дата предсказания')
        
        fig3.add_annotation(
                            x=last_predicted_date,
                            y=last_predicted_price,
                            text=f'({last_predicted_date.strftime("%b %d, %Y")}, {int(last_predicted_price)}) <span style="#98331c">Прогноз</span>',
                            showarrow=True,
                            arrowhead=1,
                            ax=0,  # Нет смещения по оси x
                            ay=-40,  # Смещение вниз от точки
                            arrowcolor='orange',  # Оранжевый цвет стрелки
                            font=dict(color="black", size=12),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1.4,
                            borderpad=10,
                            align="right"
                            )
        st.plotly_chart(fig3)

sums_data = pd.read_csv('export/streamlit/00_sums_data.csv')
stock_list = sums_data.name.apply(lambda x: str(x).upper()).to_list()

TICKER = st.sidebar.selectbox("Выберите строку для отображения графика:", stock_list)
data, forecast, income, forcast_income, dividends, correlation_table = data_loader(TICKER)
correlation_table.index = correlation_table.index.str.rstrip(', млрд. руб.')
correlation_table = correlation_table.iloc[:,:2].rename(columns={'normalized_rmse':'RMSE_norm', 'Корреляция':'Чистая прибыль'}).rename({'Операционный денежный поток':'Опер.ден.поток'})

n_years = st.sidebar.slider('Горизонт планирования', 1, 365)

# Боковой фейдер для управления количеством отображаемых строк и колонок
num_displayed = 5

# Отображение графика на главной части страницы
plot_loader(TICKER, data)

# Загрузка данных из CSV-файла
data = pd.read_csv('export/streamlit/00_sums_data.csv')
data.name = data.name.apply(str.upper)
data.set_index('name', inplace=True)

# Отображение DataFrame
st.subheader('ТОП НАИБОЛЕЕ ВЛИЯЮЩИХ НА СТОИМОСТЬ КОМПАНИИ ПАРАМЕТРОВ')
st.write('ПАРАМЕТРЫ РАСПОЛОЖЕНЫ ПО УБЫВАНИЮ ВАЖНОСТИ С ПОПРАВКОЙ НА ПРОГНОЗИРУЕМОСЬ')
left, right = st.columns([1, 1])

# with right:
#         # Создаем подграфик
#     fig1 = go.Figure()

#     # Добавляем первый и второй столбчатые графики
#     fig1.add_trace(
#         go.Bar(
#             y=data.reset_index().sort_values('%_change', ascending=True)['potential_yield_return'].iloc[:10],
#             x=data.reset_index().sort_values('%_change', ascending=True)['name'].iloc[:10],
#             orientation='v',
#             name='Прогнозный процент роста',
#             marker=dict(color='orange'),
#         )
#     )

#     fig1.add_trace(
#         go.Bar(
#             y=data.reset_index().sort_values('%_change', ascending=True)['potential_risk'].iloc[:10],
#             x=data.reset_index().sort_values('%_change', ascending=True)['name'].iloc[:10],
#             orientation='v',
#             name='Потенциальный риск',
#             marker=dict(color='rgba(82, 126, 233, 0.8)')
#         )
#     )

#     # Обновляем макет графика
#     fig1.update_layout(
#             title='Прогнозный процент роста и Потенциальный риск',
#             template='plotly_white',
#             height=300,
#             width=600,
#             barmode='group'
#     )
#     fig1.update_xaxes(side='top')
#     st.plotly_chart(fig1)

with right:
    st.write("")
    st.dataframe(data.drop(['date', 'predicted_date', 'lowest', 'highest', 'potential_yield_return'], axis=1).rename(columns={'predicted_date':'ПРОГНОЗНАЯ ЦЕНА'}).sort_values('%_change', ascending=False) \
            .applymap(lambda x: f'{x:.2f}')  \
            .round(2) \
            .style.background_gradient(cmap='RdYlGn', low=0, high=2, axis=0, subset=['%_change']),
            width=700, height=None)
    
with left:
    fig2, ax = plt.subplots(figsize=(7, 4), facecolor='none')
    #correlation_table.index = correlation_table.index.str.rstrip(', млрд руб')
    # Рисуем тепловую карту
    heatmap = sns.heatmap(correlation_table.iloc[:6], ax=ax, annot=True, cmap='RdYlGn', fmt=".3f", alpha=0.7, annot_kws={"fontsize": 6},
                        vmin=0, vmax=2)  # Устанавливаем минимальное и максимальное значение для цветовой карты

    #plt.title("Тепловая карта корреляции лучших параметров с 'Капитализация, млрд руб'", color='white')
    plt.grid(False)  # Отключаем отображение сетки, если необходимо

    # Устанавливаем цвет и размер шрифта подписей на осях x и y
    ax.set_xticklabels(ax.get_xticklabels(), color='white', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), color='white', fontsize=7)

    # Получаем объект цветовой шкалы и изменяем его цвет и размер шрифта
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors='white', labelcolor='white', labelsize=4)

    # Отображаем фигуру в Streamlit
    
    st.pyplot(fig2)
    

num_displayed = st.slider("Select number of items to display", 1, 10)
# Вывод DataFrame с выравниванием по центру для колонки "Дивиденд"
st.sidebar.header('ИСТОРИЯ ДИВИДЕНДОВ')
dividends.iloc[:,:2] = dividends.iloc[:,:2].applymap(lambda x: pd.to_datetime(x).date())
st.sidebar.dataframe(dividends.set_index('Объяв. див.') \
                .head(10).style.format({'Дивиденд': '{:^10.2f}'}),
                width=280, height=None)


import plotly.graph_objects as go

def income_loader(TICKER, forecast, income):
    left, right = st.columns([3, 2])
    with right:
                st.markdown(f"<div style='font-size: 20px; padding-left: 300px; text-align: left;'><br><br><br>P/E при будущих доходах и текущей цене P/E<span style='font-size: 100px;padding-left: 30px;'>{np.round(income['Капитализация, млрд руб'].iloc[-1] / (forcast_income.iloc[-4:].yhat.sum()),2):.2f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 20px; padding-left: 300px; text-align: left;'>P/E при оценке текущей цене акции и доходе P/E<span style='font-size: 100px;padding-left: 20px;'>{np.round(income['Капитализация, млрд руб'].iloc[-1]/income.y.iloc[-4:].sum()):.2f}</span></div>", unsafe_allow_html=True)

    with left:    
        fig4 = go.Figure(data=[
            # Предсказанные значения
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'],
                mode='lines',  
                name='Прогноз',
                marker_color='orange'
            ),
            # График цен акций
            go.Bar(
                x=income['ds'],  # Временные метки
                y=income['y'],   # Фактические значения
                name='Факт. знач.',
                marker=dict(color='rgba(82, 126, 233, 0.8)'),
            ),
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_upper'], 
                mode='lines',
                showlegend=False,
                fill='tonexty',
                fillcolor='rgba(186, 191, 204, 0.05)', 
                line=dict(color='rgba(186, 191, 204, 0.1)')
            ),
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_lower'],
                name='Гран.прогн.', 
                mode='lines', 
                fill='tonexty',
                fillcolor='rgba(186, 191, 204, 0.05)',  
                line=dict(color='rgba(186, 191, 204, 0.1)')
            )
        ])
        
        fig4.update_layout(
                title=dict(text='Прогноз дохода комании', font=dict(size=30)),
                xaxis_title='Дата',
                yaxis_title='Цена',
                height=600,
                width=1150,
                xaxis_rangeslider_visible=True
        ) 

        fig4.update_yaxes(
                range=[forecast['yhat'].min()-50, 
                        max(max(income['y']), 
                        forecast['yhat'].max()+50)]
        )
        
        last_predicted_price = forecast['yhat'].iloc[-1]
        last_predicted_date = forecast['ds'].max()

        # Добавление горизонтальной линии к последней предсказанной цене акции
        fig4.add_shape(type="line",
                x0=income['ds'].min(), 
                y0=last_predicted_price,
                x1=forecast['ds'].max(), 
                y1=last_predicted_price,
                opacity=0.5,  # Прозрачность
                line=dict(color="grey", dash="dashdot"),
                name='Последняя предсказанная цена'
                        )

        fig4.add_shape(
                type="line",
                x0=last_predicted_date, 
                y0=0,
                x1=last_predicted_date, 
                y1=last_predicted_price,
                line=dict(color="grey", dash="dashdot"),
                opacity=0.5,  # Прозрачность
                name='Последняя дата предсказания'
        )
        
        fig4.add_annotation(
                x=last_predicted_date,
                y=last_predicted_price,
                text=f'({last_predicted_date.strftime("%b %d, %Y")}, {int(last_predicted_price)} Прогноз {int(((last_predicted_price/ income.iloc[-2:].y.mean() - 1)*100).round()):+d}%)',
                showarrow=True,
                arrowhead=1,
                ax=0,  # Нет смещения по оси x
                ay=-40,  # Смещение вниз от точки
                arrowcolor='orange',  # Оранжевый цвет стрелки
                font=dict(color="black", size=18),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1.4,
                borderpad=10,
                align="right"
        )
        
        return st.plotly_chart(fig4)

# Вызов функции с вашими данными
income_loader(TICKER, forcast_income, income)


# Создаем фигуру и оси


fig2, ax = plt.subplots(figsize=(2, 6), facecolor='none')
st.sidebar.subheader('ОПРЕДЕЛЯЮЩИЕ ПАРАМЕТРЫ С КОРРЕЛЯЦИЕЙ КАПИТАЛИЗАЦИЯ')
# Рисуем тепловую карту
correlation_table.index = correlation_table.index.str.rstrip(', млрд руб')
heatmap = sns.heatmap(correlation_table.iloc[:,:1], ax=ax, annot=True,  fmt=".3f", alpha=0.7, annot_kws={"fontsize": 10},
                    vmin=1, vmax=2)  # Устанавливаем минимальное и максимальное значение для цветовой карты

#plt.title("Тепловая карта корреляции лучших параметров с 'Капитализация, млрд руб'", color='white')
plt.grid(False)  # Отключаем отображение сетки, если необходимо

# Устанавливаем цвет и размер шрифта подписей на осях x и y
ax.set_xticklabels(ax.get_xticklabels(), color='white', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), color='white', fontsize=10)

# Получаем объект цветовой шкалы и изменяем его цвет и размер шрифта
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(colors='white', labelcolor='white', labelsize=1)

# Отображаем фигуру в Streamlit

st.sidebar.pyplot(fig2)
