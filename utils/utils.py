import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def create_summary_table_and_correlation_heatmap(data, correlation_table):
    """
    Функция для создания таблицы с суммарной информацией о прогнозах и тепловой карты с корреляцией.

    Args:
        data (pd.DataFrame): DataFrame с данными о прогнозах.
        correlation_table (pd.DataFrame): DataFrame с корреляцией метрик и капитализации.
        st (streamlit): Объект streamlit.

    Returns:
        None
    """
    # Разделение страницы на два столбца для таблицы и тепловой карты
    left, right = st.columns([1, 1])

    # Создание таблицы с суммарной информацией о прогнозах
    with right:
        st.write("")
        st.dataframe(data.drop(['date', 'predicted_date', 'lowest', 'highest', 'potential_yield_return'], axis=1) \
                        .rename(columns={'name':'ТИКЕР',
                                        'price':'ТЕКУЩАЯ ЦЕНА',
                                        'predicted_price':'ПРОГНОЗНАЯ ЦЕНА', 
                                        'potential_risk':'РИСК'}) \
                        .sort_values('%_change', ascending=False) \
                        .applymap(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)  \
                        .round(2) \
                        .style.background_gradient(cmap='RdYlGn', 
                                                low=0, 
                                                high=2, 
                                                axis=0, 
                                                subset=['%_change']),
                        width=700, height=None)

    # Создание тепловой карты с корреляцией метрик и капитализации
    with left:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='none')
        heatmap = sns.heatmap(correlation_table.iloc[:6], 
                                ax=ax, 
                                annot=True, 
                                cmap='RdYlGn', 
                                fmt=".3f", 
                                alpha=0.7, 
                                annot_kws={"fontsize": 6},
                                vmin=0, vmax=2) 
        plt.grid(False) 

        # Цвет и размер шрифта, осей x, y и графика
        ax.set_xticklabels(ax.get_xticklabels(), color='white', fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), color='white', fontsize=7)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors='white', labelcolor='white', labelsize=4)
        
        st.pyplot(fig) # Отображаем фигуру в Streamlit
