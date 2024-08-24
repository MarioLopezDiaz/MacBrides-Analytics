import pandas as pd
import yfinance as yf
from alpaca_trade_api import REST
import os
import warnings
warnings.filterwarnings('ignore')


def get_news(df, symbol, api):
    """
        Obtiene noticias relacionadas con un símbolo de acciones específico a partir de la API de alpaca.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.
        - symbol: Símbolo de las acciones del cual se desea obtener las noticias.
        - api: Objeto de la API utilizada para obtener las noticias.

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df
          añadiendo una columna llamada "News" que contiene las noticias obtenidas.
    """
    news = []
    for d in df.loc[df.index >= pd.Timestamp(2015, 1, 1)].iterrows():
        en = d[0].date()
        st = en - pd.Timedelta(days=1)
        news.append(pd.Series({d[0]: api.get_news(symbol, st, en)}))

    index = [day.index[0] for day in news]
    values = [[n.__dict__['_raw'] for n in day[0]] for day in news]
    df['News'] = pd.Series(values, index=index)


def get_data_symbol(symbol, start, api):
    """
        Descarga los datos históricos de precios de acciones y noticias relacionadas con un símbolo específico.

        Parámetros:
        - symbol: Símbolo de las acciones del cual se desea obtener los datos.
        - start: Fecha de inicio para obtener los datos (en formato 'YYYY-MM-DD').
        - api: Objeto de la API utilizada para obtener las noticias.

        Devuelve:
        - DataFrame: DataFrame que contiene los datos de las acciones y las noticias.
    """
    df = yf.download(symbol, start)
    get_news(df, symbol, api)
    return df


def get_data(symbols, start, api, name_file):
    """
        Descarga los datos históricos de precios de acciones y noticias relacionadas con varios símbolos.

        Parámetros:
        - symbols: Lista de símbolos de acciones de los cuales se desea obtener los datos.
        - start: Fecha de inicio para obtener los datos (en formato 'YYYY-MM-DD').
        - api: Objeto de la API utilizada para obtener las noticias.

        Devuelve:
        - No devuelve ningún valor explícito, pero guarda los datos en archivos .parquet dentro
          de la carpeta 'data'.
    """
    # Verificar si la carpeta 'data' existe. Si no existe, la crea
    if not os.path.exists('data'):
        os.mkdir('data')

    for s in symbols:
        print('Loading stocks from ', s, '...', sep='')
        aux = get_data_symbol(s, start, api)
        aux.reset_index(inplace=True)
        aux.to_parquet(f'./data/{name_file}_{s}.parquet')


if __name__ == "__main__":
    SYMBOLS = ['AAPL', 'F', 'GOOG', 'META', 'MSFT', 'TSLA']
    START = '2010-01-01'
    API_KEY = 'PKQYAQ73E58SXZXCV500'
    API_SECRET = '90vWP5CMCWHww9FeeIGGjMVZfP5T52WD7Xq7B3Il'
    API = REST(key_id=API_KEY, secret_key=API_SECRET)
    get_data(SYMBOLS, START, API, 'raw')
