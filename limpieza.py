import ta
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import descarga_drive
import warnings
warnings.filterwarnings('ignore')


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)
LABELS = ["positive", "negative", "neutral"]


def add_past_returns(df, lags=3):
    """
        Agrega columnas al DataFrame con los retornos pasados de las acciones.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.
        - lags: Número de períodos pasados para los cuales se calcularán los retornos (por defecto 3).

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df añadiendo
          columnas con los retornos pasados (Ret_1, Ret_2, Ret_3, etc).
    """
    for i in range(1, lags + 1):
        df['Ret_' + str(i)] = df['Rets'].shift(i)


def add_vma(df, max_window=60):
    """
        Agrega columnas al DataFrame con el promedio móvil de los retornos de las acciones.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.
        - max_window: Tamaño máximo de la ventana para calcular el promedio móvil (por defecto 60).

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df añadiendo
          columnas con el promedio móvil de los retornos.
    """
    for i in range(10, max_window + 1, 10):
        df['VMA' + str(i)] = df['Rets'].rolling(i).mean()


def add_vma_week(df, num_weeks=5):
    """
        Agrega columnas al DataFrame con el promedio móvil semanal de los retornos de las acciones.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.
        - num_weeks: Número de semanas para las cuales se calculará el promedio móvil (por defecto 5).

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df añadiendo
          columnas con el promedio móvil semanal de los retornos.
    """
    for i in range(num_weeks):
        df['VMA_' + str(i + 1) + 'ªsem'] = df['Rets'].shift(7 * i).rolling(7).mean()


def add_features(df):
    """
        Agrega diversas características de la biblioteca ta y otras medias moviles al DataFrame de datos de acciones.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df
          agregando múltiples columnas con diferentes características.
    """
    add_past_returns(df)
    add_vma(df)
    add_vma_week(df)
    ta.add_all_ta_features(df=df, open='Open', high='High', low='Low', close='Close', volume='Volume')


def estimate_sentiment(news):
    """
        Estima el sentimiento de un conjunto de noticias utilizando un modelo de IA de clasificación.

        Parámetros:
        - news: Cadena de texto que representa las noticias o listas de cadenas de texto.

        Devuelve:
        - probability: Probabilidad de que el sentimiento sea positivo, negativo o neutro.
        - sentiment: Etiqueta del sentimiento (positivo, negativo o neutro).
    """
    if news:
        tokens = TOKENIZER(news, return_tensors="pt", padding=True).to(DEVICE)
        result = MODEL(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = LABELS[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, LABELS[-1]


def process_row(row):
    """
        Procesa una fila de datos para estimar el sentimiento de las noticias y devuelve la etiqueta del sentimiento.

        Parámetros:
        - row: Fila de datos que contiene las noticias.

        Devuelve:
        - sentiment: Etiqueta del sentimiento (positivo, negativo o neutro).
    """
    return estimate_sentiment(row['News_headlines'])[1]


def process_news_sentiment(df):
    """
        Procesa las noticias en el DataFrame para estimar el sentimiento y agrega una columna 'Sent' con las etiquetas del sentimiento.

        Parámetros:
        - df: DataFrame que contiene las noticias.

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df
          agregando una columna 'Sent' con las etiquetas del sentimiento.
    """
    rows = []
    for i, row in df.iterrows():
        rows.append(process_row(row))
    df['Sent'] = rows


def transform_df(df):
    """
        Transforma el DataFrame de datos de acciones agregando nuevas columnas y preprocesando los datos.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - df_transform: DataFrame transformado con nuevas columnas y datos preprocesados.
    """
    df_transform = df.copy()
    df_transform['Rets'] = (df_transform['Close'] / df_transform['Open']) - 1
    add_features(df_transform)
    df_transform['Tmrw_rets'] = df_transform['Rets'].shift(-1)
    df_transform['Invest'] = np.where(df_transform['Tmrw_rets'] > 0, 1, 0)
    df_transform['News'] = [[] if day is None else day for day in df_transform['News']]
    df_transform['News_headlines'] = [[new['headline'] for new in day] for day in df_transform['News']]
    return df_transform


def gestion_trend_psar(df):
    """
        Calcula la dirección de la tendencia basada en los valores de PSAR y agrega una columna 'trend_direction'.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df
          agregando una columna 'trend_direction' con la dirección de la tendencia.
    """
    df['trend_dir_val'] = (np.where(df['trend_psar_down'].isna(),
                                    0,
                                    (df['trend_psar_down']) * -df['trend_psar_down'].notna().astype(int)) +
                           (np.where(df['trend_psar_up'].isna(),
                                     0,
                                     df['trend_psar_up']) * (df['trend_psar_up'].notna().astype(int))))
    df['trend_direction'] = np.where(df['trend_dir_val'] > 0, 1, 0)
    df.drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)


def gestion_primeros_dias(df):
    """
        Gestiona los primeros días del DataFrame eliminando filas nulas y columnas no necesarias.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - No devuelve ningún valor explícito, pero modifica el DataFrame df
          eliminando filas nulas y columnas no necesarias.
    """
    df.dropna(inplace=True)
    df.drop(['News', 'News_headlines', 'Tmrw_rets'], axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)


def drop_na(df):
    """
        Elimina filas nulas y gestiona los primeros días del DataFrame.

        Parámetros:
        - df: DataFrame que contiene los datos de las acciones.

        Devuelve:
        - df: DataFrame modificado sin filas nulas y con los primeros días gestionados.
    """
    gestion_trend_psar(df)
    gestion_primeros_dias(df)
    return df


if __name__ == "__main__":
    descarga_drive.main('archivos_raw.txt')
    symbols = ['AAPL', 'F', 'GOOG', 'META', 'MSFT', 'TSLA']
    for s in symbols:
        df_s = pd.read_parquet(Path(Path.cwd(), f'data/raw_{s}.parquet'))
        df_trans = transform_df(df_s)
        process_news_sentiment(df_trans)
        df_sinnan = drop_na(df_trans)
        df_sinnan.to_parquet(f'./data/final_{s}.parquet')
