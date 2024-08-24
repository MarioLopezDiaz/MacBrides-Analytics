"""
Descarga de Archivos desde Google Drive sin Autenticación

Este script permite descargar archivos específicos desde Google Drive directamente a una ubicación local,
utilizando el ID de Google Drive del archivo (el archivo debe ser accesible para cualquier persona que tenga la URL).

El script procesa un archivo de texto ('archivos_raw.txt') que contiene las IDs de Google Drive de los archivos,
los nombres con los que se desean guardar localmente, y las rutas locales de almacenamiento, todo separado por comas.

Formato esperado de 'archivos_raw.txt':
ID_de_Google_Drive,Nombre_Archivo_Local,Ruta_Local

Ejemplo:
1hGvKmNAkK...,mi_archivo.txt,./descargas

Requisitos:
- Módulo 'requests' instalado en el entorno Python donde se ejecute este script.
"""
from pathlib import Path
import requests
import os


def descargar_archivo_directo(id_archivo, directorio_destino, archivo_destino):
    """
    Descarga un archivo directamente desde Google Drive y lo guarda localmente.

    Parámetros:
    - id_archivo (str): ID del archivo en Google Drive.
    - directorio_destino (str): Ruta del directorio local donde se guardará el archivo.
    - archivo_destino (str): Nombre deseado para el archivo en local

    Devuelve:
    - archivo_destino (str): Nombre del archivo guardado.
    - ruta_completa (str): Ruta completa del archivo guardado.
    """
    # Construye la URL de descarga directa utilizando el ID del archivo
    url = f"https://drive.google.com/uc?export=download&id={id_archivo}"

    # Realiza la petición HTTP GET para descargar el archivo
    respuesta = requests.get(url, allow_redirects=True)

    # Comprueba que el directorio destino existe, si no, lo crea
    os.makedirs(directorio_destino, exist_ok=True)

    # Construye la ruta completa donde se guardará el archivo en local
    ruta_completa = os.path.join(directorio_destino, archivo_destino)

    # Guarda el contenido del archivo descargado en local
    with open(ruta_completa, 'wb') as archivo:
        archivo.write(respuesta.content)

    return archivo_destino, ruta_completa


def procesar_archivo_info(ruta_archivo_info):
    """
    Procesa un archivo de texto que contiene información sobre los archivos a descargar.

    Parámetros:
    - ruta_archivo_info (str): Ruta del archivo de texto que contiene los IDs de Google Drive,
                               los nombres de los archivos locales y las rutas locales.

    Devuelve:
    - Una lista de tuplas con el ID de Google Drive, el nombre local del archivo, y la ruta local.
    """
    archivos_info = []
    with open(ruta_archivo_info, 'r') as archivo:
        for linea in archivo:
            id_archivo, nombre_archivo, directorio_destino = linea.strip().split(',')
            archivos_info.append((id_archivo, nombre_archivo, directorio_destino))
    return archivos_info


def main(archivos_info):
    # Ruta al archivo que contiene la información de los archivos a descargar
    dir_act = Path.cwd()
    while not dir_act.name == 'MBA-Finance':
        dir_act = dir_act.parent
    ruta_archivo_info = Path(Path(dir_act, 'archivos'), archivos_info)

    # Obtenemos la lista con la info de los archivos del fichero
    archivos_a_descargar = procesar_archivo_info(ruta_archivo_info)

    # Descargamos cada archivo de la lista
    for id_archivo, nombre_archivo, directorio_destino in archivos_a_descargar:
        dir_act = Path.cwd()
        while not dir_act.name == 'MBA-Finance':
            dir_act = dir_act.parent
        directorio_destino = Path(dir_act, directorio_destino)
        nombre_archivo_descargado, ruta_archivo_guardado = descargar_archivo_directo(id_archivo, directorio_destino,
                                                                                     nombre_archivo)
        print(f"Archivo {nombre_archivo_descargado} guardado en: {ruta_archivo_guardado}")


if __name__ == "__main__":
    main('archivos_final.txt')
