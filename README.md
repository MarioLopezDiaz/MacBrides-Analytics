# MACBRIDES ANALYTICS

### DESCRIPCIÓN DEL PROYECTO

Este proyecto se centra en el desarrollo e implementación de un sistema de aprendizaje automático dirigido a MacBrides Analytics (MBA), una entidad ficticia dedicada al análisis y predicción del comportamiento de acciones en el mercado financiero, con especial énfasis en las empresas del S&P 500.

### PROPÓSITO DEL PROYECTO

El objetivo principal de este proyecto es mejorar la estrategia de inversión de MBA mediante el uso de herramientas avanzadas de análisis de datos y de inteligencia artificial (IA). Además, se busca reducir el número de analistas en plantilla para optimizar los recursos financieros de la empresa.

### ESTRUCTURA DEL REPOSITORIO

 • **`archivos/`**: Contiene conjuntos de claves para acceder a los archivos .parquet que están subidos a google drive.
   - **`archivos_final.txt`**: Contiene las claves de los datos de train finales.
   - **`archivos_raw.txt`**: Contiene las claves de los datos de train en crudo.
   - **`archivos_final_test.txt`**: Contiene las claves de los datos de test finales.
   - **`archivos_raw_test.txt`**: Contiene las claves de los datos de test en crudo.

 • **`modelizacion/`**: Contiene los distintos archivos relacionados con la modelización del proyecto.
   - **`dbs/`**: Contiene las bases de datos de cada clasificador.
      - **`LR1.db`** : Base de datos de la regresión logística.
      - **`MLP1.db`** : Base de datos de la red neuronal.
      - **`RF1.db`** : Base de datos del random forest.
   - **`Analisis resultados.ipynb`**: Contiene el notebook del análisis de resultados que tiene el formato .ipynb para representar mejor las gráficas.
   - **`evaluar.py`**: Contiene el código fuente modularizado para la evaluación de modelos.
   - **`feature_selection.py`**: Contiene el código fuente a cerca de la selección de las mejores k para los modelos.
   - **`main_LR.py`**: Contiene los experimentos con cada modelo y sus hiperparámetros con el clasificador de regresión logística.
   - **`main_MLP.py`**: Contiene los experimentos con cada modelo y sus hiperparámetros con el clasificador de red neuronal.
   - **`main_RF.py`**: Contiene los experimentos con cada modelo y sus hiperparámetros con el clasificador de random forest.
   - **`preparacion.py`**: Contiene el código fuente para la preparación de los datos.
   - **`train.py`**: Contiene el código fuente modularizado para el entrenamiento de modelos.
 
 • **`gitignore/`**: Se utiliza para especificar archivos y directorios que Git debe ignorar al realizar seguimiento de cambios en un repositorio.

 • **`Exploracion.ipynb`**: Contiene el notebook de exploracion que tiene el formato .ipynb para representar mejor las gráficas.

 • **`README.md`**: Este archivo que proporciona una visión general del proyecto, su propósito y la estructura del repositorio.
 
 • **`Requirements.txt`**: Contiene las dependencias del proyecto, es decir, las bibliotecas de Python necesarias para ejecutar el código correctamente. Estas dependencias pueden instalarse fácilmente utilizando un administrador de paquetes como pip.
 
 • **`adquisición.py`**: Contiene el código fuente del sistema de aprendizaje automático, en este caso, los scripts de adquisición.

 • **`descarga_drive.py`**: Contiene el código modularizado para descargar los archivos del drive.

 • **`despliegue.ipynb`**: Contiene un notebook con el despliegue de los datos.

 • **`limpieza.py`**: Contiene el código fuente del sistema de aprendizaje automático, en este caso, los scripts de limpieza.


 
### INTEGRANTES DEL PROYECTO

- [Mario López Díaz](https://github.com/MarioLopezDiaz)
    
