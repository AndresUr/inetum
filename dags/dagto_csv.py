# Importar las librerías necesarias para el DAG
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from clients.spaceflightnews_client import ExtractAPI

import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import StringIO
import os
import logging
# Configurar los logs de Airflow
logging.basicConfig(
                    level=logging.INFO,  # Definicion del nivel de logs
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato de logs
                    handlers=[
                        logging.FileHandler("extraction_log.log"),  # Guardar los logs
                        logging.StreamHandler()  #mostrar los logs en consola
                            ]
                    )

# Definir las variables de conexión a Azure Blob Storage
connection_string = "connection_string"
container_name = "articles" 
blob_name = "archivo.json"  

# Definir la ruta del archivo temporal
TEMP_FILE = "/tmp/df.json"

# Definir los argumentos por defecto del DAG
default_args = {
    "owner":"Andres Urrea",
    "depends_on_past" : False,
    "email_on_failture" :False,
    "email_on_retry" :False,
    "retries" : 1
    
}

#La presente funcion extrae los datos de la API de Spaceflightnews y los guarda en un archivo temporal
def extract_data(ti):
    try:        
        client_spaceflight = ExtractAPI()
        articles = client_spaceflight.extract_all_articles()    
        ti.xcom_push(key="articles", value=articles)
        logging.info(f"Proceso de extraccion de data realizado con exito")
    except Exception as e:
        logging.error(f"Error en la extraccion de datos: {e}")
        raise e
#La presente funcion realiza la limpieza de los datos extraidos de la API de Spaceflightnews
def preprocess_data(ti):
    try:
        
        articles = ti.xcom_pull(task_ids='extract_data', key='articles')    
        if not articles:
            logging.error("No se encontraron datos para procesar")
            return
    
        df_articles  = pd.DataFrame(articles["results"])
        df_articles['type'] = 'article'
    
        df_articles = df_articles.drop_duplicates(subset=['id'], keep='first')
    
        df_articles.to_json(TEMP_FILE, orient="records", lines=True)
        logging.info(f"Proceso de preprocess_data realizado con exito, se extrajeron {len(df_articles)} registros unicos")
        logging.info(f"El archivo temporal se guardo en la ruta {TEMP_FILE}")
    except Exception as e:
        logging.error(f"Error en la funcion preprocess_data: {e}")
        raise e
    
    
#La presente funcion guarda el archivo temporal en Azure Blob Storage
def save_to_df():
   try:
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            json_data = f.read()
        # Conectarse a Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Subir el archivo Json al Blob Storage
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(json_data, overwrite=True)

        logging.info(f"Proceso de almacenamiento en la nube se realizado con exito, en la ruta {blob_client.url}")
   except Exception as e:
        logging.error(f"Error en la funcion save_to_df: {e}")
        raise e

#La presente funcion elimina el archivo temporal creado en el proceso de extraccion y transformacion
def clear_temp():    
    
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        print(f"Archivo {TEMP_FILE} eliminado correctamente.")
        logging.info(f"El archivo temporal {TEMP_FILE} eliminado correctamente.")
    else:
        print(f"El archivo {TEMP_FILE} no fue encontrado, probablemente ya fue eliminado.")

# Definir el DAG principal con el nombre "csv_dag"
with DAG(
    "INETUM_ETL",
    default_args=default_args,
    description = "Creacion Dag ETL para INETUM",
    tags=["ETL","Inetum", "Databricks", "ETL"]
) as dag:
    
    # Definir los operadores del DAG usando los operadores DummyOperator, PythonOperator y DatabricksRunNowOperator
    start = DummyOperator(task_id="start")
    
    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data
    )
    
    preprocess_data_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )
    
    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=save_to_df
    )
    
    clear_data_task = PythonOperator(
        task_id="clear_data_task",
        python_callable=clear_temp
    )
    
    run_job_spark = DatabricksRunNowOperator(
        task_id = 'run_now_job',
        databricks_conn_id = 'Databricks-connection-airflow',
        job_id = 882479244738536
    )
    
    end = DummyOperator(task_id="end")
    #definir
    # Definir el orden de ejecución de los operadores
    start >> extract_data >> preprocess_data_task >> load_data_task >> clear_data_task >> run_job_spark  >> end