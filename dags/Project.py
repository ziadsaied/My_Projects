from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from Files.extract import extract_data
from Files.assess_quality import assess_data_quality
from Files.clean import clean_data
from Files.visualize import create_visualizations
from Files.database import create_database
from Files.load import load_data_to_database

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'gold_price_analysis_pipeline',
    default_args=default_args,
    description='A data pipeline for gold price analysis',
    schedule_interval='@daily',
)

extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

assess_quality = PythonOperator(
    task_id='assess_data_quality',
    python_callable=assess_data_quality,
    provide_context=True, 
    dag=dag,
)

clean = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    provide_context=True,
    dag=dag,
)

visualize = PythonOperator(
    task_id='create_visualizations',
    python_callable=create_visualizations,
    provide_context=True,
    dag=dag,
)

create_db = PythonOperator(
    task_id='create_database',
    python_callable=create_database,
    dag=dag,
)

load_data = PythonOperator(
    task_id='load_data_to_database',
    python_callable=load_data_to_database,
    provide_context=True,
    dag=dag,
)

extract >> assess_quality >> clean
clean >> [visualize, create_db]
[visualize, create_db] >> load_data