def extract_data(**kwargs):
    import pandas as pd

    input_path = "/opt/airflow/dags/Files/Gold.csv"
    output_path = "/opt/airflow/dags/Files/processed_gold.csv"

    df = pd.read_csv(input_path)
    print(f"Extracted {len(df)} records from Gold.csv")

    df.to_csv(output_path, index=False)

    ti = kwargs['ti']
    ti.xcom_push(key='processed_file_path', value=output_path)