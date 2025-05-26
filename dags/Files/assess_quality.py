def assess_data_quality(**kwargs):
    import pandas as pd

    ti = kwargs['ti']

    input_path = ti.xcom_pull(task_ids='extract_data', key='processed_file_path')

    df = pd.read_csv(input_path)
    print(f"Loaded data for quality check. Shape: {df.shape}")

    print(f"Dataset shape: {df.shape}")

    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    print("\nChecking for ERROR values:")
    for column in ['price_usd', 'usd_to_eur']:
        error_count = df[df[column] == 'ERROR'].shape[0]
        print(f"{column}: {error_count} errors")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"\nDate Range: {df['date'].min()} to {df['date'].max()}")

    sentiment_counts = df['market_sentiment'].value_counts()
    print("\nMarket Sentiment Distribution:")
    print(sentiment_counts)

    output_path = "/opt/airflow/dags/Files/quality_checked_gold.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved quality-checked data to {output_path}")

    ti.xcom_push(key='quality_checked_file_path', value=output_path)