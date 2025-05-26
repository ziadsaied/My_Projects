def clean_data(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    input_path = ti.xcom_pull(task_ids='assess_data_quality', key='quality_checked_file_path')

    df = pd.read_csv(input_path)
    print(f"Loaded data for cleaning. Shape: {df.shape}")

    # Clean data
    df = df[df["price_usd"] != "ERROR"].copy()
    df["price_usd"] = df["price_usd"].astype(float)
    df.dropna(inplace=True)
    df = df[df['price_usd'] > 0]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['price_change'] = df['price_usd'].diff()
    df['price_change_pct'] = df['price_usd'].pct_change() * 100
    df['price_7d_avg'] = df['price_usd'].rolling(window=7).mean()
    df['price_30d_avg'] = df['price_usd'].rolling(window=30).mean()

    output_path = "/opt/airflow/dags/Files/cleaned_gold.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

    ti.xcom_push(key='cleaned_file_path', value=output_path)