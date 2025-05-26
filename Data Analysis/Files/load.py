def load_data_to_database(**kwargs):
    import pandas as pd
    import psycopg2
    from datetime import datetime

    ti = kwargs['ti']

    # Pull file path and db name from XCom
    input_path = ti.xcom_pull(task_ids='clean_data', key='cleaned_file_path')
    db_name = ti.xcom_pull(task_ids='create_database', key='db_name')

    print(f"[{datetime.now()}] Loading data from {input_path} into database {db_name}")

    if not input_path or not db_name:
        raise ValueError("Missing required input for loading data")

    # Load DataFrame
    df = pd.read_csv(input_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user="airflow",
        password="airflow",
        host="postgres",
        port="5432"
    )
    cursor = conn.cursor()

    print(f"[{datetime.now()}] Connected to database {db_name}")

    # Insert sentiment dimension
    sentiments = df['market_sentiment'].unique()
    sentiment_ids = {}

    for sentiment in sentiments:
        cursor.execute(
            "INSERT INTO market_sentiment (sentiment_name) VALUES (%s) RETURNING sentiment_id",
            (sentiment,)
        )
        sentiment_id = cursor.fetchone()[0]
        sentiment_ids[sentiment] = sentiment_id

    print(f"[{datetime.now()}] Loaded unique sentiments")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO date_dimension (date, year, month, day_of_week, is_weekend)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date) DO NOTHING
        """, (
            row['date'].date(),  
            row['year'],
            row['month'],
            row['day_of_week'],
            row['is_weekend']
        ))

    cursor.execute("SELECT date, date_id FROM date_dimension")
    date_ids = {str(row[0]): row[1] for row in cursor.fetchall()}

    print(f"[{datetime.now()}] Loaded date dimension")

    for _, row in df.iterrows():
        date_key = str(row['date'].date())  
        sentiment = row['market_sentiment']
        sentiment_id = sentiment_ids.get(sentiment)

        date_id = date_ids.get(date_key)

        if not date_id or not sentiment_id:
            continue

        cursor.execute("""
            INSERT INTO gold_prices (
                date_id, sentiment_id, price_usd, usd_to_eur, 
                price_change, price_change_pct
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            date_id,
            sentiment_id,
            float(row['price_usd']),
            float(row['usd_to_eur']),
            float(row['price_change']) if pd.notna(row['price_change']) else None,
            float(row['price_change_pct']) if pd.notna(row['price_change_pct']) else None
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[{datetime.now()}] Data loaded successfully")