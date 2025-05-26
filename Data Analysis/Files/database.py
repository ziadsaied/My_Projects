def create_database(**kwargs):
    import psycopg2
    from datetime import datetime

    DB_NAME = "gold_prices_db"
    USER = "airflow"
    PASSWORD = "airflow"
    HOST = "postgres"   
    PORT = "5432"

    print(f"[{datetime.now()}] Starting database setup...")

    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{DB_NAME}' AND pid <> pg_backend_pid();
        """)
        print(f"[{datetime.now()}] Terminated existing connections to {DB_NAME}")

        # Drop and recreate the database
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"[{datetime.now()}] Database {DB_NAME} recreated successfully")

        cursor.close()
        conn.close()

        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS date_dimension (
            date_id SERIAL PRIMARY KEY,
            date DATE UNIQUE NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            is_weekend INTEGER NOT NULL
        )
        """)
        print(f"[{datetime.now()}] Created table: date_dimension")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_sentiment (
            sentiment_id SERIAL PRIMARY KEY,
            sentiment_name VARCHAR(20) UNIQUE NOT NULL
        )
        """)
        print(f"[{datetime.now()}] Created table: market_sentiment")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_prices (
            price_id SERIAL PRIMARY KEY,
            date_id INTEGER REFERENCES date_dimension(date_id),
            sentiment_id INTEGER REFERENCES market_sentiment(sentiment_id),
            price_usd NUMERIC(12, 6) NOT NULL,
            usd_to_eur NUMERIC(12, 10) NOT NULL,
            price_change NUMERIC(12, 6),
            price_change_pct NUMERIC(8, 4)
        )
        """)
        print(f"[{datetime.now()}] Created table: gold_prices")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[{datetime.now()}] Database schema created successfully")

        ti = kwargs['ti']
        ti.xcom_push(key='db_name', value=DB_NAME)

    except Exception as e:
        print(f"[{datetime.now()}] Error during database setup: {e}")
        raise