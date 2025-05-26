def create_visualizations(**kwargs):
    import pandas as pd
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    ti = kwargs['ti']
    input_path = ti.xcom_pull(task_ids='clean_data', key='cleaned_file_path')

    df = pd.read_csv(input_path)
    print(f"Loaded data for visualization. Shape: {df.shape}")

    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
    df = df.dropna(subset=['price_usd'])

    df['year'] = df['date'].dt.year

    price_by_year = df.groupby('year')['price_usd'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.bar(price_by_year['year'], price_by_year['price_usd'], color='purple')
    plt.title("Average Gold Price (USD) per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Price (USD)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'BarChart.png'))
    plt.close()

    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'HeatMap.png'))
    plt.close()

    ti.xcom_push(key='visualization_dir', value=viz_dir)