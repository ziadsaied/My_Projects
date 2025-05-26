import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

class MLAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Analyzer")
        self.root.geometry("1920x1080")
        
        # Initialize variables
        self.df = None
        self.feature_cols = []
        self.target_col = tk.StringVar()
        self.task = tk.StringVar(value="classification")
        self.algorithm = tk.StringVar()
        self.label_encoders = {}
        
        # Create notebook for pages
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create pages
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)
        self.page3 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.page1, text="Welcome")
        self.notebook.add(self.page2, text="Dataset & Config")
        self.notebook.add(self.page3, text="Model & Evaluation")
        
        # Setup pages
        self.setup_welcome_page()
        self.setup_dataset_page()
        self.setup_model_page()
    
    def setup_welcome_page(self):
        # Simple welcome page
        label = tk.Label(self.page1, text="ML Analyzer: Predict & Classify Any Dataset", font=("Arial", 16))
        label.pack(pady=20)
        
        start_button = tk.Button(self.page1, text="Start", command=lambda: self.notebook.select(1))
        start_button.pack(pady=10)
    
    def setup_dataset_page(self):
        # File upload
        upload_frame = tk.LabelFrame(self.page2, text="File Upload")
        upload_frame.pack(fill="x", padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        file_path_entry = tk.Entry(upload_frame, textvariable=self.file_path_var, width=50)
        file_path_entry.pack(side="left", padx=5, pady=5)
        
        upload_button = tk.Button(upload_frame, text="Upload CSV", command=self.upload_csv)
        upload_button.pack(side="left", padx=5, pady=5)
        
        # Column display
        columns_frame = tk.LabelFrame(self.page2, text="Columns")
        columns_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.columns_listbox = tk.Listbox(columns_frame)
        self.columns_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(columns_frame, orient="vertical", command=self.columns_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.columns_listbox.config(yscrollcommand=scrollbar.set)
        
        # Target selection
        target_frame = tk.LabelFrame(self.page2, text="Target Column")
        target_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(target_frame, text="Select target:").pack(side="left", padx=5, pady=5)
        self.target_combobox = ttk.Combobox(target_frame, textvariable=self.target_col)
        self.target_combobox.pack(side="left", padx=5, pady=5)
        
        # Task selection
        task_frame = tk.LabelFrame(self.page2, text="Task")
        task_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Radiobutton(task_frame, text="Classification", variable=self.task, value="classification", 
                      command=self.update_algorithm_dropdown).pack(side="left", padx=5, pady=5)
        tk.Radiobutton(task_frame, text="Regression", variable=self.task, value="regression",
                      command=self.update_algorithm_dropdown).pack(side="left", padx=5, pady=5)
        tk.Radiobutton(task_frame, text="Clustering", variable=self.task, value="clustering",
                      command=self.update_algorithm_dropdown).pack(side="left", padx=5, pady=5)
        
        # Algorithm selection
        algo_frame = tk.LabelFrame(self.page2, text="Algorithm")
        algo_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(algo_frame, text="Select algorithm:").pack(side="left", padx=5, pady=5)
        self.algorithm_combobox = ttk.Combobox(algo_frame, textvariable=self.algorithm)
        self.algorithm_combobox.pack(side="left", padx=5, pady=5)
        
        # Data preview
        preview_frame = tk.LabelFrame(self.page2, text="Data Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10)
        self.preview_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Continue button
        continue_button = tk.Button(self.page2, text="Continue", command=self.continue_to_model)
        continue_button.pack(pady=10)
        
        # Initialize algorithm dropdown
        self.update_algorithm_dropdown()
    
    def setup_model_page(self):
        # Left frame for controls
        left_frame = tk.Frame(self.page3)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Right frame for output
        right_frame = tk.Frame(self.page3)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Train model
        train_frame = tk.LabelFrame(left_frame, text="Train Model")
        train_frame.pack(fill="x", padx=5, pady=5)
        
        train_button = tk.Button(train_frame, text="Train", command=self.train_model)
        train_button.pack(padx=5, pady=5)
        
        # Random data generation
        random_frame = tk.LabelFrame(left_frame, text="Random Data")
        random_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.random_data_text = scrolledtext.ScrolledText(random_frame, height=8)
        self.random_data_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        generate_button = tk.Button(random_frame, text="Generate Random Data", command=self.generate_random_data)
        generate_button.pack(padx=5, pady=5)
        
        # Predict
        predict_frame = tk.LabelFrame(left_frame, text="Predict")
        predict_frame.pack(fill="x", padx=5, pady=5)
        
        predict_button = tk.Button(predict_frame, text="Predict", command=self.predict)
        predict_button.pack(padx=5, pady=5)
        
        # Output
        output_frame = tk.LabelFrame(right_frame, text="Evaluation Output")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Plot frame
        self.plot_frame = tk.Frame(right_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_var.set(file_path)
            try:
                self.df = pd.read_csv(file_path)
                self.update_column_display()
                self.update_data_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def update_column_display(self):
        if self.df is not None:
            self.columns_listbox.delete(0, tk.END)
            for col in self.df.columns:
                self.columns_listbox.insert(tk.END, col)
            
            # Update target column dropdown
            self.target_combobox['values'] = list(self.df.columns)
    
    def update_data_preview(self):
        if self.df is not None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, self.df.head().to_string())
    
    def update_algorithm_dropdown(self):
        task = self.task.get()
        if task == "classification":
            self.algorithm_combobox['values'] = ["KNN", "SVM", "Decision Tree"]
            self.algorithm_combobox.current(0)
        elif task == "regression":
            self.algorithm_combobox['values'] = ["Linear Regression"]
            self.algorithm_combobox.current(0)
        else:  # clustering
            self.algorithm_combobox['values'] = ["KMeans"]
            self.algorithm_combobox.current(0)
    
    def continue_to_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload a CSV file first.")
            return
        
        if not self.target_col.get() and self.task.get() != "clustering":
            messagebox.showerror("Error", "Please select a target column.")
            return
        
        # Preprocess data
        self.preprocess_data()
        
        # Update feature columns
        if self.task.get() != "clustering":
            self.feature_cols = [col for col in self.df.columns if col != self.target_col.get()]
        else:
            self.feature_cols = list(self.df.columns)
        
        # Show model page
        self.notebook.select(2)
    
    def preprocess_data(self):
        """
        Preprocess data with label encoding for categorical features
        """
        if self.df is None:
            return
        
        # Reset encoders
        self.label_encoders = {}
        
        # Handle missing values
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Label encode categorical columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
    
    def generate_random_data(self):
        if self.df is None or not self.feature_cols:
            messagebox.showerror("Error", "Please upload CSV and configure first.")
            return
        
        try:
            # Generate random data based on feature columns
            random_data = {}
            random_data_original = {}
            
            for feature in self.feature_cols:
                if feature in self.df.columns:
                    if pd.api.types.is_numeric_dtype(self.df[feature]):
                        min_val = self.df[feature].min()
                        max_val = self.df[feature].max()
                        
                        if pd.api.types.is_integer_dtype(self.df[feature]):
                            value = np.random.randint(int(min_val), int(max_val) + 1)
                        else:
                            value = np.random.uniform(min_val, max_val)
                        
                        random_data[feature] = value
                        random_data_original[feature] = value
            
            # Store random data
            self.random_data = random_data
            
            # Display random data
            self.random_data_text.delete(1.0, tk.END)
            for feature, value in random_data.items():
                self.random_data_text.insert(tk.END, f"{feature}: {value}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating random data: {str(e)}")
    
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload CSV first.")
            return
        
        task = self.task.get()
        algorithm = self.algorithm.get()
        
        try:
            # Clear previous output
            self.output_text.delete(1.0, tk.END)
            
            # Clear previous plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            if task == "classification":
                self.train_classification_model(algorithm)
            elif task == "regression":
                self.train_regression_model()
            else:  # clustering
                self.train_clustering_model()
            
            # Generate random data after training
            self.generate_random_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def train_classification_model(self, algorithm):
        # Get data
        X = self.df[self.feature_cols]
        y = self.df[self.target_col.get()]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model based on selected algorithm
        if algorithm == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=3)
        elif algorithm == "SVM":
            self.model = svm.SVC(kernel='rbf')
        else:  # Decision Tree
            self.model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display metrics
        self.output_text.insert(tk.END, f"Classification Metrics:\n")
        self.output_text.insert(tk.END, f"Accuracy: {accuracy:.3f}\n")
        self.output_text.insert(tk.END, f"Precision: {precision:.3f}\n")
        self.output_text.insert(tk.END, f"Recall: {recall:.3f}\n")
        self.output_text.insert(tk.END, f"F1 Score: {f1:.3f}\n\n")
        self.output_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")
        
        # Plot confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def train_regression_model(self):
        # Get data
        X = self.df[self.feature_cols]
        y = self.df[self.target_col.get()]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        self.output_text.insert(tk.END, f"Regression Metrics:\n")
        self.output_text.insert(tk.END, f"MAE: {mae:.3f}\n")
        self.output_text.insert(tk.END, f"RMSE: {rmse:.3f}\n")
        self.output_text.insert(tk.END, f"R² Score: {r2:.3f}\n\n")
        
        # Create comparison dataframe
        df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        self.output_text.insert(tk.END, f"Actual vs Predicted (first 5 rows):\n{df_compare.head().to_string()}\n")
        
        # Plot scatter plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Scatter Plot")
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def train_clustering_model(self):
        # Get data (use first two numeric columns for visualization)
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns[:2]
        if len(numeric_cols) < 2:
            messagebox.showerror("Error", "Need at least 2 numeric columns for clustering visualization")
            return
        
        X = self.df[numeric_cols]
        
        # Train KMeans model
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Store model
        self.model = kmeans
        
        # Display info
        self.output_text.insert(tk.END, f"KMeans Clustering:\n")
        self.output_text.insert(tk.END, f"Number of clusters: 3\n")
        self.output_text.insert(tk.END, f"Cluster centers:\n{kmeans.cluster_centers_}\n\n")
        
        # Count samples in each cluster
        unique, counts = np.unique(clusters, return_counts=True)
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            self.output_text.insert(tk.END, f"Cluster {cluster}: {count} samples\n")
        
        # Plot clusters
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Plot each cluster
        for i in range(3):
            cluster_points = X[clusters == i]
            ax.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {i}')
        
        # Plot centroids
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                  s=200, marker='*', c='red', label='Centroids')
        
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title("KMeans Clustering")
        ax.legend()
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def predict(self):
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please train a model first.")
            return
        
        if not hasattr(self, 'random_data') or self.random_data is None:
            messagebox.showerror("Error", "Please generate random data first.")
            return
        
        try:
            # Create input dataframe from random data
            input_df = pd.DataFrame([self.random_data])
            
            task = self.task.get()
            
            if task == "classification":
                # Make prediction
                prediction = self.model.predict(input_df)
                
                # Simplified output
                pred_value = prediction[0]
                if pred_value == 1 or pred_value == True:
                    result = "Yes"
                else:
                    result = "No"
                
                self.output_text.insert(tk.END, f"\nPrediction: {result}\n")
                
            elif task == "regression":
                # Make prediction
                prediction = self.model.predict(input_df)
                self.output_text.insert(tk.END, f"\nPredicted value: {prediction[0]:.2f}\n")
                
            else:  # clustering
                # Make prediction
                cluster = self.model.predict(input_df)
                self.output_text.insert(tk.END, f"\nPredicted cluster: {cluster[0]}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLAnalyzerApp(root)
    root.mainloop()
