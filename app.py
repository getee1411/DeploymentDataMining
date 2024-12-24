import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Tuple, List

# Constants
MODEL_FILE = "student_depression_model.pkl"
NUMERIC_COLUMNS = [
    'Age', 'Academic Pressure', 'Work Pressure', 
    'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
    'Work/Study Hours', 'Financial Stress'
]
CATEGORICAL_COLUMNS = [
    'Gender', 'City', 'Profession', 'Sleep Duration', 
    'Dietary Habits', 'Degree', 
    'Have you ever had suicidal thoughts ?', 
    'Family History of Mental Illness'
]

class DataProcessor:
    @staticmethod
    def load_data(file_path: str = 'Student Depression Dataset.csv') -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            st.error(f"Dataset not found at {file_path}")
            return None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None

        df = df.copy()
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        # Remove outliers using IQR method
        for col in ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        return df

class ModelBuilder:
    def __init__(self):
        self.preprocessor = self._create_preprocessor()
        self.model = self._create_pipeline()

    def _create_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_COLUMNS),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_COLUMNS)
            ])

    def _create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

    def save_model(self) -> None:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model() -> Pipeline:
        try:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

    @staticmethod
    def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray) -> None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

    @staticmethod
    def plot_clusters(X_transformed: np.ndarray, clusters: np.ndarray) -> None:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_transformed)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Cluster Visualization (PCA)')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.colorbar(scatter)
        st.pyplot(fig)

def main():
    st.title("Student Depression Analysis")
    
    # Initialize classes
    processor = DataProcessor()
    visualizer = Visualizer()
    
    # Load and preprocess data
    df = processor.load_data()
    if df is None:
        return
    
    df = processor.preprocess_data(df)
    if df is None:
        return

    # Prepare data for modeling
    X = df.drop('Depression', axis=1)
    y = df['Depression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training/loading
    builder = ModelBuilder()
    if not os.path.exists(MODEL_FILE):
        st.info("Training new model...")
        builder.model.fit(X_train, y_train)
        builder.save_model()
        st.success("Model trained successfully!")
    else:
        builder.model = builder.load_model()
        if builder.model is None:
            st.error("Failed to load model")
            return

    # Model evaluation
    y_pred = builder.model.predict(X_test)
    y_prob = builder.model.predict_proba(X_test)[:, 1]

    # Display metrics
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Visualizations
    st.write("Model Performance Visualizations")
    visualizer.plot_confusion_matrix(y_test, y_pred)
    visualizer.plot_roc_curve(y_test, y_prob)

    # Clustering analysis with fixed number of clusters
    st.write("Clustering Analysis")
    X_transformed = builder.preprocessor.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_transformed)
        
    visualizer.plot_clusters(X_transformed, clusters)
        
    # Cluster analysis
    df['Cluster'] = clusters
    st.write("#### Cluster Statistics")
    for cluster in range(3):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster} Size: {len(cluster_data)}")
        st.write("Average values:")
        st.write(cluster_data[NUMERIC_COLUMNS].mean())

if __name__ == "__main__":
    main()