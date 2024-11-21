import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de MLflow
mlflow.set_experiment("/wine-classification-experiment")

# Función para cargar y preprocesar datos
def load_and_preprocess_data():
    """Load and preprocess dataset from the specified URL."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    names_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names'
    
    # Cargar datos
    column_names = [
        'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 
        'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
        'proanthocyanins', 'color_intensity', 'hue', 
        'od280/od315_of_diluted_wines', 'proline'
    ]
    data = pd.read_csv(url, header=None, names=column_names)
    
    # Separar características y target
    X = data.drop('class', axis=1)
    y = data['class'] - 1  # Ajustar etiquetas a 0, 1, 2
    
    # Verificar balanceo de clases
    print("Distribución de clases:")
    print(y.value_counts())
    
    # Identificar y manejar valores atípicos con IQR
    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_clean
    
    X_clean = remove_outliers(X)
    y_clean = y[X_clean.index]
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    return train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)

# Función para entrenar y evaluar modelos
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Registrar métricas en MLflow
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Registrar modelo
            mlflow.sklearn.log_model(model, name)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    
    return results

# Función para graficar resultados
def plot_model_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Modelos')
    plt.ylabel('Puntuación')
    plt.title('Comparación de Modelos')
    plt.xticks(x + width*1.5, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Función principal
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Entrenar y evaluar modelos
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Graficar resultados
    plot_model_comparison(results)
    
    # Seleccionar mejor modelo
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    # Predecir nuevas muestras
    new_samples = np.array([
        [13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285],
        [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]
    ])
    
    # Escalar nuevas muestras
    scaler = StandardScaler()
    scaler.fit(X_train)
    new_samples_scaled = scaler.transform(new_samples)
    
    # Predecir
    predictions = best_model.predict(new_samples_scaled)
    print("\nPredicciones para nuevas muestras:")
    for i, pred in enumerate(predictions, 1):
        print(f"Muestra {i}: Variedad {pred}")

if __name__ == "__main__":
    main()