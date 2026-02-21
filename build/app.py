import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, average_precision_score, 
    PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - Prediccion de Desercion Estudiantil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">Dashboard - Prediccion de Desercion Estudiantil</div>', unsafe_allow_html=True)
st.markdown("### Comparacion de Modelos de Machine Learning")
st.markdown("---")

# Cache para funciones pesadas
@st.cache_data
def load_data(file_path, max_samples=10000):
    """Carga y prepara el dataset"""
    df = pd.read_csv(file_path)
    
    # Muestreo estratificado si hay muchos datos
    if len(df) > max_samples:
        df = df.groupby("Deserto", group_keys=False).apply(
            lambda x: x.sample(
                n=int(max_samples * len(x) / len(df)),
                random_state=42
            )
        )
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Limpieza
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates()
    
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    
    df.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "None": np.nan}, inplace=True)
    
    return df

@st.cache_data
def identify_features(df, target_col="Deserto"):
    """Identifica variables categóricas y numéricas"""
    nunique = df.nunique()
    low_cardinality_cols = nunique[nunique <= 30].index.tolist()
    
    if target_col in low_cardinality_cols:
        low_cardinality_cols.remove(target_col)
    
    categorical_cols = [c for c in low_cardinality_cols if df[c].dtype == 'object' or df[c].dtype == 'int64']
    
    forced_categorical = ["InstitucionBach", "Carrera", "Plan", "IdCampus", "Sexo"]
    forced_numeric = ["MateriasInscritas_C1", "MateriasAprobadas_C1", "MateriasReprobadas_C1", 
                     "MateriasInscritas_C2", "MateriasAprobadas_C2", "MateriasReprobadas_C2", 
                     "TotalMateriasInscritas_Anio1", "TotalMateriasAprobadas_Anio1", "TotalMateriasReprobadas_Anio1"]
    
    for col in forced_categorical:
        if col in df.columns and col not in categorical_cols:
            categorical_cols.append(col)
    
    for col in forced_numeric:
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]
    
    for col in forced_numeric:
        if col in df.columns and col not in numeric_cols:
            numeric_cols.append(col)
    
    return numeric_cols, categorical_cols

def create_preprocessor(numeric_cols, categorical_cols):
    """Crea el preprocesador de datos"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def train_models(X_train, y_train, preprocessor, progress_bar=None):
    """Entrena todos los modelos"""
    ratio_desbalance = (y_train == 0).sum() / (y_train == 1).sum()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_config = [
        {
            "name": "Regresion Logistica",
            "pipeline": Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
            ]),
            "params": {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__penalty': ['l2']
            }
        },
        {
            "name": "K-Nearest Neighbors",
            "pipeline": Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier())
            ]),
            "params": {
                'classifier__n_neighbors': [3, 5, 7, 11],
                'classifier__weights': ['uniform', 'distance']
            }
        },
        {
            "name": "XGBoost",
            "pipeline": Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight=ratio_desbalance,
                    verbosity=0
                ))
            ]),
            "params": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0]
            }
        }
    ]
    
    trained_models = {}
    results = []
    
    for i, config in enumerate(models_config):
        if progress_bar:
            progress_bar.progress((i + 1) / len(models_config), text=f"Entrenando {config['name']}...")
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            config['pipeline'], 
            config['params'], 
            cv=skf, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        trained_models[config['name']] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'training_time': end_time - start_time
        }
        
        results.append({
            'Modelo': config['name'],
            'Mejor F1 (CV)': round(grid_search.best_score_, 4),
            'Tiempo (s)': round(end_time - start_time, 2)
        })
    
    return trained_models, pd.DataFrame(results)

def evaluate_model(model, X_test, y_test):
    """Evalúa un modelo en el conjunto de test"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'F1-Score': f1_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
        metrics['PR-AUC'] = average_precision_score(y_test, y_proba)
    
    return metrics, y_pred, y_proba

# Sidebar - Configuración
st.sidebar.title("Configuracion")

# Selector de archivo
data_file_options = {
    "Dataset Principal (2015-2019)": "Tbl_DesercionEstudiantil_PrimerAnio_2015_2019.csv",
    "Dataset Completo": "Tbl_DesercionEstudiantil_PrimerAnio_.csv"
}

selected_dataset = st.sidebar.selectbox(
    "Seleccionar Dataset",
    options=list(data_file_options.keys())
)

data_file = data_file_options[selected_dataset]

max_samples = st.sidebar.slider(
    "Maximo de muestras",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
    help="Numero maximo de registros a procesar"
)

test_size = st.sidebar.slider(
    "Tamano del conjunto de prueba (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
    help="Porcentaje de datos para el conjunto de test"
)

# Botón para entrenar modelos
train_button = st.sidebar.button("Entrenar Modelos", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre este Dashboard")
st.sidebar.info(
    "Este dashboard compara tres modelos de clasificacion para predecir "
    "la desercion estudiantil en el primer año acade")
train_button = st.sidebar.button("🚀 Entrenar Modelos", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre este Dashboard")
st.sidebar.info(
    "Este dashboard compara tres modelos de clasificación para predecir "
    "la deserción estudiantil en el primer año académico."
)

# Main content
if train_button:
    try:
        # Cargar datos
        with st.spinner("Cargando datos..."):
            df = load_data(data_file, max_samples)
        
        st.success(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
        
        # Mostrar información del dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        
        with col2:
            desertion_rate = df['Deserto'].mean() * 100
            st.metric("Tasa de Deserción", f"{desertion_rate:.1f}%")
        
        with col3:
            st.metric("Total de Features", df.shape[1] - 1)
        
        with col4:
            missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("% Valores Faltantes", f"{missing_pct:.1f}%")
        
        st.markdown("---")
        
        # Distribución de clases
        st.markdown('<div class="sub-header">Distribucion de la Variable Objetivo</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            class_counts = df['Deserto'].value_counts()
            st.dataframe(
                pd.DataFrame({
                    'Clase': ['No Desertó (0)', 'Desertó (1)'],
                    'Cantidad': class_counts.values,
                    'Porcentaje': [f"{v/class_counts.sum()*100:.1f}%" for v in class_counts.values]
                }),
                hide_index=True,
                width='stretch'
            )
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(['No Desertó (0)', 'Desertó (1)'], class_counts.values, 
                         color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.7)
            ax.set_ylabel("Cantidad de estudiantes", fontsize=10)
            ax.set_title("Distribucion de Desercion", fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, class_counts.values)):
                height = bar.get_height()
                pct = val / class_counts.sum() * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Preparar datos
        with st.spinner("Preparando features..."):
            numeric_cols, categorical_cols = identify_features(df)
            X = df.drop(columns=['Deserto'])
            y = df['Deserto']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size/100,
                random_state=42,
                stratify=y
            )
            
            preprocessor = create_preprocessor(numeric_cols, categorical_cols)
        
        st.success(f"Features identificados: {len(numeric_cols)} numericos, {len(categorical_cols)} categoricos")
        
        # Entrenar modelos
        st.markdown('<div class="sub-header">Entrenamiento de Modelos</div>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0, text="Iniciando entrenamiento...")
        
        with st.spinner("Entrenando modelos..."):
            trained_models, cv_results = train_models(X_train, y_train, preprocessor, progress_bar)
        
        progress_bar.empty()
        st.success("Todos los modelos entrenados exitosamente!")
        
        st.markdown("---")
        
        # Evaluación en Test
        st.markdown('<div class="sub-header">Resultados en Conjunto de Prueba</div>', unsafe_allow_html=True)
        
        test_results = []
        all_predictions = {}
        
        for name, model_info in trained_models.items():
            metrics, y_pred, y_proba = evaluate_model(model_info['model'], X_test, y_test)
            
            all_predictions[name] = {
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            test_results.append({
                'Modelo': name,
                'F1-Score': round(metrics['F1-Score'], 4),
                'Recall': round(metrics['Recall'], 4),
                'Precision': round(metrics['Precision'], 4),
                'Accuracy': round(metrics['Accuracy'], 4),
                'ROC-AUC': round(metrics.get('ROC-AUC', 0), 4),
                'PR-AUC': round(metrics.get('PR-AUC', 0), 4),
                'Tiempo (s)': model_info['training_time']
            })
        
        df_test_results = pd.DataFrame(test_results).sort_values('F1-Score', ascending=False)
        
        # Tabla de resultados
        st.dataframe(
            df_test_results.style.highlight_max(
                subset=['F1-Score', 'Recall', 'Precision', 'Accuracy', 'ROC-AUC', 'PR-AUC'],
                color='lightgreen'
            ).highlight_min(
                subset=['Tiempo (s)'],
                color='lightblue'
            ),
            hide_index=True,
            width='stretch'
        )
        
        # Gráficos comparativos
        st.markdown("### Comparacion Visual de Metricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras - Métricas principales
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_to_plot = ['F1-Score', 'Recall', 'Precision', 'Accuracy']
            x = np.arange(len(df_test_results))
            width = 0.2
            
            for i, metric in enumerate(metrics_to_plot):
                offset = width * (i - 1.5)
                ax.bar(x + offset, df_test_results[metric], width, 
                      label=metric, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Modelos', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Comparacion de Metricas de Clasificacion', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df_test_results['Modelo'], rotation=15, ha='right')
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Scatter plot - Precision vs Recall
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, row in df_test_results.iterrows():
                ax.scatter(row['Recall'], row['Precision'], 
                          s=300, alpha=0.6, edgecolor='black',
                          color=colors[i % len(colors)],
                          label=row['Modelo'])
                ax.annotate(row['Modelo'], 
                           (row['Recall'], row['Precision']),
                           fontsize=9, ha='center', va='bottom')
            
            ax.set_xlabel('Recall (Sensibilidad)', fontsize=11)
            ax.set_ylabel('Precision', fontsize=11)
            ax.set_title('Trade-off Precision vs Recall', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower left')
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Análisis detallado por modelo
        st.markdown('<div class="sub-header">Analisis Detallado por Modelo</div>', unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Selecciona un modelo para análisis detallado:",
            options=list(trained_models.keys()),
            index=0
        )
        
        model_info = trained_models[selected_model]
        model_preds = all_predictions[selected_model]
        
        # Información del modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hiperparametros Optimos")
            st.json(model_info['best_params'])
        
        with col2:
            st.markdown("#### Metricas de Rendimiento")
            model_metrics = df_test_results[df_test_results['Modelo'] == selected_model].iloc[0]
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("F1-Score", f"{model_metrics['F1-Score']:.4f}")
                st.metric("Recall", f"{model_metrics['Recall']:.4f}")
            with metric_col2:
                st.metric("Precision", f"{model_metrics['Precision']:.4f}")
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
        
        # Matriz de confusión y curvas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Matriz de Confusion")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, model_preds['y_pred'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Deserto', 'Deserto'])
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title(f'Matriz de Confusion - {selected_model}', fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            if model_preds['y_proba'] is not None:
                st.markdown("#### Curva ROC")
                fig, ax = plt.subplots(figsize=(8, 6))
                RocCurveDisplay.from_predictions(y_test, model_preds['y_proba'], ax=ax)
                ax.set_title(f'Curva ROC - {selected_model}', fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        # Curva Precision-Recall
        if model_preds['y_proba'] is not None:
            st.markdown("#### Curva Precision-Recall")
            fig, ax = plt.subplots(figsize=(10, 5))
            PrecisionRecallDisplay.from_predictions(y_test, model_preds['y_proba'], ax=ax)
            ax.set_title(f'Curva Precision-Recall - {selected_model}', fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Reporte de clasificación
        st.markdown("#### Reporte de Clasificacion")
        report = classification_report(y_test, model_preds['y_pred'], 
                                      target_names=['No Deserto', 'Deserto'],
                                      output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
        
        # Feature Importance (solo para XGBoost)
        if selected_model == "XGBoost":
            st.markdown("---")
            st.markdown("#### Feature Importance")
            
            try:
                clf = model_info['model'].named_steps.get('classifier')
                pre = model_info['model'].named_steps.get('preprocessor')
                
                if hasattr(clf, 'feature_importances_') and pre is not None:
                    feat_names = pre.get_feature_names_out()
                    importances = pd.Series(clf.feature_importances_, index=feat_names)
                    importances = importances.sort_values(ascending=False).head(20)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    importances.sort_values().plot(kind='barh', ax=ax, color='steelblue', 
                                                   edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Importancia', fontsize=11)
                    ax.set_title('Top 20 Features Más Importantes', fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Tabla de importancias
                    st.dataframe(
                        pd.DataFrame({
                            'Feature': importances.index,
                            'Importancia': importances.values
                        }).reset_index(drop=True),
                        width='stretch',
                        height=400
                    )
            except Exception as e:
                st.warning(f"No se pudo extraer feature importance: {str(e)}")
        
        st.markdown("---")
        
        # Conclusiones
        st.markdown('<div class="sub-header">Conclusiones</div>', unsafe_allow_html=True)
        
        best_model_name = df_test_results.iloc[0]['Modelo']
        best_f1 = df_test_results.iloc[0]['F1-Score']
        best_recall = df_test_results.iloc[0]['Recall']
        best_precision = df_test_results.iloc[0]['Precision']
        
        st.success(f"""
        **Modelo Recomendado: {best_model_name}**
        
        - El modelo {best_model_name} obtuvo el mejor F1-Score ({best_f1:.4f}) en el conjunto de prueba.
        - Recall de {best_recall:.4f} indica que detecta {best_recall*100:.1f}% de los casos de desercion.
        - Precision de {best_precision:.4f} significa que {best_precision*100:.1f}% de las predicciones positivas son correctas.
        - Este modelo balancea adecuadamente la deteccion de estudiantes en riesgo con la precision de las alertas.
        """)
        
        st.info("""
        **Consideraciones para la Implementacion:**
        
        1. **Monitoreo Continuo**: El rendimiento del modelo debe monitorearse regularmente ante cambios en la poblacion estudiantil.
        2. **Ajuste de Umbral**: El umbral de decision (0.5 por defecto) puede ajustarse segun el costo relativo de falsos positivos vs. falsos negativos.
        3. **Actualizacion Periodica**: Re-entrenar el modelo con datos nuevos para mantener su precision.
        4. **Intervencion Temprana**: Usar las predicciones para implementar programas de retencion focalizados.
        """)
        
    except FileNotFoundError:
        st.error(f"No se encontro el archivo: {data_file}")
        st.info("Por favor, verifica la ruta del archivo en la barra lateral.")
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")
        st.exception(e)

else:
    # Pantalla inicial
    st.info("Configura los parametros en la barra lateral y presiona 'Entrenar Modelos' para comenzar.")
    
    st.markdown("""
    ### Instrucciones de Uso
    
    1. **Seleccionar Dataset**: En la barra lateral, elige entre el dataset principal (2015-2019) o el completo.
    2. **Ajustar Parametros**: Define el numero maximo de muestras y el tamano del conjunto de prueba.
    3. **Entrenar Modelos**: Presiona el boton para iniciar el entrenamiento y comparacion de modelos.
    4. **Analizar Resultados**: Explora las metricas, graficos y analisis detallados de cada modelo.
    
    ### Modelos Incluidos
    
    - **Regresion Logistica**: Modelo baseline, interpretable y rapido
    - **K-Nearest Neighbors**: Modelo basado en similitud
    - **XGBoost**: Modelo de ensamble con gradient boosting
    
    ### Metricas Evaluadas
    
    - **F1-Score**: Balance entre precision y recall
    - **Recall**: Capacidad de detectar casos positivos
    - **Precision**: Exactitud de las predicciones positivas
    - **Accuracy**: Precision general del modelo
    - **ROC-AUC**: Area bajo la curva ROC
    - **PR-AUC**: Area bajo la curva Precision-Recall
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Dashboard de Prediccion de Desercion Estudiantil | Machine Learning Supervisado | 2026"
    "</div>",
    unsafe_allow_html=True
)
