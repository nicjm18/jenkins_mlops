import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import pickle
import os
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Credit Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-metric {
        border-left-color: #28a745;
    }
    
    .warning-metric {
        border-left-color: #ffc107;
    }
    
    .danger-metric {
        border-left-color: #dc3545;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class ModelEvaluator:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.model = None
        self.pipeline = None
        
    def load_local_artifacts(self):
        """Cargar modelo y pipeline localmente para evaluación offline"""
        try:
            if os.path.exists("models/best_model_xgboost.pkl"):
                with open("models/best_model_xgboost.pkl", 'rb') as f:
                    self.model = pickle.load(f)
                st.success("✅ Modelo cargado localmente")
            
            if os.path.exists("models/pipeline_ml.pkl"):
                with open("models/pipeline_ml.pkl", 'rb') as f:
                    self.pipeline = pickle.load(f)
                st.success("✅ Pipeline cargado localmente")
                
            return True
        except Exception as e:
            st.error(f"❌ Error cargando artefactos: {str(e)}")
            return False
    
    def test_api_connection(self):
        """Probar conexión con la API"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self):
        """Obtener información del modelo desde la API"""
        try:
            response = requests.get(f"{self.api_url}/model/info")
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def evaluate_with_api(self, test_data):
        """Evaluar usando la API"""
        predictions = []
        for _, row in test_data.iterrows():
            try:
                data = row.to_dict()
                # Limpiar valores NaN
                data = {k: v for k, v in data.items() if pd.notna(v)}
                
                response = requests.post(f"{self.api_url}/predict", json=data)
                if response.status_code == 200:
                    result = response.json()
                    predictions.append({
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'risk_level': result['risk_level'],
                        'recommendation': result['recommendation']
                    })
                else:
                    predictions.append({
                        'prediction': 0,
                        'probability': 0.5,
                        'risk_level': 'Error',
                        'recommendation': 'Error en API'
                    })
            except Exception as e:
                st.error(f"Error en predicción: {str(e)}")
                predictions.append({
                    'prediction': 0,
                    'probability': 0.5,
                    'risk_level': 'Error',
                    'recommendation': 'Error de conexión'
                })
        
        return pd.DataFrame(predictions)
    
    def evaluate_locally(self, test_data, y_true):
        """Evaluar usando modelo local"""
        if self.model is None or self.pipeline is None:
            return None
        
        try:
            # Preprocesar datos
            X_processed = self.pipeline.transform(test_data)
            
            # Predicciones
            y_pred = self.model.predict(X_processed)
            y_proba = self.model.predict_proba(X_processed)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            return y_pred, y_proba
        except Exception as e:
            st.error(f"Error en evaluación local: {str(e)}")
            return None, None

def create_confusion_matrix_plot(y_true, y_pred):
    """Crear matriz de confusión interactiva"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicción", y="Real", color="Frecuencia"),
                    x=['No Pagará (0)', 'Pagará (1)'],
                    y=['No Pagará (0)', 'Pagará (1)'])
    
    fig.update_layout(
        title="Matriz de Confusión",
        title_x=0.5,
        width=500,
        height=400
    )
    
    return fig

def create_roc_curve_plot(y_true, y_proba):
    """Crear curva ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    
    return fig

def create_precision_recall_plot(y_true, y_proba):
    """Crear curva Precision-Recall"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='green', width=2)
    ))
    
    baseline = np.mean(y_true)
    fig.add_hline(y=baseline, 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Baseline = {baseline:.3f}")
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Crear gráfico de importancia de features"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(feature_df, 
                     x='importance', 
                     y='feature',
                     orientation='h',
                     title='Top 15 Feature Importances',
                     color='importance',
                     color_continuous_scale='viridis')
        
        fig.update_layout(height=600)
        return fig
    
    return None

def create_risk_distribution_plot(predictions_df):
    """Crear distribución de niveles de riesgo"""
    risk_counts = predictions_df['risk_level'].value_counts()
    
    fig = px.pie(values=risk_counts.values, 
                 names=risk_counts.index,
                 title='Distribución de Niveles de Riesgo',
                 color_discrete_map={
                     'Bajo': '#28a745',
                     'Medio-Bajo': '#6f42c1',
                     'Medio': '#ffc107',
                     'Alto': '#fd7e14',
                     'Muy Alto': '#dc3545'
                 })
    
    return fig

def main():
    # Header principal
    st.markdown('<div class="main-header">📊 Credit Model Evaluation Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Inicializar evaluador
    evaluator = ModelEvaluator()
    
    # Sidebar para configuración
    st.sidebar.title("⚙️ Configuración")
    
    # Verificar conexión API
    api_connected = evaluator.test_api_connection()
    if api_connected:
        st.sidebar.success("🟢 API Conectada")
    else:
        st.sidebar.error("🔴 API Desconectada")
    
    # Cargar artefactos locales
    #local_loaded = evaluator.load_local_artifacts()
    
    # Información del modelo
    if api_connected:
        model_info = evaluator.get_model_info()
        if model_info:
            st.sidebar.info(f"**Modelo:** {model_info.get('model_name', 'N/A')}")
            st.sidebar.info(f"**Features:** {model_info.get('features_count', 'N/A')}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Métricas Generales", 
        "🔍 Análisis Detallado", 
        "📊 Visualizaciones", 
        "🎯 Evaluación en Vivo",
        "📋 Reportes"
    ])
    
    with tab1:
        st.header("Métricas de Rendimiento del Modelo")
        
        # Upload de datos de test
        uploaded_file = st.file_uploader(
            "Sube tus datos de test (CSV con columna 'Pago_atiempo')",
            type=['csv']
        )
        
        if uploaded_file is not None:
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Datos cargados: {len(test_data)} registros")
            
            if 'Pago_atiempo' in test_data.columns:
                y_true = test_data['Pago_atiempo']
                X_test = test_data.drop('Pago_atiempo', axis=1)
                
                # Evaluar con API o localmente
                if api_connected:
                    st.info("🔄 Evaluando con API...")
                    api_predictions = evaluator.evaluate_with_api(X_test)
                    y_pred = api_predictions['prediction'].values
                    y_proba = api_predictions['probability'].values
                else:
                    st.info("🔄 Evaluando localmente...")
                    y_pred, y_proba = evaluator.evaluate_locally(X_test, y_true)
                
                if y_pred is not None:
                    # Calcular métricas
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted')
                    recall = recall_score(y_true, y_pred, average='weighted')
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    
                    # Mostrar métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy:.3f}", 
                                delta=f"{(accuracy-0.5):.3f}" if accuracy > 0.5 else None)
                    
                    with col2:
                        st.metric("🔍 Precision", f"{precision:.3f}",
                                delta=f"{(precision-0.5):.3f}" if precision > 0.5 else None)
                    
                    with col3:
                        st.metric("📡 Recall", f"{recall:.3f}",
                                delta=f"{(recall-0.5):.3f}" if recall > 0.5 else None)
                    
                    with col4:
                        st.metric("⚖️ F1-Score", f"{f1:.3f}",
                                delta=f"{(f1-0.5):.3f}" if f1 > 0.5 else None)
                    
                    # Métricas por clase
                    st.subheader("📊 Métricas por Clase")
                    
                    report = classification_report(y_true, y_pred, 
                                                 target_names=['No Pago (0)', 'Pago (1)'], 
                                                 output_dict=True)
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
    
    with tab2:
        st.header("🔍 Análisis Detallado del Modelo")
        
        if uploaded_file is not None and y_pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Matriz de confusión
                cm_fig = create_confusion_matrix_plot(y_true, y_pred)
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                # Distribución de predicciones
                pred_dist = pd.DataFrame({
                    'Real': y_true,
                    'Predicción': y_pred
                })
                
                fig = px.histogram(pred_dist, x='Predicción', color='Real',
                                 title='Distribución de Predicciones por Clase Real',
                                 barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de errores
            st.subheader("❌ Análisis de Errores")
            
            errors_df = test_data.copy()
            errors_df['Predicción'] = y_pred
            errors_df['Error'] = (y_true != y_pred)
            
            error_rate = errors_df['Error'].mean()
            st.metric("Tasa de Error", f"{error_rate:.1%}")
            
            if error_rate > 0:
                st.write("**Casos con errores:**")
                error_cases = errors_df[errors_df['Error'] == True].head(10)
                st.dataframe(error_cases, use_container_width=True)
    
    with tab3:
        st.header("📊 Visualizaciones Avanzadas")
        
        if uploaded_file is not None and y_pred is not None and y_proba is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Curva ROC
                roc_fig = create_roc_curve_plot(y_true, y_proba)
                st.plotly_chart(roc_fig, use_container_width=True)
            
            with col2:
                # Curva Precision-Recall
                pr_fig = create_precision_recall_plot(y_true, y_proba)
                st.plotly_chart(pr_fig, use_container_width=True)
            
            # Distribución de probabilidades
            st.subheader("📈 Distribución de Probabilidades")
            
            prob_df = pd.DataFrame({
                'Probabilidad': y_proba,
                'Clase_Real': ['Pago' if x == 1 else 'No Pago' for x in y_true]
            })
            
            fig = px.histogram(prob_df, x='Probabilidad', color='Clase_Real',
                             title='Distribución de Probabilidades por Clase Real',
                             barmode='overlay', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance si está disponible
            if evaluator.model and hasattr(evaluator.model, 'feature_importances_'):
                if evaluator.pipeline and hasattr(evaluator.pipeline, 'get_feature_names_out'):
                    try:
                        feature_names = evaluator.pipeline.get_feature_names_out()
                        fi_fig = create_feature_importance_plot(evaluator.model, feature_names)
                        if fi_fig:
                            st.plotly_chart(fi_fig, use_container_width=True)
                    except:
                        st.info("No se pudo mostrar feature importance")
    
    with tab4:
        st.header("🎯 Evaluación en Tiempo Real")
        
        st.write("Prueba el modelo con datos individuales:")
        
        # Formulario para predicción individual
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tipo_credito = st.selectbox("Tipo Crédito", ["04", "12", "06"])
                capital_prestado = st.number_input("Capital Prestado", min_value=0, value=1500000)
                plazo_meses = st.number_input("Plazo (meses)", min_value=1, max_value=60, value=12)
                edad_cliente = st.number_input("Edad Cliente", min_value=18, max_value=80, value=30)
                saldo_principal = st.number_input("Saldo Principal", min_value=0, value=0)
                fecha_prestamo = st.date_input("Fecha Préstamo", value=datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
            

            with col2:
                tipo_laboral = st.selectbox("Tipo Laboral", ["Empleado", "Independiente"])
                salario_cliente = st.number_input("Salario", min_value=0, value=3000000)
                puntaje_datacredito = st.number_input("Puntaje Datacrédito", min_value=0, max_value=950, value=750)
                cant_creditosvigentes = st.number_input("Créditos Vigentes", min_value=0, value=0)
                saldo_total = st.number_input("Saldo Total", min_value=0, value=0)
            
            with col3:
                total_otros_prestamos = st.number_input("Otros Préstamos", min_value=0, value=0)
                cuota_pactada = st.number_input("Cuota Pactada", min_value=0, value=150000)
                huella_consulta = st.number_input("Huella Consulta", min_value=0, value=2)
                saldo_mora = st.number_input("Saldo Mora", min_value=0, value=0)
                creditos_sectorFinanciero = st.number_input("Créditos Sector Financiero", min_value=0, value=0)
            
            submitted = st.form_submit_button("🔮 Realizar Predicción")
            
            if submitted:
                if api_connected:
                    test_case = {
                        "tipo_credito": tipo_credito,
                        "fecha_prestamo": fecha_prestamo,
                        "capital_prestado": capital_prestado,
                        "plazo_meses": plazo_meses,
                        "edad_cliente": edad_cliente,
                        "tipo_laboral": tipo_laboral,
                        "salario_cliente": salario_cliente,
                        "puntaje_datacredito": puntaje_datacredito,
                        "cant_creditosvigentes": cant_creditosvigentes,
                        "total_otros_prestamos": total_otros_prestamos,
                        "cuota_pactada": cuota_pactada,
                        "huella_consulta": huella_consulta,
                        "saldo_mora": saldo_mora,
                        "saldo_principal": saldo_principal,
                        "creditos_sectorFinanciero": creditos_sectorFinanciero,
                        "saldo_total": saldo_total
                    }
                    
                    try:
                        response = requests.post(f"{evaluator.api_url}/predict", json=test_case)
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Mostrar resultado
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pred_text = "✅ PAGARÁ A TIEMPO" if result['prediction'] == 1 else "❌ NO PAGARÁ"
                                st.success(pred_text) if result['prediction'] == 1 else st.error(pred_text)
                            
                            with col2:
                                st.metric("Probabilidad", f"{result['probability']:.1%}")
                            
                            with col3:
                                risk_color = "success" if "Bajo" in result['risk_level'] else "warning" if "Medio" in result['risk_level'] else "error"
                                st.metric("Nivel de Riesgo", result['risk_level'])
                            
                            st.info(f"**Recomendación:** {result['recommendation']}")
                        
                        else:
                            st.error("Error en la predicción")
                    
                    except Exception as e:
                        st.error(f"Error de conexión: {str(e)}")
                else:
                    st.error("API no disponible")
    
    with tab5:
        st.header("📋 Reportes y Resúmenes")
        
        if uploaded_file is not None and y_pred is not None:
            # Resumen ejecutivo
            st.subheader("📊 Resumen Ejecutivo")
            
            total_cases = len(y_true)
            correct_predictions = (y_true == y_pred).sum()
            accuracy_pct = (correct_predictions / total_cases) * 100
            
            # Métricas de negocio
            true_positives = ((y_true == 1) & (y_pred == 1)).sum()
            false_positives = ((y_true == 0) & (y_pred == 1)).sum()
            false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Casos", total_cases)
            with col2:
                st.metric("Predicciones Correctas", correct_predictions)
            with col3:
                st.metric("Accuracy", f"{accuracy_pct:.1f}%")
            with col4:
                st.metric("Casos de Alto Riesgo Detectados", true_positives)
            
            # Tabla de rendimiento por segmento
            st.subheader("📈 Rendimiento por Segmento")
            
            if 'tipo_laboral' in test_data.columns:
                segment_analysis = []
                for segment in test_data['tipo_laboral'].unique():
                    mask = test_data['tipo_laboral'] == segment
                    segment_acc = accuracy_score(y_true[mask], y_pred[mask])
                    segment_precision = precision_score(y_true[mask], y_pred[mask], average='weighted')
                    segment_recall = recall_score(y_true[mask], y_pred[mask], average='weighted')
                    
                    segment_analysis.append({
                        'Segmento': segment,
                        'Casos': mask.sum(),
                        'Accuracy': f"{segment_acc:.3f}",
                        'Precision': f"{segment_precision:.3f}",
                        'Recall': f"{segment_recall:.3f}"
                    })
                
                segment_df = pd.DataFrame(segment_analysis)
                st.dataframe(segment_df, use_container_width=True)
            
            # Botón para descargar reporte
            if st.button("📥 Generar Reporte PDF"):
                st.info("Funcionalidad de PDF en desarrollo...") #Implementar luego

if __name__ == "__main__":
    main()