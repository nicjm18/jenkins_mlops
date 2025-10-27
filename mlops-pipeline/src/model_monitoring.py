import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from scipy import stats
import logging
import plotly.express as px
import plotly.graph_objects as go
import sys
import streamlit as st
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMonitor:
    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path
        self.setup_database()
        logger.info("Monitor inicializado")
    
    def setup_database(self):
        """Crear tabla para capturar predicciones de la API"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    capital_prestado REAL,
                    plazo_meses INTEGER,
                    edad_cliente INTEGER,
                    salario_cliente REAL,
                    puntaje_datacredito REAL
                )
            ''')
    
    def capture_prediction(self, input_data: dict, api_response: dict):
        """Capturar una predicción real de la API"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO api_predictions 
                    (timestamp, input_data, prediction, probability, risk_level,
                     capital_prestado, plazo_meses, edad_cliente, salario_cliente, puntaje_datacredito)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    json.dumps(input_data),
                    api_response.get('prediction', 0),
                    api_response.get('probability', 0.5),
                    api_response.get('risk_level', 'Unknown'),
                    input_data.get('capital_prestado'),
                    input_data.get('plazo_meses'),
                    input_data.get('edad_cliente'),
                    input_data.get('salario_cliente'),
                    input_data.get('puntaje_datacredito')
                ))
            logger.info("Predicción capturada")
        except Exception as e:
            logger.error(f"Error capturando predicción: {e}")
    
    def get_data(self, days: int = 7) -> pd.DataFrame:
        """Obtener datos de los últimos días"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM api_predictions 
                    WHERE timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            if 'streamlit' not in sys.modules:
                print(f"Error conectando a la base de datos: {e}")
            return pd.DataFrame()
    
    def check_drift(self) -> dict:
        """Detección simple de drift"""
        recent = self.get_data(days=3)
        older = self.get_data(days=14)
        
        if len(recent) < 5 or len(older) < 10:
            return {"status": "insufficient_data", "message": "Necesitas más predicciones"}
        
        cutoff = datetime.now() - timedelta(days=3)
        older = older[older['timestamp'] < cutoff]
        
        if len(older) < 5:
            return {"status": "insufficient_data", "message": "Necesitas más datos históricos"}
        
        drift_results = {}
        features = ['capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente', 'puntaje_datacredito']
        
        for feature in features:
            if feature in recent.columns and feature in older.columns:
                recent_vals = recent[feature].dropna()
                older_vals = older[feature].dropna()
                
                if len(recent_vals) > 2 and len(older_vals) > 2:
                    recent_mean = recent_vals.mean()
                    older_mean = older_vals.mean()
                    
                    if older_mean != 0:
                        change_pct = ((recent_mean - older_mean) / older_mean) * 100
                    else:
                        change_pct = 0
                    
                    try:
                        _, p_value = stats.ttest_ind(recent_vals, older_vals)
                    except:
                        p_value = 1.0
                    
                    drift_results[feature] = {
                        'recent_mean': recent_mean,
                        'older_mean': older_mean,
                        'change_pct': change_pct,
                        'p_value': p_value,
                        'drift_detected': abs(change_pct) > 15 or p_value < 0.05
                    }
        
        return drift_results
    
    def get_summary(self) -> dict:
        """Resumen simple del modelo"""
        data = self.get_data(days=7)
        
        if data.empty:
            return {"status": "no_data", "message": "No hay predicciones en los últimos 7 días"}
        
        summary = {
            'total_predictions': len(data),
            'approval_rate': (data['prediction'] == 1).mean() * 100,
            'avg_probability': data['probability'].mean(),
            'risk_distribution': data['risk_level'].value_counts().to_dict(),
            'latest_prediction': data.iloc[0]['timestamp'] if len(data) > 0 else None,
            'avg_loan_amount': data['capital_prestado'].mean(),
            'avg_credit_score': data['puntaje_datacredito'].mean()
        }
        
        return summary
    
    def create_charts(self):
        """Crear gráficos básicos"""
        data = self.get_data(days=14)
        
        if data.empty:
            return None
        
        charts = {}
        
        # Predicciones por día
        daily = data.groupby(data['timestamp'].dt.date).size().reset_index()
        daily.columns = ['date', 'count']
        charts['daily'] = px.line(daily, x='date', y='count', title='Predicciones por Día')
        
        # Distribución de riesgos
        risk_counts = data['risk_level'].value_counts()
        charts['risk'] = px.pie(values=risk_counts.values, names=risk_counts.index, 
                               title='Distribución de Riesgos',
                               color_discrete_map={
                                   'Bajo': '#28a745',
                                   'Medio-Bajo': '#17a2b8', 
                                   'Medio': '#ffc107',
                                   'Alto': '#fd7e14',
                                   'Muy Alto': '#dc3545'
                               })
        
        # Evolución de probabilidades
        data_sorted = data.sort_values('timestamp')
        charts['probability'] = px.line(data_sorted, x='timestamp', y='probability', 
                                      title='Evolución de Probabilidades')
        
        # Distribución de montos
        charts['amounts'] = px.histogram(data, x='capital_prestado', nbins=15,
                                       title='Distribución de Montos de Préstamo')
        
        return charts

# Función para integrar con la API
def log_api_prediction(input_data: dict, api_response: dict):
    """Función para llamar desde la API"""
    monitor = SimpleMonitor()
    monitor.capture_prediction(input_data, api_response)

# INTERFAZ DE CONSOLA
def console_interface():
    """Función principal para interfaz de consola"""
    monitor = SimpleMonitor()
    
    print("\n" + "="*40)
    print("  MONITOR SIMPLE DE MODELO")
    print("="*40)
    
    while True:
        print("\n1. Ver resumen")
        print("2. Detectar drift")
        print("3. Ver gráficos")
        print("4. Datos recientes")
        print("5. Salir")
        
        choice = input("\nOpción (1-5): ").strip()
        
        if choice == '1':
            print("\n--- RESUMEN ---")
            summary = monitor.get_summary()
            
            if summary.get('status') == 'no_data':
                print("❌ No hay datos. Haz algunas predicciones primero.")
            else:
                print(f"Predicciones (7 días): {summary['total_predictions']}")
                print(f"Tasa de aprobación: {summary['approval_rate']:.1f}%")
                print(f"Probabilidad promedio: {summary['avg_probability']:.3f}")
                print(f"Última predicción: {summary['latest_prediction'].strftime('%Y-%m-%d %H:%M') if summary['latest_prediction'] else 'N/A'}")
                
                print("\nDistribución de riesgos:")
                for risk, count in summary['risk_distribution'].items():
                    print(f"  {risk}: {count}")
        
        elif choice == '2':
            print("\n--- DETECCIÓN DE DRIFT ---")
            drift = monitor.check_drift()
            
            if drift.get('status') == 'insufficient_data':
                print("❌ " + drift['message'])
            else:
                print("Comparando últimos 3 días vs histórico:")
                
                for feature, result in drift.items():
                    status = "🚨 DRIFT" if result['drift_detected'] else "✅ OK"
                    print(f"\n{feature}: {status}")
                    print(f"  Media reciente: {result['recent_mean']:.0f}")
                    print(f"  Media histórica: {result['older_mean']:.0f}")
                    print(f"  Cambio: {result['change_pct']:.1f}%")
        
        elif choice == '3':
            print("\n--- GRÁFICOS ---")
            charts = monitor.create_charts()
            
            if charts is None:
                print("❌ No hay datos para gráficos")
            else:
                for name, fig in charts.items():
                    fig.write_html(f"monitor_{name}.html")
                print("✅ Gráficos guardados como HTML:")
                for name in charts.keys():
                    print(f"  - monitor_{name}.html")
        
        elif choice == '4':
            print("\n--- DATOS RECIENTES ---")
            data = monitor.get_data(days=3)
            
            if data.empty:
                print("❌ No hay datos recientes")
            else:
                print(f"Últimas {min(5, len(data))} predicciones:")
                recent = data[['timestamp', 'capital_prestado', 'prediction', 'probability', 'risk_level']].head()
                for _, row in recent.iterrows():
                    print(f"  {row['timestamp'].strftime('%m-%d %H:%M')} | "
                          f"${row['capital_prestado']:,.0f} | "
                          f"Pred: {row['prediction']} | "
                          f"Prob: {row['probability']:.3f} | "
                          f"Risk: {row['risk_level']}")
        
        elif choice == '5':
            print("Saliendo...")
            break
        
        else:
            print("❌ Opción no válida")

# INTERFAZ DE STREAMLIT
def streamlit_interface():
    """Interfaz de Streamlit"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Model Monitoring Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            font-size: 2rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        
        .alert-high {
            background: #ffe6e6;
            border-left: 4px solid #dc3545;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .alert-ok {
            background: #e6f7e6;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🏦 Model Monitoring Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Inicializar monitor
    monitor = SimpleMonitor()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    
    # Botón de refresh
    if st.sidebar.button("🔄 Actualizar Datos"):
        st.experimental_rerun()
    
    # Información de BD
    try:
        total_predictions = len(monitor.get_data(days=365))
        st.sidebar.info(f"Total predicciones: {total_predictions}")
    except:
        st.sidebar.error("Error conectando a BD")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen", 
        "🔍 Data Drift", 
        "📈 Visualizaciones", 
        "🕒 Datos Recientes"
    ])
    
    with tab1:
        st.subheader("📈 Resumen del Modelo")
        
        summary = monitor.get_summary()
        
        if summary.get('status') == 'no_data':
            st.warning("No hay datos. Realiza predicciones usando la API.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predicciones", summary['total_predictions'])
            with col2:
                st.metric("Tasa de Aprobación", f"{summary['approval_rate']:.1f}%")
            with col3:
                st.metric("Probabilidad Promedio", f"{summary['avg_probability']:.3f}")
            with col4:
                latest = summary['latest_prediction']
                if latest:
                    latest_str = latest.strftime('%m-%d %H:%M')
                else:
                    latest_str = "N/A"
                st.metric("Última Predicción", latest_str)
            
            # Distribución de riesgos
            if summary['risk_distribution']:
                st.subheader("🎯 Distribución de Riesgos")
                
                risk_df = pd.DataFrame(list(summary['risk_distribution'].items()), 
                                      columns=['Nivel de Riesgo', 'Cantidad'])
                
                fig = px.pie(risk_df, values='Cantidad', names='Nivel de Riesgo',
                            color_discrete_map={
                                'Bajo': '#28a745',
                                'Medio-Bajo': '#17a2b8', 
                                'Medio': '#ffc107',
                                'Alto': '#fd7e14',
                                'Muy Alto': '#dc3545'
                            })
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Detección de Data Drift")
        
        drift_results = monitor.check_drift()
        
        if drift_results.get('status') == 'insufficient_data':
            st.info("Necesitas más predicciones para detectar drift.")
        else:
            drift_features = [f for f, r in drift_results.items() if r.get('drift_detected', False)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Features Analizados", len(drift_results))
            with col2:
                st.metric("Features con Drift", len(drift_features))
            
            if len(drift_features) == 0:
                st.markdown('<div class="alert-ok">✅ No se detectó drift</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-high">⚠️ Drift detectado en {len(drift_features)} feature(s)</div>', 
                           unsafe_allow_html=True)
            
            # Detalle por feature
            for feature, result in drift_results.items():
                with st.expander(f"{feature} {'🚨' if result['drift_detected'] else '✅'}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Media Reciente", f"{result['recent_mean']:,.0f}")
                    with col2:
                        st.metric("Media Histórica", f"{result['older_mean']:,.0f}")
                    with col3:
                        change_pct = result['change_pct']
                        st.metric("Cambio", f"{change_pct:+.1f}%")
    
    with tab3:
        st.subheader("📊 Visualizaciones")
        
        charts = monitor.create_charts()
        
        if charts is None:
            st.info("No hay datos para visualizar")
        else:
            # Predicciones por día
            st.plotly_chart(charts['daily'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['probability'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['amounts'], use_container_width=True)
    
    with tab4:
        st.subheader("🕒 Predicciones Recientes")
        
        data = monitor.get_data(days=3)
        
        if data.empty:
            st.info("No hay predicciones recientes")
        else:
            # Mostrar tabla
            recent_data = data.head(10)
            display_data = recent_data[['timestamp', 'capital_prestado', 'plazo_meses', 
                                       'edad_cliente', 'prediction', 'probability', 'risk_level']].copy()
            
            display_data['timestamp'] = display_data['timestamp'].dt.strftime('%m-%d %H:%M')
            display_data['capital_prestado'] = display_data['capital_prestado'].apply(lambda x: f"${x:,.0f}")
            display_data['probability'] = display_data['probability'].apply(lambda x: f"{x:.3f}")
            display_data['prediction'] = display_data['prediction'].apply(lambda x: "Aprobado" if x == 1 else "Rechazado")
            
            display_data.columns = ['Fecha/Hora', 'Monto', 'Plazo', 'Edad', 'Predicción', 'Probabilidad', 'Riesgo']
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)

def main():
    """Detectar si correr como Streamlit o consola"""
    if 'streamlit' in sys.modules:
        streamlit_interface()
    else:
        console_interface()

if __name__ == "__main__":
    main()