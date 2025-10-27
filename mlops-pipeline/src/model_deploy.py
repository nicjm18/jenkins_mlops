import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import io
import json
from datetime import datetime
import logging
import os

# Pipeline de feature engineering
from src.feature_engineering import pipeline_ml

from src.model_monitoring import log_api_prediction

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Créditos",
    description="API para predecir la probabilidad de pago a tiempo de créditos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Variables globales para el modelo y pipeline
model = None
pipeline_ml = None

# Modelo de datos para la API basado en tu dataset
class CreditFeatures(BaseModel):
    """Esquema para predicción de crédito basado en tu dataset"""
    tipo_credito: Optional[str] = Field(None, description="Tipo de crédito", example="04")
    fecha_prestamo: Optional[str] = Field(None, description="Fecha del préstamo (YYYY-MM-DD)", example="2025-01-07")
    capital_prestado: Optional[float] = Field(None, description="Capital prestado", ge=0, example=1852560.0)
    plazo_meses: Optional[int] = Field(None, description="Plazo en meses", ge=1, le=120, example=12)
    edad_cliente: Optional[int] = Field(None, description="Edad del cliente", ge=18, le=99, example=32)
    tipo_laboral: Optional[str] = Field(None, description="Tipo laboral", example="Empleado")
    salario_cliente: Optional[float] = Field(None, description="Salario del cliente", ge=0, example=3500000)
    total_otros_prestamos: Optional[float] = Field(None, description="Total otros préstamos", ge=0, example=1000000)
    cuota_pactada: Optional[float] = Field(None, description="Cuota pactada", ge=0, example=128650)
    puntaje: Optional[float] = Field(None, description="Puntaje", example=95.23)
    puntaje_datacredito: Optional[float] = Field(None, description="Puntaje Datacrédito", example=795.0)
    cant_creditosvigentes: Optional[int] = Field(None, description="Cantidad créditos vigentes", ge=0, example=0)
    huella_consulta: Optional[int] = Field(None, description="Huella consulta", ge=0, example=2)
    saldo_mora: Optional[float] = Field(None, description="Saldo en mora", ge=0, example=0.0)
    saldo_total: Optional[float] = Field(None, description="Saldo total", ge=0, example=0.0)
    saldo_principal: Optional[float] = Field(None, description="Saldo principal", ge=0, example=0.0)
    saldo_mora_codeudor: Optional[float] = Field(None, description="Saldo mora codeudor", ge=0, example=0.0)
    creditos_sectorFinanciero: Optional[int] = Field(None, description="Créditos sector financiero", ge=0, example=0)
    creditos_sectorCooperativo: Optional[int] = Field(None, description="Créditos sector cooperativo", ge=0, example=0)
    creditos_sectorReal: Optional[int] = Field(None, description="Créditos sector real", ge=0, example=0)
    promedio_ingresos_datacredito: Optional[float] = Field(None, description="Promedio ingresos Datacrédito", example=916148.0)
    tendencia_ingresos: Optional[str] = Field(None, description="Tendencia ingresos", example="Creciente")
    
    class Config:
        schema_extra = {
            "example": {
                "tipo_credito": "04",
                "fecha_prestamo": "2025-01-07",
                "capital_prestado": 1852560.0,
                "plazo_meses": 12,
                "edad_cliente": 32,
                "tipo_laboral": "Empleado",
                "salario_cliente": 3500000,
                "total_otros_prestamos": 1000000,
                "cuota_pactada": 128650,
                "puntaje": 95.23,
                "puntaje_datacredito": 795.0,
                "cant_creditosvigentes": 0,
                "huella_consulta": 2,
                "saldo_mora": 0.0,
                "saldo_total": 0.0,
                "saldo_principal": 0.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 0,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 916148.0,
                "tendencia_ingresos": "Creciente"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Esquema para predicciones en lote"""
    data: List[Dict] = Field(..., description="Lista de registros para predecir")

class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    prediction: int = Field(..., description="Predicción: 0 = Moroso, 1 = Paga a tiempo")
    probability: float = Field(..., description="Probabilidad de pagar a tiempo")
    risk_level: str = Field(..., description="Nivel de riesgo: Alto, Medio, Bajo")
    risk_score: float = Field(..., description="Score de riesgo (0-100)")
    recommendation: str = Field(..., description="Recomendación crediticia")
    timestamp: str = Field(..., description="Timestamp de la predicción")

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote"""
    predictions: List[PredictionResponse]
    summary: Dict = Field(..., description="Resumen de las predicciones")

class ModelInfo(BaseModel):
    """Información del modelo cargado"""
    model_name: str
    model_type: str
    features_count: Optional[int] = None
    loaded_at: str
    status: str
    pipeline_stages: Optional[List[str]] = None

# Funciones auxiliares
def load_model_and_pipeline(model_path: str, pipeline_path: str = None):
    """Cargar modelo y pipeline de preprocessing"""
    global model, pipeline_ml
    
    try:
        # Cargar modelo
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modelo cargado desde: {model_path}")
        
        # Cargar pipeline si se proporciona
        if pipeline_path and os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                pipeline_ml = pickle.load(f)
            logger.info(f"Pipeline cargado desde: {pipeline_path}")
        else:
            logger.info("No se proporcionó pipeline o no existe")
            
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False

def get_risk_level_and_recommendation(probability: float) -> tuple:
    """Determinar nivel de riesgo y recomendación basado en probabilidad"""
    risk_score = probability * 100
    
    if probability >= 0.8:
        return "Bajo", risk_score, "APROBADO - Cliente de bajo riesgo"
    elif probability >= 0.6:
        return "Medio-Bajo", risk_score, "APROBADO CON CONDICIONES - Monitoreo regular"
    elif probability >= 0.4:
        return "Medio", risk_score, "REVISAR - Análisis adicional requerido"
    elif probability >= 0.2:
        return "Alto", risk_score, "RECHAZADO - Alto riesgo de impago"
    else:
        return "Muy Alto", risk_score, "RECHAZADO - Riesgo extremo"

def preprocess_features(data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """Preprocesar features usando tu pipeline"""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Asegurar que las columnas de fecha estén en formato correcto
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce')
    
    # Usar el pipeline si está disponible
    if pipeline_ml is not None:
        try:
            df_processed = pipeline_ml.transform(df)
            return df_processed
        except Exception as e:
            logger.error(f"Error en pipeline: {str(e)}")

# Endpoints de la API

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la aplicación"""
    # Intentar cargar el modelo
    default_model_paths = [
        "models/best_model_xgboost.pkl",
        "models/model_random_forest.pkl",
        "models/model_adaboost.pkl",
        "models/model_logistic.pkl",
        "models/model_linear_svc.pkl",
        "models/model_decision_tree.pkl",
        "models/model_naive_bayes.pkl",
        "models/model_bagging.pkl"
    ]
    
    for model_path in default_model_paths:
        if os.path.exists(model_path):
            success = load_model_and_pipeline(model_path, "models/pipeline_ml.pkl")
            if success:
                logger.info(f"Modelo cargado exitosamente: {model_path}")
                break
    else:
        logger.warning("No se pudo cargar ningún modelo por defecto")

@app.get("/", summary="Health Check")
async def root():
    """Endpoint de salud de la API"""
    return {
        "message": "API de Predicción de Créditos",
        "status": "activo",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline_ml is not None,
        "version": "1.0.0"
    }

@app.get("/model/info", response_model=ModelInfo, summary="Información del Modelo")
async def get_model_info():
    """Obtener información del modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    pipeline_stages = []
    if pipeline_ml is not None:
        pipeline_stages = [step[0] for step in pipeline_ml.steps] if hasattr(pipeline_ml, 'steps') else []
    
    return ModelInfo(
        model_name=str(type(model).__name__),
        model_type=str(type(model).__module__),
        features_count=getattr(model, 'n_features_in_', None),
        loaded_at=datetime.now().isoformat(),
        status="activo",
        pipeline_stages=pipeline_stages
    )

# Cargar pipeline o modelo diferente para pruebas

@app.post("/model/load", summary="Cargar Modelo")
async def load_model_endpoint(model_path: str, pipeline_path: Optional[str] = None):
    """Cargar un modelo específico"""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Archivo de modelo no encontrado: {model_path}")
    
    success = load_model_and_pipeline(model_path, pipeline_path)
    
    if success:
        return {
            "message": f"Modelo cargado exitosamente desde {model_path}",
            "pipeline_loaded": pipeline_path is not None and os.path.exists(pipeline_path or ""),
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo")

@app.post("/predict", response_model=PredictionResponse, summary="Predicción Individual")
async def predict_single(features: CreditFeatures):
    """Realizar predicción para un solo registro"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Convertir a diccionario y luego a DataFrame
        data_dict = features.dict(exclude_none=True)
        df = preprocess_features(data_dict)
        
        # Realizar predicción
        prediction = int(model.predict(df)[0])
        
        # Obtener probabilidad si es posible
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(df)[0, 1])
        elif hasattr(model, 'decision_function'):
            # Para SVM, convertir decision function a probabilidad
            decision = model.decision_function(df)[0]
            probability = 1 / (1 + np.exp(-decision))  # sigmoid
        else:
            probability = 0.5 if prediction == 1 else 0.3  # default basado en predicción
        
        risk_level, risk_score, recommendation = get_risk_level_and_recommendation(probability)

        
        result = PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            risk_score=risk_score,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )

        try:
            log_api_prediction(data_dict, result.__dict__)
        except Exception as e:
            logger.warning(f"Error logging to monitoring: {e}")

        return result
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predicción en Lote")
async def predict_batch(request: BatchPredictionRequest):
    """Realizar predicciones para múltiples registros"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Convertir a DataFrame
        df = pd.DataFrame(request.data)
        df_processed = preprocess_features(df)
        
        # Realizar predicciones
        predictions = model.predict(df_processed)
        
        # Obtener probabilidades
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_processed)[:, 1]
        elif hasattr(model, 'decision_function'):
            decisions = model.decision_function(df_processed)
            probabilities = 1 / (1 + np.exp(-decisions))
        else:
            probabilities = [0.5 if pred == 1 else 0.3 for pred in predictions]
        
        # Crear respuestas
        responses = []
        for pred, prob in zip(predictions, probabilities):
            risk_level, risk_score, recommendation = get_risk_level_and_recommendation(float(prob))
            responses.append(PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level,
                risk_score=risk_score,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat()
            ))
        
        # Crear resumen
        total = len(responses)
        pagaran = sum(1 for r in responses if r.prediction == 1)
        no_pagaran = total - pagaran
        riesgo_alto = sum(1 for r in responses if "Alto" in r.risk_level)
        aprobados = sum(1 for r in responses if "APROBADO" in r.recommendation)
        
        summary = {
            "total_predictions": total,
            "pagaran_a_tiempo": pagaran,
            "no_pagaran_a_tiempo": no_pagaran,
            "porcentaje_pago": round((pagaran / total) * 100, 2),
            "riesgo_alto": riesgo_alto,
            "creditos_aprobados": aprobados,
            "tasa_aprobacion": round((aprobados / total) * 100, 2),
            "probabilidad_promedio": round(np.mean([r.probability for r in responses]), 4),
            "risk_score_promedio": round(np.mean([r.risk_score for r in responses]), 2)
        }
        
        return BatchPredictionResponse(
            predictions=responses,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en predicción batch: {str(e)}")

@app.post("/predict/csv", summary="Predicción desde CSV")
async def predict_from_csv(file: UploadFile = File(...)):
    """Realizar predicciones desde archivo CSV"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
    
    try:
        # Leer CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocesar
        df_processed = preprocess_features(df)
        
        # Realizar predicciones
        predictions = model.predict(df_processed)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_processed)[:, 1]
        else:
            probabilities = [0.5 if pred == 1 else 0.3 for pred in predictions]
        
        # Agregar resultados al DataFrame original
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = probabilities
        
        # Agregar análisis de riesgo
        risk_analysis = [get_risk_level_and_recommendation(p) for p in probabilities]
        result_df['risk_level'] = [r[0] for r in risk_analysis]
        result_df['risk_score'] = [r[1] for r in risk_analysis]
        result_df['recommendation'] = [r[2] for r in risk_analysis]
        
        # Convertir a JSON para la respuesta
        results = result_df.to_dict('records')
        
        # Crear resumen detallado
        total = len(results)
        pagaran = sum(1 for r in results if r['prediction'] == 1)
        aprobados = sum(1 for r in results if "APROBADO" in r['recommendation'])
        
        return {
            "filename": file.filename,
            "total_records": total,
            "predictions_summary": {
                "pagaran_a_tiempo": pagaran,
                "no_pagaran_a_tiempo": total - pagaran,
                "porcentaje_pago": round((pagaran / total) * 100, 2),
                "creditos_aprobados": aprobados,
                "tasa_aprobacion": round((aprobados / total) * 100, 2),
                "risk_score_promedio": round(np.mean([r['risk_score'] for r in results]), 2)
            },
            "results": results[:100],  # Limitar respuesta a 100 registros
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error procesando CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error procesando archivo: {str(e)}")

@app.get("/health", summary="Health Check Detallado")
async def health_check():
    """Endpoint de salud detallado"""
    return {
        "api_status": "activo",
        "model_status": "cargado" if model is not None else "no_cargado",
        "pipeline_status": "cargado" if pipeline_ml is not None else "no_cargado",
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": "3.x",
            "fastapi_version": "0.104.x"
        }
    }

# Configuración para ejecutar el servidor
if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )