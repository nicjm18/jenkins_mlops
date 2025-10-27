# MLOps Credit Risk Prediction

Sistema de **predicción de riesgo crediticio** implementado bajo un enfoque **MLOps**, que integra desde el análisis exploratorio hasta el despliegue de un modelo en producción con monitoreo de desempeño.

---

## Índice
1. [Descripción](#-descripción)
2. [Tecnologías](#-tecnologías)
3. [Estructura del proyecto](#-estructura-del-proyecto)
4. [Instalación](#-instalación)
5. [Uso](#-uso)
   - [API con FastAPI](#api-con-fastapi)
   - [Informe de evaluación (Streamlit)](#informe-de-evaluación-streamlit)
   - [Monitoreo del modelo (Streamlit)](#monitoreo-del-modelo-streamlit)
6. [Características](#-características)
7. [Autores](#-autores)

---

## Descripción
Este proyecto implementa un **pipeline completo** para el modelado de riesgo crediticio.  
Se parte de un dataset de créditos, pasando por **EDA, feature engineering, modelado y despliegue**, hasta llegar a la **exposición de la API** y la creación de herramientas interactivas de **evaluación y monitoreo**.

---

## Tecnologías
- [Python](https://www.python.org/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Streamlit](https://streamlit.io/)  
- [Docker](https://www.docker.com/)  
- [scikit-learn](https://scikit-learn.org/)  

---

## Estructura del proyecto

├── .github/workflows/sonar.yml      # Workflow de CI/CD          
├── src/
│   ├── cargar_datos.py              # Módulo de carga de datos
│   ├── comprension_eda.ipynb        # Exploratory Data Analysis
│   ├── config.json                  # Configuración (rutas/parámetros)
│   ├── feature_engineering.py       # Feature engineering
│   ├── heuristic_model.py           # Modelo heurístico/base
│   ├── model_training.py            # Entrenamiento de modelos ML
│   ├── model_deploy.py              # API con FastAPI
│   ├── model_evaluation.py          # Informe interactivo en Streamlit
│   └── model_monitoring.py          # Monitoreo de drift en Streamlit
├── Dockerfile                       # Imagen para despliegue de la API
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Documentación del proyecto
├── set_up.bat                       # Script de inicialización en Windows


---

## Instalación
1. Clonar el repositorio:

   ```bash
   git clone https://github.com/nicjm18/Repositorio_Modelado_Nico.git
   cd Repositorio_Modelado_Nico
   
   ```

2. Crear entorno virtual e instalar dependencias:
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows

   pip install -r requirements.txt #Instalar dependencias
   ```

---

## Uso

### API con FastAPI
Levantar la API:
```bash
   uvicorn src.model_deploy:app --reload

   ```

Documentación interactiva: http://127.0.0.1:8000/docs

### Informe de evaluación (Streamlit)
    ```bash
   streamlit run src/model_evaluation.py
   ```

Accede en http://localhost:8501.

### Monitoreo del modelo (Streamlit)
    ```bash
   streamlit run src/model_monitoring.py
   ```

Accede en http://localhost:8501.

---

## Características
-  **EDA**: exploración inicial del dataset.  
-  **Feature Engineering**: preparación, construcción y transformación de variables.  
-  **Modelo Base y Entrenamiento**: modelos de ML con scikit-learn.  
-  **API con FastAPI**: predicciones individuales, por batch o desde archivo csv en tiempo real.  
-  **Informe en Streamlit**: evaluación de métricas y visualización.  
-  **Monitoreo**: seguimiento del drift de datos y desempeño del modelo.  

---

## Autores
- **Nicolás Jaramillo** – Autor principal  
- **Pablo Peralta** – Revisor  

---
