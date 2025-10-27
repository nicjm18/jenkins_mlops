import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, ShuffleSplit, learning_curve, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
from src.feature_engineering import pipeline_ml
from src.cargar_datos import cargar_dataset
warnings.filterwarnings('ignore')


# Funciones para validacion cruzada

def crossval_report(model, X_train: pd.DataFrame, y_train: pd.Series, scoring_metrics=None, cv=5, random_state=42):
    """
    Reporte de validación cruzada 
    """
    if scoring_metrics is None:
        # Metricas 
        scoring_metrics = ["roc_auc", "average_precision", "recall", "precision", "f1"]
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}
    
    print(f"\nVALIDACIÓN CRUZADA ({cv} folds)")
    print("="*50)
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=metric, n_jobs=-1)
            results[metric] = scores
            print(f"{metric:20}: {scores.mean():.4f} ± {scores.std():.4f} | min: {scores.min():.4f} | max: {scores.max():.4f}")
        except Exception as e:
            print(f"{metric:20}: Error - {str(e)}")
            results[metric] = np.array([np.nan] * cv)
    
    return pd.DataFrame(results)

def crossval_detailed_metrics(model, X_train, y_train, cv=5, random_state=42):
    """
    Validación cruzada con métricas específicas para cada clase
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    metrics_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Entrenar en el fold
        model_fold = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        model_fold.fit(X_tr, y_tr)
        
        # Predicciones
        y_pred = model_fold.predict(X_val)
        
        # Probabilidades
        if hasattr(model_fold, "predict_proba"):
            y_proba = model_fold.predict_proba(X_val)[:, 1]
        elif hasattr(model_fold, "decision_function"):
            y_proba = model_fold.decision_function(X_val)
        else:
            y_proba = None
        
        # Calcular metricas
        fold_metrics = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision_0': precision_score(y_val, y_pred, pos_label=0, zero_division=0),
            'recall_0': recall_score(y_val, y_pred, pos_label=0, zero_division=0),
            'f1_0': f1_score(y_val, y_pred, pos_label=0, zero_division=0),
            'precision_1': precision_score(y_val, y_pred, pos_label=1, zero_division=0),
            'recall_1': recall_score(y_val, y_pred, pos_label=1, zero_division=0),
            'f1_1': f1_score(y_val, y_pred, pos_label=1, zero_division=0),
        }
        
        if y_proba is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_val, y_proba)
            fold_metrics['pr_auc'] = average_precision_score(y_val, y_proba)
        else:
            fold_metrics['roc_auc'] = np.nan
            fold_metrics['pr_auc'] = np.nan
        
        metrics_per_fold.append(fold_metrics)
    
    cv_results = pd.DataFrame(metrics_per_fold)
    
    # Resumen estadistico
    print(f"\nRESUMEN VALIDACIÓN CRUZADA DETALLADA")
    print("="*60)
    
    key_metrics = ['accuracy', 'roc_auc', 'pr_auc', 'recall_0', 'precision_0', 'f1_0']
    
    for metric in key_metrics:
        if metric in cv_results.columns:
            values = cv_results[metric].dropna()
            if len(values) > 0:
                print(f"{metric:12}: {values.mean():.4f} ± {values.std():.4f}")
    
    return cv_results

def plot_learning_curves(estimator, X, y, scoring="recall", model_name="Model"):
    """
    Curvas de aprendizaje mejoradas con múltiples métricas
    """
    print(f"\nGenerando curvas de aprendizaje para {model_name}...")
    
    # Curva principal
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        scoring=scoring,
        return_times=True,
    )
    
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    fit_times_mean, fit_times_std = fit_times.mean(axis=1), fit_times.std(axis=1)
    score_times_mean, score_times_std = score_times.mean(axis=1), score_times.std(axis=1)
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Análisis de Curvas de Aprendizaje - {model_name}', fontsize=16)
    
    # 1. Curva de aprendizaje principal
    axes[0, 0].plot(train_sizes, train_mean, "o-", label="Entrenamiento", color='blue')
    axes[0, 0].plot(train_sizes, test_mean, "o-", label="Validación", color='red')
    axes[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3, color='blue')
    axes[0, 0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color='red')
    axes[0, 0].set_title(f"Curva de aprendizaje ({scoring})")
    axes[0, 0].set_xlabel("Ejemplos de entrenamiento")
    axes[0, 0].set_ylabel(scoring.upper())
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Tiempo de entrenamiento
    axes[0, 1].plot(train_sizes, fit_times_mean, "o-", color='green')
    axes[0, 1].fill_between(train_sizes, fit_times_mean - fit_times_std, 
                           fit_times_mean + fit_times_std, alpha=0.3, color='green')
    axes[0, 1].set_title("Escalabilidad - Tiempo de entrenamiento")
    axes[0, 1].set_xlabel("Ejemplos de entrenamiento")
    axes[0, 1].set_ylabel("Tiempo (segundos)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Tiempo de predicción
    axes[1, 0].plot(train_sizes, score_times_mean, "o-", color='orange')
    axes[1, 0].fill_between(train_sizes, score_times_mean - score_times_std, 
                           score_times_mean + score_times_std, alpha=0.3, color='orange')
    axes[1, 0].set_title("Tiempo de predicción")
    axes[1, 0].set_xlabel("Ejemplos de entrenamiento")
    axes[1, 0].set_ylabel("Tiempo (segundos)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Gap entre entrenamiento y validación
    gap = train_mean - test_mean
    axes[1, 1].plot(train_sizes, gap, "o-", color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title("Gap Entrenamiento-Validación")
    axes[1, 1].set_xlabel("Ejemplos de entrenamiento")
    axes[1, 1].set_ylabel(f"Diferencia {scoring}")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    final_gap = gap[-1]
    print(f"Gap final: {final_gap:.3f})")


# Evaluacion

def evaluate_credit_model_enhanced(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluación completa con validación cruzada 
    """
    print(f"\nEVALUACIÓN COMPLETA: {model_name}")
    print("="*60)
    
    # Validacion cruzada
    cv_results = crossval_detailed_metrics(model, X_train, y_train, cv=5)
    
    # Entrenamiento final en todo el conjunto
    model.fit(X_train, y_train)
    
    # Predicciones en test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # AGREGAR CLASSIFICATION REPORT AQUÍ
    print(f"\n=== CLASSIFICATION REPORT - {model_name.upper()} ===")
    print(classification_report(y_test, y_test_pred, target_names=['No Pago (0)', 'Pago (1)']))
    
    # Probabilidades
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_train_proba = model.decision_function(X_train)
        y_test_proba = model.decision_function(X_test)
    else:
        y_train_proba = None
        y_test_proba = None
    
    # Metricas finales
    metrics = {
        "model": model_name,
        
        # Métricas de entrenamiento
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision_0": precision_score(y_train, y_train_pred, pos_label=0, zero_division=0),
        "train_recall_0": recall_score(y_train, y_train_pred, pos_label=0, zero_division=0),
        "train_f1_0": f1_score(y_train, y_train_pred, pos_label=0, zero_division=0),
        
        # Métricas de test
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_0": precision_score(y_test, y_test_pred, pos_label=0, zero_division=0),
        "test_recall_0": recall_score(y_test, y_test_pred, pos_label=0, zero_division=0),
        "test_f1_0": f1_score(y_test, y_test_pred, pos_label=0, zero_division=0),
        
        # AUCs
        "train_roc_auc": np.nan,
        "test_roc_auc": np.nan,
        "train_pr_auc": np.nan,
        "test_pr_auc": np.nan,
        
        # Métricas de validación cruzada (promedio)
        "cv_roc_auc_mean": cv_results['roc_auc'].mean(),
        "cv_roc_auc_std": cv_results['roc_auc'].std(),
        "cv_recall_0_mean": cv_results['recall_0'].mean(),
        "cv_recall_0_std": cv_results['recall_0'].std(),
        "cv_precision_0_mean": cv_results['precision_0'].mean(),
        "cv_precision_0_std": cv_results['precision_0'].std(),
    }
    
    # Calcular AUCs
    if y_train_proba is not None:
        metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)
        metrics["train_pr_auc"] = average_precision_score(y_train, y_train_proba)
    
    if y_test_proba is not None:
        metrics["test_roc_auc"] = roc_auc_score(y_test, y_test_proba)
        metrics["test_pr_auc"] = average_precision_score(y_test, y_test_proba)
    
    # Matriz de confusión y análisis
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"\nMÉTRICAS CLAVE:")
    print(f"• ROC-AUC Test: {metrics['test_roc_auc']:.4f}")
    print(f"• ROC-AUC CV: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"• Recall-0 Test: {metrics['test_recall_0']:.4f}")
    print(f"• Recall-0 CV: {metrics['cv_recall_0_mean']:.4f} ± {metrics['cv_recall_0_std']:.4f}")
    
    # Análisis de estabilidad
    roc_auc_diff = abs(metrics['test_roc_auc'] - metrics['cv_roc_auc_mean'])
    recall_0_diff = abs(metrics['test_recall_0'] - metrics['cv_recall_0_mean'])
    
    print(f"\nANÁLISIS DE ESTABILIDAD:")
    print(f"• Diferencia ROC-AUC (test vs CV): {roc_auc_diff:.4f}")
    print(f"• Diferencia Recall-0 (test vs CV): {recall_0_diff:.4f}")
    
    if roc_auc_diff < 0.05 and recall_0_diff < 0.1:
        print("Modelo estable y confiable")
    elif roc_auc_diff < 0.1 and recall_0_diff < 0.15:
        print("Modelo moderadamente estable")
    else:
        print("Modelo inestable, revisar overfitting")
    
    return metrics, cv_results

# Entrenamiento

def run_training_credit_complete(
    data_path: str = "BD_creditos.xlsx",
    target_col: str = "Pago_atiempo",
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: str = "models",
    primary_metric: str = "cv_roc_auc_mean",
    secondary_metric: str = "cv_recall_0_mean",
    plot_learning_curves_flag: bool = True
):
    """
    Sistema completo de entrenamiento
    """
    print("INICIANDO ENTRENAMIENTO")
    print("="*70)
    
    # Cargar datos y dividir
    df = cargar_dataset()
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Preprocesar
    X_train_ready = pipeline_ml.fit_transform(X_train_raw)
    X_test_ready = pipeline_ml.transform(X_test_raw)
    
    y_train_ready = y_train.loc[X_train_ready.index].astype(int)
    y_test_ready = y_test.loc[X_test_ready.index].astype(int)

    print(f"INFORMACIÓN DEL DATASET:")
    print(f"• Entrenamiento: {X_train_ready.shape[0]} muestras, {X_train_ready.shape[1]} features")
    print(f"• Test: {X_test_ready.shape[0]} muestras")
    print(f"• Distribución clase 0 (no pago): {(y_train_ready == 0).mean():.1%}")
    print(f"• Distribución clase 1 (pago): {(y_train_ready == 1).mean():.1%}")

    # Configurar modelos
    pos = int((y_train_ready == 1).sum())
    neg = int((y_train_ready == 0).sum())
    scale_pos_weight = (neg / max(pos, 1))

    models = {
        "logistic": LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced", random_state=random_state),
        "linear_svc": LinearSVC(C=1.0, max_iter=5000, dual=False, class_weight="balanced", random_state=random_state),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", random_state=random_state,
            scale_pos_weight=scale_pos_weight, verbosity=0
        ),
        "bagging": BaggingClassifier(
            n_estimators=100,
            random_state=42,
            max_samples=0.8,
            max_features=0.8
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        ),
    }

    # Entrenar y evaluar todos los modelos
    results = []
    trained = {}
    cv_results_all = {}
    
    for name, clf in models.items():
        print(f"\n{'='*50}")
        print(f"EVALUANDO: {name.upper()}")
        print(f"{'='*50}")
        
        # Evaluación completa con CV
        metrics, cv_detailed = evaluate_credit_model_enhanced(
            clf, X_train_ready, y_train_ready, 
            X_test_ready, y_test_ready, name
        )
        
        results.append(metrics)
        trained[name] = clf
        cv_results_all[name] = cv_detailed
        
        # Curvas de aprendizaje para modelos seleccionados
        if plot_learning_curves_flag and name in ["naive_bayes", "linear_svc", "logistic", "decision_tree", "random_forest", "xgboost", "bagging", "adaboost"]:
            plot_learning_curves(clf, X_train_ready, y_train_ready, 
                                scoring="recall", model_name=name)
    
    # Ranking de modelos
    leaderboard = pd.DataFrame(results)
    leaderboard_ranked = rank_credit_models_enhanced(
        leaderboard, primary_metric, secondary_metric
    )
    
    # Modelo ganador
    best_name = leaderboard_ranked.loc[0, "model"]
    best_clf = trained[best_name]
    
    print(f"\nMODELO GANADOR: {best_name.upper()}")
    print(f"Score compuesto: {leaderboard_ranked.loc[0, 'credit_score']:.4f}")
    
    # CLASSIFICATION REPORT FINAL PARA EL MEJOR MODELO
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT FINAL - MEJOR MODELO: {best_name.upper()}")
    print(f"{'='*60}")
    y_best_pred = best_clf.predict(X_test_ready)
    print(classification_report(y_test_ready, y_best_pred, target_names=['No Pago (0)', 'Pago (1)']))

    # Crear directorio si no existe
    os.makedirs(out_dir, exist_ok=True)

    #Guardar todos los modelos entrenados
    for name, model in trained.items():
        model_path = os.path.join(out_dir, f"model_{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Guardar el mejor modelo
    model_filename = f"{out_dir}/best_model_{best_name}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(best_clf, f)

    #Guardar el pipeline
    with open('models/pipeline_ml.pkl', 'wb') as f:
        pickle.dump(pipeline_ml, f)

    print(f"\nModelo guardado en: {model_filename}")
    
    return {
        "leaderboard": leaderboard_ranked,
        "best_name": best_name,
        "best_model": best_clf,
        "cv_results": cv_results_all,
        "trained_models": trained,
        "X_train": X_train_ready,
        "X_test": X_test_ready,
        "y_train": y_train_ready,
        "y_test": y_test_ready
    }

def rank_credit_models_enhanced(results_df, primary_metric="cv_roc_auc_mean", secondary_metric="cv_recall_0_mean"):
  
    results_df = results_df.copy()
    
    # Normalizar métricas 
    primary_norm = (results_df[primary_metric] - results_df[primary_metric].min()) / \
                   (results_df[primary_metric].max() - results_df[primary_metric].min())
    
    secondary_norm = (results_df[secondary_metric] - results_df[secondary_metric].min()) / \
                     (results_df[secondary_metric].max() - results_df[secondary_metric].min())
    
    # Penalizar si tienen alta desviacion estándar)
    std_penalty_roc = results_df.get('cv_roc_auc_std', 0) * 2
    std_penalty_recall = results_df.get('cv_recall_0_std', 0) * 2
    
    # Score compuesto
    results_df['credit_score'] = (0.6 * primary_norm + 0.3 * secondary_norm 
                                 - 0.05 * std_penalty_roc - 0.05 * std_penalty_recall)
    
    # Aseguramos que el score no sea negativo
    results_df['credit_score'] = np.maximum(results_df['credit_score'], 0)
    
    ranked = results_df.sort_values('credit_score', ascending=False).reset_index(drop=True)
    
    print(f"\nRANKING FINAL DE MODELOS")
    print(f"Score = 60% {primary_metric} + 30% {secondary_metric} - 10% penalización por inestabilidad")
    print("="*80)
    
    for i, row in ranked.iterrows():
        print(f"{i+1}. {row['model']:15} | Score: {row['credit_score']:.4f} | "
              f"ROC-AUC: {row[primary_metric]:.3f}±{row.get('cv_roc_auc_std', 0):.3f} | "
              f"Recall-0: {row[secondary_metric]:.3f}±{row.get('cv_recall_0_std', 0):.3f}")
    
    return ranked


if __name__ == "__main__":
    
    results = run_training_credit_complete(
        primary_metric="test_f1_0",#"cv_recall_0_mean",
        secondary_metric="test_accuracy",#"cv_roc_auc_mean",
        plot_learning_curves_flag=True
    )
    
    print(f"\nENTRENAMIENTO COMPLETADO")
    print(f"Mejor modelo: {results['best_name']}")
    print(f"Resultados disponibles en: results['leaderboard']")