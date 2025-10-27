from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    StratifiedKFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from src.feature_engineering import pipeline_basemodel
from src.cargar_datos import cargar_dataset

# Modelo Heuristico

class HeuristicCreditModel(BaseEstimator, ClassifierMixin):

    """
    Clasificador heurístico simple. Reglas:
      - puntaje_datacredito < puntaje_threshold -> 0 (riesgo)
      - huella_consulta > huellas_threshold -> 0
      - antiguedad_credito_meses > antiguedad_threshold -> 0
      - tipo_credito == tipo_credito_riesgoso -> 0
      - else -> 1 (paga a tiempo)
    """

    def __init__(
        self,
        puntaje_threshold: float = 750,
        huellas_threshold: int = 6,
        antiguedad_threshold: int = 6,
        tipo_credito_riesgoso: int | str = 6,
        target_positive: int | str = 1,
        target_negative: int | str = 0,
    ):
        self.puntaje_threshold = puntaje_threshold
        self.huellas_threshold = huellas_threshold
        self.antiguedad_threshold = antiguedad_threshold
        self.tipo_credito_riesgoso = tipo_credito_riesgoso
        self.target_positive = target_positive
        self.target_negative = target_negative
        self.classes_ = None

    def fit(self, X, y=None):
      if y is not None:
          self.classes_ = np.unique(y)
          # guarda dtype para devolver igual tipo
          self._y_dtype_ = pd.Series(y).dtype
      else:
          self._y_dtype_ = np.int64
      return self

    def predict(self, X):
        y_hat = np.full(len(X), fill_value=self.target_positive)  # ints por defecto
        mask_riesgo = np.zeros(len(X), dtype=bool)

        if "puntaje_datacredito" in X:
            mask_riesgo |= X["puntaje_datacredito"] < self.puntaje_threshold
        if "huella_consulta" in X:
            mask_riesgo |= X["huella_consulta"] > self.huellas_threshold
        if "antiguedad_credito_meses" in X:
            mask_riesgo |= X["antiguedad_credito_meses"] > self.antiguedad_threshold
        if "tipo_credito" in X:
            mask_riesgo |= X["tipo_credito"] == self.tipo_credito_riesgoso

        y_hat[mask_riesgo] = self.target_negative
        return y_hat.astype(self._y_dtype_)


# Funciones para evaluación

def crossval_report(model, X_train: pd.DataFrame, y_train: pd.Series, scoring_metrics=None, cv=5, random_state=42):
    if scoring_metrics is None:
        scoring_metrics = ["accuracy", "precision", "recall", "f1"]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}
    for m in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=m, n_jobs=-1)
        results[m] = scores
        print(f"{m}: mean={scores.mean():.3f}  std={scores.std():.3f}")

    return pd.DataFrame(results)


def plot_learning_curves(estimator, X, y, scoring="recall"):
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        n_jobs=-1,
        scoring=scoring,
        return_times=True,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    fit_times_mean, fit_times_std = fit_times.mean(axis=1), fit_times.std(axis=1)
    score_times_mean, score_times_std = score_times.mean(axis=1), score_times.std(axis=1)

    # Curva de aprendizaje
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Entrenamiento")
    ax.plot(train_sizes, test_mean, "o-", label="Cross-val")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3)
    ax.set_title("Curva de aprendizaje")
    ax.set_xlabel("Ejemplos de entrenamiento")
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Tiempos de fit/score
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 8), sharex=True)
    ax[0].plot(train_sizes, fit_times_mean, "o-")
    ax[0].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.3)
    ax[0].set_ylabel("Tiempo de entrenamiento (s)")
    ax[0].set_title("Escalabilidad")

    ax[1].plot(train_sizes, score_times_mean, "o-")
    ax[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha=0.3)
    ax[1].set_ylabel("Tiempo de evaluacion (s)")
    ax[1].set_xlabel("Ejemplos de entrenamiento")
    plt.tight_layout()
    plt.show()

# Entrenamiento

def run_baseline(data_path: str = "", target_col: str = "Pago_atiempo", test_size: float = 0.2, seed: int = 42):

    df = cargar_dataset()

    # Pipeline
    df_base = pipeline_basemodel.fit_transform(df)

    # X -> Features / y -> Target
    X = df_base.drop(columns=[target_col]) if target_col in df_base.columns else df_base.copy()
    y = df.loc[X.index, target_col]

    # División de los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Entrenar modelo base
    model = HeuristicCreditModel(
        puntaje_threshold=750,
        huellas_threshold=6,
        antiguedad_threshold=6,
        tipo_credito_riesgoso=6,
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    print("\n=== Reporte Modelo Heuristico ===")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Matriz de Confusion - Heuristic")
    plt.tight_layout()
    plt.show()

    # Cross-validation
    print("\n=== Validacion Cruzada (accuracy, precision, recall, f1) ===")
    cv_df = crossval_report(model, X_train, y_train, scoring_metrics=["accuracy", "precision", "recall", "f1"], cv=10)
    print("\nCV resumen:\n", cv_df.describe().round(3))

    # Curva de aprendizaje
    plot_learning_curves(model, X_train, y_train, scoring="recall")

    return {
        "df_base": df_base,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "cv_df": cv_df,
    }


if __name__ == "__main__":
    _ = run_baseline()