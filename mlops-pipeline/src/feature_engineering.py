import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

class ToDF(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ct_ = None

    def fit(self, X, y=None):
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", ohe, self.categorical_features),
            ]
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.ct_.transform(X)

        try:
            feat_names = self.ct_.get_feature_names_out()
        except AttributeError:
            feat_names = []
            for name, trans, cols in self.ct_.transformers_:
                if name == "remainder" and trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    feat_names.extend(trans.get_feature_names_out(cols))
                else:
                    feat_names.extend(cols)
        return pd.DataFrame(Xt, columns=feat_names, index=X.index)


class ColumnasNulos(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")

class Imputacion(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.median_saldo_principal = X["saldo_principal"].median()
        self.median_saldo_mora = X["saldo_mora"].median()
        self.mean_puntaje = X["puntaje_datacredito"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["saldo_principal"] = X["saldo_principal"].fillna(self.median_saldo_principal)
        X["saldo_mora"] = X["saldo_mora"].fillna(self.median_saldo_mora)
        X["puntaje_datacredito"] = X["puntaje_datacredito"].fillna(self.mean_puntaje)
        return X

class Outliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X["edad_cliente"] < 100].copy()

class NuevasVariables(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Crear mes del préstamo
        X["mes_prestamo"] = pd.to_datetime(X["fecha_prestamo"]).dt.month.astype("category")
        # Antigüedad en meses
        X["antiguedad_credito_meses"] = ((pd.to_datetime("today") - pd.to_datetime(X["fecha_prestamo"])).dt.days // 30).astype(int)
        return X

class ToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X

class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")

class EliminarCategorias(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, cats_to_drop):
        self.target_col = target_col
        self.cats_to_drop = cats_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[~X[self.target_col].isin(self.cats_to_drop)].copy()

class AgregarTarget(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="Pago_atiempo"):
        self.target_col = target_col
        self._y = None

    def fit(self, X, y=None):
        self._y = pd.Series(y, index=getattr(X, "index", None), name=self.target_col) if y is not None else None
        return self

    def transform(self, X):
        if self._y is None:
            return X
        X = X.copy()
        X[self.target_col] = self._y.reindex(X.index)
        return X


# 2. Pipeline Base

pipeline_basemodel = Pipeline(steps=[
    ("eliminar_nulos", ColumnasNulos(cols_to_drop=["tendencia_ingresos", "promedio_ingresos_datacredito"])),
    ("imputacion", Imputacion()),
    ("outliers", Outliers()),
    ("nuevas_variables", NuevasVariables()),
    ("to_category", ToCategory(cols=["tipo_laboral", "tipo_credito"])),
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop=["fecha_prestamo", "creditos_sectorCooperativo",
                                                    "creditos_sectorReal", "puntaje", "saldo_mora_codeudor",
                                                    "saldo_total"])),
    ("eliminar_categorias", EliminarCategorias(target_col="tipo_credito", cats_to_drop=["7", "68"]))
])


# 3. Pipeline ML

# Definir columnas numéricas y categóricas
numeric_features = ["saldo_principal", "saldo_mora", "puntaje_datacredito", "edad_cliente", "antiguedad_credito_meses", "capital_prestado", "plazo_meses",
                    "salario_cliente", "total_otros_prestamos", "cuota_pactada", "huella_consulta", "cant_creditosvigentes", "creditos_sectorFinanciero"]
categorical_features = ["mes_prestamo", "tipo_credito", "tipo_laboral"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

pipeline_ml = Pipeline(steps=[
    ("basemodel", pipeline_basemodel),
    ("preprocessor", ToDF(numeric_features=numeric_features, categorical_features=categorical_features)),
    #("agregar_target", AgregarTarget(target_col="Pago_atiempo"))
])