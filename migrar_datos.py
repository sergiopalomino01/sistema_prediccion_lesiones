"""Entrena el modelo experimental sin fabricar ni sobrescribir etiquetas.

Riesgo_Lesion debe proceder de una fuente previamente documentada. Este script no
convierte el resultado en un modelo clínicamente validado.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "Peso(kg)", "Talla (m)", "IMC", "SUM 3 PL", "SUM 6 PL",
    "Perimetro de cintura", "Perimetro Brazo relajado", "Perimetro de muslo",
    "Perimetro de pantorrilla", "Diametro de Humero", "Diametro de Femur",
    "Indice_Sobrecarga_Muscular", "Horas_Sueno", "Carga_Entrenamiento",
]
TARGET = "Riesgo_Lesion"


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar clasificador experimental")
    parser.add_argument(
        "archivo", nargs="?", type=Path,
        default=BASE_DIR / "BASE_DATOS_120_JUGADORES_CON_RIESGO.xlsx",
    )
    parser.add_argument("--salida", type=Path, default=BASE_DIR / "modelo_lesiones.pkl")
    return parser.parse_args()


def cargar_dataset(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "No se generan variables o etiquetas sintéticas. Faltan: " + ", ".join(missing)
        )
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or df[TARGET].isna().any():
        raise ValueError("El dataset contiene valores vacíos o no numéricos")
    if df[TARGET].nunique() < 2:
        raise ValueError("La variable objetivo necesita al menos dos clases")
    return X, df[TARGET].astype(str)


def entrenar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_split=2,
        max_features="sqrt", class_weight="balanced", random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"Evaluación sobre {len(y_test)} registros reservados:")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    return model


if __name__ == "__main__":
    args = parse_args()
    try:
        X_data, y_data = cargar_dataset(args.archivo.resolve())
        modelo = entrenar(X_data, y_data)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, args.salida)
        print(f"Modelo guardado en {args.salida.resolve()}")
        print("AVISO: resultado experimental; requiere validación con lesiones reales y datos externos.")
    except Exception as exc:
        raise SystemExit(f"Error de entrenamiento: {exc}") from exc
