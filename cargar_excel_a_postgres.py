"""Importa jugadores y evaluaciones desde un archivo explícito.

Por seguridad, los datos existentes solo se eliminan al usar --reemplazar.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg2

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "NOMBRES", "APELLIDOS", "Posición", "Edad", "Peso(kg)", "Talla (m)", "IMC",
    "SUM 3 PL", "SUM 6 PL", "Perimetro de cintura", "Perimetro Brazo relajado",
    "Perimetro de muslo", "Perimetro de pantorrilla", "Diametro de Humero",
    "Diametro de Femur", "Horas_Sueno", "Carga_Entrenamiento", "Riesgo_Lesion",
}


def argumentos():
    parser = argparse.ArgumentParser(description="Importar evaluaciones a PostgreSQL")
    parser.add_argument(
        "archivo", nargs="?", type=Path,
        default=BASE_DIR / "BASE_DATOS_120_JUGADORES_CON_RIESGO.xlsx",
        help="Ruta del archivo .xlsx o .csv",
    )
    parser.add_argument(
        "--reemplazar", action="store_true",
        help="Borra jugadores y evaluaciones existentes antes de importar",
    )
    return parser.parse_args()


def leer_datos(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError("El archivo debe ser .xlsx o .csv")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Hay valores vacíos en columnas obligatorias")
    return df


def conectar():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "sistema_lesiones"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        port=int(os.getenv("DB_PORT", "5432")),
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
    )


def importar(df: pd.DataFrame, reemplazar: bool) -> int:
    with conectar() as conn, conn.cursor() as cursor:
        if reemplazar:
            cursor.execute("TRUNCATE TABLE evaluaciones, jugadores RESTART IDENTITY CASCADE")

        for _, row in df.iterrows():
            nombre = f"{str(row['NOMBRES']).strip().title()} {str(row['APELLIDOS']).strip().title()}"
            cursor.execute(
                "INSERT INTO jugadores (nombre, edad, posicion) VALUES (%s, %s, %s) RETURNING id",
                (nombre, int(row["Edad"]), str(row["Posición"]).strip().capitalize()),
            )
            jugador_id = cursor.fetchone()[0]
            cursor.execute("""
                INSERT INTO evaluaciones (
                    jugador_id, peso, talla, imc, sum_3_pl, sum_6_pl, perimetro_cintura,
                    perimetro_brazo, perimetro_muslo, perimetro_pantorrilla, diametro_humero,
                    diametro_femur, horas_sueno, carga_entrenamiento, riesgo_predicho
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                jugador_id, float(row["Peso(kg)"]), float(row["Talla (m)"]), float(row["IMC"]),
                float(row["SUM 3 PL"]), float(row["SUM 6 PL"]),
                float(row["Perimetro de cintura"]), float(row["Perimetro Brazo relajado"]),
                float(row["Perimetro de muslo"]), float(row["Perimetro de pantorrilla"]),
                float(row["Diametro de Humero"]), float(row["Diametro de Femur"]),
                float(row["Horas_Sueno"]), float(row["Carga_Entrenamiento"]),
                str(row["Riesgo_Lesion"]),
            ))
    return len(df)


if __name__ == "__main__":
    args = argumentos()
    try:
        total = importar(leer_datos(args.archivo.resolve()), args.reemplazar)
        print(f"Importación completada: {total} jugadores")
    except Exception as exc:
        raise SystemExit(f"Error de importación: {exc}") from exc
