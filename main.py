import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
import psycopg2
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from dotenv import load_dotenv
except ImportError:  # Permite diagnósticos antes de instalar requirements.txt.
    load_dotenv = None

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "modelo_lesiones.pkl"))
API_KEY = os.getenv("API_KEY", "").strip()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "sistema_lesiones"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", "5432")),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
}

EXPECTED_FEATURES = [
    "Peso(kg)", "Talla (m)", "IMC", "SUM 3 PL", "SUM 6 PL",
    "Perimetro de cintura", "Perimetro Brazo relajado", "Perimetro de muslo",
    "Perimetro de pantorrilla", "Diametro de Humero", "Diametro de Femur",
    "Indice_Sobrecarga_Muscular", "Horas_Sueno", "Carga_Entrenamiento",
]


def cargar_modelo():
    if not MODEL_PATH.is_file():
        logger.error("No se encontró el modelo en %s", MODEL_PATH)
        return None
    try:
        model = joblib.load(MODEL_PATH)
        features = list(getattr(model, "feature_names_in_", []))
        if features and features != EXPECTED_FEATURES:
            raise ValueError("Las variables del modelo no coinciden con las esperadas")
        return model
    except Exception:
        logger.exception("No se pudo cargar el modelo")
        return None


modelo_inteligente = cargar_modelo()


def conectar_db():
    return psycopg2.connect(**DB_CONFIG)


def inicializar_db() -> None:
    with conectar_db() as conexion, conexion.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jugadores (
                id SERIAL PRIMARY KEY,
                nombre VARCHAR(150) NOT NULL,
                edad INTEGER CHECK (edad BETWEEN 5 AND 100),
                posicion VARCHAR(50)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluaciones (
                id SERIAL PRIMARY KEY,
                jugador_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
                fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                peso REAL, talla REAL, imc REAL, sum_3_pl REAL, sum_6_pl REAL,
                perimetro_cintura REAL, perimetro_brazo REAL, perimetro_muslo REAL,
                perimetro_pantorrilla REAL, diametro_humero REAL, diametro_femur REAL,
                horas_sueno REAL, carga_entrenamiento REAL, riesgo_predicho VARCHAR(50)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluaciones_jugador_fecha "
            "ON evaluaciones (jugador_id, fecha DESC)"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        inicializar_db()
        logger.info("Base de datos verificada")
    except Exception:
        logger.exception("La base de datos no está disponible al iniciar")
    yield


app = FastAPI(
    title="Sistema de evaluación de riesgo de lesiones",
    description=(
        "Prototipo experimental. Sus resultados no constituyen un diagnóstico médico "
        "ni deben utilizarse de forma aislada para tomar decisiones clínicas."
    ),
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def verificar_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and (not x_api_key or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autorizado")


protegido = Depends(verificar_api_key)


class JugadorNuevo(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    nombre: str = Field(min_length=2, max_length=150)
    edad: int = Field(ge=5, le=100)
    posicion: Literal["Arquero", "Defensa", "Mediocampista", "Delantero"]

    @field_validator("nombre")
    @classmethod
    def validar_nombre(cls, value: str) -> str:
        if not any(char.isalpha() for char in value):
            raise ValueError("El nombre debe contener letras")
        return " ".join(value.split())


class EvaluacionNueva(BaseModel):
    peso_kg: float = Field(gt=20, le=200)
    talla_m: float = Field(ge=0.8, le=2.5)
    imc: float | None = Field(default=None, ge=8, le=80)
    sum_3_pl: float = Field(ge=0, le=300)
    sum_6_pl: float = Field(ge=0, le=500)
    perimetro_cintura: float = Field(gt=20, le=250)
    perimetro_brazo_relajado: float = Field(gt=5, le=100)
    perimetro_muslo: float = Field(gt=5, le=150)
    perimetro_pantorrilla: float = Field(gt=5, le=100)
    diametro_humero: float = Field(gt=1, le=30)
    diametro_femur: float = Field(gt=1, le=30)
    horas_sueno: float = Field(ge=0, le=24)
    carga_entrenamiento: float = Field(ge=0, le=5000)

    @model_validator(mode="after")
    def calcular_imc(self):
        self.imc = round(self.peso_kg / (self.talla_m ** 2), 2)
        return self


@app.get("/", response_class=HTMLResponse)
def cargar_interfaz():
    ruta_html = BASE_DIR / "index.html"
    if ruta_html.is_file():
        return ruta_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Interfaz no encontrada")


@app.get("/api/salud")
def salud():
    db_ok = False
    try:
        with conectar_db() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            db_ok = cursor.fetchone()[0] == 1
    except Exception:
        logger.warning("Comprobación de base de datos fallida", exc_info=True)
    return {"estado": "ok" if db_ok and modelo_inteligente is not None else "degradado",
            "base_datos": db_ok, "modelo": modelo_inteligente is not None}


@app.post("/api/jugadores", dependencies=[protegido], status_code=201)
def registrar_jugador(j: JugadorNuevo):
    try:
        with conectar_db() as conn, conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO jugadores (nombre, edad, posicion) VALUES (%s, %s, %s) RETURNING id",
                (j.nombre, j.edad, j.posicion),
            )
            jugador_id = cursor.fetchone()[0]
        return {"id": jugador_id, "mensaje": "Jugador registrado con éxito"}
    except Exception:
        logger.exception("Error al registrar jugador")
        raise HTTPException(status_code=503, detail="No se pudo registrar el jugador")


@app.get("/api/jugadores", dependencies=[protegido])
def listar_jugadores():
    try:
        with conectar_db() as conn, conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT j.id, j.nombre, j.edad, j.posicion,
                    e.riesgo_predicho, e.peso, e.talla, e.imc, e.sum_3_pl, e.sum_6_pl,
                    e.perimetro_cintura, e.perimetro_brazo, e.perimetro_muslo,
                    e.perimetro_pantorrilla, e.diametro_humero, e.diametro_femur,
                    e.horas_sueno, e.carga_entrenamiento
                FROM jugadores j
                LEFT JOIN LATERAL (
                    SELECT * FROM evaluaciones
                    WHERE jugador_id = j.id
                    ORDER BY fecha DESC, id DESC LIMIT 1
                ) e ON TRUE
                ORDER BY j.id ASC
            """)
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        logger.exception("Error al listar jugadores")
        raise HTTPException(status_code=503, detail="No se pudo consultar la base de datos")


@app.post("/api/evaluar/{jugador_id}", dependencies=[protegido])
def evaluar_y_guardar(jugador_id: int, ev: EvaluacionNueva):
    if modelo_inteligente is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    indice_sobrecarga = ev.sum_6_pl / ev.perimetro_muslo
    valores = [
        ev.peso_kg, ev.talla_m, ev.imc, ev.sum_3_pl, ev.sum_6_pl,
        ev.perimetro_cintura, ev.perimetro_brazo_relajado, ev.perimetro_muslo,
        ev.perimetro_pantorrilla, ev.diametro_humero, ev.diametro_femur,
        indice_sobrecarga, ev.horas_sueno, ev.carga_entrenamiento,
    ]
    try:
        prediccion = str(modelo_inteligente.predict(pd.DataFrame([valores], columns=EXPECTED_FEATURES))[0])
        with conectar_db() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM jugadores WHERE id = %s", (jugador_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Jugador no encontrado")
            cursor.execute("""
                INSERT INTO evaluaciones (
                    jugador_id, peso, talla, imc, sum_3_pl, sum_6_pl, perimetro_cintura,
                    perimetro_brazo, perimetro_muslo, perimetro_pantorrilla, diametro_humero,
                    diametro_femur, horas_sueno, carga_entrenamiento, riesgo_predicho
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                jugador_id, ev.peso_kg, ev.talla_m, ev.imc, ev.sum_3_pl, ev.sum_6_pl,
                ev.perimetro_cintura, ev.perimetro_brazo_relajado, ev.perimetro_muslo,
                ev.perimetro_pantorrilla, ev.diametro_humero, ev.diametro_femur,
                ev.horas_sueno, ev.carga_entrenamiento, prediccion,
            ))
        return {"riesgo": prediccion, "imc_calculado": ev.imc,
                "advertencia": "Resultado experimental; no constituye diagnóstico médico."}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error al evaluar al jugador %s", jugador_id)
        raise HTTPException(status_code=503, detail="No se pudo completar la evaluación")
