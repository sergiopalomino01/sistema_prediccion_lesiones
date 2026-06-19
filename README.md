# Sistema experimental de evaluación de riesgo de lesiones

Prototipo con FastAPI, PostgreSQL, una interfaz HTML y un clasificador Random Forest.

> **Advertencia:** el modelo incluido fue entrenado con categorías de riesgo derivadas de
> reglas sobre el propio dataset. No predice lesiones observadas y no constituye un
> diagnóstico médico. No debe decidir por sí solo si un deportista entrena o compite.

## Instalación local

Requiere Python 3.11+ y PostgreSQL.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env`; la aplicación lo carga automáticamente. También puedes sobrescribir
cualquier valor en la sesión de PowerShell:

```powershell
$env:DB_PASSWORD = "tu_clave"
$env:DB_NAME = "sistema_lesiones"
python -m uvicorn main:app --reload
```

Abre <http://127.0.0.1:8000>. La documentación de la API queda en `/docs` y el estado
en `/api/salud`.

Si configuras `API_KEY`, los endpoints de datos exigirán la cabecera `X-API-Key`. Para
probar la interfaz localmente puedes establecerla para la pestaña actual desde la consola
del navegador:

```javascript
sessionStorage.setItem('apiKey', 'la-misma-clave')
```

## Importación

El importador añade registros por defecto. Solo borra la base si se indica explícitamente:

```powershell
python cargar_excel_a_postgres.py .\datos.xlsx
python cargar_excel_a_postgres.py .\datos.xlsx --reemplazar
```

## Entrenamiento experimental

```powershell
python migrar_datos.py .\datos.xlsx --salida .\modelo_lesiones.pkl
```

El script exige que todas las variables y `Riesgo_Lesion` ya existan: no fabrica datos ni
sobrescribe el Excel. Para un modelo válido se necesitan lesiones observadas, ventanas
temporales claras, separación por jugador/temporada y validación externa.

## Pruebas

```powershell
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## Privacidad

Los archivos de datos contienen información personal y posiblemente datos de menores.
El `.gitignore` excluye hojas de cálculo, bases locales y modelos; aun así, aplica control
de acceso, cifrado, copias de seguridad y la normativa correspondiente antes de desplegar.
