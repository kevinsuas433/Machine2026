
# Importamos las herramientas necesarias para crear la API.
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Obtenemos la ubicación de la carpeta donde está este archivo.
CARPETA_API = Path(__file__).resolve().parent

# Indicamos dónde se encuentra la configuración generada por el notebook.
RUTA_MODELO = CARPETA_API / "modelo_mp.json"


# Comprobamos que el archivo del modelo exista.
if not RUTA_MODELO.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo del modelo: {RUTA_MODELO}. "
        "Ejecuta primero el notebook para generarlo."
    )


# Leemos la configuración guardada por el notebook.
with RUTA_MODELO.open("r", encoding="utf-8") as archivo:
    modelo = json.load(archivo)


# Recuperamos los parámetros que necesita la neurona.
UMBRAL = int(modelo["threshold"])
CARACTERISTICAS = modelo["feature_names"]
PUNTOS_CORTE = modelo["puntos_corte"]


# Creamos la aplicación FastAPI.
app = FastAPI(
    title="API de la Neurona MP",
    description=(
        "API académica para probar la neurona de McCulloch y Pitts "
        "con el conjunto Breast Cancer Wisconsin."
    ),
    version="1.0.0"
)


# Permitimos solicitudes desde la futura página HTML.
# Más adelante podemos restringir los orígenes permitidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Definimos la estructura de los datos que recibirá la API.
class SolicitudPrediccion(BaseModel):
    caracteristicas: dict[str, float] = Field(
        ...,
        description="Las 30 características numéricas de una muestra."
    )


# Ruta principal para comprobar que la API funciona.
@app.get("/")
def inicio():
    return {
        "mensaje": "API de la Neurona MP funcionando correctamente",
        "documentacion": "/docs"
    }


# Ruta para comprobar que el modelo fue cargado.
@app.get("/salud")
def comprobar_estado():
    return {
        "estado": "correcto",
        "modelo_cargado": True,
        "numero_caracteristicas": len(CARACTERISTICAS),
        "umbral": UMBRAL
    }


# Ruta que recibe las características y devuelve la predicción.
@app.post("/predecir")
def predecir(solicitud: SolicitudPrediccion):

    datos = solicitud.caracteristicas

    # Comprobamos que se hayan enviado todas las características.
    faltantes = [
        nombre for nombre in CARACTERISTICAS
        if nombre not in datos
    ]

    # Rechazamos características que no pertenecen al modelo.
    adicionales = [
        nombre for nombre in datos
        if nombre not in CARACTERISTICAS
    ]

    if faltantes or adicionales:
        raise HTTPException(
            status_code=422,
            detail={
                "mensaje": "Las características recibidas no coinciden "
                           "con las esperadas por el modelo.",
                "faltantes": faltantes,
                "adicionales": adicionales
            }
        )

    entradas_binarias = []

    # Aplicamos los mismos puntos de corte utilizados en el notebook.
    for nombre in CARACTERISTICAS:
        valor = datos[nombre]

        # Rechazamos valores que no sean números finitos.
        # La validación de Pydantic ya convierte los valores numéricos.
        if valor != valor or abs(valor) == float("inf"):
            raise HTTPException(
                status_code=422,
                detail=f"El valor de '{nombre}' debe ser un número finito."
            )

        # Conservamos la regla de transformación del notebook:
        # valor menor o igual al punto de corte -> 1; si no -> 0.
        entrada = int(valor <= PUNTOS_CORTE[nombre])
        entradas_binarias.append(entrada)

    # La neurona suma las entradas binarias y compara con el umbral.
    suma_entradas = sum(entradas_binarias)
    prediccion = int(suma_entradas >= UMBRAL)

    # Interpretamos la salida usando las etiquetas de scikit-learn.
    clase = "benigno" if prediccion == 1 else "maligno"

    return {
        "prediccion": prediccion,
        "clase": clase,
        "suma_entradas": suma_entradas,
        "umbral": UMBRAL,
        "nota": (
            "Resultado experimental con fines educativos. "
            "No constituye un diagnóstico médico."
        )
    }