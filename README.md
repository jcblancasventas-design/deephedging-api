# Deep Hedging API — Sector Minero Peruano

API REST para simulaciones de cobertura dinámica en acciones mineras peruanas, basada en arquitectura Actor-Crítico con aprendizaje por refuerzo (A2C) y Block Bootstrap.

Desarrollada por **Juan Carlos Blancas Garcia**  
Registrada en **INDECOPI** — Lima, Perú

---

## ¿Qué hace esta API?

Dado un archivo CSV con precios históricos de activos mineros, la API:

1. Calcula los retornos diarios
2. Genera trayectorias sintéticas mediante Block Bootstrap
3. Evalúa la política de cobertura óptima con un agente Actor-Crítico
4. Devuelve una recomendación: cobertura **activa** o **inacción**

---

## Activos analizados

| Activo | Descripción |
|--------|-------------|
| `SCCO` | Southern Copper Corporation |
| `MINSUR` | Minsur S.A. |
| `COPPER` | Cobre (commodity, USD/ton) |

---

## Base teórica

El modelo se fundamenta en:

- **Lema de Itô** — base del cálculo estocástico aplicado a precios de activos
- **Ecuación de Hamilton-Jacobi-Bellman (HJB)** — control óptimo de la política de cobertura
- **Deep Hedging** (Buehler et al., 2019) — cobertura dinámica con redes neuronales
- **Block Bootstrap** — simulación de trayectorias preservando dependencia temporal

---

## Endpoints

### `GET /`
Información general de la API.

### `GET /health`
Verifica que el servicio esté activo.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "API funcionando correctamente"
}
```

### `POST /simular`
Ejecuta el análisis de cobertura sobre un archivo CSV.

**Autenticación:** requiere header `X-API-Key`

**Formato del CSV:**
```
date,SCCO,MINSUR,COPPER
2015-01-02,100.5,5.1,8520.0
2015-01-05,101.2,5.0,8490.0
...
```

**Respuesta:**
```json
{
  "posicion_promedio_abs": 0.0312,
  "proporcion_mayor_01": 0.021,
  "posicion_maxima": 0.4821,
  "posicion_minima": -0.4654,
  "conclusion": "INACCION",
  "recomendacion": "No implementar Deep Hedging. Usar forwards USD/PEN y futuros de cobre LME.",
  "registros_analizados": 2500,
  "activos": ["SCCO", "MINSUR", "COPPER"]
}
```

---

## Ejemplo de uso

### Con Python

```python
import requests

url = "https://deephedging-api.onrender.com/simular"

headers = {
    "X-API-Key": "tu_api_key_aqui"
}

with open("precios_mineras.csv", "rb") as f:
    response = requests.post(url, headers=headers, files={"file": f})

print(response.json())
```

### Con curl

```bash
curl -X POST https://deephedging-api.onrender.com/simular \
  -H "X-API-Key: tu_api_key_aqui" \
  -F "file=@precios_mineras.csv"
```

---

## Instalación local

```bash
git clone https://github.com/jcblancasventas-design/deephedging-api.git
cd deephedging-api
pip install -r requirements.txt
python api_flask.py
```

La API estará disponible en `http://localhost:10000`

---

## Requisitos

Ver `requirements.txt`. Principales dependencias:

- Python 3.9+
- Flask
- PyTorch
- pandas
- numpy

---

## Nota sobre el plan de despliegue

La API está desplegada en **Render** (plan gratuito). El servicio puede tardar
hasta 60 segundos en responder si estuvo inactivo. Esto es normal y no indica
un error.

---

## Limitaciones

- Los resultados se limitan a los activos analizados: SCCO, MINSUR y COPPER
- No son extensibles a otros sectores de la BVL sin reentrenamiento
- El modelo utiliza datos sintéticos (GBM + Block Bootstrap)
- No constituye asesoría financiera

---

## Licencia y propiedad intelectual

Software registrado en **INDECOPI** — Dirección de Derechos de Autor  
© 2026 Juan Carlos Blancas Garcia. Todos los derechos reservados.

Para consultas de licenciamiento o colaboración académica, contactar al autor.
