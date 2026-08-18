# TP — RAG sobre Legislación Argentina

Sistema de preguntas y respuestas (Q&A) sobre un corpus de **legislación argentina**, construido con **RAG (Retrieval-Augmented Generation)**. El asistente responde consultas buscando primero la información en los documentos oficiales cargados y generando luego la respuesta con un modelo de lenguaje (Gemini), en lugar de responder "de memoria".

> **Opción del TP:** A — RAG
> **Corpus:** leyes y normas publicadas en fuentes oficiales (InfoLEG / Boletín Oficial).

---

## 1. Fundamentos teóricos

**RAG (Retrieval-Augmented Generation)** combina dos etapas:

1. **Retrieval (recuperación):** ante una pregunta, se buscan en una base de documentos los fragmentos más relevantes.
2. **Generation (generación):** esos fragmentos se le pasan al modelo de lenguaje como *contexto*, y el modelo redacta la respuesta basándose en ellos.

¿Por qué RAG? Un LLM solo "sabe" lo que vio durante su entrenamiento y puede **alucinar** (inventar datos). RAG resuelve dos problemas:

- **Conocimiento específico:** permite responder sobre documentos que el modelo nunca vio (legislación puntual, documentos internos, etc.).
- **Menos alucinaciones:** al obligar al modelo a responder con un texto concreto, las respuestas quedan ancladas a una fuente verificable.

**Analogía:** es la diferencia entre un examen *de memoria* y uno *a libro abierto*. Con RAG, el modelo "abre el libro" (los documentos), busca la parte pertinente y recién ahí responde.

### Componentes clave

| Componente | Rol |
|------------|-----|
| **Chunking** | Divide los documentos largos en fragmentos ("chunks") con solapamiento. |
| **Embeddings** | Convierten cada chunk en un vector numérico que representa su significado. |
| **Base vectorial (Chroma)** | Almacena los vectores y permite buscar por similitud semántica. |
| **Top-k** | Recupera los *k* fragmentos más parecidos a la pregunta. |
| **LLM (Gemini)** | Genera la respuesta final usando los fragmentos recuperados como contexto. |

---

## 2. Arquitectura del sistema

```
                         INDEXACIÓN (una vez)
   PDFs  ─►  Chunking  ─►  Embeddings  ─►  Base vectorial (Chroma)

                         CONSULTA (por pregunta)
   Pregunta ─► Embedding ─► Búsqueda top-k ─► Contexto + Pregunta ─► LLM ─► Respuesta
```

- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, funciona bien en español).
- **Base vectorial:** ChromaDB.
- **LLM:** Gemini `gemini-2.5-flash` (vía la librería oficial `google-genai`).

---

## 3. Instalación y ejecución

El proyecto está pensado para correr en **Google Colab** (no requiere GPU).

### Requisitos

- Una **API key gratuita** de Gemini: se obtiene en https://aistudio.google.com/apikey
- Al menos **5 PDFs** de legislación (descargados de InfoLEG o el Boletín Oficial), ubicados en la carpeta `documentos/`.

### Pasos

1. Abrir `notebook_rag.ipynb` en Google Colab.
2. Ejecutar la celda de instalación de dependencias.
3. Pegar la API key cuando el notebook lo solicite.
4. Subir los PDFs a la carpeta `documentos/` (hay una celda de subida para Colab).
5. Ejecutar el resto de las celdas en orden.

Instalación local de dependencias (opcional):

```bash
pip install -r requirements.txt
```

---

## 4. Ejemplos de uso

Una vez cargados los documentos, se consulta con:

```python
# Sin RAG (el modelo responde de memoria)
responder_sin_rag("¿Cuál es la autoridad de aplicación?")

# Con RAG (el modelo responde usando los documentos)
respuesta, fuentes = responder_con_rag("¿Cuál es la autoridad de aplicación?")
print(respuesta)
print("Fuentes:", fuentes)
```

La celda de **evaluación** ejecuta 5 preguntas y arma una tabla comparando las respuestas CON vs SIN RAG, indicando de qué documento salió cada respuesta.

---

## 5. Evaluación

Se comparan 5 preguntas respondidas **con** y **sin** RAG. Resultado esperado:

- **Con RAG:** respuestas precisas, ajustadas a los documentos y con la fuente identificada.
- **Sin RAG:** respuestas más genéricas; el modelo puede equivocarse en datos específicos (artículos, fechas, autoridad de aplicación) o inventar.

---

## 6. Análisis crítico

### Limitaciones

- La calidad depende del **chunking**: fragmentos demasiado grandes o chicos degradan la búsqueda.
- Los **PDFs escaneados** (imágenes) no se leen bien sin OCR.
- El sistema solo conoce lo que está en el corpus: no puede responder sobre normas ausentes.
- La recuperación por similitud puede traer fragmentos relacionados pero no exactos.

### Mejoras posibles

- Ajustar tamaño de chunk, solapamiento y valor de *k*.
- Incluir **citas exactas** (número de artículo y norma) en cada respuesta.
- Incorporar un *re-ranker* o un modelo de embeddings más potente.
- Ampliar y mantener actualizado el corpus desde InfoLEG / Boletín Oficial.

---

## Estructura del repositorio

```
tp-rag-legislacion/
├── notebook_rag.ipynb     # Notebook ejecutable de punta a punta
├── README.md              # Este archivo
├── requirements.txt       # Dependencias
└── documentos/            # PDFs de legislación (mínimo 5)
```
