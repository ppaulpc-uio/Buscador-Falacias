import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(
    page_title="Buscador de Falacias y Sesgos",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Buscador de Falacias Lógicas y Sesgos Cognitivos")
st.write(
    "Explora una base de datos de casos ecuatorianos de falacias lógicas, "
    "sesgos cognitivos y heurísticas. Ingresa tu consulta en lenguaje natural "
    "para encontrar los casos más relevantes."
)

# ── Carga del dataset (se cachea para no releer el Excel en cada interacción) ──
@st.cache_data
def load_data():
    df = pd.read_excel(
        "Pensamiento Crítico Ecuatoriano_ 100 Ejemplos Multidisciplinarios Penaherrera.xlsx"
    )
    df["Caso Texto (Contexto Ecuador)"] = df["Caso Texto (Contexto Ecuador)"].astype(str)
    return df

# ── Modelo y índice FAISS (se cachean: solo se calculan una vez por sesión) ──
@st.cache_resource
def load_model_and_index(_df):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = _df["Caso Texto (Contexto Ecuador)"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False).astype("float32")

    # Normalizar para usar similitud coseno (más precisa que distancia L2)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])   # Inner Product = coseno tras normalizar
    index.add(embeddings)
    return model, index

# ── Inicialización (con spinner para indicar que está cargando) ──
with st.spinner("Cargando dataset y modelo de embeddings… (solo la primera vez)"):
    df_cases = load_data()
    model, index = load_model_and_index(df_cases)

# ── Interfaz de búsqueda ──
user_query = st.text_input(
    "Ingresa tu consulta (ej. 'argumentos engañosos en política', "
    "'manipulación mediática', 'sesgo en economía'):",
    placeholder="Escribe aquí tu consulta en español…"
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🔍 Buscar", use_container_width=True)
with col2:
    k = st.slider("Número de resultados", min_value=1, max_value=10, value=5)

# ── Función de búsqueda semántica ──
def retrieve_cases(query: str, k: int = 5):
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if not (0 <= idx < len(df_cases)):
            continue
        case = df_cases.iloc[idx]

        # Extraer fuente y sector (convertir a str por si el valor es float/NaN)
        raw_val = case.get("Fuente / Sector", "")
        raw = str(raw_val) if pd.notna(raw_val) else "N/A / N/A"
        parts = [p.strip() for p in raw.split("/")]
        fuente = parts[0] if len(parts) > 0 else "N/A"
        sector = parts[1] if len(parts) > 1 else "N/A"

        # Generar etiquetas desde los campos de texto
        words = []
        for campo in ["Subcategoría", "Categoría", "Explicación Lógica"]:
            val = case.get(campo, "")
            if isinstance(val, str):
                words.extend(val.lower().replace(",", "").split())
        etiquetas = sorted(set(w for w in words if len(w) > 3))[:8]

        results.append({
            "tipo":       case.get("Subcategoría", "N/A"),
            "categoria":  case.get("Categoría", "N/A"),
            "caso":       case.get("Caso Texto (Contexto Ecuador)", "N/A"),
            "explicacion":case.get("Explicación Lógica", "N/A"),
            "sector":     sector,
            "fuente":     fuente,
            "etiquetas":  etiquetas,
            "similitud":  float(score),
        })
    return results

# ── Mostrar resultados ──
if search_button and user_query.strip():
    with st.spinner("Buscando casos relevantes…"):
        results = retrieve_cases(user_query.strip(), k)

    if results:
        st.success(f"Se encontraron {len(results)} casos para: **{user_query}**")
        for i, case in enumerate(results):
            sim_pct = round(case["similitud"] * 100, 1)
            label = f"**{i+1}. {case['tipo']}** | Sector: {case['sector']} | Similitud: {sim_pct}%"
            with st.expander(label):
                st.markdown(f"**Categoría:** {case['categoria']}")
                st.markdown(f"**Explicación lógica:** {case['explicacion']}")
                st.markdown(f"**Caso (contexto Ecuador):** {case['caso']}")
                st.markdown(f"**Fuente:** {case['fuente']}")
                if case["etiquetas"]:
                    tags_str = " · ".join(f"`{t}`" for t in case["etiquetas"])
                    st.markdown(f"**Etiquetas:** {tags_str}")
    else:
        st.warning("No se encontraron casos. Intenta con otra consulta.")

elif search_button and not user_query.strip():
    st.warning("Por favor escribe una consulta antes de buscar.")

# ── Sección informativa ──
st.markdown("---")
with st.expander("📚 Sobre el dataset y la metodología"):
    st.write(
        "Este sistema usa el modelo **paraphrase-multilingual-MiniLM-L12-v2** "
        "para convertir cada caso en un vector de 384 dimensiones. "
        "La búsqueda se realiza con **FAISS** usando similitud coseno, "
        "lo que permite encontrar casos semánticamente relacionados "
        "aunque no compartan palabras exactas con tu consulta."
    )

with st.expander("🔒 Lineamientos éticos y atribución de fuentes"):
    st.write(
        "El dataset fue curado siguiendo criterios de verificación y "
        "atribución clara de cada fuente. Todos los casos provienen de "
        "contextos reales ecuatorianos documentados. "
        "El sistema es de carácter educativo y no busca descalificar "
        "a personas o instituciones."
    )
