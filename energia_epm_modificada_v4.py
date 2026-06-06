# ============================================================
# DASHBOARD STREAMLIT - ANÁLISIS TARIFAS ENERGÍA EPM
# VERSIÓN V4 MODIFICADA
# Gráficos simples: barras, torta, puntos, dispersión,
# predicción, correlación simple y KMeans.
# ============================================================

import io
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VERSION_APP = "V4 - gráficos modificados + observaciones + KMeans"

st.set_page_config(
    page_title="Energía EPM - Dashboard V4",
    page_icon="⚡",
    layout="wide"
)

MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

# ============================================================
# FUNCIONES DE LIMPIEZA
# ============================================================


def limpiar_texto(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def quitar_acentos(texto) -> str:
    texto = str(texto).lower().strip()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }
    for a, b in reemplazos.items():
        texto = texto.replace(a, b)
    return texto


def normalizar_mes(x):
    x = quitar_acentos(limpiar_texto(x))
    return MESES.get(x, np.nan)


def limpiar_numero(x):
    """Convierte textos monetarios o números a float sin romper Streamlit."""
    if isinstance(x, pd.Series):
        return np.nan

    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    x = str(x).strip()
    x = x.replace("$", "")
    x = x.replace(" ", "")
    x = x.replace("\xa0", "")

    # Caso 1,234.56
    if "," in x and "." in x:
        x = x.replace(",", "")
    # Caso 1234,56
    elif "," in x and "." not in x:
        x = x.replace(",", ".")

    try:
        return float(x)
    except Exception:
        return np.nan


def buscar_hoja_energia(archivo) -> str:
    excel = pd.ExcelFile(archivo, engine="openpyxl")
    hojas = excel.sheet_names

    for hoja in hojas:
        if quitar_acentos(hoja) == "energia #":
            return hoja

    for hoja in hojas:
        h = quitar_acentos(hoja)
        if "energia" in h and "#" in h:
            return hoja

    raise ValueError("No se encontró la hoja 'Energia #' en el archivo.")


@st.cache_data(show_spinner=False)
def cargar_energia(archivo_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    """Carga y transforma la hoja Energia # en formato largo."""
    archivo = io.BytesIO(archivo_bytes)
    hoja = buscar_hoja_energia(archivo)

    archivo.seek(0)
    raw = pd.read_excel(
        archivo,
        sheet_name=hoja,
        header=None,
        engine="openpyxl"
    )

    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    if raw.shape[0] < 5 or raw.shape[1] < 3:
        raise ValueError("La hoja Energia # no tiene suficientes filas o columnas para procesar.")

    fila_grupo = raw.iloc[1].ffill()
    fila_rango = raw.iloc[2].ffill()
    fila_propiedad = raw.iloc[3].ffill()

    data = raw.iloc[4:].copy().reset_index(drop=True)
    data = data.dropna(how="all").reset_index(drop=True)

    anio = data.iloc[:, 0].ffill()
    mes = data.iloc[:, 1].ffill()

    base_fechas = pd.DataFrame({
        "Año": pd.to_numeric(anio, errors="coerce"),
        "Mes": mes,
    })

    base_fechas["Mes_num"] = base_fechas["Mes"].apply(normalizar_mes)
    base_fechas = base_fechas.dropna(subset=["Año", "Mes_num"])

    indices_validos = base_fechas.index
    data = data.loc[indices_validos].reset_index(drop=True)
    base_fechas = base_fechas.reset_index(drop=True)

    base_fechas["Año"] = base_fechas["Año"].astype(int)
    base_fechas["Mes_num"] = base_fechas["Mes_num"].astype(int)

    base_fechas["Fecha"] = pd.to_datetime(
        {
            "year": base_fechas["Año"],
            "month": base_fechas["Mes_num"],
            "day": 1,
        },
        errors="coerce"
    )

    registros = []

    for pos in range(2, raw.shape[1]):
        grupo = limpiar_texto(fila_grupo.iloc[pos]) or "Sin grupo"
        rango = limpiar_texto(fila_rango.iloc[pos]) or "Todo el consumo"
        propiedad = limpiar_texto(fila_propiedad.iloc[pos]) or "Sin propiedad"

        serie_completa = f"{grupo} - {rango} - {propiedad}"
        valores = data.iloc[:, pos].apply(limpiar_numero)

        if valores.notna().sum() < 6:
            continue

        temp = base_fechas.copy()
        temp["Tarifa"] = valores.values
        temp["Grupo"] = grupo
        temp["Rango"] = rango
        temp["Propiedad"] = propiedad
        temp["Serie_completa"] = serie_completa
        temp["Columna_origen"] = pos
        temp = temp.dropna(subset=["Fecha", "Tarifa"])
        registros.append(temp)

    if len(registros) == 0:
        raise ValueError("No se encontraron columnas numéricas válidas en la hoja Energia #.")

    df = pd.concat(registros, ignore_index=True)
    df = df.sort_values(["Serie_completa", "Fecha"]).reset_index(drop=True)

    return df, hoja


# ============================================================
# FUNCIONES ANALÍTICAS
# ============================================================


def formato_moneda(x) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return "$0.00"


def formato_pct(x) -> str:
    try:
        return f"{x:,.2f}%"
    except Exception:
        return "0.00%"


def mostrar_observacion(titulo: str, texto: str):
    st.info(f"**Observación - {titulo}:** {texto}")


def calcular_metricas_series(df: pd.DataFrame) -> pd.DataFrame:
    registros = []

    for serie, temp in df.groupby("Serie_completa"):
        temp = temp.sort_values("Fecha").dropna(subset=["Tarifa"])

        if len(temp) < 6:
            continue

        inicial = temp["Tarifa"].iloc[0]
        final = temp["Tarifa"].iloc[-1]
        promedio = temp["Tarifa"].mean()
        minimo = temp["Tarifa"].min()
        maximo = temp["Tarifa"].max()
        variaciones = temp["Tarifa"].pct_change() * 100
        volatilidad = variaciones.std(skipna=True)

        if inicial > 0:
            crecimiento = ((final / inicial) - 1) * 100
        else:
            crecimiento = np.nan

        modelo = LinearRegression()
        x = np.arange(len(temp)).reshape(-1, 1)
        y = temp["Tarifa"].values
        modelo.fit(x, y)
        tendencia = float(modelo.coef_[0])

        registros.append({
            "Serie_completa": serie,
            "Grupo": temp["Grupo"].iloc[-1],
            "Rango": temp["Rango"].iloc[-1],
            "Propiedad": temp["Propiedad"].iloc[-1],
            "Fecha_inicial": temp["Fecha"].min(),
            "Fecha_final": temp["Fecha"].max(),
            "Tarifa_inicial": inicial,
            "Tarifa_actual": final,
            "Tarifa_promedio": promedio,
            "Tarifa_minima": minimo,
            "Tarifa_maxima": maximo,
            "Crecimiento_%": crecimiento,
            "Volatilidad_%": volatilidad,
            "Tendencia_mensual": tendencia,
            "Datos": len(temp),
        })

    metricas = pd.DataFrame(registros)
    if not metricas.empty:
        metricas = metricas.replace([np.inf, -np.inf], np.nan)
        metricas = metricas.dropna(subset=["Tarifa_actual", "Crecimiento_%", "Volatilidad_%", "Tendencia_mensual"])

    return metricas


def proyectar_serie(df_serie: pd.DataFrame, meses_futuros: int = 6):
    df_modelo = df_serie.sort_values("Fecha").dropna(subset=["Tarifa"]).copy()

    if len(df_modelo) < 6:
        raise ValueError("No hay datos suficientes para proyectar la serie seleccionada.")

    df_modelo["Periodo"] = np.arange(len(df_modelo))

    x = df_modelo[["Periodo"]]
    y = df_modelo["Tarifa"]

    modelo = LinearRegression()
    modelo.fit(x, y)

    df_modelo["Tendencia_modelo"] = modelo.predict(x)
    r2 = r2_score(y, df_modelo["Tendencia_modelo"])

    ultimo_periodo = int(df_modelo["Periodo"].max())
    ultima_fecha = df_modelo["Fecha"].max()

    futuros = pd.DataFrame({
        "Periodo": np.arange(ultimo_periodo + 1, ultimo_periodo + meses_futuros + 1)
    })
    futuros["Fecha"] = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=meses_futuros,
        freq="MS"
    )
    futuros["Proyeccion"] = modelo.predict(futuros[["Periodo"]]).clip(min=0)

    return df_modelo, futuros, r2


def crear_excel_descarga(
    df: pd.DataFrame,
    df_mensual: pd.DataFrame,
    metricas: pd.DataFrame,
    proyeccion: pd.DataFrame,
    correlaciones: pd.DataFrame,
    resumen_cluster: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Base_limpia", index=False)
        df_mensual.to_excel(writer, sheet_name="Tendencia_mensual", index=False)
        metricas.to_excel(writer, sheet_name="Metricas_series", index=False)
        proyeccion.to_excel(writer, sheet_name="Proyeccion", index=False)
        correlaciones.to_excel(writer, sheet_name="Correlacion_simple", index=False)
        resumen_cluster.to_excel(writer, sheet_name="KMeans_resumen", index=False)
    return output.getvalue()


# ============================================================
# INTERFAZ
# ============================================================

st.title("⚡ Análisis gráfico de tarifas de energía EPM")
st.caption(VERSION_APP)

st.markdown(
    """
Esta versión reemplaza los gráficos extensos por visuales más directos: barras, torta, puntos,
dispersión, predicción, correlación simple y KMeans. Cada gráfico incluye una observación automática
para interpretar el resultado.
"""
)

with st.sidebar:
    st.header("Carga de información")
    archivo = st.file_uploader(
        "Carga el archivo Excel con la hoja Energia #",
        type=["xlsx", "xlsm", "xls"]
    )

    st.divider()
    meses_a_proyectar = st.slider("Meses a proyectar", 3, 12, 6)
    clusters_solicitados = st.slider("Cantidad de grupos KMeans", 2, 5, 3)

if archivo is None:
    st.warning("Carga el archivo Excel para iniciar el análisis.")
    st.stop()

try:
    archivo_bytes = archivo.getvalue()
    df, hoja_usada = cargar_energia(archivo_bytes)
except Exception as e:
    st.error(f"No fue posible procesar el archivo: {e}")
    st.stop()

# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

fecha_min = df["Fecha"].min()
fecha_max = df["Fecha"].max()
series_total = df["Serie_completa"].nunique()
registros_total = len(df)
metricas = calcular_metricas_series(df)

if metricas.empty:
    st.error("No hay suficientes datos por serie para construir métricas, predicciones y KMeans.")
    st.stop()

ultima_fecha = df["Fecha"].max()
df_ultimo = df[df["Fecha"] == ultima_fecha].copy()

# Serie por defecto: tarifa más alta del último mes
serie_default = df_ultimo.sort_values("Tarifa", ascending=False).iloc[0]["Serie_completa"]
lista_series = sorted(df["Serie_completa"].unique())
indice_default = lista_series.index(serie_default) if serie_default in lista_series else 0

with st.sidebar:
    serie_seleccionada = st.selectbox(
        "Serie para predicción y correlación",
        lista_series,
        index=indice_default
    )

# ============================================================
# INDICADORES GENERALES
# ============================================================

st.subheader("1. Indicadores generales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Hoja analizada", hoja_usada)
col2.metric("Periodo", f"{fecha_min:%Y-%m} a {fecha_max:%Y-%m}")
col3.metric("Series", f"{series_total:,.0f}")
col4.metric("Registros", f"{registros_total:,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Tarifa promedio", formato_moneda(df["Tarifa"].mean()))
col6.metric("Tarifa mínima", formato_moneda(df["Tarifa"].min()))
col7.metric("Tarifa máxima", formato_moneda(df["Tarifa"].max()))
col8.metric("Último mes", f"{ultima_fecha:%Y-%m}")

# ============================================================
# GRÁFICO 1 - BARRAS ANUALES
# ============================================================

st.subheader("2. Barras - tarifa promedio anual")

df_anual = df.groupby("Año", as_index=False)["Tarifa"].mean()
df_anual["Tarifa"] = df_anual["Tarifa"].round(2)

fig_anual = px.bar(
    df_anual,
    x="Año",
    y="Tarifa",
    text="Tarifa",
    title="Tarifa promedio anual"
)
fig_anual.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
fig_anual.update_layout(yaxis_title="Tarifa promedio", xaxis_title="Año", height=480)
st.plotly_chart(fig_anual, use_container_width=True)

anio_mayor = df_anual.sort_values("Tarifa", ascending=False).iloc[0]
anio_menor = df_anual.sort_values("Tarifa", ascending=True).iloc[0]
mostrar_observacion(
    "barras anuales",
    f"El año con mayor tarifa promedio es {int(anio_mayor['Año'])} con {formato_moneda(anio_mayor['Tarifa'])}. "
    f"El menor promedio está en {int(anio_menor['Año'])} con {formato_moneda(anio_menor['Tarifa'])}."
)

# ============================================================
# GRÁFICO 2 - PUNTOS Y TENDENCIA MENSUAL
# ============================================================

st.subheader("3. Puntos - tendencia mensual general")

df_mensual = df.groupby("Fecha", as_index=False)["Tarifa"].mean().sort_values("Fecha")
df_mensual["Variacion_mensual_%"] = df_mensual["Tarifa"].pct_change() * 100
df_mensual["Media_movil_3m"] = df_mensual["Tarifa"].rolling(3).mean()

fig_mensual = go.Figure()
fig_mensual.add_trace(go.Scatter(
    x=df_mensual["Fecha"],
    y=df_mensual["Tarifa"],
    mode="lines+markers",
    name="Tarifa promedio mensual"
))
fig_mensual.add_trace(go.Scatter(
    x=df_mensual["Fecha"],
    y=df_mensual["Media_movil_3m"],
    mode="lines",
    name="Media móvil 3 meses"
))
fig_mensual.update_layout(
    title="Tendencia mensual con puntos y media móvil",
    xaxis_title="Fecha",
    yaxis_title="Tarifa promedio",
    height=520
)
st.plotly_chart(fig_mensual, use_container_width=True)

crecimiento_general = ((df_mensual["Tarifa"].iloc[-1] / df_mensual["Tarifa"].iloc[0]) - 1) * 100
mostrar_observacion(
    "tendencia mensual",
    f"La tarifa promedio pasó de {formato_moneda(df_mensual['Tarifa'].iloc[0])} a "
    f"{formato_moneda(df_mensual['Tarifa'].iloc[-1])}, con una variación acumulada de "
    f"{formato_pct(crecimiento_general)}."
)

# ============================================================
# GRÁFICO 3 - BARRAS VARIACIÓN MENSUAL
# ============================================================

st.subheader("4. Barras - variación mensual porcentual")

fig_var = px.bar(
    df_mensual,
    x="Fecha",
    y="Variacion_mensual_%",
    title="Variación mensual promedio %"
)
fig_var.update_layout(xaxis_title="Fecha", yaxis_title="Variación %", height=470)
st.plotly_chart(fig_var, use_container_width=True)

var_valida = df_mensual.dropna(subset=["Variacion_mensual_%"])
mes_mayor_subida = var_valida.sort_values("Variacion_mensual_%", ascending=False).iloc[0]
mes_mayor_bajada = var_valida.sort_values("Variacion_mensual_%", ascending=True).iloc[0]
mostrar_observacion(
    "variación mensual",
    f"La mayor subida ocurrió en {mes_mayor_subida['Fecha']:%Y-%m} con {formato_pct(mes_mayor_subida['Variacion_mensual_%'])}. "
    f"La mayor caída ocurrió en {mes_mayor_bajada['Fecha']:%Y-%m} con {formato_pct(mes_mayor_bajada['Variacion_mensual_%'])}."
)

# ============================================================
# GRÁFICO 4 - TORTA POR GRUPO
# ============================================================

st.subheader("5. Torta - participación visual por grupo en el último mes")

df_torta = df_ultimo.groupby("Grupo", as_index=False)["Tarifa"].mean()
df_torta = df_torta.sort_values("Tarifa", ascending=False)

fig_torta = px.pie(
    df_torta,
    names="Grupo",
    values="Tarifa",
    title=f"Participación relativa de tarifa promedio por grupo - {ultima_fecha:%Y-%m}",
    hole=0.35
)
fig_torta.update_traces(textposition="inside", textinfo="percent+label")
fig_torta.update_layout(height=520)
st.plotly_chart(fig_torta, use_container_width=True)

grupo_mayor = df_torta.iloc[0]
mostrar_observacion(
    "torta por grupo",
    f"El grupo con mayor peso relativo en el último mes es {grupo_mayor['Grupo']} con tarifa promedio de "
    f"{formato_moneda(grupo_mayor['Tarifa'])}. Esta torta es comparativa, no representa consumo real."
)

# ============================================================
# GRÁFICO 5 - BARRAS TOP TARIFAS ÚLTIMO MES
# ============================================================

st.subheader("6. Barras - top tarifas del último mes")

top_tarifas = df_ultimo.sort_values("Tarifa", ascending=False).head(10)

fig_top = px.bar(
    top_tarifas,
    x="Tarifa",
    y="Serie_completa",
    orientation="h",
    text="Tarifa",
    title=f"Top 10 tarifas más altas - {ultima_fecha:%Y-%m}"
)
fig_top.update_traces(texttemplate="%{text:,.2f}")
fig_top.update_yaxes(autorange="reversed")
fig_top.update_layout(xaxis_title="Tarifa", yaxis_title="Serie", height=560)
st.plotly_chart(fig_top, use_container_width=True)

serie_top = top_tarifas.iloc[0]
mostrar_observacion(
    "top tarifas",
    f"La tarifa más alta del último mes es {serie_top['Serie_completa']} con {formato_moneda(serie_top['Tarifa'])}."
)

# ============================================================
# KMEANS + DISPERSIÓN
# ============================================================

st.subheader("7. Dispersión + KMeans - grupos automáticos de tarifas")

features_kmeans = ["Tarifa_actual", "Crecimiento_%", "Volatilidad_%", "Tendencia_mensual"]
metricas_km = metricas.dropna(subset=features_kmeans).copy()

if len(metricas_km) >= 2:
    n_clusters = min(clusters_solicitados, len(metricas_km))
    scaler = StandardScaler()
    matriz = scaler.fit_transform(metricas_km[features_kmeans])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    metricas_km["Cluster"] = kmeans.fit_predict(matriz) + 1
    metricas_km["Cluster"] = "Grupo " + metricas_km["Cluster"].astype(str)
    metricas_km["Volatilidad_plot"] = metricas_km["Volatilidad_%"].abs().fillna(0) + 0.1

    fig_kmeans = px.scatter(
        metricas_km,
        x="Tarifa_actual",
        y="Crecimiento_%",
        color="Cluster",
        size="Volatilidad_plot",
        hover_name="Serie_completa",
        hover_data={
            "Grupo": True,
            "Tarifa_actual": ":,.2f",
            "Crecimiento_%": ":.2f",
            "Volatilidad_%": ":.2f",
            "Tendencia_mensual": ":.2f",
            "Volatilidad_plot": False,
        },
        title="Dispersión: tarifa actual vs crecimiento, agrupada con KMeans"
    )
    fig_kmeans.update_layout(
        xaxis_title="Tarifa actual",
        yaxis_title="Crecimiento acumulado %",
        height=620
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    resumen_cluster = metricas_km.groupby("Cluster", as_index=False).agg(
        Series=("Serie_completa", "count"),
        Tarifa_actual_prom=("Tarifa_actual", "mean"),
        Crecimiento_prom=("Crecimiento_%", "mean"),
        Volatilidad_prom=("Volatilidad_%", "mean"),
        Tendencia_mensual_prom=("Tendencia_mensual", "mean"),
    ).sort_values("Tarifa_actual_prom", ascending=False)

    st.dataframe(resumen_cluster, use_container_width=True)

    cluster_critico = resumen_cluster.sort_values(
        ["Tarifa_actual_prom", "Crecimiento_prom"],
        ascending=False
    ).iloc[0]
    mostrar_observacion(
        "KMeans",
        f"El grupo más crítico por tarifa actual y crecimiento es {cluster_critico['Cluster']}. "
        f"Tiene {int(cluster_critico['Series'])} series, tarifa promedio actual de "
        f"{formato_moneda(cluster_critico['Tarifa_actual_prom'])} y crecimiento promedio de "
        f"{formato_pct(cluster_critico['Crecimiento_prom'])}."
    )
else:
    metricas_km = pd.DataFrame()
    resumen_cluster = pd.DataFrame()
    st.warning("No hay suficientes series para aplicar KMeans.")

# ============================================================
# PREDICCIÓN SERIE SELECCIONADA
# ============================================================

st.subheader("8. Predicción - serie seleccionada")

st.write(f"**Serie seleccionada:** {serie_seleccionada}")

df_serie = df[df["Serie_completa"] == serie_seleccionada].sort_values("Fecha").copy()

try:
    historico_serie, proyeccion_serie, r2_serie = proyectar_serie(df_serie, meses_a_proyectar)

    df_hist_plot = historico_serie[["Fecha", "Tarifa", "Tendencia_modelo"]].rename(columns={
        "Tarifa": "Histórico",
        "Tendencia_modelo": "Tendencia modelo"
    })
    df_hist_melt = df_hist_plot.melt(id_vars="Fecha", var_name="Tipo", value_name="Valor")

    df_fut_plot = proyeccion_serie[["Fecha", "Proyeccion"]].rename(columns={"Proyeccion": "Valor"})
    df_fut_plot["Tipo"] = "Predicción"

    df_pred_plot = pd.concat(
        [df_hist_melt, df_fut_plot[["Fecha", "Tipo", "Valor"]]],
        ignore_index=True
    )

    fig_pred = px.line(
        df_pred_plot,
        x="Fecha",
        y="Valor",
        color="Tipo",
        markers=True,
        title=f"Predicción a {meses_a_proyectar} meses"
    )
    fig_pred.update_layout(xaxis_title="Fecha", yaxis_title="Tarifa", height=560)
    st.plotly_chart(fig_pred, use_container_width=True)

    tarifa_actual = historico_serie["Tarifa"].iloc[-1]
    tarifa_pred_final = proyeccion_serie["Proyeccion"].iloc[-1]
    variacion_pred = ((tarifa_pred_final / tarifa_actual) - 1) * 100 if tarifa_actual > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Tarifa actual", formato_moneda(tarifa_actual))
    c2.metric("Predicción final", formato_moneda(tarifa_pred_final))
    c3.metric("Variación proyectada", formato_pct(variacion_pred))

    mostrar_observacion(
        "predicción",
        f"El modelo lineal proyecta que la serie llegue a {formato_moneda(tarifa_pred_final)} en "
        f"{meses_a_proyectar} meses. El R² del ajuste es {r2_serie:.3f}; entre más cercano a 1, "
        f"mejor explica la tendencia histórica."
    )
except Exception as e:
    proyeccion_serie = pd.DataFrame()
    st.warning(f"No fue posible proyectar la serie seleccionada: {e}")

# ============================================================
# CORRELACIÓN SIMPLE EN BARRAS
# ============================================================

st.subheader("9. Barras - correlación simple con la serie seleccionada")

pivot = df.pivot_table(
    index="Fecha",
    columns="Serie_completa",
    values="Tarifa",
    aggfunc="mean"
).sort_index()

minimo_datos = max(6, int(len(pivot) * 0.5))
pivot = pivot.dropna(axis=1, thresh=minimo_datos)

if serie_seleccionada in pivot.columns and pivot.shape[1] >= 2:
    corr_serie = pivot.corr()[serie_seleccionada].drop(labels=[serie_seleccionada]).dropna()
    corr_serie = corr_serie.sort_values(ascending=False).head(8).reset_index()
    corr_serie.columns = ["Serie_completa", "Correlacion"]

    fig_corr = px.bar(
        corr_serie,
        x="Correlacion",
        y="Serie_completa",
        orientation="h",
        text="Correlacion",
        title="Top 8 series que más se mueven parecido a la serie seleccionada"
    )
    fig_corr.update_traces(texttemplate="%{text:.3f}")
    fig_corr.update_yaxes(autorange="reversed")
    fig_corr.update_layout(xaxis_title="Correlación Pearson", yaxis_title="Serie", height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    if not corr_serie.empty:
        mejor_corr = corr_serie.iloc[0]
        mostrar_observacion(
            "correlación simple",
            f"La serie que más se mueve parecido a la seleccionada es {mejor_corr['Serie_completa']}, "
            f"con correlación de {mejor_corr['Correlacion']:.3f}. Valores cercanos a 1 indican movimiento muy similar."
        )
else:
    corr_serie = pd.DataFrame(columns=["Serie_completa", "Correlacion"])
    st.warning("La serie seleccionada no tiene suficientes datos para calcular correlación simple.")

# ============================================================
# CONCLUSIÓN EJECUTIVA
# ============================================================

st.subheader("10. Conclusión ejecutiva")

serie_mayor_actual = metricas.sort_values("Tarifa_actual", ascending=False).iloc[0]
serie_mayor_crecimiento = metricas.sort_values("Crecimiento_%", ascending=False).iloc[0]
serie_mayor_volatilidad = metricas.sort_values("Volatilidad_%", ascending=False).iloc[0]

st.success(
    f"Periodo analizado: {fecha_min:%Y-%m} a {fecha_max:%Y-%m}. "
    f"La tarifa promedio general tuvo una variación acumulada de {formato_pct(crecimiento_general)}. "
    f"La serie con tarifa actual más alta es {serie_mayor_actual['Serie_completa']} "
    f"con {formato_moneda(serie_mayor_actual['Tarifa_actual'])}. "
    f"La serie con mayor crecimiento es {serie_mayor_crecimiento['Serie_completa']} "
    f"con {formato_pct(serie_mayor_crecimiento['Crecimiento_%'])}. "
    f"La serie más volátil es {serie_mayor_volatilidad['Serie_completa']} "
    f"con volatilidad de {formato_pct(serie_mayor_volatilidad['Volatilidad_%'])}."
)

# ============================================================
# DESCARGA
# ============================================================

st.subheader("11. Descargar análisis")

try:
    excel_bytes = crear_excel_descarga(
        df=df,
        df_mensual=df_mensual,
        metricas=metricas,
        proyeccion=proyeccion_serie,
        correlaciones=corr_serie,
        resumen_cluster=resumen_cluster,
    )
    st.download_button(
        label="Descargar Excel del análisis",
        data=excel_bytes,
        file_name="analisis_energia_epm_v4.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.warning(f"No fue posible generar el Excel de descarga: {e}")

st.caption("Archivo principal para Streamlit: energia_epm_modificada_v4.py")
