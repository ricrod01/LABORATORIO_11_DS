"""Observatorio interactivo de turismo sostenible de Guatemala.

Ejecucion local:
    streamlit run app.py
"""

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Observatorio de Turismo Sostenible | Guatemala",
    page_icon="GT",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "datos" / "DATOS_LIMPIOS.csv"
VERSION_PATH = ROOT / "resultados" / "version_vigente.json"
APROBADOS_DIR = ROOT / "resultados" / "aprobados"
MODELOS_DIR = ROOT / "modelos_guardados"
REPORTES_DIR = ROOT / "reportes"

# Únicamente las dos vías que utiliza el pipeline de MLOps.
VIAS = {"AEREA": "Aérea", "TERRESTRE": "Terrestre"}

PETROLEO = "#123B4A"
TURQUESA = "#19A7A0"
GRIS = "#F2F0EB"
CORAL = "#E45B4F"
BLANCO = "#FFFFFF"
TEXTO_SUAVE = "#5E6D72"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
def verificar_version_publicada() -> dict | None:
    """Resuelve la última versión *vigente* (incluye restauraciones).

    Nunca selecciona una carpeta por fecha, porque un run fallido o rechazado
    podría ser posterior a la última publicación aprobada.
    """
    if not VERSION_PATH.is_file():
        return None

    with VERSION_PATH.open("r", encoding="utf-8") as archivo:
        puntero = json.load(archivo)

    identificador = puntero.get("id_ejecucion", "")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}", identificador):
        raise ValueError("El identificador en version_vigente.json no es válido.")

    carpeta_aprobada = APROBADOS_DIR / identificador
    carpeta_modelos = MODELOS_DIR / identificador
    ruta_predicciones = carpeta_aprobada / "estimaciones.csv"
    ruta_metadatos = carpeta_aprobada / "metadatos.json"
    for ruta in (ruta_predicciones, ruta_metadatos):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el archivo de la versión vigente: {ruta}")

    with ruta_metadatos.open("r", encoding="utf-8") as archivo:
        metadatos = json.load(archivo)

    if metadatos.get("id_ejecucion") != identificador:
        raise ValueError("El identificador de los metadatos no coincide con la versión vigente.")
    if not metadatos.get("aceptacion", {}).get("aprobado", False):
        raise ValueError("La versión señalada no figura como aprobada.")

    huella = hashlib.sha256(ruta_predicciones.read_bytes()).hexdigest()
    if huella != metadatos.get("sha256_estimaciones"):
        raise ValueError("La huella SHA256 de estimaciones.csv no coincide con los metadatos.")

    return {
        "id": identificador,
        "puntero": puntero,
        "metadatos": metadatos,
        "ruta_predicciones": ruta_predicciones,
        "carpeta_modelos": carpeta_modelos,
    }


@st.cache_data(show_spinner="Preparando indicadores...")
def load_data(path: str, modified_ns: int) -> pd.DataFrame:
    """Carga los datos limpios reales del pipeline, sin volver a procesarlos."""
    del modified_ns  # Su valor sirve para invalidar el caché cuando cambia el CSV.
    data = pd.read_csv(path, low_memory=False)
    data = data.rename(columns={
        "fecha": "Fecha", "via": "Via", "país": "Pais", "pais": "Pais",
        "tipo_viajero": "Tipo de Viajero", "cantidad_viajero": "Viajero",
        "Vía": "Via", "País": "Pais",
    })
    requeridas = {"Fecha", "Via", "Pais", "Tipo de Viajero", "Viajero"}
    faltantes = requeridas.difference(data.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en DATOS_LIMPIOS.csv: {sorted(faltantes)}")

    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="raise")
    data["Viajero"] = pd.to_numeric(data["Viajero"], errors="raise")
    data["Via"] = (data["Via"].astype("string").str.strip()
                   .str.normalize("NFKD").str.replace(r"[\u0300-\u036f]", "", regex=True)
                   .str.upper())
    data["Pais"] = data["Pais"].astype("string").str.strip().str.upper()
    data["Tipo de Viajero"] = data["Tipo de Viajero"].astype("string").str.strip().str.title()
    data = data.loc[
        data["Via"].isin(VIAS)
        & data["Tipo de Viajero"].isin(["Turista", "Excursionista"])
        & (data["Pais"] != "GUATEMALA")
    ].copy()
    if data.empty:
        raise ValueError("La fuente limpia no contiene datos para AEREA y TERRESTRE.")
    data["Anio"] = data["Fecha"].dt.year
    data["Mes"] = data["Fecha"].dt.month
    return data


@st.cache_data(show_spinner="Leyendo estimaciones aprobadas...")
def load_published_forecasts(path: str, modified_ns: int, identifier: str) -> pd.DataFrame:
    """Lee únicamente estimaciones que ya fueron autorizadas por el pipeline."""
    del modified_ns
    data = pd.read_csv(path, parse_dates=["fecha"])
    required = {"via", "fecha", "valor", "id_ejecucion", "tipo_valor", "modelo"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas en estimaciones.csv: {sorted(missing)}")
    if data.empty or not data["id_ejecucion"].eq(identifier).all():
        raise ValueError("Las estimaciones no corresponden a la ejecución publicada.")
    if not data["tipo_valor"].eq("estimado").all():
        raise ValueError("El archivo contiene valores que no son estimaciones.")
    if set(data["via"]) != set(VIAS):
        raise ValueError("El archivo aprobado no contiene exactamente AEREA y TERRESTRE.")
    if data.duplicated(["via", "fecha"]).any():
        raise ValueError("Hay duplicados de vía y fecha en la versión aprobada.")
    data["valor"] = pd.to_numeric(data["valor"], errors="raise")
    if not np.isfinite(data["valor"].to_numpy()).all() or (data["valor"] < 0).any():
        raise ValueError("Las estimaciones contienen valores negativos o no finitos.")
    return data.sort_values(["via", "fecha"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Cargando modelos de la versión aprobada...")
def load_models(model_specs: tuple[tuple[str, str, int], ...]) -> dict:
    """Carga los modelos serializados de la misma versión que el CSV publicado.

    NO se llaman .fit(), .predict() ni .forecast(): las cifras aprobadas vienen
    del CSV publicado y no se recalculan al interactuar con el tablero.
    """
    return {via: joblib.load(path) for via, path, _ in model_specs}


@st.cache_data(show_spinner="Leyendo métricas de validación...")
def load_evaluation(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    data = pd.read_csv(path)
    required = {"periodo", "observado", "estimacion", "referencia"}
    if not required.issubset(data.columns):
        raise ValueError(f"Reporte de evaluación incompleto: {path}")
    data["Fecha"] = pd.to_datetime(data["periodo"], format="%Y-%m")
    data["Error"] = pd.to_numeric(data["observado"]) - pd.to_numeric(data["estimacion"])
    return data.sort_values("Fecha")


def monthly_series(data: pd.DataFrame, via: str | None = None, foreign_only: bool = False) -> pd.Series:
    subset = data
    if via:
        subset = subset[subset["Via"] == via]
    if foreign_only:
        subset = subset[subset["Pais"] != "GUATEMALA"]
    result = subset.groupby("Fecha")["Viajero"].sum(min_count=1).sort_index()
    if result.empty:
        return result
    # Los meses sin cobertura son valores desconocidos, no ceros.
    index = pd.date_range(result.index.min(), result.index.max(), freq="MS")
    return result.reindex(index)


def chart_layout(fig: go.Figure, height: int = 350, percent_axis: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=18, b=8),
        paper_bgcolor=BLANCO,
        plot_bgcolor=BLANCO,
        font=dict(family="Arial, sans-serif", color=PETROLEO, size=12),
        hoverlabel=dict(bgcolor=BLANCO, font_color=PETROLEO),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(color=PETROLEO, size=12),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#8B9A9D",
        tickfont=dict(color=PETROLEO, size=12),
        title_font=dict(color=PETROLEO, size=13),
    )
    fig.update_yaxes(
        gridcolor="#D5DDDB",
        zerolinecolor="#8B9A9D",
        rangemode="tozero",
        tickfont=dict(color=PETROLEO, size=12),
        title_font=dict(color=PETROLEO, size=13),
    )
    if percent_axis:
        fig.update_yaxes(ticksuffix="%")
    return fig


def section_title(number: int, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
          <span>{number}</span><div><b>{title}</b><small>{subtitle}</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0) or pd.isna(current) or pd.isna(previous):
        return None
    return (current / previous - 1) * 100


def kpi_card(title: str, context: str, value: float | None, footer: str) -> None:
    display = "N/D" if value is None else f"{value:+.1f}%"
    color = TEXTO_SUAVE if value is None else (TURQUESA if value >= 0 else CORAL)
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-icon">↗</div><div>
            <b>{title}</b><small>{context}</small>
            <strong style="color:{color}">{display}</strong><em>{footer}</em>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
      :root {{ --petroleo:{PETROLEO}; --turquesa:{TURQUESA}; --gris:{GRIS}; --coral:{CORAL}; }}
      .stApp {{ background: var(--gris); color: var(--petroleo); }}
      .block-container {{ max-width: 1480px; padding-top: 1rem; padding-bottom: 3rem; }}
      html, body, [class*="st-"] {{ font-family: Inter, Arial, sans-serif; }}
      h1, h2, h3 {{ color: var(--petroleo); letter-spacing: -.025em; }}
      div[data-testid="stMarkdownContainer"] p:not(.hero-subtitle),
      div[data-testid="stCaptionContainer"],
      div[data-testid="stWidgetLabel"] p,
      div[role="radiogroup"] label p,
      details summary {{ color: var(--petroleo) !important; }}
      .hero {{ background:linear-gradient(110deg, rgba(18,59,74,.98), rgba(25,167,160,.82));
        border-radius:18px; padding:30px 34px; color:white; margin-bottom:14px; position:relative; overflow:hidden; }}
      .hero:after {{ content:""; position:absolute; width:250px; height:250px; border:38px solid rgba(255,255,255,.08);
        border-radius:50%; right:-65px; top:-100px; }}
      .hero h1 {{ color:white; margin:0; font-size:clamp(1.8rem, 4vw, 3rem); }}
      .hero .hero-subtitle {{ margin:.45rem 0 0; opacity:.88; max-width:760px; color:#FFFFFF !important; }}
      .eyebrow {{ text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:800; opacity:.75; }}
      .kpi {{ background:white; border:1px solid #DDE3E1; border-radius:15px; padding:18px; min-height:142px;
        display:flex; gap:15px; box-shadow:0 3px 15px rgba(18,59,74,.05); }}
      .kpi-icon {{ width:48px; height:48px; flex:0 0 48px; display:grid; place-items:center; border-radius:12px;
        background:#DDF4F1; color:var(--turquesa); font-size:24px; font-weight:800; }}
      .kpi b, .kpi small, .kpi strong, .kpi em {{ display:block; }}
      .kpi b {{ color:var(--petroleo); font-size:.92rem; }} .kpi small {{ color:#6C797D; margin-top:3px; }}
      .kpi strong {{ font-size:2rem; line-height:1.2; margin-top:5px; }} .kpi em {{ color:#819094; font-size:.75rem; font-style:normal; }}
      div[data-testid="stPlotlyChart"] {{ background:white; border:1px solid #DDE3E1; border-radius:15px;
        padding:8px; box-shadow:0 3px 15px rgba(18,59,74,.04); }}
      .section-title {{ display:flex; gap:10px; align-items:flex-start; margin:18px 4px 8px; color:var(--petroleo); }}
      .section-title span {{ background:var(--turquesa); color:white; border-radius:50%; min-width:27px; height:27px;
        display:grid; place-items:center; font-weight:800; font-size:.82rem; }}
      .section-title b, .section-title small {{ display:block; }} .section-title small {{ color:#69777B; margin-top:2px; }}
      div[data-testid="stMetric"] {{ background:white; border:1px solid #DDE3E1; border-radius:14px; padding:15px; }}
      .note {{ border-left:4px solid var(--turquesa); background:white; padding:12px 15px; border-radius:0 10px 10px 0;
        color:#526166; font-size:.88rem; margin:8px 0; }}
      @media (max-width:700px) {{ .hero {{ padding:24px 20px; }} .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

if not CSV_PATH.is_file():
    st.error("No se encuentra datos/DATOS_LIMPIOS.csv. Ejecuta primero el pipeline normal.")
    st.stop()

try:
    df = load_data(str(CSV_PATH), CSV_PATH.stat().st_mtime_ns)
except Exception as error:
    st.error(f"No fue posible leer los datos limpios: {error}")
    st.stop()

# No usar las rutas absolutas guardadas por Windows en version_vigente.json:
# se reconstruyen todas a partir de ROOT y del ID de la versión aprobada.
try:
    publicacion = verificar_version_publicada()
    publicacion_error = None
except (ValueError, FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as error:
    publicacion = None
    publicacion_error = str(error)

st.markdown(
    """
    <div class="hero"><div class="eyebrow">Observatorio de turismo sostenible</div>
      <h1>Panorama del turismo en Guatemala</h1>
      <div class="hero-subtitle">Indicadores, patrones de llegada y proyecciones para apoyar decisiones de planificacion turistica.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

available_years = sorted(df["Anio"].unique().astype(int), reverse=True)
default_year = 2024 if 2024 in available_years else available_years[0]
control_a, control_b, control_c = st.columns([1.1, 1.35, 3.2], vertical_alignment="bottom")
with control_a:
    selected_year = st.selectbox("Año de analisis", available_years, index=available_years.index(default_year))
with control_b:
    detail_range = st.selectbox("Detalle del historico", ["Ultimos 5 años", "Ultimos 10 años", "Serie completa"], index=1)
with control_c:
    visible_groups = st.multiselect(
        "Secciones visibles",
        ["Panorama", "Origen y estacionalidad", "Modelos predictivos"],
        default=["Panorama", "Origen y estacionalidad", "Modelos predictivos"],
        help="Permite elegir que grupos de graficas desea consultar.",
    )

annual_months = sorted(df.loc[df["Anio"] == selected_year, "Mes"].unique())
last_month = max(annual_months)
comparable_months = list(range(1, last_month + 1))
current_total = df[(df["Anio"] == selected_year) & (df["Mes"].isin(comparable_months))]["Viajero"].sum()
previous_total = df[(df["Anio"] == selected_year - 1) & (df["Mes"].isin(comparable_months))]["Viajero"].sum()
annual_change = pct_change(current_total, previous_total if previous_total else None)
month_current = df[(df["Anio"] == selected_year) & (df["Mes"] == last_month)]["Viajero"].sum()
month_previous = df[(df["Anio"] == selected_year - 1) & (df["Mes"] == last_month)]["Viajero"].sum()
monthly_change = pct_change(month_current, month_previous if month_previous else None)

kpi_left, kpi_right = st.columns(2)
with kpi_left:
    period = f"{selected_year} vs. {selected_year - 1}"
    if last_month < 12:
        period += f" (ene-{MESES[last_month - 1].lower()})"
    kpi_card("Crecimiento interanual de visitantes", period, annual_change, "Periodos equivalentes")
with kpi_right:
    kpi_card(
        "Crecimiento del ultimo mes disponible",
        f"{MESES[last_month - 1]} {selected_year} vs. {MESES[last_month - 1]} {selected_year - 1}",
        monthly_change,
        "Mismo mes del año anterior",
    )

plot_config = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d"]}
history = monthly_series(df)

if "Panorama" in visible_groups:
    left, right = st.columns([1.55, 1])
    with left:
        section_title(1, "Visitantes historicos", "Seleccione un punto para actualizar el detalle mensual enlazado.")
        years_back = {"Ultimos 5 años": 5, "Ultimos 10 años": 10, "Serie completa": 99}[detail_range]
        start = history.index.max() - pd.DateOffset(years=years_back)
        shown_history = history[history.index >= start] if years_back < 99 else history
        historical_fig = go.Figure(go.Scatter(
            x=shown_history.index,
            y=shown_history.values,
            mode="lines+markers",
            marker=dict(size=5, color=PETROLEO),
            line=dict(color=PETROLEO, width=2.5),
            name="Visitantes",
            customdata=shown_history.index.strftime("%Y-%m"),
            hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> visitantes<extra></extra>",
        ))
        historical_fig.add_vrect(
            x0="2020-03-01", x1="2022-12-01", fillcolor=CORAL, opacity=0.10, line_width=0,
            annotation_text="Periodo atipico", annotation_position="top left",
        )
        chart_layout(historical_fig, 390).update_yaxes(title="Visitantes")
        selection = st.plotly_chart(
            historical_fig,
            width="stretch",
            config=plot_config,
            key="historical_selection",
            on_select="rerun",
            selection_mode="points",
        )

    selected_date = pd.Timestamp(f"{selected_year}-{last_month:02d}-01")
    try:
        points = selection.selection.points
        if points:
            selected_date = pd.Timestamp(points[0]["customdata"] + "-01")
    except (AttributeError, KeyError, TypeError):
        pass

    with right:
        section_title(2, "Detalle mensual enlazado", f"Composicion de {MESES[selected_date.month - 1]} {selected_date.year} por via.")
        month_detail = (
            df[df["Fecha"] == selected_date].groupby("Via", as_index=False)["Viajero"].sum().sort_values("Viajero")
        )
        if month_detail.empty:
            st.info("No hay informacion disponible para el mes seleccionado.")
        else:
            detail_fig = px.bar(
                month_detail, x="Viajero", y="Via", orientation="h",
                color_discrete_sequence=[TURQUESA], text_auto=",.0f",
            )
            detail_fig.update_traces(hovertemplate="%{y}<br><b>%{x:,.0f}</b> visitantes<extra></extra>")
            chart_layout(detail_fig, 390).update_layout(showlegend=False)
            detail_fig.update_xaxes(title="Visitantes")
            detail_fig.update_yaxes(title=None)
            st.plotly_chart(detail_fig, width="stretch", config=plot_config)

if "Origen y estacionalidad" in visible_groups:
    col_country, col_month, col_route = st.columns([1.1, 1, 1])
    year_data = df[df["Anio"] == selected_year]

    with col_country:
        section_title(3, "Top 5 paises de origen", f"Mayores mercados internacionales en {selected_year}.")
        countries = (
            year_data[year_data["Pais"] != "Guatemala"]
            .groupby("Pais", as_index=False)["Viajero"].sum()
            .nlargest(5, "Viajero").sort_values("Viajero")
        )
        country_fig = px.bar(countries, x="Viajero", y="Pais", orientation="h", color_discrete_sequence=[PETROLEO], text_auto=",.0f")
        country_fig.update_traces(hovertemplate="%{y}<br><b>%{x:,.0f}</b> visitantes<extra></extra>")
        chart_layout(country_fig).update_layout(showlegend=False)
        country_fig.update_xaxes(title="Visitantes")
        country_fig.update_yaxes(title=None)
        st.plotly_chart(country_fig, width="stretch", config=plot_config)

    with col_month:
        section_title(4, "Estacionalidad mensual", "Promedio por mes; excluye 2020-2022 por su caracter atipico.")
        normal_years = df[~df["Anio"].isin([2020, 2021, 2022])]
        by_year_month = normal_years.groupby(["Anio", "Mes"])["Viajero"].sum().reset_index()
        season = by_year_month.groupby("Mes", as_index=False)["Viajero"].mean()
        season["Mes_nombre"] = [MESES[int(month) - 1] for month in season["Mes"]]
        month_fig = px.bar(season, x="Mes_nombre", y="Viajero", color_discrete_sequence=[TURQUESA])
        month_fig.update_traces(hovertemplate="%{x}<br>Promedio: <b>%{y:,.0f}</b><extra></extra>")
        chart_layout(month_fig).update_layout(showlegend=False)
        month_fig.update_xaxes(title=None)
        month_fig.update_yaxes(title="Promedio de visitantes")
        st.plotly_chart(month_fig, width="stretch", config=plot_config)

    with col_route:
        section_title(5, "Vias de ingreso", f"Distribucion de visitantes durante {selected_year}.")
        routes = year_data.groupby("Via", as_index=False)["Viajero"].sum()
        route_fig = go.Figure(go.Pie(
            labels=routes["Via"], values=routes["Viajero"], hole=.58,
            marker=dict(colors=[PETROLEO, TURQUESA, "#7B9A98"], line=dict(color=BLANCO, width=2)),
            textinfo="percent", hovertemplate="%{label}<br><b>%{value:,.0f}</b> (%{percent})<extra></extra>",
        ))
        route_fig.add_annotation(text=f"<b>{routes['Viajero'].sum():,.0f}</b><br><span style='font-size:11px'>Total</span>", showarrow=False)
        chart_layout(route_fig).update_layout(legend=dict(orientation="h", y=-.08, x=.03))
        st.plotly_chart(route_fig, width="stretch", config=plot_config)

    section_title(6, "Crecimiento anual", "Cambio frente al año anterior; el ultimo año se compara a igual numero de meses.")
    annual_rows = []
    for year in sorted(df["Anio"].unique()):
        months = sorted(df.loc[df["Anio"] == year, "Mes"].unique())
        if year == min(df["Anio"]) or not months:
            continue
        upto = max(months)
        current = df[(df["Anio"] == year) & (df["Mes"] <= upto)]["Viajero"].sum()
        previous = df[(df["Anio"] == year - 1) & (df["Mes"] <= upto)]["Viajero"].sum()
        change = pct_change(current, previous if previous else None)
        if change is not None:
            annual_rows.append({"Anio": int(year), "Cambio": change})
    annual = pd.DataFrame(annual_rows)
    annual["Color"] = np.where(annual["Cambio"] < 0, CORAL, TURQUESA)
    growth_fig = go.Figure(go.Bar(
        x=annual["Anio"].astype(str), y=annual["Cambio"], marker_color=annual["Color"],
        text=annual["Cambio"].map(lambda value: f"{value:+.1f}%"), textposition="outside",
        hovertemplate="%{x}<br><b>%{y:+.1f}%</b><extra></extra>",
    ))
    chart_layout(growth_fig, 340, percent_axis=True).update_layout(showlegend=False)
    growth_fig.update_xaxes(title=None)
    growth_fig.update_yaxes(title="Variacion")
    st.plotly_chart(growth_fig, width="stretch", config=plot_config)

if "Modelos predictivos" in visible_groups:
    st.markdown("## Modelos predictivos")
    if publicacion_error:
        st.error(f"No se pudo consultar la versión publicada: {publicacion_error}")
    elif publicacion is None:
        st.info("Todavía no existe una versión aprobada. Las estimaciones no están disponibles.")
    else:
        version_id = publicacion["id"]
        meta = publicacion["metadatos"]
        try:
            ruta_csv = publicacion["ruta_predicciones"]
            forecasts = load_published_forecasts(
                str(ruta_csv), ruta_csv.stat().st_mtime_ns, version_id
            )
            modelos_info = meta["modelos"]
            if set(modelos_info) != set(VIAS):
                raise ValueError("Los modelos publicados no coinciden con las dos vías autorizadas.")
            model_specs = tuple(
                (via, str(publicacion["carpeta_modelos"] / f"{via.lower()}.joblib"),
                 (publicacion["carpeta_modelos"] / f"{via.lower()}.joblib").stat().st_mtime_ns)
                for via in VIAS
            )
            modelos_cargados = load_models(model_specs)
            if set(modelos_cargados) != set(VIAS):
                raise ValueError("No se pudieron cargar todos los modelos aprobados.")
        except Exception as error:
            st.error(f"La versión aprobada no está completa o no se puede cargar: {error}")
            st.stop()

        fecha_pub = pd.to_datetime(
            publicacion["puntero"].get("fecha_actualizacion"), utc=True, errors="coerce"
        )
        actualizacion = fecha_pub.strftime("%d/%m/%Y %H:%M UTC") if pd.notna(fecha_pub) else "No disponible"
        st.caption(
            f"Versión publicada: {version_id} · Actualización: {actualizacion} · "
            f"Procedimiento: {meta.get('version_procedimiento', 'N/D')}"
        )
        st.caption(
            "Se utilizan modelos guardados de esta misma ejecución; las cifras mostradas "
            "son las estimaciones aprobadas por el pipeline. Cambiar los filtros no vuelve a entrenar ni pronosticar."
        )
        selected_via = st.radio(
            "Modelo y vía",
            list(VIAS),
            format_func=lambda via: f"{modelos_info[via]['tipo']} · {VIAS[via]}",
            horizontal=True,
        )

        modelo_seleccionado = modelos_info[selected_via]
        futuro = forecasts.loc[forecasts["via"] == selected_via]
        observado = monthly_series(df, via=selected_via, foreign_only=True)
        ultimo_observado = observado.index.max()
        historico = observado[observado.index >= ultimo_observado - pd.DateOffset(months=35)]

        section_title(
            7, "Proyección de visitantes",
            f"{modelo_seleccionado['tipo']} · vía {VIAS[selected_via]} · "
            f"{len(futuro)} meses aprobados para publicación",
        )
        projection_fig = go.Figure()
        projection_fig.add_trace(go.Scatter(
            x=historico.index, y=historico.values, mode="lines+markers", name="Observado",
            line=dict(color=PETROLEO, width=2.5), marker=dict(size=4),
            hovertemplate="%{x|%b %Y}<br>Observado: <b>%{y:,.0f}</b><extra></extra>",
        ))
        projection_fig.add_trace(go.Scatter(
            x=futuro["fecha"], y=futuro["valor"], mode="lines+markers", name="Estimado (aprobado)",
            line=dict(color=CORAL, width=2.5, dash="dash"), marker=dict(size=7),
            hovertemplate="%{x|%b %Y}<br>Estimación aprobada: <b>%{y:,.0f}</b><extra></extra>",
        ))
        chart_layout(projection_fig, 420).update_yaxes(title="Visitantes internacionales")
        st.plotly_chart(projection_fig, width="stretch", config=plot_config)

        st.markdown(
            "<div class='note'><b>Interpretación:</b> la línea oscura representa registros observados; "
            "la línea coral muestra únicamente los meses estimados y aprobados. "
            "No se muestran intervalos de confianza porque el pipeline actual no los publica.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Evaluación de los modelos publicados")
        filas_metricas = []
        evaluaciones = {}
        for via, etiqueta in VIAS.items():
            item = modelos_info[via]
            metricas = item["metricas"]
            path_eval = REPORTES_DIR / version_id / f"evaluacion_{via.lower()}.csv"
            eval_df = None
            if path_eval.is_file():
                try:
                    eval_df = load_evaluation(str(path_eval), path_eval.stat().st_mtime_ns)
                    evaluaciones[via] = eval_df
                except (ValueError, OSError) as error:
                    st.warning(f"No se pudo consultar la evaluación de {etiqueta}: {error}")

            promedio = eval_df["observado"].mean() if eval_df is not None else np.nan
            mae = float(metricas["mae_modelo"])
            rmse = float(np.sqrt(np.mean(eval_df["Error"].to_numpy() ** 2))) if eval_df is not None else np.nan
            mae_relativo = mae / promedio * 100 if np.isfinite(promedio) and promedio > 0 else np.nan
            filas_metricas.append({
                "Modelo": item["tipo"], "Vía": etiqueta, "Código": via,
                "MAE": mae, "MAE referencia": float(metricas["mae_referencia"]),
                "RMSE": rmse, "MAE relativo": mae_relativo,
                "Meses": len(item.get("meses_evaluados", [])),
                "Período evaluado": ", ".join(item.get("meses_evaluados", [])[::11])
                if item.get("meses_evaluados") else "N/D",
            })
        metrics_df = pd.DataFrame(filas_metricas)

        performance_col, error_col = st.columns(2)
        with performance_col:
            section_title(8, "Desempeño de los dos modelos", "MAE del procedimiento frente a la referencia histórica.")
            performance_long = metrics_df.melt(
                id_vars=["Vía"], value_vars=["MAE", "MAE referencia"],
                var_name="Procedimiento", value_name="Error absoluto medio",
            )
            performance_fig = px.bar(
                performance_long, x="Vía", y="Error absoluto medio", color="Procedimiento",
                barmode="group", color_discrete_map={"MAE": PETROLEO, "MAE referencia": TURQUESA},
                text_auto=",.0f",
            )
            chart_layout(performance_fig, 390)
            performance_fig.update_xaxes(title=None)
            performance_fig.update_yaxes(title="MAE (visitantes)")
            st.plotly_chart(performance_fig, width="stretch", config=plot_config)

        with error_col:
            section_title(9, "Error a través del tiempo", f"Observado menos estimado · {VIAS[selected_via]}.")
            if selected_via in evaluaciones:
                eval_df = evaluaciones[selected_via]
                error_fig = go.Figure(go.Bar(
                    x=eval_df["Fecha"], y=eval_df["Error"],
                    marker_color=np.where(eval_df["Error"] < 0, CORAL, TURQUESA),
                    hovertemplate="%{x|%b %Y}<br>Error: <b>%{y:,.0f}</b><extra></extra>",
                ))
                error_fig.add_hline(y=0, line_color=PETROLEO, line_width=1)
                chart_layout(error_fig, 390).update_layout(showlegend=False)
                error_fig.update_yaxes(title="Error (visitantes)", rangemode="normal")
                st.plotly_chart(error_fig, width="stretch", config=plot_config)
            else:
                st.info("El reporte temporal de esta ejecución no está disponible en reportes/.")

        display_metrics = metrics_df.drop(columns="Código").copy()
        st.dataframe(
            display_metrics,
            width="stretch", hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%,.0f"),
                "MAE referencia": st.column_config.NumberColumn(format="%,.0f"),
                "RMSE": st.column_config.NumberColumn(format="%,.0f"),
                "MAE relativo": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        meses_prueba = modelo_seleccionado.get("meses_evaluados", [])
        periodo_prueba = (
            f"{meses_prueba[0]} a {meses_prueba[-1]}" if meses_prueba else "período no disponible"
        )
        st.caption(
            f"El MAE indica el error absoluto promedio en visitantes; se evaluó sobre {periodo_prueba}. "
            "El MAE de referencia corresponde al método histórico sencillo del pipeline. "
            "La comparación se realiza sobre los mismos meses observados; no es una evaluación del pronóstico futuro."
        )

st.markdown("## Metodología, alcance y limitaciones")
st.markdown(
    """
    **Fuente y alcance.** Registros mensuales del observatorio, de acuerdo con la cobertura de
    `datos/DATOS_LIMPIOS.csv`. Las visualizaciones se limitan a vías aérea y terrestre y a
    turistas y excursionistas internacionales, según los filtros del pipeline.

    **Interacción.** Los filtros modifican únicamente la presentación. El tablero consulta
    la última versión vigente y aprobada, indicada en `resultados/version_vigente.json`,
    incluidos los casos de restauración de versiones anteriores.

    **Limitaciones.** La ausencia de cobertura no equivale a cero. El ajuste excepcional
    de la pandemia se realiza dentro del pipeline, no en las visualizaciones.
    Los pronósticos no son observaciones y pueden diferir de los resultados reales.
    La aplicación no extiende el horizonte aprobado ni calcula intervalos de confianza
    no publicados; los detalles técnicos e hiperparámetros se conservan en MLflow.
    """
)

st.caption("Observatorio de Turismo Sostenible · Guatemala · Elaborado con Streamlit y Plotly")
