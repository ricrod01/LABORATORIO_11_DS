"""Observatorio interactivo de turismo sostenible de Guatemala.

Ejecucion local:
    streamlit run app.py
"""

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
CSV_PATH = ROOT / "DATOS_LIMPIOS.csv"
MODEL_PATHS = {
    "Aerea": ROOT / "modelo_aereo.joblib",
    "Maritima": ROOT / "modelo_maritimo.joblib",
    "Terrestre": ROOT / "modelo_terrestre.joblib",
}

PETROLEO = "#123B4A"
TURQUESA = "#19A7A0"
GRIS = "#F2F0EB"
CORAL = "#E45B4F"
BLANCO = "#FFFFFF"
TEXTO_SUAVE = "#5E6D72"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MODEL_INFO = {
    "Aerea": {
        "via": "A\u00e9rea",
        "nombre": "Holt-Winters",
        "corte": pd.Timestamp("2021-04-01"),
        "intervalo": "95% aproximado",
    },
    "Maritima": {
        "via": "Mar\u00edtima",
        "nombre": "SARIMAX",
        "corte": pd.Timestamp("2016-08-01"),
        "intervalo": "95%",
    },
    "Terrestre": {
        "via": "Terrestre",
        "nombre": "Prophet",
        "corte": pd.Timestamp("2021-04-01"),
        "intervalo": "80%",
    },
}


@st.cache_data(show_spinner="Preparando indicadores...")
def load_data(path: str, modified: float) -> pd.DataFrame:
    """Carga y normaliza la fuente; modified invalida el cache al cambiar el CSV."""
    data = pd.read_csv(path, low_memory=False)
    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce")
    data["Viajero"] = pd.to_numeric(data["Viajero"], errors="coerce")
    data = data.dropna(subset=["Fecha", "Viajero"])
    data = data[data["Tipo de Viajero"].isin(["Turista", "Excursionista"])].copy()
    data["Anio"] = data["Fecha"].dt.year
    data["Mes"] = data["Fecha"].dt.month
    return data


@st.cache_resource(show_spinner="Cargando modelos existentes...")
def load_models(model_stamps: tuple[tuple[str, float], ...]) -> dict:
    del model_stamps
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


def monthly_series(data: pd.DataFrame, via: str | None = None, foreign_only: bool = False) -> pd.Series:
    subset = data
    if via:
        subset = subset[subset["Via"] == via]
    if foreign_only:
        subset = subset[subset["Pais"] != "Guatemala"]
    result = subset.groupby("Fecha")["Viajero"].sum().sort_index()
    if result.empty:
        return result
    index = pd.date_range(result.index.min(), result.index.max(), freq="MS")
    return result.reindex(index, fill_value=0.0)


@st.cache_data(show_spinner="Calculando pronosticos...")
def model_forecast(model_name: str, final_date: pd.Timestamp, model_stamp: float) -> pd.DataFrame:
    """Extiende un modelo guardado, sin ajustarlo nuevamente."""
    del model_stamp
    models = load_models(tuple((name, path.stat().st_mtime) for name, path in MODEL_PATHS.items()))
    model = models[model_name]
    info = MODEL_INFO[model_name]
    dates = pd.date_range(info["corte"] + pd.offsets.MonthBegin(1), final_date, freq="MS")
    steps = len(dates)

    if model_name == "Maritima":
        result = model.get_forecast(steps=steps)
        interval = result.conf_int(alpha=0.05)
        frame = pd.DataFrame({
            "Fecha": dates,
            "Pronostico": np.asarray(result.predicted_mean, dtype=float),
            "Inferior": np.asarray(interval.iloc[:, 0], dtype=float),
            "Superior": np.asarray(interval.iloc[:, 1], dtype=float),
        })
    elif model_name == "Aerea":
        predicted = np.asarray(model.forecast(steps=steps), dtype=float)
        sigma = float(np.std(np.asarray(model.resid, dtype=float), ddof=1))
        uncertainty = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        frame = pd.DataFrame({
            "Fecha": dates,
            "Pronostico": predicted,
            "Inferior": predicted - uncertainty,
            "Superior": predicted + uncertainty,
        })
    else:
        # Prophet usa muestreo para los limites; la semilla hace estable el cache visual.
        np.random.seed(42)
        predicted = model.predict(pd.DataFrame({"ds": dates}))
        frame = pd.DataFrame({
            "Fecha": dates,
            "Pronostico": predicted["yhat"].to_numpy(),
            "Inferior": predicted["yhat_lower"].to_numpy(),
            "Superior": predicted["yhat_upper"].to_numpy(),
        })
    frame[["Pronostico", "Inferior", "Superior"]] = frame[
        ["Pronostico", "Inferior", "Superior"]
    ].clip(lower=0)
    return frame


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

if not CSV_PATH.exists():
    st.error("No se encontro DATOS_LIMPIOS.csv junto a app.py.")
    st.stop()
missing_models = [path.name for path in MODEL_PATHS.values() if not path.exists()]
if missing_models:
    st.error("Faltan modelos requeridos: " + ", ".join(missing_models))
    st.stop()

try:
    df = load_data(str(CSV_PATH), CSV_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"No fue posible cargar los datos: {error}")
    st.stop()

# Normaliza nombres internos para evitar tildes en expresiones y mantener los textos originales al mostrar.
df = df.rename(columns={"Via": "Via", "Vía": "Via", "Pais": "Pais", "País": "Pais"})
if not {"Via", "Pais"}.issubset(df.columns):
    st.error("El archivo no contiene las columnas Via y Pais esperadas.")
    st.stop()

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
    st.caption("Resultados de tres modelos ya entrenados. Cada uno representa una via distinta; no se construyen modelos nuevos.")
    model_col, horizon_col = st.columns([1.5, 1], vertical_alignment="bottom")
    with model_col:
        selected_model = st.radio(
            "Modelo y via",
            list(MODEL_INFO),
            format_func=lambda name: f"{MODEL_INFO[name]['nombre']} · {name}",
            horizontal=True,
        )
    with horizon_col:
        forecast_horizon = st.slider("Horizonte visible", 6, 24, 24, 6, format="%d meses")

    latest_date = df["Fecha"].max()
    final_date = latest_date + pd.offsets.MonthBegin(forecast_horizon)
    try:
        forecasts = {
            name: model_forecast(name, final_date, MODEL_PATHS[name].stat().st_mtime)
            for name in MODEL_INFO
        }
    except Exception as error:
        st.error(f"No fue posible generar los pronosticos guardados: {error}")
        st.stop()

    info = MODEL_INFO[selected_model]
    model_series = monthly_series(df, via=info["via"], foreign_only=True)
    selected_forecast = forecasts[selected_model]
    future = selected_forecast[selected_forecast["Fecha"] > latest_date].head(forecast_horizon)
    recent_actual = model_series[model_series.index >= latest_date - pd.DateOffset(months=35)]

    section_title(7, "Proyeccion de visitantes", f"{info['nombre']} para via {selected_model.lower()}; siguientes {forecast_horizon} meses.")
    projection_fig = go.Figure()
    projection_fig.add_trace(go.Scatter(
        x=future["Fecha"], y=future["Superior"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    projection_fig.add_trace(go.Scatter(
        x=future["Fecha"], y=future["Inferior"], mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(25,167,160,.16)", name=f"Intervalo {info['intervalo']}", hoverinfo="skip",
    ))
    projection_fig.add_trace(go.Scatter(
        x=recent_actual.index, y=recent_actual.values, mode="lines", name="Observado",
        line=dict(color=PETROLEO, width=2.5), hovertemplate="%{x|%b %Y}<br>Observado: <b>%{y:,.0f}</b><extra></extra>",
    ))
    projection_fig.add_trace(go.Scatter(
        x=future["Fecha"], y=future["Pronostico"], mode="lines+markers", name="Pronostico",
        line=dict(color=CORAL, width=2.5, dash="dash"), marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>Estimacion: <b>%{y:,.0f}</b><extra></extra>",
    ))
    projection_fig.add_vline(x=latest_date.timestamp() * 1000, line_dash="dot", line_color=TEXTO_SUAVE)
    chart_layout(projection_fig, 420).update_yaxes(title="Visitantes extranjeros")
    st.plotly_chart(projection_fig, width="stretch", config=plot_config)
    months_from_cut = (latest_date.year - info["corte"].year) * 12 + latest_date.month - info["corte"].month
    st.markdown(
        f"<div class='note'><b>Como leerlo:</b> la linea coral es una estimacion, no un dato observado. "
        f"El modelo fue entrenado hasta {info['corte']:%b %Y}; el primer mes futuro mostrado esta a "
        f"{months_from_cut + 1} pasos de ese corte, por lo que la incertidumbre aumenta con el horizonte.</div>",
        unsafe_allow_html=True,
    )

    metrics = []
    validation_frames = {}
    for name, model_info in MODEL_INFO.items():
        actual = monthly_series(df, via=model_info["via"], foreign_only=True)
        predicted = forecasts[name]
        validation = predicted[predicted["Fecha"].isin(actual.index) & (predicted["Fecha"] <= latest_date)].copy()
        validation["Observado"] = validation["Fecha"].map(actual)
        validation["Error"] = validation["Observado"] - validation["Pronostico"]
        validation = validation.dropna(subset=["Observado", "Pronostico"])
        mae = validation["Error"].abs().mean()
        rmse = np.sqrt(np.mean(validation["Error"] ** 2))
        relative_mae = mae / validation["Observado"].mean() * 100 if validation["Observado"].mean() else np.nan
        metrics.append({
            "Modelo": model_info["nombre"], "Via": name, "MAE": mae, "RMSE": rmse,
            "MAE relativo": relative_mae, "Meses": len(validation), "Corte": model_info["corte"],
        })
        validation_frames[name] = validation
    metrics_df = pd.DataFrame(metrics)

    performance_col, error_col = st.columns(2)
    with performance_col:
        section_title(8, "Desempeño de los tres modelos", "MAE relativo: error absoluto medio como porcentaje del promedio observado.")
        performance_fig = px.bar(
            metrics_df, x="Modelo", y="MAE relativo", color="Via", barmode="group",
            color_discrete_map={"Aerea": PETROLEO, "Maritima": "#7B9A98", "Terrestre": TURQUESA},
            text_auto=".1f",
            custom_data=["Via", "MAE", "RMSE", "Meses"],
        )
        performance_fig.update_traces(
            hovertemplate="%{x} · %{customdata[0]}<br>MAE relativo: <b>%{y:.1f}%</b><br>MAE: %{customdata[1]:,.0f}<br>RMSE: %{customdata[2]:,.0f}<br>Meses: %{customdata[3]}<extra></extra>"
        )
        chart_layout(performance_fig, 390, percent_axis=True)
        performance_fig.update_xaxes(title=None)
        performance_fig.update_yaxes(title="MAE relativo")
        st.plotly_chart(performance_fig, width="stretch", config=plot_config)

    with error_col:
        section_title(9, "Error a traves del tiempo", f"Diferencia observado menos estimado para {info['nombre']}.")
        validation = validation_frames[selected_model]
        error_fig = go.Figure(go.Bar(
            x=validation["Fecha"], y=validation["Error"],
            marker_color=np.where(validation["Error"] < 0, CORAL, TURQUESA),
            hovertemplate="%{x|%b %Y}<br>Error: <b>%{y:,.0f}</b><extra></extra>",
        ))
        error_fig.add_hline(y=0, line_color=PETROLEO, line_width=1)
        chart_layout(error_fig, 390).update_layout(showlegend=False)
        error_fig.update_yaxes(title="Error (visitantes)", rangemode="normal")
        st.plotly_chart(error_fig, width="stretch", config=plot_config)

    display_metrics = metrics_df.copy()
    display_metrics["Corte"] = display_metrics["Corte"].dt.strftime("%b %Y")
    display_metrics["MAE"] = display_metrics["MAE"].round(0)
    display_metrics["RMSE"] = display_metrics["RMSE"].round(0)
    display_metrics["MAE relativo"] = display_metrics["MAE relativo"].map(lambda value: f"{value:.1f}%")
    st.dataframe(
        display_metrics[["Modelo", "Via", "Corte", "Meses", "MAE", "RMSE", "MAE relativo"]],
        width="stretch",
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%,.0f"),
            "RMSE": st.column_config.NumberColumn(format="%,.0f"),
        },
    )
    st.caption(
        "MAE es el error promedio en visitantes; RMSE penaliza mas los errores grandes. "
        "Los modelos representan escalas y vias diferentes, por eso el MAE relativo es la comparacion mas util."
    )

st.markdown("## Metodologia, alcance y limitaciones")
st.markdown(
    """
    **Fuente y alcance.** Registros mensuales de INGUAT entre enero de 2009 y junio de 2026. Los indicadores
    descriptivos incluyen turistas y excursionistas. Los rankings y modelos excluyen residentes de Guatemala
    para aproximar llegadas internacionales.

    **Interaccion.** El selector anual actualiza las tarjetas, paises y vias. El nivel de detalle modifica la
    ventana historica. Seleccionar un punto del historico actualiza el desglose mensual por via. Tambien se pueden
    ocultar grupos completos de visualizaciones y cambiar el modelo u horizonte del pronostico.

    **Limitaciones.** Los datos describen asociaciones, no causas. Los años 2020-2022 se excluyen solamente del
    promedio estacional por ser atipicos. Los tres modelos tienen cortes distintos, no usan informacion posterior
    al entrenamiento y sus proyecciones lejanas deben interpretarse como escenarios orientativos. Holt-Winters
    utiliza una banda aproximada del 95%, SARIMAX una banda del 95% y Prophet su intervalo configurado del 80%.
    """
)

st.caption("Observatorio de Turismo Sostenible · Guatemala · Elaborado con Streamlit y Plotly")
