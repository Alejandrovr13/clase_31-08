import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LAT_DEFECTO = 6.2766
LON_DEFECTO = -75.5901
API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"
LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"
CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]

st.set_page_config(
    page_title="Análisis de nivel — CORNARE",
    page_icon="🌊",
    layout="wide"
)

def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {
        "desde": desde,
        "hasta": hasta,
        "calidad": calidad
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*"
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=False
        )

        if resp.status_code == 200:
            return resp.json(), None

        return None, f"HTTP {resp.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"

def obtener_todas_las_paginas(datos_json, timeout=30):
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")

    while siguiente_url:
        try:
            resp = requests.get(
                siguiente_url,
                timeout=timeout,
                verify=False
            )
        except requests.exceptions.RequestException:
            break

        if resp.status_code != 200:
            break

        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")

    return registros

def detectar_coordenadas(datos_json):
    if not isinstance(datos_json, dict):
        return LAT_DEFECTO, LON_DEFECTO, False

    lat = next(
        (datos_json[k] for k in CANDIDATOS_LAT if k in datos_json),
        None
    )

    lon = next(
        (datos_json[k] for k in CANDIDATOS_LON if k in datos_json),
        None
    )

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), True
        except (TypeError, ValueError):
            pass

    return LAT_DEFECTO, LON_DEFECTO, False

def preparar_dataframe(registros):
    df = pd.DataFrame(registros)

    df = df.rename(
        columns={
            LLAVE_FECHA: "fecha",
            LLAVE_VALOR: "nivel"
        }
    )

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="coerce"
    )

    df["nivel"] = pd.to_numeric(
        df["nivel"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["fecha"]
    ).sort_values("fecha").reset_index(drop=True)

    return df

def detectar_frecuencia(df):
    diferencias = df["fecha"].sort_values().diff().dropna()

    if diferencias.empty:
        return None

    frecuencia = diferencias.mode()

    if frecuencia.empty:
        return None

    return frecuencia.iloc[0]

def crear_serie_regular(df):
    frecuencia = detectar_frecuencia(df)

    if frecuencia is None:
        return df.copy(), None

    inicio = df["fecha"].min()
    fin = df["fecha"].max()

    indice_regular = pd.date_range(
        start=inicio,
        end=fin,
        freq=frecuencia
    )

    df_regular = (
        df.set_index("fecha")
        .reindex(indice_regular)
        .rename_axis("fecha")
        .reset_index()
    )

    return df_regular, frecuencia

def detectar_outliers(df):
    datos = df["nivel"].dropna()

    if datos.empty:
        return pd.Series(False, index=df.index), 0, 0

    q1 = datos.quantile(0.25)
    q3 = datos.quantile(0.75)

    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = (
        (df["nivel"] < limite_inferior) |
        (df["nivel"] > limite_superior) |
        (df["nivel"] < 0)
    )

    return (
        outliers,
        limite_inferior,
        limite_superior
    )

def calcular_indice_calidad(df):
    if df.empty:
        return 0.0, 0, 0

    df_regular, _ = crear_serie_regular(df)

    huecos = df_regular["nivel"].isna().sum()

    total_esperado = len(df_regular)

    if total_esperado > 0:
        completitud = 1 - (huecos / total_esperado)
    else:
        completitud = 0

    outliers, _, _ = detectar_outliers(df_regular)

    datos_validos = df_regular["nivel"].notna().sum()

    if datos_validos > 0:
        proporcion_outliers = outliers.sum() / datos_validos
    else:
        proporcion_outliers = 0

    indice = (
        completitud * 0.7 +
        (1 - proporcion_outliers) * 0.3
    ) * 100

    return (
        round(indice, 1),
        int(huecos),
        int(outliers.sum())
    )

st.sidebar.header("Parámetros de tu consulta")

nombre_estudiante = st.sidebar.text_input(
    "Nombre del estudiante",
    "Tu Nombre Aquí"
)

codigo_estacion = st.sidebar.text_input(
    "Código de estación",
    "18"
)

fecha_desde = st.sidebar.date_input(
    "Desde",
    pd.to_datetime("2026-08-23")
).strftime("%Y-%m-%d")

fecha_hasta = st.sidebar.date_input(
    "Hasta",
    pd.to_datetime("2026-08-30")
).strftime("%Y-%m-%d")

calidad = st.sidebar.selectbox(
    "Calidad",
    [1, 0],
    index=0,
    help="1 = solo datos validados"
)

consultar = st.sidebar.button(
    "🔍 Consultar",
    type="primary"
)

st.title("🌊 Análisis de nivel de ríos y quebradas — CORNARE")

st.caption(
    f"Estudiante: **{nombre_estudiante}** · "
    f"Estación: **{codigo_estacion}**"
)

if consultar:

    with st.spinner("Consultando la API..."):

        datos_crudos, error = obtener_serie_nivel(
            codigo_estacion,
            fecha_desde,
            fecha_hasta,
            calidad
        )

    if error:

        st.error(f"❌ {error}")

    else:

        registros = obtener_todas_las_paginas(
            datos_crudos
        )

        if not registros:

            st.warning(
                "No hay registros para esta estación y rango de fechas. "
                "Prueba otro código u otro rango."
            )

        else:

            df = preparar_dataframe(registros)

            lat, lon, coords_reales = detectar_coordenadas(
                datos_crudos
            )

            indice_calidad, huecos, n_outliers = calcular_indice_calidad(
                df
            )

            st.subheader("1. DataFrame de la estación")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Lecturas",
                len(df)
            )

            col2.metric(
                "Nivel promedio",
                f"{df['nivel'].mean():.2f}"
            )

            col3.metric(
                "Índice de calidad",
                f"{indice_calidad} / 100"
            )

            col4.metric(
                "Outliers",
                n_outliers
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.subheader("2. Tipos de datos y orden temporal")

            tipos = pd.DataFrame({
                "Columna": df.columns,
                "Tipo de dato": [
                    str(df[col].dtype)
                    for col in df.columns
                ]
            })

            st.dataframe(
                tipos,
                use_container_width=True
            )

            st.write(
                f"Fecha inicial: **{df['fecha'].min()}**"
            )

            st.write(
                f"Fecha final: **{df['fecha'].max()}**"
            )

            st.subheader("3. Missing values reales")

            df_regular, frecuencia = crear_serie_regular(df)

            if frecuencia is not None:

                st.write(
                    f"Frecuencia detectada: **{frecuencia}**"
                )

                missing = df_regular["nivel"].isna().sum()

                col1, col2 = st.columns(2)

                col1.metric(
                    "Registros esperados",
                    len(df_regular)
                )

                col2.metric(
                    "Missing values",
                    int(missing)
                )

                st.dataframe(
                    df_regular[df_regular["nivel"].isna()],
                    use_container_width=True
                )

            else:

                st.warning(
                    "No fue posible determinar una frecuencia regular."
                )

            st.subheader("4. Serie de nivel")

            st.line_chart(
                df.set_index("fecha")["nivel"]
            )

            st.subheader("5. Outliers con IQR + límites físicos")

            outliers, limite_inferior, limite_superior = detectar_outliers(
                df_regular
            )

            df_outliers = df_regular.copy()

            df_outliers["outlier"] = outliers

            st.write(
                f"Límite inferior IQR: **{limite_inferior:.2f}**"
            )

            st.write(
                f"Límite superior IQR: **{limite_superior:.2f}**"
            )

            st.write(
                f"Outliers encontrados: **{int(outliers.sum())}**"
            )

            st.dataframe(
                df_outliers[df_outliers["outlier"]],
                use_container_width=True
            )

            st.subheader("6. Normalización y estandarización")

            df_transformaciones = df_regular.copy()

            datos_nivel = df_transformaciones["nivel"]

            minimo = datos_nivel.min()
            maximo = datos_nivel.max()

            if maximo != minimo:

                df_transformaciones["nivel_normalizado"] = (
                    (datos_nivel - minimo) /
                    (maximo - minimo)
                )

            else:

                df_transformaciones["nivel_normalizado"] = 0

            media = datos_nivel.mean()
            desviacion = datos_nivel.std()

            if desviacion != 0:

                df_transformaciones["nivel_estandarizado"] = (
                    (datos_nivel - media) /
                    desviacion
                )

            else:

                df_transformaciones["nivel_estandarizado"] = 0

            st.dataframe(
                df_transformaciones,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write("Normalización Min-Max")

                st.line_chart(
                    df_transformaciones.set_index("fecha")[
                        "nivel_normalizado"
                    ]
                )

            with col2:

                st.write("Estandarización Z-score")

                st.line_chart(
                    df_transformaciones.set_index("fecha")[
                        "nivel_estandarizado"
                    ]
                )

            st.subheader(
                "7. Train / Validation / Test — Split cronológico"
            )

            df_modelo = df_transformaciones.dropna(
                subset=["nivel"]
            ).copy()

            total = len(df_modelo)

            limite_train = int(total * 0.70)
            limite_val = int(total * 0.85)

            train = df_modelo.iloc[
                :limite_train
            ]

            validation = df_modelo.iloc[
                limite_train:limite_val
            ]

            test = df_modelo.iloc[
                limite_val:
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Train",
                len(train)
            )

            col2.metric(
                "Validation",
                len(validation)
            )

            col3.metric(
                "Test",
                len(test)
            )

            st.write(
                f"Train: **{train['fecha'].min()}** "
                f"→ **{train['fecha'].max()}**"
            )

            st.write(
                f"Validation: **{validation['fecha'].min()}** "
                f"→ **{validation['fecha'].max()}**"
            )

            st.write(
                f"Test: **{test['fecha'].min()}** "
                f"→ **{test['fecha'].max()}**"
            )

            st.subheader("8. Estadística descriptiva")

            estadisticas = df["nivel"].describe()

            st.dataframe(
                estadisticas.to_frame(
                    name="Nivel"
                ),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Media",
                f"{df['nivel'].mean():.2f}"
            )

            col2.metric(
                "Mediana",
                f"{df['nivel'].median():.2f}"
            )

            st.subheader("9. Ubicación de la estación")

            if not coords_reales:

                st.caption(
                    "La API no trajo las coordenadas de la estación. "
                    "Se muestra el punto de referencia de Pascual Bravo."
                )

            st.map(
                pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon]
                }),
                zoom=10
            )

            st.subheader("10. Resumen de calidad")

            with st.expander(
                "Ver detalle del análisis"
            ):

                st.write(
                    f"**Estación:** {codigo_estacion}"
                )

                st.write(
                    f"**Rango:** {fecha_desde} → {fecha_hasta}"
                )

                st.write(
                    f"**Calidad consultada:** {calidad}"
                )

                st.write(
                    f"**Lecturas:** {len(df)}"
                )

                st.write(
                    f"**Missing values:** {huecos}"
                )

                st.write(
                    f"**Outliers:** {n_outliers}"
                )

                st.write(
                    f"**Índice de calidad:** {indice_calidad} / 100"
                )

            csv = df_transformaciones.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Descargar CSV completo",
                csv,
                file_name=f"analisis_estacion_{codigo_estacion}.csv",
                mime="text/csv"
            )

else:

    st.info(
        "Ajusta los parámetros en el sidebar y presiona **Consultar**."
    )
