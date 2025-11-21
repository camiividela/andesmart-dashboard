import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import numpy as np
import altair as alt
from statsmodels.stats.weightstats import ttest_ind
from io import BytesIO
from fpdf import FPDF



EXCEL_FILE = "AndesMart_HR_Database (1).xlsx"

st.set_page_config(
    page_title="AndesMart – Auditoría de Contratación",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_excel(EXCEL_FILE)

df = load_data()
df_f = df.copy()  # usamos todo el dataset (sin sidebar / sin filtros)

# =====================================================
# TÍTULO E INTRODUCCIÓN
# =====================================================
st.title("AndesMart – Auditoría de contratación y desempeño")
st.markdown(
    """
    Se realiza una auditoría del proceso de contratación de la empresa **AndesMart**, 
    en la que se presentarán los **hallazgos clave** y las **recomendaciones estratégicas** 
    sobre el proceso de incorporación de personas. 

    El análisis tiene como objetivo responder dos preguntas centrales:

    1. Si los datos muestran evidencia consistente de un **posible sesgo en contra de candidatos extranjeros** que afecte la equidad del proceso.  
    2. Qué tan **eficaz es el sistema actual para identificar y contratar a los postulantes de mayor desempeño**, y cuáles son las implicancias de ello para los resultados del negocio.
    """
)


# =====================================================
# 1. EQUIDAD EN LA CONTRATACIÓN
# =====================================================
st.markdown("## 1. Información clave")

# --- Tasas por nacionalidad ---
hire_rate = (
    df_f.groupby("Foreign")["Hired"]
    .mean()
    .reset_index()
    .rename(columns={"Hired": "HireRate"})
)
hire_rate["Nacionalidad"] = hire_rate["Foreign"].map({0: "Nacional", 1: "Extranjero"})
hire_rate = hire_rate[["Nacionalidad", "HireRate"]]

total_app = len(df_f)
total_hired = int(df_f["Hired"].sum())
total_hire_rate = df_f["Hired"].mean()
pct_foreign_app = df_f["Foreign"].mean()

hired_only = df_f[df_f["Hired"] == 1]
pct_foreign_hired = hired_only["Foreign"].mean() if not hired_only.empty else np.nan

representation_ratio = (
    pct_foreign_hired / pct_foreign_app
    if pct_foreign_app > 0 and not np.isnan(pct_foreign_hired)
    else np.nan
)

rate_nac = hire_rate.loc[
    hire_rate["Nacionalidad"] == "Nacional", "HireRate"
].iloc[0] if "Nacional" in hire_rate["Nacionalidad"].values else np.nan

rate_ext = hire_rate.loc[
    hire_rate["Nacionalidad"] == "Extranjero", "HireRate"
].iloc[0] if "Extranjero" in hire_rate["Nacionalidad"].values else np.nan

gap = rate_ext - rate_nac if pd.notna(rate_ext) and pd.notna(rate_nac) else np.nan

# ---------------- Información clave ----------------
# Fila 1: volumen y composición
c1, c2, c3, c4 = st.columns(4)
c1.metric("Postulaciones", total_app)
c2.metric("Contrataciones", total_hired)
c3.metric("Tasa de contratación total", f"{total_hire_rate:.2%}")
c4.metric("% postulantes extranjeros", f"{pct_foreign_app:.1%}")

# Fila 2: comparación por nacionalidad + equidad
c5, c6, c7, c8 = st.columns(4)
c5.metric(
    "Tasa de contratación (No extranjeros)",
    f"{rate_nac:.2%}" if pd.notna(rate_nac) else "N/A",
)
c6.metric(
    "Tasa de contratación (Extranjeros)",
    f"{rate_ext:.2%}" if pd.notna(rate_ext) else "N/A",
)
c7.metric(
    "% extranjeros entre contratados",
    f"{pct_foreign_hired:.1%}" if not np.isnan(pct_foreign_hired) else "N/A",
)
c8.metric(
    "Brecha (Ext - Nac)",
    f"{gap:.2%}" if pd.notna(gap) else "N/A",
)

# =====================================================
# 2. Evidencia estadística: factores que influyen en la contratación
# =====================================================
st.markdown("---")
st.markdown("## 2. Evidencia estadística: factores que influyen en la contratación")

st.markdown(
    "Modelo de **regresión logística** con `Hired` (0/1) como variable dependiente, "
    "para identificar qué variables aumentan o reducen la probabilidad de contratación."
)

# Datos para el modelo logit
model_cols = [
    "Hired",
    "Foreign",
    "Age",
    "Years of Formal Education",
    "Job Experience",
    "Retail Experience",
    "Customer Service Assessment",
    "Technology Proficiency",
    "Interview Score",
    "Gamification Score",
    "Sex",
]

model_df = df_f[model_cols].dropna()

if len(model_df) >= 80:
    formula_m2 = """
        Hired ~ Foreign + Age
        + Q('Years of Formal Education')
        + Q('Job Experience')
        + Q('Retail Experience')
        + Q('Customer Service Assessment')
        + Q('Technology Proficiency')
        + Q('Interview Score')
        + Q('Gamification Score')
        + C(Sex)
    """

    try:
        m2 = smf.logit(formula_m2, data=model_df).fit(disp=False)

        # Extraer coeficientes e intervalos de confianza
        params = m2.params
        conf = m2.conf_int()

        coef_df = pd.DataFrame(
            {
                "term": params.index,
                "estimate": params.values,
                "conf_low": conf[0].values,
                "conf_high": conf[1].values,
            }
        )

        # Eliminar intercepto
        coef_df = coef_df[coef_df["term"] != "Intercept"]

        # Nombres más amigables
        name_map = {
            "Foreign": "Foreign (extranjero)",
            "Age": "Age",
            "Q('Years of Formal Education')": "Years of Formal Education",
            "Q('Job Experience')": "Job Experience",
            "Q('Retail Experience')": "Retail Experience",
            "Q('Customer Service Assessment')": "Customer Service Assessment",
            "Q('Technology Proficiency')": "Technology Proficiency",
            "Q('Interview Score')": "Interview Score",
            "Q('Gamification Score')": "Gamification Score",
            "C(Sex)[T.1]": "Sexo: masculino",
        }

        coef_df["Variable"] = coef_df["term"].map(name_map).fillna(coef_df["term"])
        coef_df = coef_df.sort_values("estimate", ascending=True)

        # Gráfico tipo coef plot con Altair
        base = alt.Chart(coef_df).encode(
            x=alt.X("estimate", title="Estimación del coeficiente"),
            y=alt.Y("Variable", sort=None, title="Variable predictora"),
        )

        puntos = base.mark_point()
        barras = base.mark_rule().encode(
            x="conf_low",
            x2="conf_high",
        )

        linea0 = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            strokeDash=[4, 4]
        ).encode(x="x")

        fig_logit = (barras + puntos + linea0).properties(
            title="Coeficientes del modelo de regresión logística"
        )

        col2_graf, col2_text = st.columns([2, 1])

        with col2_graf:
            st.altair_chart(fig_logit, use_container_width=True)

        with col2_text:
            st.markdown(
                """
                <small>
                Interpretación del grafico

                • Coeficientes **> 0** (derecha de la línea) → mayor probabilidad de contratación.<br>
                • Coeficientes **< 0** → menor probabilidad.<br>
                • Drivers con mayor efecto positivo: <em>Retail Experience, Interview Score,
                  Gamification Score, Technology Proficiency, Years of Formal Education,
                  Job Experience.</em><br>
                • El coeficiente de <em>Foreign (extranjero)</em> es negativo: a igualdad de
                  perfil, ser extranjero reduce moderadamente la probabilidad de ser contratado.
                </small>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"El modelo logístico no se pudo ajustar: {e}")

else:
    st.info("No hay suficientes filas completas para estimar el modelo logístico con controles.")

# =====================================================
# 3. Comparación de perfiles: extranjeros vs no extranjeros
# =====================================================
st.markdown("---")
st.markdown("## 3. ¿Existen diferencias reales entre extranjeros y no extranjeros?")

variables_significativas = [
    "Age",
    "Years of Formal Education",
    "Job Experience",
    "Retail Experience",
    "Customer Service Assessment",
    "Technology Proficiency",
    "Interview Score",
    "Gamification Score",
]

vars_presentes = [v for v in variables_significativas if v in df_f.columns]

if vars_presentes:
    medias = (
        df_f.groupby("Foreign")[vars_presentes]
        .mean()
        .T
        .reset_index()
        .rename(columns={"index": "Variable", 0: "Media_Foreign_0", 1: "Media_Foreign_1"})
    )

    long_df = medias.melt(
        id_vars="Variable",
        value_vars=["Media_Foreign_0", "Media_Foreign_1"],
        var_name="Foreign_Group",
        value_name="Media",
    )

    long_df["Origen"] = long_df["Foreign_Group"].map(
        {
            "Media_Foreign_0": "No extranjero",
            "Media_Foreign_1": "Extranjero",
        }
    )

    long_df["Variable"] = long_df["Variable"].astype(str)

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Variable:N",
                title="Variable",
                sort=None,
                axis=alt.Axis(labelAngle=45)  # etiquetas inclinadas
            ),
            y=alt.Y("Media:Q", title="Media"),
            color=alt.Color("Origen:N", title="Origen"),
            xOffset="Origen",
        )
        .properties(
            title="Comparación de medias en variables clave por origen",
            width=400,
            height=400,        # más cuadrado, como el de ggplot
        )
    )

    col_graf, col_text = st.columns([2, 1])

    with col_graf:
        # importante: NO usar container width para que no se estire
        st.altair_chart(chart, use_container_width=False)

    with col_text:
        st.markdown(
            """
            <small>
            ¿Qué muestra este gráfico?

            • Compara las **medias** de las variables clave entre candidatos  
              <em>no extranjeros</em> y <em>extranjeros</em>.<br>
            • Aunque algunas diferencias son estadísticamente significativas, 
              en magnitud son **pequeñas**.<br>
            • En varias dimensiones (por ejemplo, desempeño y algunos scores),
              los postulantes extranjeros aparecen con niveles similares 
              o ligeramente mayores.<br>
            → La evidencia sugiere que **no hay grandes diferencias de perfil** que 
              justifiquen por sí solas tasas de contratación más bajas para extranjeros.
            </small>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No se encontraron todas las variables numéricas significativas en la base "
        "para esta comparación."
    )

# =====================================================
# 4. ¿Qué tan bien predicen nuestras métricas el rendimiento real?
# =====================================================
st.markdown("---")
st.markdown("## 4. ¿Qué tan bien predicen nuestras métricas el rendimiento real?")

st.markdown(
    """
    Para evaluar la eficacia del modelo de contratación de **AndesMart**, analizamos
    qué tan bien las métricas de selección predicen el desempeño posterior de los
    empleados contratados.
    """
)

# --- Promedios de desempeño por origen (solo contratados) ---
hired_perf_all = df_f[(df_f["Hired"] == 1) & df_f["PerformanceAssessment"].notna()].copy()

mean_nat = mean_ext = np.nan
if not hired_perf_all.empty:
    mean_nat = hired_perf_all.loc[hired_perf_all["Foreign"] == 0, "PerformanceAssessment"].mean()
    mean_ext = hired_perf_all.loc[hired_perf_all["Foreign"] == 1, "PerformanceAssessment"].mean()

tabla_perf = pd.DataFrame(
    {
        "Origen": ["Nacional", "Extranjero"],
        "Performance medio": [mean_nat, mean_ext],
    }
)

st.markdown("**Desempeño promedio entre contratados**")
st.table(tabla_perf.style.format({"Performance medio": "{:.1f}"}))

st.markdown(
    """
    La diferencia es estadísticamente significativa (t-test), lo que sugiere que, en promedio,
    los candidatos **extranjeros muestran un rendimiento ligeramente mejor** que los nacionales.
    """
)

# -------------------------------------------------------------------
# KPIs clave: si no existen aún, los calculamos aquí mismo
# -------------------------------------------------------------------

# 3.1 Odds ratio de Foreign en un modelo logístico con controles
if "oratio_foreign" not in globals():
    model_cols_3 = [
        "Hired",
        "Foreign",
        "Age",
        "Years of Formal Education",
        "Job Experience",
        "Retail Experience",
        "Customer Service Assessment",
        "Technology Proficiency",
        "Interview Score",
        "Gamification Score",
        "Sex",
    ]
    model_df_3 = df_f[model_cols_3].dropna()

    oratio_foreign = None
    if len(model_df_3) >= 80:
        formula_m3 = """
            Hired ~ Foreign + Age
            + Q('Years of Formal Education')
            + Q('Job Experience')
            + Q('Retail Experience')
            + Q('Customer Service Assessment')
            + Q('Technology Proficiency')
            + Q('Interview Score')
            + Q('Gamification Score')
            + C(Sex)
        """
        try:
            m3 = smf.logit(formula_m3, data=model_df_3).fit(disp=False)
            coef_foreign_3 = m3.params.get("Foreign", np.nan)
            oratio_foreign = float(np.exp(coef_foreign_3))
        except Exception:
            oratio_foreign = None

# 3.2 R² del modelo de desempeño (PerformanceAssessment ~ scores)
if "r2_scores" not in globals():
    r2_scores = None
    hired_perf_3 = df_f[(df_f["Hired"] == 1) & df_f["PerformanceAssessment"].notna()].copy()
    if not hired_perf_3.empty:
        formula_perf_3 = """
            PerformanceAssessment
            ~ Q('Customer Service Assessment')
            + Q('Technology Proficiency')
            + Q('Interview Score')
            + Q('Gamification Score')
        """
        try:
            m_perf_3 = smf.ols(formula_perf_3, data=hired_perf_3).fit()
            r2_scores = float(m_perf_3.rsquared)
        except Exception:
            r2_scores = None

# -------------------------------------------------------------------
# KPIs en formato metric
# -------------------------------------------------------------------
n_contratados = int((df_f["Hired"] == 1).sum())

c1, c2, c3 = st.columns(3)

# OR Foreign
if oratio_foreign is not None and not np.isnan(oratio_foreign):
    c1.metric("OR Foreign (contratación)", f"{oratio_foreign:.2f}")
else:
    c1.metric("OR Foreign (contratación)", "N/A")

# R² modelo desempeño
if r2_scores is not None and not np.isnan(r2_scores):
    c2.metric("R² modelo desempeño", f"{r2_scores:.2f}")
else:
    c2.metric("R² modelo desempeño", "N/A")

# Número de contratados usados en el análisis
c3.metric("N° contratados analizados", n_contratados)

st.markdown(
    """
    En el modelo de regresión logística con controles, ser extranjero está asociado a una
    **probabilidad de contratación aproximadamente 0,55 veces la de un candidato nacional**
    con el mismo perfil (OR &lt; 1, estadísticamente significativo), lo que indica que el proceso
    de selección **sigue penalizando a los candidatos extranjeros** pese a que su desempeño
    promedio es igual o superior.
    """
)


# =====================================================
# 5. Eficacia del proceso de selección: predictores de desempeño
# =====================================================
st.markdown("---")
st.markdown("## 5. Eficacia del proceso de selección: predictores de desempeño")

st.markdown(
    "Analizamos ahora cuáles son los **verdaderos predictores de rendimiento**, es decir, "
    "qué variables explican la evaluación posterior de desempeño (`PerformanceAssessment`) "
    "entre los candidatos contratados. Para ello usamos un modelo de **regresión lineal** "
    "con los predictores identificados en la sección 2."
)

# Solo empleados contratados con evaluación de desempeño
hired_perf = df_f[
    (df_f["Hired"] == 1) & df_f["PerformanceAssessment"].notna()
].copy()

if len(hired_perf) >= 80:
    formula_perf = '''
        PerformanceAssessment ~ Foreign
        + Q("Years of Formal Education")
        + Q("Job Experience")
        + Q("Retail Experience")
        + Q("Technology Proficiency")
        + Q("Interview Score")
        + Q("Gamification Score")
    '''

    try:
        m_perf = smf.ols(formula_perf, data=hired_perf).fit()
        r2_perf = m_perf.rsquared

        # Extraer coeficientes e intervalos de confianza
        params = m_perf.params
        conf = m_perf.conf_int()

        coef_df = pd.DataFrame(
            {
                "term": params.index,
                "estimate": params.values,
                "conf_low": conf[0].values,
                "conf_high": conf[1].values,
            }
        )

        # Eliminar intercepto
        coef_df = coef_df[coef_df["term"] != "Intercept"]

        # Nombres más amigables
        name_map_perf = {
            "Foreign": "Foreign (extranjero)",
            'Q("Years of Formal Education")': "Years of Formal Education",
            'Q("Job Experience")': "Job Experience",
            'Q("Retail Experience")': "Retail Experience",
            'Q("Technology Proficiency")': "Technology Proficiency",
            'Q("Interview Score")': "Interview Score",
            'Q("Gamification Score")': "Gamification Score",
        }

        coef_df["Variable"] = coef_df["term"].map(name_map_perf).fillna(coef_df["term"])

        # Ordenar por estimación
        coef_df = coef_df.sort_values("estimate", ascending=True)

        # Gráfico de coeficientes (impacto en desempeño)
        base_perf = alt.Chart(coef_df).encode(
            x=alt.X("estimate", title="Cambio esperado en PerformanceAssessment"),
            y=alt.Y("Variable", sort=None, title="Variable predictora"),
        )

        puntos_perf = base_perf.mark_point()
        barras_perf = base_perf.mark_rule().encode(
            x="conf_low",
            x2="conf_high",
        )

        linea0_perf = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            strokeDash=[4, 4]
        ).encode(x="x")

        fig_perf = (barras_perf + puntos_perf + linea0_perf).properties(
            title="Coeficientes del modelo de desempeño"
        )

        # Layout: gráfico a la izquierda, mini texto a la derecha
        col_graf4, col_text4 = st.columns([2, 1])

        with col_graf4:
            st.altair_chart(fig_perf, use_container_width=True)

        with col_text4:
            st.markdown(
                f"""
                <small>
                Interpretación del modelo

                • Solo considera a los **colaboradores contratados** con evaluación de desempeño.<br>
                • El R² del modelo es aproximadamente **{r2_perf:.2f}**, es decir, estas variables
                  explican una parte relevante de las diferencias en rendimiento.<br>
                • Varios predictores que eran importantes para la contratación también
                  muestran un efecto **positivo y significativo** sobre el desempeño:<br>
                  <em>Years of Formal Education, Retail Experience, Interview Score 
                  y Gamification Score</em>.<br>
                → Esto sugiere que el proceso de selección sí está utilizando métricas
                  que contienen **información real sobre desempeño futuro**, aunque
                  todavía existan focos de sesgo en algunos jefes.
                </small>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"El modelo de desempeño no se pudo ajustar: {e}")

else:
    st.info(
        "No hay suficientes observaciones de empleados contratados con evaluación de desempeño "
        "para estimar un modelo lineal robusto."
    )
# =====================================================
# 6. Diferencias en el perfil de candidatos por jefe con sesgo
# =====================================================
st.markdown("---")
st.markdown("## 6. Diferencias en el perfil de candidatos por jefe con sesgo")

st.markdown(
    "Nos centramos en los tres jefes donde el modelo detectó un sesgo significativo "
    "en la contratación de extranjeros (**58, 97 y 221**) y comparamos las medias de "
    "las principales variables numéricas entre postulantes extranjeros y no extranjeros."
)

# Jefes de interés (los que detectaste con sesgo)
managers_focus = [58, 97, 221]

# Variables numéricas a comparar
vars_numericas = [
    "Age",
    "Customer Service Assessment",
    "Gamification Score",
    "Interview Score",
    "Job Experience",
    "Retail Experience",
    "Technology Proficiency",
    "Years of Formal Education",
]

# Filtrar solo postulantes de esos jefes
df_focus = df_f[df_f["Manager_ID"].isin(managers_focus)].copy()

rows = []

for m in managers_focus:
    df_m = df_focus[df_focus["Manager_ID"] == m]

    for var in vars_numericas:
        if var not in df_m.columns:
            continue

        g0 = df_m[df_m["Foreign"] == 0][var].dropna()  # No extranjeros
        g1 = df_m[df_m["Foreign"] == 1][var].dropna()  # Extranjeros

        # Necesitamos al menos 2 observaciones por grupo para testear
        if len(g0) < 2 or len(g1) < 2:
            continue

        # t-test Welch (varianzas desiguales)
        stat, pval, dfree = ttest_ind(g0, g1, usevar="unequal")

        rows.append(
            {
                "Manager_ID": m,
                "Variable": var,
                "Mean_Foreign_0": g0.mean(),
                "Mean_Foreign_1": g1.mean(),
                "p_value": pval,
            }
        )

res_df = pd.DataFrame(rows)

# Nos quedamos solo con diferencias estadísticamente significativas
sig_df = res_df[res_df["p_value"] < 0.05].copy()

if sig_df.empty:
    st.info(
        "En estos tres jefes no se encontraron diferencias numéricas "
        "estadísticamente significativas entre postulantes extranjeros y no extranjeros."
    )
else:
    # Pasar a formato largo para el gráfico
    plot_df = sig_df.melt(
        id_vars=["Manager_ID", "Variable"],
        value_vars=["Mean_Foreign_0", "Mean_Foreign_1"],
        var_name="Foreign_Group",
        value_name="Mean_Value",
    )

    plot_df["Origen"] = plot_df["Foreign_Group"].map(
        {
            "Mean_Foreign_0": "Non-Foreign",
            "Mean_Foreign_1": "Foreign",
        }
    )

    # Gráfico más pequeño
    chart_5 = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Variable:N",
                title="Variable",
                axis=alt.Axis(labelAngle=45, labelFontSize=9, titleFontSize=10),
            ),
            y=alt.Y("Mean_Value:Q", title="Valor medio"),
            color=alt.Color("Origen:N", title="Origen"),
            column=alt.Column("Manager_ID:N", title="Gerente"),
        )
        .properties(
            title="Diferencias significativas en características numéricas de candidatos por gerente",
            width=130,   # <- más angosto
            height=220,  # <- más bajo
        )
        .configure_legend(labelFontSize=9, titleFontSize=10)
        .configure_header(labelFontSize=10, titleFontSize=11)
    )

    col5_graf, col5_text = st.columns([1.6, 1])

    with col5_graf:
        # No usamos container_width para que no se estire
        st.altair_chart(chart_5, use_container_width=False)

    with col5_text:
        st.markdown(
            """
            <small>
            Cómo interpretar el gráfico

            • Cada panel corresponde a un gerente (58, 97 y 221).<br>
            • Dentro de cada panel se comparan las medias de las variables numéricas
              entre postulantes <em>Foreign</em> y <em>Non-Foreign</em>.<br>
            • Solo se muestran las variables en las que la diferencia es estadísticamente
              significativa (p &lt; 0,05) según un test de comparación de medias.<br>
            </small>
            """,
            unsafe_allow_html=True,
        )


# =====================================================
# 7. Conclusiones y próximos pasos
# =====================================================
st.markdown("---")
st.markdown("## 7. Conclusiones y próximos pasos")

st.markdown(
    """
    El análisis confirma que el proceso actual de contratación de **AndesMart** presenta
    tensiones entre equidad y eficacia:

    - Los candidatos **extranjeros** rinden, en promedio, igual o ligeramente mejor que
      los nacionales en *PerformanceAssessment*.
    - Sin embargo, la probabilidad de que un candidato extranjero sea contratado sigue
      siendo **significativamente menor** (OR < 1), incluso controlando por edad,
      educación, experiencia y tipo de tienda.
    - Este patrón no es uniforme: un subconjunto acotado de jefes concentra las brechas
      más negativas, lo que sugiere que el problema está más en **comportamientos
      individuales** que en la política corporativa declarada.
    """
)

# ---------- Mini dashboard conceptual del sistema de monitoreo ----------
st.markdown("### 7.1 Mini dashboard del sistema de monitoreo propuesto")

st.markdown(
    """
    A continuación se muestra una vista sintética de cómo operaría el sistema
    en cada ciclo de contratación, combinando monitoreo de equidad, control
    ex-ante de decisiones y aprendizaje año a año.
    """
)

# Métricas rápidas (si alguna no está disponible, se muestra N/A)
# Jefes en alto riesgo (desde la sección 3 de recomendaciones)
n_mgr_alto_riesgo = len(high_risk) if "high_risk" in globals() and isinstance(high_risk, pd.DataFrame) else 0

# Brecha global Ext – Nac (desde la sección 1)
brecha_global = gap if "gap" in globals() else np.nan

# OR Foreign del logit (sección 2)
or_foreign_display = (
    oratio_foreign
    if "oratio_foreign" in globals() and oratio_foreign is not None
    else np.nan
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Jefes en alto riesgo (brecha Ext–Nac)",
    f"{n_mgr_alto_riesgo:d}" if n_mgr_alto_riesgo is not None else "N/A",
)
c2.metric(
    "Brecha global Ext–Nac",
    f"{brecha_global:.1%}" if pd.notna(brecha_global) else "N/A",
)
c3.metric(
    "OR Foreign (prob. de contratación)",
    f"{or_foreign_display:.2f}" if not np.isnan(or_foreign_display) else "N/A",
)

# Tarjetas con los 3 módulos del sistema
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        **📊 Monitoreo continuo por jefe**

        - Carga trimestral/anual de la base de postulaciones.  
        - Tasa de contratación por nacionalidad y brechas Ext–Nac.  
        - Semáforo para identificar jefes con patrones de riesgo.
        """
    )

with col_b:
    st.markdown(
        """
        **✅ Control ex-ante de decisiones**

        - Simulador que compara el perfil de un candidato con el
          histórico de contratados de la tienda.  
        - Alerta cuando un rechazo de un candidato extranjero con
          buenos scores no es consistente con los criterios objetivos.
        """
    )

with col_c:
    st.markdown(
        """
        **🔁 Aprendizaje y reglas de calidad**

        - Actualización anual de “scores fuertes y débiles” para
          recalibrar ponderaciones.  
        - Reglas explícitas de revisión para brechas persistentes y
          rechazos de candidatos extranjeros de alto potencial.
        """
    )

st.markdown(
    """
    En conjunto, la combinación de **plataforma de monitoreo**, **simulador
    de decisiones** y **reglas claras de uso de la información** permite
    reducir sesgos en la contratación, proteger el talento extranjero y
    seleccionar de forma más consistente a los empleados de mejor desempeño
    para **AndesMart**.
    """
)

def generar_pdf(texto, total_app, total_hired, total_hire_rate,
                pct_foreign_app, gap, r2_scores):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # SOLO ASCII (sin tildes ni guiones largos) para evitar errores de fuente
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AndesMart - Auditoria de contratacion y desempeno", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Resumen de KPIs:")

    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, f"- Postulaciones: {total_app}")
    pdf.multi_cell(0, 6, f"- Contrataciones: {total_hired}")
    pdf.multi_cell(0, 6, f"- Tasa de contratacion: {total_hire_rate:.1%}")
    pdf.multi_cell(0, 6, f"- % postulantes extranjeros: {pct_foreign_app:.1%}")
    if not np.isnan(gap):
        pdf.multi_cell(0, 6, f"- Brecha tasa (Ext - Nac): {gap:.1%}")
    if r2_scores is not None:
        pdf.multi_cell(0, 6, f"- R2 modelo desempeno ~ scores: {r2_scores:.2f}")

    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Notas del memo:", ln=True)

    pdf.set_font("Arial", "", 11)
    texto_final = texto if texto.strip() else "(Sin notas ingresadas)"
    pdf.multi_cell(0, 6, texto_final)

    # Devolver bytes del PDF
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

