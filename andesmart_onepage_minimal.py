import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import numpy as np

EXCEL_FILE = "AndesMart_HR_Database (1).xlsx"

st.set_page_config(
    page_title="AndesMart – Auditoría de Contratación",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_excel(EXCEL_FILE)

df = load_data()

st.title("AndesMart – Auditoría de contratación y desempeño")

st.sidebar.header("Filtros")

cities = ["Todos"] + sorted(df["City"].dropna().unique().tolist())
store_types = ["Todos"] + sorted(df["Store_Type"].dropna().unique().tolist())
managers = ["Todos"] + sorted(df["Manager_ID"].dropna().unique().tolist())
months = ["Todos"] + sorted(df["Month"].dropna().unique().tolist())

city_filter = st.sidebar.selectbox("Ciudad", cities)
store_type_filter = st.sidebar.selectbox("Tipo de tienda", store_types)
manager_filter = st.sidebar.selectbox("Jefe de tienda", managers)
month_filter = st.sidebar.selectbox("Mes", months)

df_f = df.copy()
if city_filter != "Todos":
    df_f = df_f[df_f["City"] == city_filter]
if store_type_filter != "Todos":
    df_f = df_f[df_f["Store_Type"] == store_type_filter]
if manager_filter != "Todos":
    df_f = df_f[df_f["Manager_ID"] == manager_filter]
if month_filter != "Todos":
    df_f = df_f[df_f["Month"] == month_filter]

st.sidebar.write(f"Postulaciones filtradas: **{len(df_f)}**")

if df_f.empty:
    st.warning("No hay datos con estos filtros. Cambia la selección en la barra lateral.")
    st.stop()

# =====================================================
# 1. EQUIDAD EN LA CONTRATACIÓN
# =====================================================
st.markdown("## 1. Equidad en la contratación")

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Postulaciones", total_app)
c2.metric("Contrataciones", total_hired)
c3.metric("Tasa contratación", f"{total_hire_rate:.1%}")
c4.metric("% postulantes extranjeros", f"{pct_foreign_app:.1%}")

c5, c6, c7 = st.columns(3)
c5.metric("% extranjeros entre contratados",
          f"{pct_foreign_hired:.1%}" if not np.isnan(pct_foreign_hired) else "N/A")
c6.metric("Ratio representación (hires/apps)",
          f"{representation_ratio:.2f}" if not np.isnan(representation_ratio) else "N/A")
c7.metric("Brecha (Ext – Nac)",
          f"{gap:.1%}" if pd.notna(gap) else "N/A")

st.bar_chart(hire_rate.set_index("Nacionalidad")["HireRate"])

pivot_mgr = (
    df_f.pivot_table(
        index="Manager_ID",
        columns="Foreign",
        values="Hired",
        aggfunc="mean"
    )
    .rename(columns={0: "Tasa_Nacional", 1: "Tasa_Extranjero"})
)

if not pivot_mgr.empty:
    pivot_mgr["Brecha_ext_menos_nac"] = (
        pivot_mgr["Tasa_Extranjero"] - pivot_mgr["Tasa_Nacional"]
    )
    mgr_gap = pivot_mgr.reset_index().dropna(subset=["Brecha_ext_menos_nac"])

    st.markdown("**Top 5 jefes con brecha más negativa (Ext – Nac):**")
    worst_mgrs = mgr_gap.sort_values("Brecha_ext_menos_nac").head(5)
    st.dataframe(
        worst_mgrs.style.format(
            {
                "Tasa_Nacional": "{:.1%}",
                "Tasa_Extranjero": "{:.1%}",
                "Brecha_ext_menos_nac": "{:.1%}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("No hay información suficiente por jefe con estos filtros.")

# Modelo logit simple con controles
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
coef_foreign = pvalue_foreign = oratio_foreign = None

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
        coef_foreign = m2.params["Foreign"]
        pvalue_foreign = m2.pvalues["Foreign"]
        oratio_foreign = np.exp(coef_foreign)

        c8, c9, c10 = st.columns(3)
        c8.metric("Coef. Foreign (log-odds)", f"{coef_foreign:.3f}")
        c9.metric("Odds ratio Foreign", f"{oratio_foreign:.2f}")
        c10.metric("p-valor Foreign", f"{pvalue_foreign:.3g}")
    except Exception as e:
        st.error(f"Logit no se pudo ajustar: {e}")
else:
    st.info("No hay suficientes filas completas para un modelo logit robusto.")

# =====================================================
# 2. CALIDAD DEL PROCESO DE SELECCIÓN
# =====================================================
st.markdown("---")
st.markdown("## 2. Calidad del proceso de selección")

hired_perf = df_f[
    (df_f["Hired"] == 1) & df_f["PerformanceAssessment"].notna()
].copy()

r2_scores = None
strong_scores = []
weak_scores = []

if hired_perf.empty:
    st.warning("No hay empleados con evaluación de desempeño para este filtro.")
else:
    cols_corr = [
        "PerformanceAssessment",
        "Customer Service Assessment",
        "Technology Proficiency",
        "Interview Score",
        "Gamification Score",
    ]
    corr_matrix = hired_perf[cols_corr].corr()
    st.markdown("**Correlaciones con desempeño:**")
    st.dataframe(corr_matrix.style.format("{:.2f}"), use_container_width=True)

    formula_perf = """
        PerformanceAssessment
        ~ Q('Customer Service Assessment')
        + Q('Technology Proficiency')
        + Q('Interview Score')
        + Q('Gamification Score')
    """

    try:
        m_perf = smf.ols(formula_perf, data=hired_perf).fit()
        r2_scores = m_perf.rsquared

        st.metric("R² modelo desempeño ~ scores", f"{r2_scores:.2f}")

        coef_table = m_perf.params.to_frame("Coef").join(
            m_perf.pvalues.to_frame("p-valor")
        )
        st.dataframe(
            coef_table.style.format({"Coef": "{:.3f}", "p-valor": "{:.3f}"}),
            use_container_width=True,
        )

        score_vars = [
            "Customer Service Assessment",
            "Technology Proficiency",
            "Interview Score",
            "Gamification Score",
        ]
        for sv in score_vars:
            param = f"Q('{sv}')"
            beta = m_perf.params.get(param, np.nan)
            pval = m_perf.pvalues.get(param, np.nan)
            if np.isnan(beta) or np.isnan(pval):
                continue
            if pval < 0.05 and beta > 0:
                strong_scores.append((sv, beta, pval))
            elif pval >= 0.05:
                weak_scores.append((sv, beta, pval))

    except Exception as e:
        st.error(f"OLS desempeño no se pudo ajustar: {e}")

    score_choice = st.selectbox(
        "Score para ver por niveles",
        [
            "Customer Service Assessment",
            "Technology Proficiency",
            "Interview Score",
            "Gamification Score",
        ],
    )
    hired_perf["quartile"] = pd.qcut(hired_perf[score_choice], 4, labels=False)
    table_q = (
        hired_perf.groupby("quartile")["PerformanceAssessment"]
        .mean()
        .reset_index()
        .rename(columns={"PerformanceAssessment": "Desempeño promedio"})
    )
    table_q["Nivel (0=bajo, 3=alto)"] = table_q["quartile"]
    table_q = table_q[["Nivel (0=bajo, 3=alto)", "Desempeño promedio"]]

    st.dataframe(table_q.style.format({"Desempeño promedio": "{:.2f}"}), use_container_width=True)
    st.bar_chart(table_q.set_index("Nivel (0=bajo, 3=alto)"))

# =====================================================
# 3. MINI SISTEMA DE RECOMENDACIONES
# =====================================================
st.markdown("---")
st.markdown("## 3. Mini sistema de recomendaciones")

cp1, cp2, cp3 = st.columns(3)
fairness_threshold = cp1.slider(
    "Umbral brecha Ext–Nac para 'alto riesgo' (en %)",
    min_value=-50,
    max_value=0,
    value=-10,
    step=1,
) / 100.0
min_apps = cp2.number_input(
    "Mínimo postulaciones por jefe",
    min_value=10,
    max_value=1000,
    value=30,
    step=5,
)
min_foreign_share = cp3.slider(
    "Mínimo % postulantes extranjeros por jefe",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
) / 100.0

group_mgr = (
    df_f.groupby("Manager_ID")
    .agg(
        Postulaciones=("Applicant_ID", "count"),
        Contrataciones=("Hired", "sum"),
        Pct_extranjeros=("Foreign", "mean"),
        Tasa_contratación=("Hired", "mean"),
    )
)

pivot_mgr_full = (
    df_f.pivot_table(
        index="Manager_ID",
        columns="Foreign",
        values="Hired",
        aggfunc="mean",
    )
    .rename(columns={0: "Tasa_Nacional", 1: "Tasa_Extranjero"})
)

mgr_full = group_mgr.join(pivot_mgr_full, how="left")
mgr_full["Brecha_ext_menos_nac"] = (
    mgr_full["Tasa_Extranjero"] - mgr_full["Tasa_Nacional"]
)

cond_apps = mgr_full["Postulaciones"] >= min_apps
cond_foreign = mgr_full["Pct_extranjeros"] >= min_foreign_share
cond_gap = mgr_full["Brecha_ext_menos_nac"] <= fairness_threshold

high_risk = mgr_full[cond_apps & cond_foreign & cond_gap].dropna(
    subset=["Brecha_ext_menos_nac"]
)
medium_risk = mgr_full[
    cond_apps
    & cond_foreign
    & ~cond_gap
    & mgr_full["Brecha_ext_menos_nac"].notna()
]

st.markdown("### 3.1 Jefes en alto riesgo y riesgo medio")

st.markdown("**Alto riesgo (según parámetros):**")
if high_risk.empty:
    st.success("No se identifican jefes en alto riesgo con estos parámetros.")
else:
    st.dataframe(
        high_risk.reset_index()[[
            "Manager_ID",
            "Postulaciones",
            "Pct_extranjeros",
            "Tasa_Nacional",
            "Tasa_Extranjero",
            "Brecha_ext_menos_nac",
        ]]
        .sort_values("Brecha_ext_menos_nac")
        .style.format(
            {
                "Pct_extranjeros": "{:.1%}",
                "Tasa_Nacional": "{:.1%}",
                "Tasa_Extranjero": "{:.1%}",
                "Brecha_ext_menos_nac": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

st.markdown("**Riesgo medio (monitoreo):**")
if medium_risk.empty:
    st.info("No se identifican jefes con brecha moderada.")
else:
    st.dataframe(
        medium_risk.reset_index()[[
            "Manager_ID",
            "Postulaciones",
            "Pct_extranjeros",
            "Tasa_Nacional",
            "Tasa_Extranjero",
            "Brecha_ext_menos_nac",
        ]]
        .sort_values("Brecha_ext_menos_nac")
        .style.format(
            {
                "Pct_extranjeros": "{:.1%}",
                "Tasa_Nacional": "{:.1%}",
                "Tasa_Extranjero": "{:.1%}",
                "Brecha_ext_menos_nac": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

st.markdown("### 3.2 Scores fuertes y débiles (para ajustar el proceso)")

if r2_scores is None or hired_perf.empty:
    st.info("No hay suficientes datos de desempeño para evaluar scores aquí.")
else:
    st.write(f"R² del modelo de desempeño: **{r2_scores:.2f}**")

    st.markdown("**Scores clave (fuerte relación con desempeño):**")
    if not strong_scores:
        st.write("- (ninguno con efecto fuerte en este subconjunto)")
    else:
        for name, beta, pval in strong_scores:
            st.write(f"- {name}: coef ≈ {beta:.2f}, p ≈ {pval:.3g}")

    st.markdown("**Scores débiles (efecto estadístico bajo):**")
    if not weak_scores:
        st.write("- (ninguno marcado como débil)")
    else:
        for name, beta, pval in weak_scores:
            st.write(f"- {name}: coef ≈ {beta:.2f}, p ≈ {pval:.3g}")

# =====================================================
# 4. NOTAS PARA TU MEMO (ESCRIBES TÚ)
# =====================================================
st.markdown("---")
st.markdown("## 4. Notas para tu memo / presentación")

st.markdown(
    "Usa este espacio como borrador. Puedes escribir con tu estilo, "
    "copiar/pegar luego a Word/Google Docs."
)

texto_narrativa = st.text_area(
    "Escribe aquí tus ideas, conclusiones o texto del memo:",
    height=200,
    placeholder="Ej: En este subconjunto vemos que la tasa de contratación de extranjeros..."
)

st.markdown("Cuando termines, selecciona el texto de arriba y cópialo a tu documento final.")
