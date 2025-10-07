# Streamlit Dashboard — East Africa Climate × Population × Production × Land Use
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="East Africa – Climate & Agro Dashboard", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

def _find_first_csv(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    for p in DATA_DIR.glob("*.csv"):
        pl = p.name.lower()
        if any(c.replace("*","").lower() in pl for c in candidates):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_population():
    f = _find_first_csv(["Population_E_Africa.csv", "population.csv", "pop.csv"])
    if not f: return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    country_col = next((cols[k] for k in cols if k in ("country","area","pays")), None)
    year_col = next((cols[k] for k in cols if k in ("year","annee","année")), None)
    pop_col = next((cols[k] for k in cols if "pop" in k), None)
    if not (country_col and year_col and pop_col): 
        return None
    out = df[[country_col, year_col, pop_col]].copy()
    out.columns = ["country","year","population"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    return out.dropna(subset=["country","year","population"])

@st.cache_data(show_spinner=False)
def load_temperature():
    f = _find_first_csv([
        "temperature.csv",
        "Temperature_Land_Surface_E_Africa.csv",
        "Variation de température sur la superficie des terres.csv",
        "variation_temperature.csv"
    ])
    if not f: return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    country_col = next((cols[k] for k in cols if k in ("country","area","pays")), None)
    year_col = next((cols[k] for k in cols if k in ("year","annee","année")), None)
    temp_col = next((cols[k] for k in cols if "temp" in k or "°" in k), None)
    if not (country_col and year_col and temp_col):
        val_col = next((cols[k] for k in cols if k in ("value","valeur")), None)
        if country_col and year_col and val_col:
            temp_col = val_col
        else:
            return None
    out = df[[country_col, year_col, temp_col]].copy()
    out.columns = ["country","year","temperature"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
    return out.dropna(subset=["country","year","temperature"])

@st.cache_data(show_spinner=False)
def load_production():
    f = _find_first_csv(["Production_Crops_Livestock_E_Africa.csv", "production.csv"])
    if not f: return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    area = next((cols[k] for k in cols if k in ("area","country","pays")), None)
    item = next((cols[k] for k in cols if k in ("item","produit","culture")), None)
    element = next((cols[k] for k in cols if k in ("element","élément")), None)
    year = next((cols[k] for k in cols if k in ("year","annee","année")), None)
    prod = next((cols[k] for k in cols if k in ("production","value","valeur")), None)
    if not (area and item and year and prod):
        return None
    out = df[[area, item, year, prod] + ([element] if element else [])].copy()
    new_cols = ["country","item","year","production"] + (["element"] if element else [])
    out.columns = new_cols
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["production"] = pd.to_numeric(out["production"], errors="coerce")
    if "element" in out.columns:
        mask = out["element"].str.contains("prod", case=False, na=False)
        if mask.any(): out = out[mask]
    return out.dropna(subset=["country","item","year","production"])

@st.cache_data(show_spinner=False)
def load_landuse():
    f = _find_first_csv(["Inputs_LandUse_E_Africa.csv","landuse.csv","land_use.csv"])
    if not f: return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    country_col = next((cols[k] for k in cols if k in ("country","area","pays")), None)
    year_col = next((cols[k] for k in cols if k in ("year","annee","année")), None)
    if not (country_col and year_col): return None
    non_cat = {country_col, year_col}
    cats = [c for c in df.columns if c not in non_cat]
    if not cats: return None
    out = df[[country_col, year_col] + cats].copy()
    out = out.rename(columns={country_col:"country", year_col:"year"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    for c in cats: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["year"])

pop = load_population()
temp = load_temperature()
prod = load_production()
land = load_landuse()

st.title("East Africa – Climate × Population × Production × Land Use")
st.caption("Déposez vos CSV dans `data/`. Les noms et schémas les plus courants sont auto-détectés.")

st.sidebar.header("Filtres")
countries = sorted(set(
    ([] if pop is None else pop["country"].unique().tolist()) +
    ([] if temp is None else temp["country"].unique().tolist()) +
    ([] if prod is None else prod["country"].unique().tolist())
))
sel_countries = st.sidebar.multiselect("Pays", countries, default=countries[: min(5, len(countries))] if countries else [])
def _minmax(df, col):
    return (int(df[col].min()), int(df[col].max())) if df is not None and not df.empty else (1960, 2025)
mins = []; maxs = []
for d in (pop,temp,prod):
    if d is not None and not d.empty:
        mins.append(int(d["year"].min())); maxs.append(int(d["year"].max()))
year_min = min(mins) if mins else 1960
year_max = max(maxs) if maxs else 2025
yr = st.sidebar.slider("Période", min_value=year_min, max_value=year_max, value=(year_min, year_max))

col1, col2, col3 = st.columns(3)
with col1:
    if pop is not None and not pop.empty:
        df = pop[(pop["year"].between(*yr)) & ((pop["country"].isin(sel_countries)) if sel_countries else True)]
        st.metric("Population (somme période)", f"{df['population'].sum():,.0f}".replace(",", " "))
    else: st.info("Population: ajoutez le CSV.")
with col2:
    if prod is not None and not prod.empty:
        df = prod[(prod["year"].between(*yr)) & ((prod["country"].isin(sel_countries)) if sel_countries else True)]
        st.metric("Production (somme période)", f"{df['production'].sum():,.0f}".replace(",", " "))
    else: st.info("Production: ajoutez le CSV.")
with col3:
    if temp is not None and not temp.empty:
        df = temp[(temp["year"].between(*yr)) & ((temp["country"].isin(sel_countries)) if sel_countries else True)]
        st.metric("Température moyenne", f"{df['temperature'].mean():.2f}")
    else: st.info("Température: ajoutez le CSV.")

st.markdown("---")

if temp is not None and not temp.empty:
    tdf = temp[(temp["year"].between(*yr)) & ((temp["country"].isin(sel_countries)) if sel_countries else True)]
    if not tdf.empty:
        st.subheader("Température – évolution")
        st.plotly_chart(px.line(tdf, x="year", y="temperature", color="country"), use_container_width=True)

if pop is not None and not pop.empty:
    pdf = pop[(pop["year"].between(*yr)) & ((pop["country"].isin(sel_countries)) if sel_countries else True)]
    if not pdf.empty:
        st.subheader("Population – évolution")
        logy = st.toggle("Échelle logarithmique", value=False)
        st.plotly_chart(px.line(pdf, x="year", y="population", color="country", log_y=logy), use_container_width=True)

if prod is not None and not prod.empty:
    st.subheader("Production – évolution par pays et produit")
    items = sorted(prod["item"].dropna().unique().tolist())
    sel_items = st.multiselect("Produits", items, default=items[:5] if items else [])
    pr = prod[(prod["year"].between(*yr)) & ((prod["country"].isin(sel_countries)) if sel_countries else True)]
    if sel_items: pr = pr[pr["item"].isin(sel_items)]
    if not pr.empty:
        st.plotly_chart(px.line(pr, x="year", y="production", color="country"), use_container_width=True)
        st.dataframe(pr.head(50))

if land is not None and not land.empty:
    st.subheader("Occupation des terres (LULC)")
    ldf = land[(land["year"].between(*yr)) & ((land["country"].isin(sel_countries)) if sel_countries else True)]
    if not ldf.empty:
        classes = [c for c in ldf.columns if c not in ("country","year") and pd.api.types.is_numeric_dtype(ldf[c])]
        if classes:
            norm = st.toggle("Afficher en parts (%)", value=True)
            ctry = st.selectbox("Pays (LULC)", sorted(ldf["country"].unique()))
            sub = ldf[ldf["country"]==ctry].copy()
            if norm:
                s = sub[classes].sum(axis=1).replace(0, np.nan)
                sub[classes] = (sub[classes].div(s, axis=0)*100).round(2)
            m = sub.melt(id_vars=["country","year"], value_vars=classes, var_name="classe", value_name="valeur")
            st.plotly_chart(px.area(m, x="year", y="valeur", color="classe"), use_container_width=True)

st.markdown("---")
with st.expander("Aide (schémas & noms de fichiers)"):
    st.write("""
Déposez vos CSV dans `data/`. Noms reconnus :
- Population : `Population_E_Africa.csv`, `population.csv`
- Température : `Variation de température sur la superficie des terres.csv`, `temperature.csv`
- Production : `Production_Crops_Livestock_E_Africa.csv`, `production.csv`
- Land Use : `Inputs_LandUse_E_Africa.csv`, `landuse.csv`
""" )
