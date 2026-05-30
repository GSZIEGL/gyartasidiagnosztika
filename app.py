
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Gyártási Diagnosztika V1",
    page_icon="🏭",
    layout="wide"
)


# ------------------------------------------------------------
# Stílus
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: .2rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        border: 1px solid #cbd5e1;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(15,23,42,.08);
        min-height: 125px;
    }
    .kpi-label {
        font-size: .82rem;
        color: #475569;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: .04em;
    }
    .kpi-value {
        font-size: 1.85rem;
        color: #0f172a;
        font-weight: 950;
        margin-top: 6px;
    }
    .kpi-note {
        font-size: .86rem;
        color: #334155;
        margin-top: 6px;
    }
    .insight-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 7px solid #2563eb;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 6px 18px rgba(15,23,42,.06);
    }
    .danger {
        border-left-color: #dc2626;
        background: #fff7f7;
    }
    .warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    .success {
        border-left-color: #16a34a;
        background: #f0fdf4;
    }
    .small-muted {
        color:#64748b;
        font-size:.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Segédfüggvények
# ------------------------------------------------------------
REQUIRED_PROD_COLS = [
    "Dátum", "Műszak", "Dolgozó", "Gép", "Termék",
    "Gyártott_db", "Selejt_db", "Állásidő_perc"
]

REQUIRED_MACHINE_COLS = ["Gép", "Kapacitás_db_óra", "Óradíj", "Kritikus_gép"]
REQUIRED_PRODUCT_COLS = ["Termék", "Eladási_ár", "Anyagköltség"]


def fmt_num(x, digits=0):
    if pd.isna(x):
        return "-"
    if digits == 0:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:,.{digits}f}".replace(",", " ")


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}%"


def fmt_huf(x):
    if pd.isna(x):
        return "-"
    return f"{x:,.0f} Ft".replace(",", " ")


def show_kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_read_excel(uploaded_file) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(uploaded_file, sheet_name=None)


def find_sheet(sheets: Dict[str, pd.DataFrame], possible_names: List[str]) -> pd.DataFrame:
    lower_map = {name.lower(): name for name in sheets.keys()}
    for name in possible_names:
        if name.lower() in lower_map:
            return sheets[lower_map[name.lower()]]
    # fallback: first sheet
    return list(sheets.values())[0]


def validate_columns(df: pd.DataFrame, required: List[str], sheet_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"A(z) {sheet_name} munkalapon hiányzó oszlopok: {', '.join(missing)}")
        st.stop()


def prepare_data(prod: pd.DataFrame, machines: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    prod = prod.copy()
    machines = machines.copy()
    products = products.copy()

    prod["Dátum"] = pd.to_datetime(prod["Dátum"], errors="coerce")
    for col in ["Gyártott_db", "Selejt_db", "Állásidő_perc"]:
        prod[col] = pd.to_numeric(prod[col], errors="coerce").fillna(0)

    machines["Kapacitás_db_óra"] = pd.to_numeric(machines["Kapacitás_db_óra"], errors="coerce").fillna(0)
    machines["Óradíj"] = pd.to_numeric(machines["Óradíj"], errors="coerce").fillna(0)

    products["Eladási_ár"] = pd.to_numeric(products["Eladási_ár"], errors="coerce").fillna(0)
    products["Anyagköltség"] = pd.to_numeric(products["Anyagköltség"], errors="coerce").fillna(0)

    df = prod.merge(machines, on="Gép", how="left").merge(products, on="Termék", how="left")

    # V1 feltételezés: egy sor egy körülbelül 1 órás termelési blokk.
    df["Munkaóra"] = 1.0
    df["Jó_db"] = (df["Gyártott_db"] - df["Selejt_db"]).clip(lower=0)
    df["Selejt_%"] = np.where(df["Gyártott_db"] > 0, df["Selejt_db"] / df["Gyártott_db"] * 100, 0)
    df["Állásidő_%"] = (df["Állásidő_perc"] / 60 * 100).clip(upper=100)
    df["Elérhetőség_%"] = (100 - df["Állásidő_%"]).clip(lower=0)
    df["Teljesítmény_%"] = np.where(
        df["Kapacitás_db_óra"] > 0,
        df["Gyártott_db"] / df["Kapacitás_db_óra"] * 100,
        0
    ).clip(upper=140)
    df["Minőség_%"] = np.where(df["Gyártott_db"] > 0, df["Jó_db"] / df["Gyártott_db"] * 100, 0)
    df["OEE_light_%"] = df["Elérhetőség_%"] * df["Teljesítmény_%"] * df["Minőség_%"] / 10000

    df["Árbevétel"] = df["Jó_db"] * df["Eladási_ár"]
    df["Anyagköltség_össz"] = df["Gyártott_db"] * df["Anyagköltség"]
    df["Gépköltség"] = df["Óradíj"] * df["Munkaóra"]
    df["Becsült_profit"] = df["Árbevétel"] - df["Anyagköltség_össz"] - df["Gépköltség"]

    return df


def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, as_index=False).agg(
        Gyártott_db=("Gyártott_db", "sum"),
        Jó_db=("Jó_db", "sum"),
        Selejt_db=("Selejt_db", "sum"),
        Állásidő_perc=("Állásidő_perc", "sum"),
        Árbevétel=("Árbevétel", "sum"),
        Becsült_profit=("Becsült_profit", "sum"),
        Átlag_OEE=("OEE_light_%", "mean"),
        Átlag_teljesítmény=("Teljesítmény_%", "mean"),
        Sorok=("Gyártott_db", "count")
    )
    out["Selejt_%"] = np.where(out["Gyártott_db"] > 0, out["Selejt_db"] / out["Gyártott_db"] * 100, 0)
    out["Profit/db"] = np.where(out["Jó_db"] > 0, out["Becsült_profit"] / out["Jó_db"], 0)
    return out


def build_worker_machine_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pair = aggregate_metrics(df, ["Dolgozó", "Gép"])
    # Kompatibilitási pont: teljesítmény + minőség + profit/db, 0-100 környék.
    perf = pair["Átlag_teljesítmény"].clip(0, 120) / 120 * 50
    quality = (100 - pair["Selejt_%"].clip(0, 20) * 5).clip(0, 100) / 100 * 25
    profit_norm = pair["Profit/db"]
    if profit_norm.max() != profit_norm.min():
        profit_score = (profit_norm - profit_norm.min()) / (profit_norm.max() - profit_norm.min()) * 25
    else:
        profit_score = 12.5
    pair["Kompatibilitási_pont"] = (perf + quality + profit_score).round(1)
    matrix = pair.pivot_table(
        index="Dolgozó",
        columns="Gép",
        values="Kompatibilitási_pont",
        aggfunc="mean"
    ).round(1)
    return matrix, pair


def generate_recommendations(df: pd.DataFrame, pair: pd.DataFrame) -> List[Tuple[str, str]]:
    recs = []

    shift = aggregate_metrics(df, ["Műszak"])
    if len(shift) >= 2:
        worst_shift = shift.sort_values("Átlag_OEE").iloc[0]
        best_shift = shift.sort_values("Átlag_OEE", ascending=False).iloc[0]
        diff = best_shift["Átlag_OEE"] - worst_shift["Átlag_OEE"]
        if diff > 5:
            recs.append((
                "warning",
                f"A(z) {worst_shift['Műszak']} műszak OEE-je {diff:.1f} ponttal gyengébb, mint a(z) {best_shift['Műszak']} műszaké."
            ))

    machine = aggregate_metrics(df, ["Gép"])
    worst_machine = machine.sort_values(["Selejt_%", "Állásidő_perc"], ascending=False).iloc[0]
    if worst_machine["Selejt_%"] > machine["Selejt_%"].mean() * 1.2 or worst_machine["Állásidő_perc"] > machine["Állásidő_perc"].mean() * 1.2:
        recs.append((
            "danger",
            f"A(z) {worst_machine['Gép']} kiemelt beavatkozási pont: magas selejt vagy állásidő látszik."
        ))

    best_pairs = pair.sort_values("Kompatibilitási_pont", ascending=False).head(3)
    if not best_pairs.empty:
        text = "; ".join([f"{r['Dolgozó']} → {r['Gép']} ({r['Kompatibilitási_pont']:.0f} pont)" for _, r in best_pairs.iterrows()])
        recs.append(("success", f"Legjobb dolgozó–gép párosok: {text}."))

    weak_pairs = pair[pair["Sorok"] >= 5].sort_values("Kompatibilitási_pont").head(3)
    if not weak_pairs.empty:
        text = "; ".join([f"{r['Dolgozó']} + {r['Gép']} ({r['Kompatibilitási_pont']:.0f} pont)" for _, r in weak_pairs.iterrows()])
        recs.append(("warning", f"Figyelendő párosítások: {text}."))

    product = aggregate_metrics(df, ["Termék"])
    if not product.empty:
        worst_product = product.sort_values("Profit/db").iloc[0]
        recs.append((
            "warning",
            f"A(z) {worst_product['Termék']} termék hozza a legalacsonyabb becsült profitot darabonként ({fmt_huf(worst_product['Profit/db'])})."
        ))

    return recs


def recommended_assignment(pair: pd.DataFrame) -> pd.DataFrame:
    # Egyszerű V1: minden gépre a legjobb kompatibilitású dolgozót ajánlja,
    # egy dolgozó több gépre is ajánlható lehet. V2-ben jöhet optimalizáló algoritmus.
    best = pair.sort_values("Kompatibilitási_pont", ascending=False).groupby("Gép", as_index=False).head(1)
    best = best[["Gép", "Dolgozó", "Kompatibilitási_pont", "Átlag_teljesítmény", "Selejt_%", "Profit/db"]]
    return best.sort_values("Gép")


def render_recommendations(recs: List[Tuple[str, str]]):
    if not recs:
        st.info("Még nincs elég adat erős ajánláshoz.")
        return
    for cls, text in recs:
        st.markdown(f'<div class="insight-card {cls}">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<div class="main-title">🏭 Gyártási Diagnosztika V1</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Excelből működő ember–gép hatékonyság, OEE light, profitdiagnosztika és beosztási ajánlórendszer KKV-knak.</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Sidebar / feltöltés
# ------------------------------------------------------------
with st.sidebar:
    st.header("Adatfeltöltés")
    uploaded = st.file_uploader("Tölts fel egy gyártási Excelt", type=["xlsx"])

    st.markdown("### Várt munkalapok")
    st.caption("Termeles, Gepek, Termekek")

    demo_hint = st.info("A demó Excel 3 munkalapos: Termeles, Gepek, Termekek.")

if uploaded is None:
    st.info("Tölts fel egy Excelt a kezdéshez. A demó fájl: Gyartasi_Diagnosztika_Demo.xlsx")
    st.stop()


# ------------------------------------------------------------
# Adatbetöltés
# ------------------------------------------------------------
try:
    sheets = safe_read_excel(uploaded)
    prod_raw = find_sheet(sheets, ["Termeles", "Termelés"])
    machines_raw = find_sheet(sheets, ["Gepek", "Gépek"])
    products_raw = find_sheet(sheets, ["Termekek", "Termékek"])

    validate_columns(prod_raw, REQUIRED_PROD_COLS, "Termeles")
    validate_columns(machines_raw, REQUIRED_MACHINE_COLS, "Gepek")
    validate_columns(products_raw, REQUIRED_PRODUCT_COLS, "Termekek")

    df = prepare_data(prod_raw, machines_raw, products_raw)
except Exception as exc:
    st.error(f"Adatbetöltési hiba: {exc}")
    st.stop()


# ------------------------------------------------------------
# Globális szűrők
# ------------------------------------------------------------
with st.expander("Szűrők", expanded=False):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        selected_shifts = st.multiselect("Műszak", sorted(df["Műszak"].dropna().unique()), default=sorted(df["Műszak"].dropna().unique()))
    with c2:
        selected_workers = st.multiselect("Dolgozó", sorted(df["Dolgozó"].dropna().unique()), default=sorted(df["Dolgozó"].dropna().unique()))
    with c3:
        selected_machines = st.multiselect("Gép", sorted(df["Gép"].dropna().unique()), default=sorted(df["Gép"].dropna().unique()))
    with c4:
        selected_products = st.multiselect("Termék", sorted(df["Termék"].dropna().unique()), default=sorted(df["Termék"].dropna().unique()))

filtered = df[
    df["Műszak"].isin(selected_shifts)
    & df["Dolgozó"].isin(selected_workers)
    & df["Gép"].isin(selected_machines)
    & df["Termék"].isin(selected_products)
].copy()

if filtered.empty:
    st.warning("A szűrők után nincs adat.")
    st.stop()


# ------------------------------------------------------------
# Alap számítások
# ------------------------------------------------------------
total_qty = filtered["Gyártott_db"].sum()
good_qty = filtered["Jó_db"].sum()
scrap_pct = filtered["Selejt_db"].sum() / total_qty * 100 if total_qty else 0
downtime = filtered["Állásidő_perc"].sum()
avg_oee = filtered["OEE_light_%"].mean()
profit = filtered["Becsült_profit"].sum()
matrix, pair = build_worker_machine_matrix(filtered)
recs = generate_recommendations(filtered, pair)


# ------------------------------------------------------------
# Tabok
# ------------------------------------------------------------
tabs = st.tabs([
    "1. Vezetői áttekintő",
    "2. Műszakok",
    "3. Dolgozó–gép mátrix",
    "4. Gépdiagnosztika",
    "5. Termék / profit",
    "6. Ajánlórendszer",
    "7. Adatellenőrzés"
])


# ------------------------------------------------------------
# 1. Vezetői áttekintő
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Vezetői áttekintő")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        show_kpi("Gyártott db", fmt_num(total_qty), "Összes gyártott mennyiség")
    with k2:
        show_kpi("Selejt %", fmt_pct(scrap_pct), "Gyártott db arányában")
    with k3:
        show_kpi("Állásidő", f"{fmt_num(downtime)} perc", "Összes állásidő")
    with k4:
        show_kpi("OEE Light", fmt_pct(avg_oee), "Egyszerűsített OEE becslés")
    with k5:
        show_kpi("Becsült profit", fmt_huf(profit), "Árbevétel - anyag - gépköltség")

    st.markdown("### Automatikus vezetői megállapítások")
    render_recommendations(recs)

    c1, c2 = st.columns(2)
    with c1:
        daily = filtered.groupby("Dátum", as_index=False).agg(Gyártott_db=("Gyártott_db", "sum"), Becsült_profit=("Becsült_profit", "sum"))
        fig = px.line(daily, x="Dátum", y="Gyártott_db", title="Napi gyártott darabszám")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_shift = aggregate_metrics(filtered, ["Műszak"])
        fig = px.bar(by_shift, x="Műszak", y="Átlag_OEE", color="Műszak", title="OEE Light műszakonként")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 2. Műszakok
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Műszak összehasonlítás")

    shift = aggregate_metrics(filtered, ["Műszak"]).sort_values("Átlag_OEE", ascending=False)
    st.dataframe(shift, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(shift, x="Műszak", y="Gyártott_db", color="Műszak", title="Gyártott db műszakonként")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(shift, x="Műszak", y="Selejt_%", color="Műszak", title="Selejt % műszakonként")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 3. Dolgozó–gép mátrix
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Dolgozó–gép kompatibilitási mátrix")
    st.caption("A pontszám teljesítményből, selejtarányból és profit/db mutatóból képzett V1 kompatibilitási score.")

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Ki melyik gépen teljesít jól?",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Legjobb párosok")
        st.dataframe(pair.sort_values("Kompatibilitási_pont", ascending=False).head(10), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Figyelendő párosok")
        st.dataframe(pair[pair["Sorok"] >= 5].sort_values("Kompatibilitási_pont").head(10), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# 4. Gépdiagnosztika
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Gépdiagnosztika")

    machine = aggregate_metrics(filtered, ["Gép"]).sort_values("Átlag_OEE", ascending=False)
    st.dataframe(machine, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(machine, x="Gép", y="Átlag_OEE", color="Átlag_OEE", title="OEE Light gépenként", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(machine, x="Állásidő_perc", y="Selejt_%", size="Gyártott_db", color="Gép", title="Selejt és állásidő gépenként")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 5. Termék / profit
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Termék / profit elemzés")

    product = aggregate_metrics(filtered, ["Termék"]).sort_values("Becsült_profit", ascending=False)
    st.dataframe(product, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(product, x="Termék", y="Becsült_profit", color="Termék", title="Becsült profit termékenként")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(product, x="Termék", y="Profit/db", color="Termék", title="Profit/db termékenként")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 6. Ajánlórendszer
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Ajánlórendszer – Holnap kit hova tegyek?")

    st.caption("V1 logika: minden gépre azt a dolgozót ajánlja, aki a múltbeli adatok alapján ott a legjobb kompatibilitási pontot hozta.")

    assignment = recommended_assignment(pair)
    st.dataframe(assignment, use_container_width=True, hide_index=True)

    st.markdown("### Vezetői javaslatok")
    render_recommendations(recs)

    st.markdown("### Következő fejlesztési szint")
    st.info(
        "V2-ben ide jöhet valódi optimalizálás: dolgozó egyszerre csak egy gépen lehet, szabadság, gépkiesés, rendelésállomány, profitmaximalizálás."
    )


# ------------------------------------------------------------
# 7. Adatellenőrzés
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("Adatellenőrzés")
    st.markdown("### Feldolgozott adatok")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    st.markdown("### Oszlopok")
    st.write(list(filtered.columns))
