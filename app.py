
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except Exception:
    SimpleDocTemplate = None



st.set_page_config(
    page_title="Gyártási Diagnosztika V3",
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
        color: #0f172a !important;
        font-weight: 650;
        line-height: 1.45;
    }
    .insight-card * {
        color: #0f172a !important;
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
    df["Állásidő_%"] = np.minimum(df["Állásidő_perc"] / 60 * 100, 100)
    df["Elérhetőség_%"] = (100 - df["Állásidő_%"]).clip(lower=0)
    df["Teljesítmény_%"] = np.where(
        df["Kapacitás_db_óra"] > 0,
        df["Gyártott_db"] / df["Kapacitás_db_óra"] * 100,
        0
    )
    df["Teljesítmény_%"] = np.minimum(df["Teljesítmény_%"], 140)
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
    """Mindig adjon vezetői javaslatokat, ne csak extrém eltérésnél."""
    recs = []

    shift = aggregate_metrics(df, ["Műszak"])
    if len(shift) >= 2:
        worst_shift = shift.sort_values("Átlag_OEE").iloc[0]
        best_shift = shift.sort_values("Átlag_OEE", ascending=False).iloc[0]
        diff = best_shift["Átlag_OEE"] - worst_shift["Átlag_OEE"]
        recs.append((
            "warning" if diff >= 3 else "success",
            f"Műszakhatás: a(z) {best_shift['Műszak']} műszak OEE-je {diff:.1f} ponttal jobb, mint a(z) {worst_shift['Műszak']} műszaké. "
            f"Érdemes megnézni, hogy ember-, gép- vagy termékösszetétel okozza-e."
        ))

    machine = aggregate_metrics(df, ["Gép"])
    if not machine.empty:
        worst_machine = machine.sort_values(["Állásidő_perc", "Selejt_%"], ascending=False).iloc[0]
        best_machine = machine.sort_values("Átlag_OEE", ascending=False).iloc[0]
        recs.append((
            "danger" if worst_machine["Állásidő_perc"] > machine["Állásidő_perc"].median() else "warning",
            f"Gépdiagnosztika: a(z) {worst_machine['Gép']} gépen a legmagasabb az állásidő/selejt kombináció. "
            f"A legjobb OEE-t jelenleg a(z) {best_machine['Gép']} hozza."
        ))

    worker = aggregate_metrics(df, ["Dolgozó"])
    if not worker.empty:
        top_worker = worker.sort_values("Átlag_OEE", ascending=False).iloc[0]
        low_worker = worker.sort_values("Átlag_OEE").iloc[0]
        recs.append((
            "success",
            f"Dolgozói teljesítmény: {top_worker['Dolgozó']} hozza a legjobb átlagos OEE-t ({top_worker['Átlag_OEE']:.1f}%). "
            f"{low_worker['Dolgozó']} esetében érdemes megnézni, hogy rossz gépen vagy nehezebb terméken dolgozik-e."
        ))

    best_pairs = pair.sort_values("Kompatibilitási_pont", ascending=False).head(3)
    if not best_pairs.empty:
        text = "; ".join([f"{r['Dolgozó']} → {r['Gép']} ({r['Kompatibilitási_pont']:.0f} pont)" for _, r in best_pairs.iterrows()])
        recs.append(("success", f"Legjobb dolgozó–gép párosok: {text}. Ezeket a párosokat érdemes preferálni beosztáskor."))

    weak_pairs = pair[pair["Sorok"] >= 3].sort_values("Kompatibilitási_pont").head(3)
    if not weak_pairs.empty:
        text = "; ".join([f"{r['Dolgozó']} + {r['Gép']} ({r['Kompatibilitási_pont']:.0f} pont)" for _, r in weak_pairs.iterrows()])
        recs.append(("warning", f"Figyelendő párosítások: {text}. Nem biztos, hogy rossz dolgozókról van szó, lehet, hogy rossz gép–ember párosítás."))

    product = aggregate_metrics(df, ["Termék"])
    if not product.empty:
        best_product = product.sort_values("Profit/db", ascending=False).iloc[0]
        worst_product = product.sort_values("Profit/db").iloc[0]
        recs.append((
            "warning",
            f"Termék/profit: a(z) {best_product['Termék']} termék profit/db alapján a legerősebb, "
            f"a(z) {worst_product['Termék']} a leggyengébb. Gyártási prioritásnál ezt érdemes figyelembe venni."
        ))

    if len(recs) == 0:
        recs.append(("warning", "Még kevés adat van, de az app már felépítette az alap mutatókat. Tölts fel több sort vagy hosszabb időszakot."))

    return recs


def recommended_assignment(pair: pd.DataFrame) -> pd.DataFrame:
    # Egyszerű V1: minden gépre a legjobb kompatibilitású dolgozót ajánlja,
    # egy dolgozó több gépre is ajánlható lehet. V2-ben jöhet optimalizáló algoritmus.
    best = pair.sort_values("Kompatibilitási_pont", ascending=False).groupby("Gép", as_index=False).head(1)
    best = best[["Gép", "Dolgozó", "Kompatibilitási_pont", "Átlag_teljesítmény", "Selejt_%", "Profit/db"]]
    return best.sort_values("Gép")



def build_pdf_report(df: pd.DataFrame, pair: pd.DataFrame, recs: List[Tuple[str, str]]) -> bytes:
    """Egyszerű vezetői PDF riport."""
    if SimpleDocTemplate is None:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleHU", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
    h2 = ParagraphStyle("H2HU", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#1e3a8a"))
    body = ParagraphStyle("BodyHU", parent=styles["Normal"], fontSize=8.5, leading=10.5)

    def safe(x):
        return str(x or "").replace("ő", "ö").replace("Ő", "Ö").replace("ű", "ü").replace("Ű", "Ü")

    def P(x, style=body):
        return Paragraph(safe(x), style)

    total_qty = df["Gyártott_db"].sum()
    scrap_pct = df["Selejt_db"].sum() / total_qty * 100 if total_qty else 0
    downtime = df["Állásidő_perc"].sum()
    avg_oee = df["OEE_light_%"].mean()
    profit = df["Becsült_profit"].sum()

    story = []
    story.append(P("Gyártási Diagnosztika V3 – vezetői riport", title))
    story.append(P("Excelből készült automatikus ember–gép, OEE light és profitdiagnosztika.", body))
    story.append(Spacer(1, 0.25 * cm))

    kpi_data = [
        [P("Gyártott db"), P("Selejt %"), P("Állásidő"), P("OEE Light"), P("Becsült profit")],
        [P(fmt_num(total_qty)), P(fmt_pct(scrap_pct)), P(f"{fmt_num(downtime)} perc"), P(fmt_pct(avg_oee)), P(fmt_huf(profit))],
    ]
    table = Table(kpi_data, colWidths=[3.3 * cm] * 5)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#eff6ff")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(P("Vezetői javaslatok", h2))
    for cls, text in recs:
        story.append(P("• " + text))
        story.append(Spacer(1, 0.08 * cm))

    story.append(Spacer(1, 0.2 * cm))
    story.append(P("Legjobb dolgozó–gép párosok", h2))
    top_pairs = pair.sort_values("Kompatibilitási_pont", ascending=False).head(8)
    if not top_pairs.empty:
        data = [["Dolgozó", "Gép", "Pont", "Teljesítmény %", "Selejt %"]]
        for _, r in top_pairs.iterrows():
            data.append([
                safe(r["Dolgozó"]),
                safe(r["Gép"]),
                f"{r['Kompatibilitási_pont']:.0f}",
                f"{r['Átlag_teljesítmény']:.1f}",
                f"{r['Selejt_%']:.1f}",
            ])
        t = Table([[P(c) for c in row] for row in data], colWidths=[3.6*cm, 3.0*cm, 2.5*cm, 3.5*cm, 3.0*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        story.append(t)


    story.append(Spacer(1, 0.2 * cm))
    story.append(P("Optimalizált beosztási javaslat", h2))
    try:
        opt = optimized_assignment(pair, [], [], True)
        if not opt.empty:
            data = [["Gép", "Ajánlott dolgozó", "Pont", "Teljesítmény %", "Selejt %"]]
            for _, r in opt.iterrows():
                data.append([
                    safe(r["Gép"]),
                    safe(r["Ajánlott dolgozó"]),
                    f"{r['Kompatibilitási_pont']:.0f}",
                    f"{r['Várható teljesítmény_%']:.1f}",
                    f"{r['Várható selejt_%']:.1f}",
                ])
            t2 = Table([[P(c) for c in row] for row in data], colWidths=[3.2*cm, 4.2*cm, 2.4*cm, 3.5*cm, 3.0*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a8a")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f8fafc")),
            ]))
            story.append(t2)
        else:
            story.append(P("Nincs elég adat optimalizált javaslat készítéséhez."))
    except Exception as exc:
        story.append(P(f"Optimalizált javaslat nem készült el: {exc}"))


    story.append(Spacer(1, 0.25 * cm))
    story.append(P("Megjegyzés: a V3 riport döntéstámogató becslés. A pontos okok feltárásához a helyi folyamatokat és adatminőséget is érdemes ellenőrizni.", body))

    doc.build(story)
    return buffer.getvalue()



def optimized_assignment(
    pair: pd.DataFrame,
    unavailable_workers: List[str] = None,
    unavailable_machines: List[str] = None,
    one_worker_once: bool = True
) -> pd.DataFrame:
    """V2 optimalizáló: dolgozó/gép kizárás és egyszerű greedy beosztás.

    Cél: minden elérhető gépre a lehető legjobb dolgozó-gép párosítás,
    opcionálisan úgy, hogy egy dolgozó csak egyszer szerepelhet.
    """
    unavailable_workers = unavailable_workers or []
    unavailable_machines = unavailable_machines or []

    available = pair[
        ~pair["Dolgozó"].isin(unavailable_workers)
        & ~pair["Gép"].isin(unavailable_machines)
    ].copy()

    if available.empty:
        return pd.DataFrame(columns=[
            "Gép", "Ajánlott dolgozó", "Kompatibilitási_pont",
            "Várható teljesítmény_%", "Várható selejt_%", "Várható profit/db"
        ])

    assignments = []
    used_workers = set()

    # A legfontosabb gépekkel kezdünk: ahol magasabb átlagprofit/db vagy teljesítmény látszik.
    machine_priority = (
        available.groupby("Gép", as_index=False)
        .agg(
            Átlag_pont=("Kompatibilitási_pont", "mean"),
            Átlag_profit=("Profit/db", "mean"),
            Sorok=("Sorok", "sum")
        )
        .sort_values(["Átlag_profit", "Átlag_pont"], ascending=False)
    )

    for machine in machine_priority["Gép"].tolist():
        candidates = available[available["Gép"] == machine].sort_values("Kompatibilitási_pont", ascending=False)
        if one_worker_once:
            candidates = candidates[~candidates["Dolgozó"].isin(used_workers)]
        if candidates.empty:
            continue

        best = candidates.iloc[0]
        used_workers.add(best["Dolgozó"])
        assignments.append({
            "Gép": best["Gép"],
            "Ajánlott dolgozó": best["Dolgozó"],
            "Kompatibilitási_pont": round(best["Kompatibilitási_pont"], 1),
            "Várható teljesítmény_%": round(best["Átlag_teljesítmény"], 1),
            "Várható selejt_%": round(best["Selejt_%"], 2),
            "Várható profit/db": round(best["Profit/db"], 0),
        })

    return pd.DataFrame(assignments).sort_values("Gép")


def compare_assignment_scenarios(pair: pd.DataFrame, current_assignment: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    """Egyszerű összehasonlítás a mostani V1 és optimalizált beosztás között."""
    if optimized is None or optimized.empty:
        return pd.DataFrame([{
            "Mutató": "Optimalizált beosztás",
            "Érték": "Nincs elég adat / túl sok kizárás"
        }])

    current_score = current_assignment["Kompatibilitási_pont"].mean() if not current_assignment.empty else np.nan
    opt_score = optimized["Kompatibilitási_pont"].mean()
    current_profit = current_assignment["Profit/db"].mean() if "Profit/db" in current_assignment.columns and not current_assignment.empty else np.nan
    opt_profit = optimized["Várható profit/db"].mean()

    rows = [
        {"Mutató": "Átlag kompatibilitási pont", "Jelenlegi egyszerű ajánlás": round(current_score, 1), "Optimalizált": round(opt_score, 1), "Változás": round(opt_score - current_score, 1) if pd.notna(current_score) else "-"},
        {"Mutató": "Átlag profit/db", "Jelenlegi egyszerű ajánlás": round(current_profit, 0) if pd.notna(current_profit) else "-", "Optimalizált": round(opt_profit, 0), "Változás": round(opt_profit - current_profit, 0) if pd.notna(current_profit) else "-"},
        {"Mutató": "Beosztott gépek száma", "Jelenlegi egyszerű ajánlás": len(current_assignment), "Optimalizált": len(optimized), "Változás": len(optimized) - len(current_assignment)},
    ]
    return pd.DataFrame(rows)


def render_recommendations(recs: List[Tuple[str, str]]):
    if not recs:
        st.info("Még nincs elég adat erős ajánláshoz.")
        return
    for cls, text in recs:
        st.markdown(f'<div class="insight-card {cls}">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<div class="main-title">🏭 Gyártási Diagnosztika V3</div>', unsafe_allow_html=True)
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

    st.markdown("### PDF export")
    if st.button("Vezetői PDF riport elkészítése", use_container_width=True):
        pdf_bytes = build_pdf_report(filtered, pair, recs)
        if pdf_bytes is None:
            st.error("A PDF exporthoz telepíteni kell a reportlab csomagot.")
        else:
            st.download_button(
                "⬇️ PDF riport letöltése",
                data=pdf_bytes,
                file_name="gyartasi_diagnosztika_vezetoi_riport.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

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

    st.caption("V2 alaplogika: dolgozó–gép kompatibilitás alapján javasol beosztást. Már kezel dolgozó kiesést, gépkiesést és egyszeres dolgozóhasználatot.")

    st.markdown("### Alap javasolt beosztás")
    assignment = recommended_assignment(pair)
    st.dataframe(assignment, use_container_width=True, hide_index=True)

    st.markdown("### Mi lenne ha? / optimalizáló")
    c1, c2, c3 = st.columns(3)

    with c1:
        unavailable_workers = st.multiselect(
            "Kieső / nem elérhető dolgozók",
            sorted(filtered["Dolgozó"].dropna().unique()),
            default=[]
        )

    with c2:
        unavailable_machines = st.multiselect(
            "Kieső / nem használható gépek",
            sorted(filtered["Gép"].dropna().unique()),
            default=[]
        )

    with c3:
        one_worker_once = st.checkbox(
            "Egy dolgozó csak egy gépre kerüljön",
            value=True,
            help="Valós beosztáshoz általában ezt érdemes bekapcsolni."
        )

    optimized = optimized_assignment(
        pair,
        unavailable_workers=unavailable_workers,
        unavailable_machines=unavailable_machines,
        one_worker_once=one_worker_once
    )

    st.markdown("### Optimalizált javasolt beosztás")
    if optimized.empty:
        st.warning("A kiválasztott kizárások mellett nincs elég adat javaslat készítéséhez.")
    else:
        st.dataframe(optimized, use_container_width=True, hide_index=True)

    st.markdown("### Várható hatás")
    scenario = compare_assignment_scenarios(pair, assignment, optimized)
    st.dataframe(scenario, use_container_width=True, hide_index=True)

    st.markdown("### Vezetői javaslatok")
    render_recommendations(recs)

    st.markdown("### Következő fejlesztési szint")
    st.info(
        "V3-ban ide jöhet rendelésállomány, műszakórák, termékprioritás, dolgozói jogosultságok és profitmaximalizáló optimalizálás."
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
