
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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
except Exception:
    SimpleDocTemplate = None


st.set_page_config(page_title="Gyártási Diagnosztika V7", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2.1rem;font-weight:950;color:#0f172a;margin-bottom:.2rem}
.subtitle{color:#475569;font-size:1rem;margin-bottom:1.1rem}
.kpi-card{background:linear-gradient(135deg,#f8fafc,#eef2ff);border:1px solid #cbd5e1;border-radius:18px;padding:16px;box-shadow:0 8px 22px rgba(15,23,42,.08);min-height:118px}
.kpi-label{font-size:.78rem;color:#475569;text-transform:uppercase;font-weight:850;letter-spacing:.04em}
.kpi-value{font-size:1.65rem;color:#0f172a;font-weight:950;margin-top:6px}
.kpi-note{font-size:.82rem;color:#334155;margin-top:6px}
.insight-card{background:#fff;border:1px solid #cbd5e1;border-left:7px solid #2563eb;border-radius:16px;padding:13px 15px;margin-bottom:10px;box-shadow:0 6px 18px rgba(15,23,42,.06);color:#0f172a!important;font-weight:650;line-height:1.45}
.insight-card *{color:#0f172a!important}
.danger{border-left-color:#dc2626;background:#fff7f7}
.warning{border-left-color:#f59e0b;background:#fffbeb}
.success{border-left-color:#16a34a;background:#f0fdf4}
</style>
""", unsafe_allow_html=True)


REQUIRED_PROD_COLS = ["Dátum","Műszak","Dolgozó","Gép","Termék","Gyártott_db","Selejt_db","Állásidő_perc"]
REQUIRED_MACHINE_COLS = ["Gép","Kapacitás_db_óra","Óradíj","Kritikus_gép"]
REQUIRED_PRODUCT_COLS = ["Termék","Eladási_ár","Anyagköltség"]
OPTIONAL_ORDER_COLS = ["Rendelés_ID","Vevő","Termék","Rendelt_db","Határidő","Prioritás"]


def fmt_num(x, digits=0):
    if pd.isna(x): return "-"
    return f"{x:,.{digits}f}".replace(",", " ") if digits else f"{x:,.0f}".replace(",", " ")

def fmt_pct(x, digits=1):
    if pd.isna(x): return "-"
    return f"{x:.{digits}f}%"

def fmt_huf(x):
    if pd.isna(x): return "-"
    return f"{x:,.0f} Ft".replace(",", " ")

def show_kpi(label, value, note=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def render_recommendations(recs):
    if not recs:
        st.info("Nincs megjeleníthető javaslat.")
        return
    for cls, text in recs:
        st.markdown(f'<div class="insight-card {cls}">{text}</div>', unsafe_allow_html=True)

def safe_read_excel(uploaded_file):
    return pd.read_excel(uploaded_file, sheet_name=None)

def find_sheet(sheets, possible_names, fallback_index=0):
    lower = {name.lower(): name for name in sheets.keys()}
    for n in possible_names:
        if n.lower() in lower:
            return sheets[lower[n.lower()]]
    return list(sheets.values())[fallback_index]

def get_optional_sheet(sheets, possible_names):
    lower = {name.lower(): name for name in sheets.keys()}
    for n in possible_names:
        if n.lower() in lower:
            return sheets[lower[n.lower()]]
    return None

def validate_columns(df, required, sheet_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"A(z) {sheet_name} munkalapon hiányzó oszlopok: {', '.join(missing)}")
        st.stop()

def prepare_data(prod, machines, products):
    prod = prod.copy()
    machines = machines.copy()
    products = products.copy()

    prod["Dátum"] = pd.to_datetime(prod["Dátum"], errors="coerce")
    for c in ["Gyártott_db","Selejt_db","Állásidő_perc"]:
        prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)

    machines["Kapacitás_db_óra"] = pd.to_numeric(machines["Kapacitás_db_óra"], errors="coerce").fillna(0)
    machines["Óradíj"] = pd.to_numeric(machines["Óradíj"], errors="coerce").fillna(0)
    if "Elérhető_óra_nap" not in machines.columns:
        machines["Elérhető_óra_nap"] = 8
    machines["Elérhető_óra_nap"] = pd.to_numeric(machines["Elérhető_óra_nap"], errors="coerce").fillna(8)

    products["Eladási_ár"] = pd.to_numeric(products["Eladási_ár"], errors="coerce").fillna(0)
    products["Anyagköltség"] = pd.to_numeric(products["Anyagköltség"], errors="coerce").fillna(0)
    if "Prioritási_súly" not in products.columns:
        products["Prioritási_súly"] = 3
    products["Prioritási_súly"] = pd.to_numeric(products["Prioritási_súly"], errors="coerce").fillna(3)

    df = prod.merge(machines, on="Gép", how="left").merge(products, on="Termék", how="left")

    df["Munkaóra"] = 1.0
    df["Jó_db"] = (df["Gyártott_db"] - df["Selejt_db"]).clip(lower=0)
    df["Selejt_%"] = np.where(df["Gyártott_db"] > 0, df["Selejt_db"] / df["Gyártott_db"] * 100, 0)
    df["Állásidő_%"] = np.minimum(df["Állásidő_perc"] / 60 * 100, 100)
    df["Elérhetőség_%"] = np.maximum(100 - df["Állásidő_%"], 0)
    df["Teljesítmény_%"] = np.where(df["Kapacitás_db_óra"] > 0, df["Gyártott_db"] / df["Kapacitás_db_óra"] * 100, 0)
    df["Teljesítmény_%"] = np.minimum(df["Teljesítmény_%"], 140)
    df["Minőség_%"] = np.where(df["Gyártott_db"] > 0, df["Jó_db"] / df["Gyártott_db"] * 100, 0)
    df["OEE_light_%"] = df["Elérhetőség_%"] * df["Teljesítmény_%"] * df["Minőség_%"] / 10000

    df["Árbevétel"] = df["Jó_db"] * df["Eladási_ár"]
    df["Anyagköltség_össz"] = df["Gyártott_db"] * df["Anyagköltség"]
    df["Gépköltség"] = df["Óradíj"] * df["Munkaóra"]
    df["Becsült_profit"] = df["Árbevétel"] - df["Anyagköltség_össz"] - df["Gépköltség"]
    return df

def normalize_orders(orders_raw):
    if orders_raw is None or orders_raw.empty:
        return pd.DataFrame(columns=OPTIONAL_ORDER_COLS)
    orders = orders_raw.copy()
    missing = [c for c in OPTIONAL_ORDER_COLS if c not in orders.columns]
    if missing:
        return pd.DataFrame(columns=OPTIONAL_ORDER_COLS)
    orders["Rendelt_db"] = pd.to_numeric(orders["Rendelt_db"], errors="coerce").fillna(0).astype(int)
    orders["Határidő"] = pd.to_datetime(orders["Határidő"], errors="coerce")
    orders["Prioritás"] = pd.to_numeric(orders["Prioritás"], errors="coerce").fillna(3).astype(int)
    return orders

def aggregate_metrics(df, group_cols):
    out = df.groupby(group_cols, as_index=False).agg(
        Gyártott_db=("Gyártott_db","sum"),
        Jó_db=("Jó_db","sum"),
        Selejt_db=("Selejt_db","sum"),
        Állásidő_perc=("Állásidő_perc","sum"),
        Árbevétel=("Árbevétel","sum"),
        Becsült_profit=("Becsült_profit","sum"),
        Átlag_OEE=("OEE_light_%","mean"),
        Átlag_teljesítmény=("Teljesítmény_%","mean"),
        Sorok=("Gyártott_db","count")
    )
    out["Selejt_%"] = np.where(out["Gyártott_db"] > 0, out["Selejt_db"] / out["Gyártott_db"] * 100, 0)
    out["Profit/db"] = np.where(out["Jó_db"] > 0, out["Becsült_profit"] / out["Jó_db"], 0)
    return out

def build_worker_machine_matrix(df):
    pair = aggregate_metrics(df, ["Dolgozó","Gép"])
    perf = pair["Átlag_teljesítmény"].clip(0,120) / 120 * 50
    quality = (100 - pair["Selejt_%"].clip(0,20)*5).clip(0,100) / 100 * 25
    p = pair["Profit/db"]
    profit_score = ((p - p.min()) / (p.max() - p.min()) * 25) if p.max() != p.min() else 12.5
    pair["Kompatibilitási_pont"] = (perf + quality + profit_score).round(1)
    matrix = pair.pivot_table(index="Dolgozó", columns="Gép", values="Kompatibilitási_pont", aggfunc="mean").round(1)
    return matrix, pair

def product_machine_priority(df):
    out = df.groupby(["Termék","Gép"], as_index=False).agg(
        Gyártott_db=("Gyártott_db","sum"),
        Jó_db=("Jó_db","sum"),
        Selejt_db=("Selejt_db","sum"),
        Átlag_OEE=("OEE_light_%","mean"),
        Átlag_teljesítmény=("Teljesítmény_%","mean"),
        Profit=("Becsült_profit","sum"),
        Kapacitás_db_óra=("Kapacitás_db_óra","mean"),
        Profit_per_good=("Eladási_ár","mean"),
        Sorok=("Gyártott_db","count")
    )
    out["Selejt_%"] = np.where(out["Gyártott_db"] > 0, out["Selejt_db"] / out["Gyártott_db"] * 100, 0)
    out["Profit/db"] = np.where(out["Jó_db"] > 0, out["Profit"] / out["Jó_db"], 0)
    p = out["Profit/db"]
    profit_score = ((p - p.min()) / (p.max() - p.min()) * 55) if p.max() != p.min() else 27.5
    oee_score = out["Átlag_OEE"].clip(0,100) / 100 * 30
    quality_score = (100 - out["Selejt_%"].clip(0,20)*5).clip(0,100) / 100 * 15
    out["Termék_gép_pont"] = (profit_score + oee_score + quality_score).round(1)
    return out.sort_values("Termék_gép_pont", ascending=False)

def order_priority_view(orders_df):
    if orders_df is None or orders_df.empty:
        return pd.DataFrame()
    out = orders_df.copy()
    today = pd.Timestamp.today().normalize()
    out["Napok_határidőig"] = (out["Határidő"] - today).dt.days
    # Mivel demó jövőbeli/teszt dátum lehet, relatív sürgősség: prioritás + korai határidő.
    min_due = out["Határidő"].min()
    out["Relatív_nap"] = (out["Határidő"] - min_due).dt.days
    out["Sürgősségi_pont"] = (6 - out["Prioritás"].clip(1,5))*20 + np.maximum(30 - out["Relatív_nap"]*2, 0)
    return out.sort_values(["Sürgősségi_pont","Határidő"], ascending=[False, True])

def demand_from_orders(orders_df):
    if orders_df is None or orders_df.empty:
        return {}
    return {str(k): int(v) for k, v in orders_df.groupby("Termék")["Rendelt_db"].sum().to_dict().items()}

def build_order_level_plan(df, orders_df, manual_demand, planning_days=5, hours_per_machine_day=8.0, unavailable_machines=None):
    unavailable_machines = unavailable_machines or []
    priority = product_machine_priority(df)
    priority = priority[~priority["Gép"].isin(unavailable_machines)].copy()

    machines = df[["Gép","Kapacitás_db_óra","Elérhető_óra_nap"]].drop_duplicates("Gép")
    available_hours = {}
    for _, r in machines.iterrows():
        if r["Gép"] in unavailable_machines:
            continue
        available_hours[r["Gép"]] = float(min(hours_per_machine_day, r.get("Elérhető_óra_nap", hours_per_machine_day))) * float(planning_days)

    if orders_df is not None and not orders_df.empty:
        order_rows = order_priority_view(orders_df)
    else:
        order_rows = pd.DataFrame([
            {"Rendelés_ID": f"MANUAL-{p}", "Vevő": "Kézi igény", "Termék": p, "Rendelt_db": int(q), "Határidő": pd.NaT, "Prioritás": 3, "Sürgősségi_pont": 0}
            for p, q in manual_demand.items() if int(q or 0) > 0
        ])

    rows = []
    for _, order in order_rows.iterrows():
        product = order["Termék"]
        remaining = int(order["Rendelt_db"])
        if remaining <= 0:
            continue

        candidates = priority[priority["Termék"] == product].sort_values("Termék_gép_pont", ascending=False)
        if candidates.empty:
            rows.append({
                "Rendelés_ID": order["Rendelés_ID"], "Vevő": order["Vevő"], "Termék": product, "Gép": "Nincs adat",
                "Igényelt_db": int(order["Rendelt_db"]), "Tervezett_db": 0, "Hiány_db": int(order["Rendelt_db"]),
                "Becsült_óra": 0, "Becsült_profit": 0, "Határidő": order["Határidő"], "Prioritás": order["Prioritás"],
                "Megjegyzés": "Nincs múltbeli termék-gép adat"
            })
            continue

        for _, cand in candidates.iterrows():
            if remaining <= 0:
                break
            machine = cand["Gép"]
            free_hours = available_hours.get(machine, 0)
            if free_hours <= 0:
                continue

            avg_per_hour = max(float(cand["Átlag_teljesítmény"]) / 100 * float(cand["Kapacitás_db_óra"]), 1)
            possible_qty = int(avg_per_hour * free_hours)
            if possible_qty <= 0:
                continue

            planned = min(remaining, possible_qty)
            used_hours = planned / avg_per_hour
            available_hours[machine] -= used_hours
            remaining -= planned

            rows.append({
                "Rendelés_ID": order["Rendelés_ID"], "Vevő": order["Vevő"], "Termék": product, "Gép": machine,
                "Igényelt_db": int(order["Rendelt_db"]), "Tervezett_db": int(planned), "Hiány_db": 0,
                "Becsült_óra": round(used_hours, 2), "Becsült_profit": round(planned * float(cand["Profit/db"]), 0),
                "Határidő": order["Határidő"], "Prioritás": order["Prioritás"],
                "Megjegyzés": f"Pont: {cand['Termék_gép_pont']:.0f}; maradék gépóra: {available_hours[machine]:.1f}"
            })

        if remaining > 0:
            rows.append({
                "Rendelés_ID": order["Rendelés_ID"], "Vevő": order["Vevő"], "Termék": product, "Gép": "Kapacitáshiány",
                "Igényelt_db": int(order["Rendelt_db"]), "Tervezett_db": 0, "Hiány_db": int(remaining),
                "Becsült_óra": 0, "Becsült_profit": 0, "Határidő": order["Határidő"], "Prioritás": order["Prioritás"],
                "Megjegyzés": "Nem fér bele a megadott horizontba / kapacitásba"
            })

    return pd.DataFrame(rows)

def build_fulfillment(plan_df, orders_df=None, manual_demand=None):
    if plan_df is None or plan_df.empty:
        return pd.DataFrame()
    planned = plan_df[~plan_df["Gép"].isin(["Kapacitáshiány","Nincs adat"])].groupby("Termék", as_index=False).agg(Tervezett_db=("Tervezett_db","sum"))
    if orders_df is not None and not orders_df.empty:
        demand = orders_df.groupby("Termék", as_index=False).agg(Igényelt_db=("Rendelt_db","sum"))
    else:
        demand = pd.DataFrame([{"Termék": k, "Igényelt_db": int(v)} for k, v in (manual_demand or {}).items() if int(v or 0) > 0])
    out = demand.merge(planned, on="Termék", how="left").fillna({"Tervezett_db":0})
    out["Hiány_db"] = (out["Igényelt_db"] - out["Tervezett_db"]).clip(lower=0)
    out["Teljesítés_%"] = np.minimum(np.where(out["Igényelt_db"] > 0, out["Tervezett_db"] / out["Igényelt_db"] * 100, 0), 100).round(1)
    return out.sort_values("Teljesítés_%")

def build_worker_machine_plan(plan_df, pair, unavailable_workers=None):
    unavailable_workers = unavailable_workers or []
    if plan_df is None or plan_df.empty or pair is None or pair.empty:
        return pd.DataFrame()
    active = plan_df[~plan_df["Gép"].isin(["Kapacitáshiány","Nincs adat"])].copy()
    rows, used = [], set()
    for _, task in active.sort_values(["Becsült_profit","Tervezett_db"], ascending=False).iterrows():
        candidates = pair[(pair["Gép"] == task["Gép"]) & (~pair["Dolgozó"].isin(unavailable_workers))].sort_values("Kompatibilitási_pont", ascending=False)
        unused = candidates[~candidates["Dolgozó"].isin(used)]
        if not unused.empty:
            best = unused.iloc[0]
        elif not candidates.empty:
            best = candidates.iloc[0]
        else:
            rows.append({"Rendelés_ID": task["Rendelés_ID"], "Gép": task["Gép"], "Termék": task["Termék"], "Tervezett_db": task["Tervezett_db"], "Ajánlott_dolgozó": "Nincs adat", "Dolgozó-gép_pont": 0})
            continue
        used.add(best["Dolgozó"])
        rows.append({"Rendelés_ID": task["Rendelés_ID"], "Gép": task["Gép"], "Termék": task["Termék"], "Tervezett_db": task["Tervezett_db"], "Ajánlott_dolgozó": best["Dolgozó"], "Dolgozó-gép_pont": round(best["Kompatibilitási_pont"],1), "Megjegyzés": f"Telj.: {best['Átlag_teljesítmény']:.1f}%, selejt: {best['Selejt_%']:.1f}%"})
    return pd.DataFrame(rows)

def build_capacity_gap(plan_df, planning_days, hours_per_machine_day):
    if plan_df is None or plan_df.empty:
        return pd.DataFrame()
    active = plan_df[~plan_df["Gép"].isin(["Kapacitáshiány","Nincs adat"])].copy()
    if active.empty:
        return pd.DataFrame()
    out = active.groupby("Gép", as_index=False).agg(Tervezett_óra=("Becsült_óra","sum"), Becsült_profit=("Becsült_profit","sum"))
    out["Max_óra"] = planning_days * hours_per_machine_day
    out["Kihasználtság_%"] = np.where(out["Max_óra"] > 0, out["Tervezett_óra"] / out["Max_óra"] * 100, 0).round(1)
    def status(x):
        if x >= 95: return "Szűk keresztmetszet"
        if x >= 75: return "Magas kihasználtság"
        if x >= 40: return "Kiegyensúlyozott"
        return "Szabad kapacitás"
    out["Státusz"] = out["Kihasználtság_%"].apply(status)
    return out.sort_values("Kihasználtság_%", ascending=False)

def recommended_assignment(pair):
    best = pair.sort_values("Kompatibilitási_pont", ascending=False).groupby("Gép", as_index=False).head(1)
    return best[["Gép","Dolgozó","Kompatibilitási_pont","Átlag_teljesítmény","Selejt_%","Profit/db"]].sort_values("Gép")

def generate_recommendations(df, pair):
    recs = []
    shift = aggregate_metrics(df, ["Műszak"])
    if len(shift) >= 2:
        worst, best = shift.sort_values("Átlag_OEE").iloc[0], shift.sort_values("Átlag_OEE", ascending=False).iloc[0]
        recs.append(("warning", f"Műszakhatás: a(z) {best['Műszak']} OEE-je {best['Átlag_OEE']-worst['Átlag_OEE']:.1f} ponttal jobb, mint a(z) {worst['Műszak']} műszaké."))
    machine = aggregate_metrics(df, ["Gép"])
    if not machine.empty:
        worst = machine.sort_values(["Állásidő_perc","Selejt_%"], ascending=False).iloc[0]
        recs.append(("danger", f"Gépdiagnosztika: a(z) {worst['Gép']} gépen a legmagasabb az állásidő/selejt kombináció."))
    worker = aggregate_metrics(df, ["Dolgozó"])
    if not worker.empty:
        top = worker.sort_values("Átlag_OEE", ascending=False).iloc[0]
        recs.append(("success", f"Dolgozói teljesítmény: {top['Dolgozó']} hozza a legjobb átlagos OEE-t ({top['Átlag_OEE']:.1f}%)."))
    best_pairs = pair.sort_values("Kompatibilitási_pont", ascending=False).head(3)
    if not best_pairs.empty:
        text = "; ".join([f"{r['Dolgozó']} → {r['Gép']} ({r['Kompatibilitási_pont']:.0f})" for _, r in best_pairs.iterrows()])
        recs.append(("success", f"Legjobb dolgozó–gép párosok: {text}."))
    return recs

def generate_plan_insights(plan_df, fulfillment_df, capacity_df):
    recs = []
    if plan_df is None or plan_df.empty:
        return [("warning", "Nincs gyártási terv.")]
    total_planned = plan_df[~plan_df["Gép"].isin(["Kapacitáshiány","Nincs adat"])]["Tervezett_db"].sum()
    total_profit = plan_df["Becsült_profit"].sum()
    recs.append(("success", f"A terv {fmt_num(total_planned)} db gyártást és kb. {fmt_huf(total_profit)} becsült profitot mutat."))
    if fulfillment_df is not None and not fulfillment_df.empty:
        shortage = fulfillment_df["Hiány_db"].sum()
        if shortage > 0:
            recs.append(("danger", f"Kapacitáshiány: {fmt_num(shortage)} db igény nem fér bele a megadott horizontba."))
        else:
            recs.append(("success", "A jelenlegi terv termékszinten fedezi a rendelésállományt."))
    if capacity_df is not None and not capacity_df.empty:
        b = capacity_df.iloc[0]
        recs.append(("warning", f"Szűk keresztmetszet jelölt: {b['Gép']} ({b['Kihasználtság_%']:.1f}% kihasználtság)."))
    return recs

def build_excel_report(df, pair, assignment, plan_df=None, worker_plan=None, orders_df=None, fulfillment_df=None, capacity_df=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        aggregate_metrics(df, ["Műszak"]).to_excel(writer, "Muszakok", index=False)
        aggregate_metrics(df, ["Gép"]).to_excel(writer, "Gepek", index=False)
        aggregate_metrics(df, ["Dolgozó"]).to_excel(writer, "Dolgozok", index=False)
        aggregate_metrics(df, ["Termék"]).to_excel(writer, "Termekek", index=False)
        pair.sort_values("Kompatibilitási_pont", ascending=False).to_excel(writer, "Dolgozo_gep_parok", index=False)
        assignment.to_excel(writer, "Javasolt_beosztas", index=False)
        if orders_df is not None and not orders_df.empty: orders_df.to_excel(writer, "Megrendelesek", index=False)
        if plan_df is not None and not plan_df.empty: plan_df.to_excel(writer, "Gyartasi_terv", index=False)
        if worker_plan is not None and not worker_plan.empty: worker_plan.to_excel(writer, "Dolgozoi_terv", index=False)
        if fulfillment_df is not None and not fulfillment_df.empty: fulfillment_df.to_excel(writer, "Rendeles_teljesites", index=False)
        if capacity_df is not None and not capacity_df.empty: capacity_df.to_excel(writer, "Kapacitas", index=False)
    return output.getvalue()

def build_pdf_report(df, pair, recs, assignment, plan_df=None, worker_plan=None, orders_df=None, fulfillment_df=None, capacity_df=None, plan_recs=None):
    if SimpleDocTemplate is None:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.1*cm, leftMargin=1.1*cm, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], fontSize=17, leading=21, textColor=colors.HexColor("#0f172a"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#1e3a8a"))
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=8.2, leading=10.2)
    def safe(x): return str(x or "").replace("ő","ö").replace("Ő","Ö").replace("ű","ü").replace("Ű","Ü")
    def P(x, style=body): return Paragraph(safe(x), style)
    story = [P("Gyártási Diagnosztika V7 – teljes vezetői riport", title), P("Ember–gép diagnosztika, rendelésállomány, gyártási terv, dolgozói beosztás és kapacitáselemzés.", body), Spacer(1,.25*cm)]
    total_qty = df["Gyártott_db"].sum(); scrap_pct = df["Selejt_db"].sum()/total_qty*100 if total_qty else 0; downtime = df["Állásidő_perc"].sum(); avg_oee=df["OEE_light_%"].mean(); profit=df["Becsült_profit"].sum()
    kpis = [[P("Gyártott db"),P("Selejt %"),P("Állásidő"),P("OEE Light"),P("Becsült profit")],[P(fmt_num(total_qty)),P(fmt_pct(scrap_pct)),P(f"{fmt_num(downtime)} perc"),P(fmt_pct(avg_oee)),P(fmt_huf(profit))]]
    t = Table(kpis, colWidths=[3.3*cm]*5); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a8a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("BACKGROUND",(0,1),(-1,1),colors.HexColor("#eff6ff")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#cbd5e1")),("ALIGN",(0,0),(-1,-1),"CENTER")])); story += [t, Spacer(1,.25*cm)]
    story.append(P("Vezetői javaslatok", h2))
    for _, text in recs: story += [P("• "+text), Spacer(1,.06*cm)]
    if plan_recs:
        story.append(P("Gyártási terv javaslatok", h2))
        for _, text in plan_recs: story += [P("• "+text), Spacer(1,.06*cm)]
    def add_table(title_txt, df2, cols, max_rows=10):
        story.append(P(title_txt, h2))
        if df2 is None or df2.empty:
            story.append(P("Nincs adat.")); return
        data = [cols]
        for _, r in df2.head(max_rows).iterrows():
            data.append([safe(r.get(c,"")) for c in cols])
        table = Table([[P(c) for c in row] for row in data], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f766e")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cbd5e1")),("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f8fafc"))]))
        story.append(table); story.append(Spacer(1,.18*cm))
    add_table("Top dolgozó–gép párosok", pair.sort_values("Kompatibilitási_pont", ascending=False), ["Dolgozó","Gép","Kompatibilitási_pont","Átlag_teljesítmény","Selejt_%"], 8)
    add_table("Gyártási terv", plan_df, ["Rendelés_ID","Termék","Gép","Igényelt_db","Tervezett_db","Hiány_db","Becsült_óra"], 12)
    add_table("Dolgozói beosztás", worker_plan, ["Rendelés_ID","Gép","Termék","Tervezett_db","Ajánlott_dolgozó","Dolgozó-gép_pont"], 12)
    add_table("Rendelésteljesítés", fulfillment_df, ["Termék","Igényelt_db","Tervezett_db","Hiány_db","Teljesítés_%"], 8)
    add_table("Kapacitás", capacity_df, ["Gép","Tervezett_óra","Max_óra","Kihasználtság_%","Státusz"], 8)
    story.append(P("Megjegyzés: a riport döntéstámogató becslés; a helyi folyamat- és adatminőséget mindig ellenőrizni kell.", body))
    doc.build(story)
    return buf.getvalue()

# UI
st.markdown('<div class="main-title">🏭 Gyártási Diagnosztika V7</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Excelből működő ember–gép hatékonyság, rendelésállomány, kapacitásszimuláció és beosztási ajánlórendszer KKV-knak.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Adatfeltöltés")
    uploaded = st.file_uploader("Tölts fel gyártási Excelt", type=["xlsx"])
    st.caption("Várt lapok: Termeles, Gepek, Termekek. Opcionális: Megrendelesek.")

if uploaded is None:
    st.info("Tölts fel egy Excelt. Használd a V7 demó fájlt: Gyartasi_Diagnosztika_Demo_V7.xlsx")
    st.stop()

try:
    sheets = safe_read_excel(uploaded)
    prod_raw = find_sheet(sheets, ["Termeles","Termelés"])
    machines_raw = find_sheet(sheets, ["Gepek","Gépek"], 1)
    products_raw = find_sheet(sheets, ["Termekek","Termékek"], 2)
    orders_raw = get_optional_sheet(sheets, ["Megrendelesek","Megrendelések","Rendelesek","Rendelések"])
    validate_columns(prod_raw, REQUIRED_PROD_COLS, "Termeles")
    validate_columns(machines_raw, REQUIRED_MACHINE_COLS, "Gepek")
    validate_columns(products_raw, REQUIRED_PRODUCT_COLS, "Termekek")
    df = prepare_data(prod_raw, machines_raw, products_raw)
    orders_df = normalize_orders(orders_raw)
except Exception as exc:
    st.error(f"Adatbetöltési hiba: {exc}")
    st.stop()

with st.expander("Szűrők", expanded=False):
    c1,c2,c3,c4 = st.columns(4)
    with c1: selected_shifts = st.multiselect("Műszak", sorted(df["Műszak"].dropna().unique()), default=sorted(df["Műszak"].dropna().unique()))
    with c2: selected_workers = st.multiselect("Dolgozó", sorted(df["Dolgozó"].dropna().unique()), default=sorted(df["Dolgozó"].dropna().unique()))
    with c3: selected_machines = st.multiselect("Gép", sorted(df["Gép"].dropna().unique()), default=sorted(df["Gép"].dropna().unique()))
    with c4: selected_products = st.multiselect("Termék", sorted(df["Termék"].dropna().unique()), default=sorted(df["Termék"].dropna().unique()))

filtered = df[df["Műszak"].isin(selected_shifts) & df["Dolgozó"].isin(selected_workers) & df["Gép"].isin(selected_machines) & df["Termék"].isin(selected_products)].copy()
if filtered.empty:
    st.warning("A szűrők után nincs adat.")
    st.stop()

matrix, pair = build_worker_machine_matrix(filtered)
assignment = recommended_assignment(pair)
recs = generate_recommendations(filtered, pair)

# Default full export plan based on orders, 5 nap x 8 óra
default_demand = demand_from_orders(orders_df) if not orders_df.empty else {p:0 for p in sorted(filtered["Termék"].unique())}
default_plan = build_order_level_plan(filtered, orders_df, default_demand, planning_days=5, hours_per_machine_day=8)
default_worker_plan = build_worker_machine_plan(default_plan, pair)
default_fulfillment = build_fulfillment(default_plan, orders_df, default_demand)
default_capacity = build_capacity_gap(default_plan, 5, 8)
default_plan_recs = generate_plan_insights(default_plan, default_fulfillment, default_capacity)

tabs = st.tabs(["1. Vezetői áttekintő","2. Műszakok","3. Dolgozó–gép mátrix","4. Gépdiagnosztika","5. Termék / profit","6. Ajánlórendszer","7. Gyártási terv + beosztás","8. Megrendelések","9. Adatellenőrzés"])

with tabs[0]:
    st.subheader("Vezetői áttekintő")
    total_qty=filtered["Gyártott_db"].sum(); good=filtered["Jó_db"].sum(); scrap=filtered["Selejt_db"].sum()/total_qty*100 if total_qty else 0; downtime=filtered["Állásidő_perc"].sum(); avg_oee=filtered["OEE_light_%"].mean(); profit=filtered["Becsült_profit"].sum()
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: show_kpi("Gyártott db", fmt_num(total_qty), "Összes mennyiség")
    with k2: show_kpi("Selejt %", fmt_pct(scrap), "Gyártott db arányában")
    with k3: show_kpi("Állásidő", f"{fmt_num(downtime)} perc", "Összes állásidő")
    with k4: show_kpi("OEE Light", fmt_pct(avg_oee), "Egyszerűsített OEE")
    with k5: show_kpi("Becsült profit", fmt_huf(profit), "Árbevétel - anyag - gépköltség")
    st.markdown("### Automatikus vezetői megállapítások")
    render_recommendations(recs + default_plan_recs)

    st.markdown("### Teljes vezetői export")
    c1,c2 = st.columns(2)
    with c1:
        pdf_bytes = build_pdf_report(filtered, pair, recs, assignment, default_plan, default_worker_plan, orders_df, default_fulfillment, default_capacity, default_plan_recs)
        if pdf_bytes:
            st.download_button("⬇️ Teljes PDF riport letöltése", data=pdf_bytes, file_name="gyartasi_diagnosztika_v7_teljes_riport.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.error("PDF exporthoz a reportlab csomag szükséges.")
    with c2:
        excel_bytes = build_excel_report(filtered, pair, assignment, default_plan, default_worker_plan, orders_df, default_fulfillment, default_capacity)
        st.download_button("⬇️ Teljes Excel riport letöltése", data=excel_bytes, file_name="gyartasi_diagnosztika_v7_teljes_riport.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        daily = filtered.groupby("Dátum", as_index=False).agg(Gyártott_db=("Gyártott_db","sum"))
        st.plotly_chart(px.line(daily, x="Dátum", y="Gyártott_db", title="Napi gyártott darabszám"), use_container_width=True)
    with c2:
        shift = aggregate_metrics(filtered, ["Műszak"])
        st.plotly_chart(px.bar(shift, x="Műszak", y="Átlag_OEE", color="Műszak", title="OEE Light műszakonként"), use_container_width=True)

with tabs[1]:
    st.subheader("Műszak összehasonlítás")
    shift = aggregate_metrics(filtered, ["Műszak"]).sort_values("Átlag_OEE", ascending=False)
    st.dataframe(shift, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(shift, x="Műszak", y="Gyártott_db", color="Műszak", title="Gyártott db műszakonként"), use_container_width=True)
    with c2: st.plotly_chart(px.bar(shift, x="Műszak", y="Selejt_%", color="Műszak", title="Selejt % műszakonként"), use_container_width=True)

with tabs[2]:
    st.subheader("Dolgozó–gép kompatibilitási mátrix")
    st.caption("Score = teljesítmény + minőség + profit/db alapján képzett kompatibilitási pont.")
    st.plotly_chart(px.imshow(matrix, text_auto=True, aspect="auto", title="Ki melyik gépen teljesít jól?", color_continuous_scale="RdYlGn"), use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Legjobb párosok")
        st.dataframe(pair.sort_values("Kompatibilitási_pont", ascending=False).head(10), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Figyelendő párosok")
        st.dataframe(pair[pair["Sorok"]>=3].sort_values("Kompatibilitási_pont").head(10), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Gépdiagnosztika")
    machine = aggregate_metrics(filtered, ["Gép"]).sort_values("Átlag_OEE", ascending=False)
    st.dataframe(machine, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(machine, x="Gép", y="Átlag_OEE", color="Átlag_OEE", title="OEE Light gépenként", color_continuous_scale="RdYlGn"), use_container_width=True)
    with c2: st.plotly_chart(px.scatter(machine, x="Állásidő_perc", y="Selejt_%", size="Gyártott_db", color="Gép", title="Selejt és állásidő gépenként"), use_container_width=True)

with tabs[4]:
    st.subheader("Termék / profit elemzés")
    product = aggregate_metrics(filtered, ["Termék"]).sort_values("Becsült_profit", ascending=False)
    st.dataframe(product, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(product, x="Termék", y="Becsült_profit", color="Termék", title="Becsült profit termékenként"), use_container_width=True)
    with c2: st.plotly_chart(px.bar(product, x="Termék", y="Profit/db", color="Termék", title="Profit/db termékenként"), use_container_width=True)

with tabs[5]:
    st.subheader("Ajánlórendszer – holnap kit hova tegyek?")
    st.dataframe(assignment, use_container_width=True, hide_index=True)
    render_recommendations(recs)

with tabs[6]:
    st.subheader("Gyártási terv + dolgozói beosztás")
    if not orders_df.empty:
        st.success("Megrendelések munkalap felismerve: az igény a rendelésállományból indul.")
    else:
        st.info("Nincs Megrendelesek munkalap, kézi igényekkel tervezhetsz.")
    product_list = sorted(filtered["Termék"].dropna().unique())
    order_demand = demand_from_orders(orders_df) if not orders_df.empty else {}
    demand = {}
    st.markdown("### Igények és kapacitás")
    cols = st.columns(min(4, max(1, len(product_list))))
    for i,p in enumerate(product_list):
        with cols[i % len(cols)]:
            demand[p] = st.number_input(f"{p} igényelt db", min_value=0, value=int(order_demand.get(p, 800 if i==0 else 500)), step=100, key=f"demand_v7_{p}")
    c1,c2,c3,c4 = st.columns(4)
    with c1: planning_days = st.slider("Tervezési horizont (nap)", 1, 20, 5)
    with c2: hours_day = st.slider("Gépóra / gép / nap", 1.0, 24.0, 8.0, step=.5)
    with c3: down_machines = st.multiselect("Kieső gépek", sorted(filtered["Gép"].unique()), default=[])
    with c4: down_workers = st.multiselect("Kieső dolgozók", sorted(filtered["Dolgozó"].unique()), default=[])

    plan_df = build_order_level_plan(filtered, orders_df, demand, planning_days, hours_day, down_machines)
    worker_plan = build_worker_machine_plan(plan_df, pair, down_workers)
    fulfillment = build_fulfillment(plan_df, orders_df if not orders_df.empty else None, demand)
    capacity = build_capacity_gap(plan_df, planning_days, hours_day)
    plan_recs = generate_plan_insights(plan_df, fulfillment, capacity)

    st.markdown("### 1. Rendelésalapú gyártási terv")
    st.caption("A tervezett db a rendelésigényből, a tervezési horizontból, a gépenkénti kapacitásból és a múltbeli termék–gép teljesítményből jön.")
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    st.markdown("### 2. Rendelésteljesítési ellenőrzés")
    st.dataframe(fulfillment, use_container_width=True, hide_index=True)
    render_recommendations(plan_recs)

    st.markdown("### 3. Dolgozói beosztás a tervhez")
    st.dataframe(worker_plan, use_container_width=True, hide_index=True)

    st.markdown("### 4. Kapacitás / szűk keresztmetszet")
    st.dataframe(capacity, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    active = plan_df[~plan_df["Gép"].isin(["Kapacitáshiány","Nincs adat"])] if not plan_df.empty else pd.DataFrame()
    with c1:
        if not active.empty:
            st.plotly_chart(px.bar(active, x="Gép", y="Tervezett_db", color="Termék", title="Tervezett db gépenként"), use_container_width=True)
    with c2:
        if not capacity.empty:
            st.plotly_chart(px.bar(capacity, x="Gép", y="Kihasználtság_%", color="Státusz", title="Gépkapacitás kihasználtság"), use_container_width=True)

    st.markdown("### 5. Export")
    st.download_button("⬇️ Terv Excel export", data=build_excel_report(filtered, pair, assignment, plan_df, worker_plan, orders_df, fulfillment, capacity), file_name="gyartasi_terv_v7.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

with tabs[7]:
    st.subheader("Megrendelésállomány")
    if orders_df.empty:
        st.info("Nincs Megrendelesek munkalap.")
        st.write("Várt oszlopok:", OPTIONAL_ORDER_COLS)
    else:
        st.dataframe(order_priority_view(orders_df), use_container_width=True, hide_index=True)
        summary = orders_df.groupby("Termék", as_index=False).agg(Rendelt_db=("Rendelt_db","sum"), Rendelések_száma=("Rendelés_ID","count"), Legkorábbi_határidő=("Határidő","min"))
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(summary, x="Termék", y="Rendelt_db", color="Termék", title="Rendelési igény termékenként"), use_container_width=True)

with tabs[8]:
    st.subheader("Adatellenőrzés")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.write(list(filtered.columns))
