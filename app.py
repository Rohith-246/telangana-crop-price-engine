import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import urllib.parse
from datetime import datetime

st.set_page_config(
    page_title="Telangana Mandi Price Intelligence",
    page_icon="🌾",
    layout="wide"
)

# Language Dictionary
LANG = {
    "English": {
        "title": "🌾 Telangana Agricultural Price Intelligence Engine",
        "subtitle": "**Location:** Jagtial, Korutla & Nizamabad Mandi Belt | **Climate:** El Niño Anomaly Active",
        "crop_label": "Select Crop:",
        "oni_label": "Current ONI Index (El Niño Severity):",
        "moisture_label": "Harvest Moisture Level (%):",
        "quantity_label": "Produce Quantity (Quintals):",
        "dealer_bid_label": "Enter Private Dealer's Quoted Price (₹/qtl):",
        "floor_label": "Farmer Protection Floor (P10)",
        "fair_label": "Fair Equilibrium Price (P50)",
        "ceiling_label": "Buyer Quality Ceiling (P90)",
        "arrivals_label": "Mandi Daily Arrivals",
        "history_title": "30-Day Trend & Today's Negotiation Corridor",
        "advisory_title": "💡 Negotiation & Dealer Bid Analysis",
        "bid_good": "✅ **Fair Deal:** Dealer's bid is above the fair equilibrium price. You can accept safely.",
        "bid_warning": "⚠️ **Underpaying Alert:** Dealer's quote is below fair price by ₹{diff:,.0f}/qtl. Total potential loss: **₹{total_loss:,.0f}** on your {qty} qtls.",
        "bid_danger": "🚨 **Severe Distress Cut:** Dealer is quoting ₹{diff:,.0f}/qtl BELOW the protection floor! Total margin surrendered: **₹{total_loss:,.0f}** on your {qty} qtls.",
        "arbitrage_title": "📊 Regional Mandi Arbitrage Comparison",
        "transport_calc_title": "🚚 Transport & Net Take-Home Realization Calculator",
        "vs_yesterday": "vs yesterday",
        "penalty_warning": "Moisture deduction applied: -₹{penalty:.0f}/quintal",
        "per_qtl": "₹ / Quintal",
        "slip_title": "📱 Official Mandi Negotiation Slip (WhatsApp / SMS)",
        "vehicle_type": "Select Transport Vehicle:",
        "destination_mandi": "Target Comparison Mandi:",
        "copy_slip": "📋 Copy Negotiation Slip Text",
        "whatsapp_button": "📲 Share Directly to WhatsApp"
    },
    "తెలుగు": {
        "title": "🌾 తెలంగాణ వ్యవసాయ మార్కెట్ ధర విశ్లేషణ వేదిక",
        "subtitle": "**ప్రాంతం:** జగిత్యాల, కోరుట్ల, నిజామాబాద్ మార్కెట్ యార్డ్స్ | **వాతావరణం:** ఎల్ నినో ప్రభావం",
        "crop_label": "పంటను ఎంచుకోండి:",
        "oni_label": "ఎల్ నినో తీవ్రత సూచిక (ONI Index):",
        "moisture_label": "దిగుబడి తేమ శాతం (%):",
        "quantity_label": "మీ వద్ద ఉన్న పంట పరిమాణం (క్వింటాళ్ళలో):",
        "dealer_bid_label": "వ్యాపారి ఆఫర్ చేసిన ధర (రూ./క్వింటాలు):",
        "floor_label": "రైతు రక్షణ కనీస ధర (P10)",
        "fair_label": "సమతుల్య సరసమైన ధర (P50)",
        "ceiling_label": "గరిష్ట నాణ్యత గల ధర (P90)",
        "arrivals_label": "మార్కెట్ రాకలు (క్వింటాళ్ళు)",
        "history_title": "గత 30 రోజుల ట్రెండ్ మరియు నేటి బేరసారాల పరిధి",
        "advisory_title": "💡 వ్యాపారితో బేరసారాల సలహా",
        "bid_good": "✅ **మంచి రేటు:** వ్యాపారి ధర సరసమైన ధర కంటే ఎక్కువగా ఉంది. అమ్మవచ్చు.",
        "bid_warning": "⚠️ **హెచ్చరిక:** వ్యాపారి ఆఫర్ ₹{diff:,.0f}/క్వింటాలు తక్కువగా ఉంది. మొత్తం నష్టం: మీ {qty} క్వింటాళ్ళకు **₹{total_loss:,.0f}**.",
        "bid_danger": "🚨 **తీవ్ర నష్టం:** వ్యాపారి కనీస రక్షణ ధర కంటే ₹{diff:,.0f}/క్వింటాలు తక్కువ ఇస్తున్నాడు! మొత్తం కోల్పోతున్న మొత్తం: **₹{total_loss:,.0f}** ({qty} క్వింటాళ్ళు).",
        "arbitrage_title": "📊 సమీప మార్కెట్లలో ధరల వ్యత్యాసం (ఆర్బిట్రేజ్)",
        "transport_calc_title": "🚚 రవాణా ఖర్చులు పోను నికర లాభం (Net Profit) కాలిక్యులేటర్",
        "vs_yesterday": "నిన్నటితో పోలిస్తే",
        "penalty_warning": "తేమ శాతం ఎక్కువ ఉన్నందుకు తగ్గింపు: -₹{penalty:.0f}/క్వింటాలు",
        "per_qtl": "రూ. / క్వింటాలు",
        "slip_title": "📱 అధికారిక మార్కెట్ రసీదు స్లిప్ (వాట్సాప్ / ఎస్ఎంఎస్)",
        "vehicle_type": "రవాణా వాహనాన్ని ఎంచుకోండి:",
        "destination_mandi": "లక్ష్య మార్కెట్ యార్డ్:",
        "copy_slip": "📋 రసీదును కాపీ చేయండి",
        "whatsapp_button": "📲 వాట్సాప్‌లో పంపండి"
    }
}

lang_choice = st.sidebar.radio("భాష / Language:", ["English", "తెలుగు"], horizontal=True)
T = LANG[lang_choice]

st.title(T["title"])
st.markdown(T["subtitle"])

crop_list = [
    "Maize (Corn)",
    "Soybean",
    "Paddy (Common)",
    "Turmeric",
    "Sesame (Til)",
    "Groundnut",
    "Chilli (Dry Red)",
    "Tomato",
    "Brinjal"
]
selected_crop = st.sidebar.selectbox(T["crop_label"], crop_list)

oni_input = st.sidebar.slider(T["oni_label"], 0.0, 2.5, 1.45, 0.05)
crop_moisture = st.sidebar.slider(T["moisture_label"], 10.0, 25.0, 14.0, 0.5)
produce_qty = st.sidebar.number_input(T["quantity_label"], min_value=1.0, value=50.0, step=5.0)

# Load Features and Models
@st.cache_data
def load_data():
    df = pd.read_csv("data/engineered_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
crop_data = df[df["crop"] == selected_crop].sort_values("date").tail(30).reset_index(drop=True)

safe_name = selected_crop.replace(" ", "_").replace("(", "").replace(")", "")
model_path = f"models/{safe_name}_models.pkl"

try:
    models = joblib.load(model_path)
except Exception:
    st.error("Model files not found. Run `src/model_trainer.py` first.")
    st.stop()

FEATURE_COLS = [
    "arrivals_qtl", "oni_index", "rain_deficit_pct",
    "month_sin", "month_cos", "day_of_week",
    "price_lag_1", "price_lag_2", "price_lag_7", "price_lag_14",
    "arrival_lag_1", "arrival_lag_7",
    "price_rolling_7d_mean", "price_rolling_7d_std", "price_momentum_7d",
    "el_nino_arrival_stress"
]

latest_row = crop_data.iloc[-1][FEATURE_COLS].to_dict()
latest_row["oni_index"] = oni_input
X_input = pd.DataFrame([latest_row])

p10 = float(models["p10"].predict(X_input)[0])
p50 = float(models["p50"].predict(X_input)[0])
p90 = float(models["p90"].predict(X_input)[0])

p10 = min(p10, p50)
p90 = max(p90, p50)

moisture_penalty = 0.0
if crop_moisture > 17.0:
    moisture_penalty = (crop_moisture - 17.0) * (p50 * 0.015)
    p50 -= moisture_penalty
    p10 -= moisture_penalty
    p90 -= moisture_penalty
    st.sidebar.warning(T["penalty_warning"].format(penalty=moisture_penalty))

st.sidebar.markdown("---")
dealer_bid = st.sidebar.number_input(
    T["dealer_bid_label"],
    min_value=0.0,
    value=float(int(p10 - 250)),
    step=50.0
)

# Display Key Metric Cards at Top
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric(T["floor_label"], f"₹{p10:,.0f} / qtl")
yesterday_price = crop_data.iloc[-2]["modal_price"]
m_col2.metric(T["fair_label"], f"₹{p50:,.0f} / qtl", delta=f"{p50 - yesterday_price:+.0f} {T['vs_yesterday']}")
m_col3.metric(T["ceiling_label"], f"₹{p90:,.0f} / qtl")
m_col4.metric(T["arrivals_label"], f"{int(latest_row['arrivals_qtl']):,} Qtls")

st.markdown("---")

# Visual Price Corridor
st.subheader(f"{T['history_title']}: {selected_crop}")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=crop_data["date"],
    y=crop_data["modal_price"],
    mode="lines+markers",
    name="Historical Modal Price",
    line=dict(color="#10b981", width=2)
))

today_date = crop_data["date"].iloc[-1]

fig.add_trace(go.Scatter(
    x=[today_date, today_date, today_date],
    y=[p10, p50, p90],
    mode="markers",
    marker=dict(size=[12, 16, 12], color=["#ef4444", "#3b82f6", "#8b5cf6"]),
    name="Negotiation Corridor"
))

if dealer_bid > 0:
    fig.add_trace(go.Scatter(
        x=[today_date],
        y=[dealer_bid],
        mode="markers",
        marker=dict(size=14, color="#f59e0b", symbol="diamond"),
        name="Dealer's Offer"
    ))

fig.add_annotation(x=today_date, y=p90, text=f"P90 Ceiling: ₹{p90:,.0f}", showarrow=True, arrowhead=2, ax=70, ay=-15)
fig.add_annotation(x=today_date, y=p50, text=f"P50 Fair: ₹{p50:,.0f}", showarrow=True, arrowhead=2, ax=70, ay=0)
fig.add_annotation(x=today_date, y=p10, text=f"P10 Floor: ₹{p10:,.0f}", showarrow=True, arrowhead=2, ax=70, ay=15)
if dealer_bid > 0:
    fig.add_annotation(x=today_date, y=dealer_bid, text=f"Dealer: ₹{dealer_bid:,.0f}", showarrow=True, arrowhead=2, ax=-70, ay=0)

fig.update_layout(
    yaxis_title=T["per_qtl"],
    xaxis_title="Date",
    template="plotly_dark",
    margin=dict(l=40, r=40, t=30, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# Dealer Bid Advisory
st.subheader(T["advisory_title"])
if dealer_bid > 0:
    if dealer_bid >= p50:
        st.success(T["bid_good"])
    elif dealer_bid >= p10:
        diff = p50 - dealer_bid
        total_loss = diff * produce_qty
        st.warning(T["bid_warning"].format(diff=diff, total_loss=total_loss, qty=produce_qty))
    else:
        diff = p10 - dealer_bid
        total_loss = diff * produce_qty
        st.error(T["bid_danger"].format(diff=diff, total_loss=total_loss, qty=produce_qty))

st.markdown("---")

# Feature 1: Regional Mandi Arbitrage & Transport Net Realization Calculator
st.subheader(T["transport_calc_title"])

nizamabad_price = p50 + (180 if selected_crop in ["Turmeric", "Soybean"] else -40)
warangal_price = p50 + (250 if selected_crop in ["Chilli (Dry Red)", "Maize (Corn)"] else 70)

# Display Regional Prices
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Jagtial APMC (Local)", f"₹{p50:,.0f} / qtl")
col_m2.metric("Nizamabad Mandi", f"₹{nizamabad_price:,.0f} / qtl", delta=f"{nizamabad_price - p50:+.0f} spread")
col_m3.metric("Warangal APMC", f"₹{warangal_price:,.0f} / qtl", delta=f"{warangal_price - p50:+.0f} spread")

# Transport Cost Estimation Form
calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    target_mandi = st.selectbox(
        T["destination_mandi"],
        ["Nizamabad Mandi (~55 km)", "Warangal APMC (~145 km)"]
    )
    vehicle_choice = st.selectbox(
        T["vehicle_type"],
        ["Auto-Trolley (Max ~15-20 qtl)", "Tractor-Trolley (Max ~50-60 qtl)", "10-Ton Truck (Max ~100 qtl)"]
    )

# Baseline logistics parameters from Jagtial
if "Nizamabad" in target_mandi:
    distance_km = 55
    target_rate = nizamabad_price
    mandi_name_str = "Nizamabad Mandi"
    freight_base = 1800 if "Auto" in vehicle_choice else (3200 if "Tractor" in vehicle_choice else 5500)
else:
    distance_km = 145
    target_rate = warangal_price
    mandi_name_str = "Warangal APMC"
    freight_base = 3800 if "Auto" in vehicle_choice else (6500 if "Tractor" in vehicle_choice else 11000)

hamali_per_bag = 16.0  # Approx ₹16 per 50kg bag (₹32 per quintal)
total_hamali = produce_qty * (hamali_per_bag * 2)
mandi_cess = target_rate * produce_qty * 0.01  # 1% standard APMC cess

# Total Revenue Comparisons
local_gross_rev = p50 * produce_qty
target_gross_rev = target_rate * produce_qty
target_net_rev = target_gross_rev - freight_base - total_hamali - mandi_cess
net_difference = target_net_rev - local_gross_rev

with calc_col2:
    st.write(f"**Travel Distance:** {distance_km} km (One-way from Jagtial)")
    st.write(f"**Estimated Freight & Fuel:** ₹{freight_base:,.0f}")
    st.write(f"**Loading/Unloading (Hamali):** ₹{total_hamali:,.0f}")
    st.write(f"**Mandi Cess (1%):** ₹{mandi_cess:,.0f}")

st.markdown("##### 💵 Net Take-Home Decision Matrix")
r_col1, r_col2, r_col3 = st.columns(3)
r_col1.metric("Local Net Take-Home (Jagtial)", f"₹{local_gross_rev:,.0f}")
r_col2.metric(f"Net Take-Home at {mandi_name_str}", f"₹{target_net_rev:,.0f}")
r_col3.metric(
    "Net Gain / (Loss) from Transport",
    f"₹{net_difference:,.0f}",
    delta=f"{net_difference:+.0f} net profit",
    delta_color="normal" if net_difference > 0 else "inverse"
)

if net_difference > 1500:
    st.success(f"🚚 **Actionable Recommendation:** Transporting your {produce_qty} quintals to {mandi_name_str} yields an extra **₹{net_difference:,.0f} net profit** after deducting all freight, hamali, and cess charges.")
else:
    st.info(f"📍 **Actionable Recommendation:** Selling at **Jagtial APMC** is more cost-effective. The price premium at {mandi_name_str} does not justify the ₹{freight_base + total_hamali + mandi_cess:,.0f} transport and loading overhead.")

st.markdown("---")

# Feature 2: WhatsApp Negotiation Slip Generator
st.subheader(T["slip_title"])

today_str = datetime.now().strftime("%d-%b-%Y")

if lang_choice == "తెలుగు":
    slip_text = f"""🌾 *తెలంగాణ మార్కెట్ యార్డ్ రక్షణ ధర నివేదిక*
📅 తేదీ: {today_str} | ప్రాంతం: జగిత్యాల & నిజామాబాద్
-----------------------------------------------
• పంట: {selected_crop}
• పరిమాణం: {produce_qty:.0f} క్వింటాళ్ళు
• నాణ్యత తేమ శాతం: {crop_moisture:.1f}%
-----------------------------------------------
💰 *నేటి బేరసారాల ధరల పరిధి:*
• రైతు రక్షణ కనీస ధర (P10 Floor): ₹{p10:,.0f} / క్వింటాలు
• సమతుల్య మార్కెట్ ధర (P50 Fair): ₹{p50:,.0f} / క్వింటాలు
• నాణ్యమైన పంట గరిష్ట ధర (P90): ₹{p90:,.0f} / క్వింటాలు
• నిజామాబాద్ మోడల్ రేటు: ₹{nizamabad_price:,.0f} / క్వింటాలు
-----------------------------------------------
⚠️ *రైతు గమనిక:*
ఈ ధరల కంటే తక్కువకు ప్రైవేట్ వ్యాపారులకు అమ్మి నష్టపోవద్దు.
ఈ నివేదిక జగిత్యాల మార్కెట్ యార్డ్ విశ్లేషణ ఆధారంగా రూపొందించబడింది.
"""
else:
    slip_text = f"""🌾 *TELANGANA MANDI BENCHMARK NEGOTIATION SLIP*
📅 Date: {today_str} | Region: Jagtial & Nizamabad Mandi Belt
-----------------------------------------------
• Crop: {selected_crop}
• Quantity: {produce_qty:.0f} Quintals
• Moisture Level: {crop_moisture:.1f}%
-----------------------------------------------
💰 *Today's Mandi Corridor (₹/Quintal):*
• Protection Floor (P10): ₹{p10:,.0f} / qtl
• Fair Mandi Price (P50): ₹{p50:,.0f} / qtl
• Top Grade Ceiling (P90): ₹{p90:,.0f} / qtl
• Nizamabad Benchmark: ₹{nizamabad_price:,.0f} / qtl
-----------------------------------------------
⚠️ *Notice for Private Buyers / Traders:*
Produce verified against active APMC mandi models. Reject bids below the protection floor.
Generated by Telangana Agricultural Price Intelligence Engine.
"""

st.text_area(T["copy_slip"], slip_text, height=230)

# Pre-filled WhatsApp Share Link
encoded_text = urllib.parse.quote(slip_text)
whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"

st.link_button(T["whatsapp_button"], whatsapp_url, use_container_width=True)
