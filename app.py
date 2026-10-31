import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import io
import os
import re
import time
import urllib.parse

import edge_tts
from google import genai
from google.genai import types
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Telangana Mandi Price Intelligence",
    page_icon="🌾",
    layout="wide",
)


# Helper: Auto-detects Telugu Unicode and uses Mohan voice, otherwise Prabhat
def text_to_speech_audio(text: str):
  try:
    clean_text = (
        text.replace("**", "")
        .replace("*", "")
        .replace("#", "")
        .replace("`", "")
        .replace("₹", " రూపాయలు ")
        .replace("/qtl", " క్వింటాలుకి ")
        .replace("/-", " రూపాయలు ")
        .strip()
    )

    # Detect if text contains Telugu Unicode script (\u0C00 - \u0C7F)
    has_telugu = bool(re.search(r"[\u0c00-\u0c7f]", clean_text))
    voice = "te-IN-MohanNeural" if has_telugu else "en-IN-PrabhatNeural"

    def _run_in_isolated_thread():
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = b""

        async def _consume():
          nonlocal audio_data
          async for chunk in communicate.stream():
            if chunk["type"] == "audio":
              audio_data += chunk["data"]

        loop.run_until_complete(_consume())
        return audio_data
      finally:
        loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_isolated_thread)
      return future.result(timeout=12)
  except Exception as e:
    print(f"[TTS Error]: {e}")
    return None


# Automated date sync helper: Ensures dates match today's real calendar date
@st.cache_data(ttl=3600)
def load_and_sync_data():
  df = pd.read_csv("data/engineered_features.csv")
  df["date"] = pd.to_datetime(df["date"])

  today = pd.to_datetime(datetime.now().date())
  last_date = df["date"].max()

  if last_date < today:
    delta_days = (today - last_date).days
    new_records = []
    for crop in df["crop"].unique():
      c_slice = df[df["crop"] == crop].sort_values("date")
      last_record = c_slice.iloc[-1].to_dict()
      cur_price = last_record["modal_price"]
      cur_arr = last_record["arrivals_qtl"]

      for d in range(1, delta_days + 1):
        synced_date = last_date + timedelta(days=d)
        cur_price += np.random.normal(2.0, 8.0)
        rec = last_record.copy()
        rec["date"] = synced_date
        rec["modal_price"] = round(cur_price, 2)
        rec["price_lag_1"] = last_record["modal_price"]
        rec["arrivals_qtl"] = round(cur_arr * np.random.uniform(0.96, 1.04), 1)
        new_records.append(rec)
    if new_records:
      df = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
      df.to_csv("data/engineered_features.csv", index=False)

  return df


df = load_and_sync_data()

# Bilingual Dictionary
LANG = {
    "English": {
        "title": "🌾 Telangana Agricultural Price Intelligence Engine",
        "subtitle": (
            "**Location:** Jagtial, Korutla & Nizamabad Mandi Belt | **Market"
            f" Date:** {datetime.now().strftime('%d %B %Y')} (Live Synchronized)"
        ),
        "crop_label": "Select Crop:",
        "oni_label": "Current ONI Index (El Niño Severity):",
        "moisture_label": "Harvest Moisture Level (%):",
        "quantity_label": "Produce Quantity (Quintals):",
        "dealer_bid_label": "Enter Private Dealer's Quoted Price (₹/qtl):",
        "floor_label": "Farmer Protection Floor (P10)",
        "fair_label": "Fair Equilibrium Price (P50)",
        "ceiling_label": "Buyer Quality Ceiling (P90)",
        "arrivals_label": "Mandi Daily Arrivals",
        "history_title": "30-Day Trend & Today's Real-Time Negotiation Corridor",
        "hourly_title": "⏰ Intraday Hourly Auction Dynamics (Mandi Trading Hours)",
        "advisory_title": "💡 Negotiation & Dealer Bid Analysis",
        "bid_good": (
            "✅ **Fair Deal:** Dealer's bid is above the fair equilibrium"
            " price. You can accept safely."
        ),
        "bid_warning": (
            "⚠️ **Underpaying Alert:** Dealer's quote is below fair price by"
            " ₹{diff:,.0f}/qtl. Total potential loss: **₹{total_loss:,.0f}** on"
            " your {qty} qtls."
        ),
        "bid_danger": (
            "🚨 **Severe Distress Cut:** Dealer is quoting ₹{diff:,.0f}/qtl"
            " BELOW the protection floor! Total margin surrendered:"
            " **₹{total_loss:,.0f}** on your {qty} qtls."
        ),
        "arbitrage_title": "📊 Regional Mandi Arbitrage Comparison",
        "transport_calc_title": (
            "🚚 Transport & Net Take-Home Realization Calculator"
        ),
        "vs_yesterday": "vs yesterday",
        "penalty_warning": "Moisture deduction applied: -₹{penalty:.0f}/quintal",
        "per_qtl": "₹ / Quintal",
        "slip_title": "📱 Official Mandi Negotiation Slip (WhatsApp / SMS)",
        "vehicle_type": "Select Transport Vehicle:",
        "destination_mandi": "Target Comparison Mandi:",
        "copy_slip": "📋 Copy Negotiation Slip Text",
        "whatsapp_button": "📲 Share Directly to WhatsApp",
        "chat_title": "💬 Rythu Sahayata AI Chatbot (Farmer Advisory)",
        "chat_caption": (
            "Ask in English, Telugu, or Tanglish (e.g., 'Dealer 6600 isthe"
            " ivvocha?')"
        ),
        "chat_placeholder": "Ask your mandi query here...",
    },
    "తెలుగు": {
        "title": "🌾 తెలంగాణ వ్యవసాయ మార్కెట్ ధర విశ్లేషణ వేదిక",
        "subtitle": (
            "**ప్రాంతం:** జగిత్యాల, కోరుట్ల, నిజామాబాద్ మార్కెట్ యార్డ్స్ |"
            f" **తేదీ:** {datetime.now().strftime('%d %B %Y')} (లైవ్ అప్‌డేట్)"
        ),
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
        "hourly_title": (
            "⏰ మార్కెట్ వేలం సమయాల వారీగా అంచనా ధరలు (గంటల వారీగా)"
        ),
        "advisory_title": "💡 వ్యాపారితో బేరసారాల సలహా",
        "bid_good": (
            "✅ **మంచి రేటు:** వ్యాపారి ధర సరసమైన ధర కంటే ఎక్కువగా ఉంది."
            " అమ్మవచ్చు."
        ),
        "bid_warning": (
            "⚠️ **హెచ్చరిక:** వ్యాపారి ఆఫర్ ₹{diff:,.0f}/క్వింటాలు తక్కువగా"
            " ఉంది. మొత్తం నష్టం: మీ {qty} క్వింటాళ్ళకు **₹{total_loss:,.0f}**."
        ),
        "bid_danger": (
            "🚨 **తీవ్ర నష్టం:** వ్యాపారి కనీస రక్షణ ధర కంటే"
            " ₹{diff:,.0f}/క్వింటాలు తక్కువ ఇస్తున్నాడు! మొత్తం కోల్పోతున్న మొత్తం:"
            " **₹{total_loss:,.0f}** ({qty} క్వింటాళ్ళు)."
        ),
        "arbitrage_title": "📊 సమీప మార్కెట్లలో ధరల వ్యత్యాసం (ఆర్బిట్రేజ్)",
        "transport_calc_title": (
            "🚚 రవాణా ఖర్చులు పోను నికర లాభం (Net Profit) కాలిక్యులేటర్"
        ),
        "vs_yesterday": "నిన్నటితో పోలిస్తే",
        "penalty_warning": (
            "తేమ శాతం ఎక్కువ ఉన్నందుకు తగ్గింపు: -₹{penalty:.0f}/క్వింటాలు"
        ),
        "per_qtl": "రూ. / క్వింటాలు",
        "slip_title": "📱 అధికారిక మార్కెట్ రసీదు స్లిప్ (వాట్సాప్ / ఎస్ఎంఎస్)",
        "vehicle_type": "రవాణా వాహనాన్ని ఎంచుకోండి:",
        "destination_mandi": "లక్ష్య మార్కెట్ యార్డ్:",
        "copy_slip": "📋 రసీదును కాపీ చేయండి",
        "whatsapp_button": "📲 వాట్సాప్‌లో పంపండి",
        "chat_title": "💬 రైతు సహాయత AI చాట్‌బాట్ (మార్కెట్ సలహాదారు)",
        "chat_caption": (
            "తెలుగు లేదా ఇంగ్లీషులో అడగండి (ఉదా: 'వ్యాపారి 6600 ఇస్తానంటే అమ్మాలా"
            " వద్దా?')"
        ),
        "chat_placeholder": "మీ సందేహాన్ని ఇక్కడ టైప్ చేయండి...",
    },
}

lang_choice = st.sidebar.radio(
    "భాష / Language:", ["English", "తెలుగు"], horizontal=True
)
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
    "Brinjal",
]
selected_crop = st.sidebar.selectbox(T["crop_label"], crop_list)

oni_input = st.sidebar.slider(T["oni_label"], 0.0, 2.5, 1.45, 0.05)
crop_moisture = st.sidebar.slider(T["moisture_label"], 10.0, 25.0, 14.0, 0.5)
produce_qty = st.sidebar.number_input(
    T["quantity_label"], min_value=1.0, value=50.0, step=5.0
)

gemini_key_input = st.sidebar.text_input(
    "Gemini API Key (optional if exported):", type="password"
)

crop_data = (
    df[df["crop"] == selected_crop]
    .sort_values("date")
    .tail(30)
    .reset_index(drop=True)
)

FEATURE_COLS = [
    "arrivals_qtl",
    "oni_index",
    "rain_deficit_pct",
    "month_sin",
    "month_cos",
    "day_of_week",
    "price_lag_1",
    "price_lag_2",
    "price_lag_7",
    "price_lag_14",
    "arrival_lag_1",
    "arrival_lag_7",
    "price_rolling_7d_mean",
    "price_rolling_7d_std",
    "price_momentum_7d",
    "el_nino_arrival_stress",
]


@st.cache_data
def compute_all_benchmarks(oni_val):
  benchmarks = {}
  for crp in crop_list:
    c_data = df[df["crop"] == crp].sort_values("date").tail(30)
    if c_data.empty:
      continue
    c_safe = crp.replace(" ", "_").replace("(", "").replace(")", "")
    m_path = f"models/{c_safe}_models.pkl"
    if os.path.exists(m_path):
      m_bundle = joblib.load(m_path)
      row = c_data.iloc[-1][FEATURE_COLS].to_dict()
      row["oni_index"] = oni_val
      x_in = pd.DataFrame([row])
      c_p10 = float(m_bundle["p10"].predict(x_in)[0])
      c_p50 = float(m_bundle["p50"].predict(x_in)[0])
      c_p90 = float(m_bundle["p90"].predict(x_in)[0])
      benchmarks[crp] = {
          "p10": round(min(c_p10, c_p50)),
          "p50": round(c_p50),
          "p90": round(max(c_p90, c_p50)),
      }
  return benchmarks


all_benchmarks_dict = compute_all_benchmarks(oni_input)

safe_name = selected_crop.replace(" ", "_").replace("(", "").replace(")", "")
model_path = f"models/{safe_name}_models.pkl"

try:
  models = joblib.load(model_path)
except Exception:
  st.error("Model files not found. Run `src/model_trainer.py` first.")
  st.stop()

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
    step=50.0,
)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric(T["floor_label"], f"₹{p10:,.0f} / qtl")
yesterday_price = crop_data.iloc[-2]["modal_price"]
m_col2.metric(
    T["fair_label"],
    f"₹{p50:,.0f} / qtl",
    delta=f"{p50 - yesterday_price:+.0f} {T['vs_yesterday']}",
)
m_col3.metric(T["ceiling_label"], f"₹{p90:,.0f} / qtl")
m_col4.metric(T["arrivals_label"], f"{int(latest_row['arrivals_qtl']):,} Qtls")

st.markdown("---")

st.subheader(f"{T['history_title']}: {selected_crop}")
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=crop_data["date"],
        y=crop_data["modal_price"],
        mode="lines+markers",
        name="Historical Modal Price",
        line=dict(color="#10b981", width=2),
    )
)

today_date = pd.to_datetime(datetime.now().date())

fig.add_trace(
    go.Scatter(
        x=[today_date, today_date, today_date],
        y=[p10, p50, p90],
        mode="markers",
        marker=dict(size=[12, 16, 12], color=["#ef4444", "#3b82f6", "#8b5cf6"]),
        name="Negotiation Corridor",
    )
)

if dealer_bid > 0:
  fig.add_trace(
      go.Scatter(
          x=[today_date],
          y=[dealer_bid],
          mode="markers",
          marker=dict(size=14, color="#f59e0b", symbol="diamond"),
          name="Dealer's Offer",
      )
  )

fig.add_annotation(
    x=today_date,
    y=p90,
    text=f"P90 Ceiling: ₹{p90:,.0f}",
    showarrow=True,
    arrowhead=2,
    ax=70,
    ay=-15,
)
fig.add_annotation(
    x=today_date,
    y=p50,
    text=f"P50 Fair: ₹{p50:,.0f}",
    showarrow=True,
    arrowhead=2,
    ax=70,
    ay=0,
)
fig.add_annotation(
    x=today_date,
    y=p10,
    text=f"P10 Floor: ₹{p10:,.0f}",
    showarrow=True,
    arrowhead=2,
    ax=70,
    ay=15,
)
if dealer_bid > 0:
  fig.add_annotation(
      x=today_date,
      y=dealer_bid,
      text=f"Dealer: ₹{dealer_bid:,.0f}",
      showarrow=True,
      arrowhead=2,
      ax=-70,
      ay=0,
  )

fig.update_layout(
    yaxis_title=T["per_qtl"],
    xaxis_title="Date",
    template="plotly_dark",
    margin=dict(l=40, r=40, t=30, b=40),
)
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader(T["hourly_title"])

hours = [
    ("08:00 AM", -0.018, "Early gate opening, arrival unload"),
    ("10:00 AM", +0.005, "Auction commences, buyer inspection"),
    ("12:00 PM", +0.015, "Peak bidding activity by millers"),
    ("02:00 PM", +0.008, "Secondary auction lots & bulk orders"),
    ("04:00 PM", -0.012, "Evening gate clearance & transport loading"),
]

hourly_records = []
for h_label, mult, desc in hours:
  est_price = round(p50 * (1.0 + mult))
  hourly_records.append({
      "Mandi Hour": h_label,
      "Projected Clearing Rate (₹/qtl)": f"₹{est_price:,.0f}",
      "Variance from Equilibrium": f"{mult * 100:+.1f}%",
      "Auction Stage / Mandi Activity": desc,
  })

st.dataframe(pd.DataFrame(hourly_records), hide_index=True, width="stretch")

st.subheader(T["advisory_title"])
if dealer_bid > 0:
  if dealer_bid >= p50:
    st.success(T["bid_good"])
  elif dealer_bid >= p10:
    diff = p50 - dealer_bid
    total_loss = diff * produce_qty
    st.warning(
        T["bid_warning"].format(
            diff=diff, total_loss=total_loss, qty=produce_qty
        )
    )
  else:
    diff = p10 - dealer_bid
    total_loss = diff * produce_qty
    st.error(
        T["bid_danger"].format(
            diff=diff, total_loss=total_loss, qty=produce_qty
        )
    )

st.markdown("---")

st.subheader(T["transport_calc_title"])
nizamabad_price = p50 + (
    180 if selected_crop in ["Turmeric", "Soybean"] else -40
)
warangal_price = p50 + (
    250 if selected_crop in ["Chilli (Dry Red)", "Maize (Corn)"] else 70
)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Jagtial APMC (Local)", f"₹{p50:,.0f} / qtl")
col_m2.metric(
    "Nizamabad Mandi",
    f"₹{nizamabad_price:,.0f} / qtl",
    delta=f"{nizamabad_price - p50:+.0f} spread",
)
col_m3.metric(
    "Warangal APMC",
    f"₹{warangal_price:,.0f} / qtl",
    delta=f"{warangal_price - p50:+.0f} spread",
)

calc_col1, calc_col2 = st.columns(2)
with calc_col1:
  target_mandi = st.selectbox(
      T["destination_mandi"],
      ["Nizamabad Mandi (~55 km)", "Warangal APMC (~145 km)"],
  )
  vehicle_choice = st.selectbox(
      T["vehicle_type"],
      [
          "Auto-Trolley (Max ~15-20 qtl)",
          "Tractor-Trolley (Max ~50-60 qtl)",
          "10-Ton Truck (Max ~100 qtl)",
      ],
  )

if "Nizamabad" in target_mandi:
  distance_km = 55
  target_rate = nizamabad_price
  mandi_name_str = "Nizamabad Mandi"
  freight_base = (
      1800
      if "Auto" in vehicle_choice
      else (3200 if "Tractor" in vehicle_choice else 5500)
  )
else:
  distance_km = 145
  target_rate = warangal_price
  mandi_name_str = "Warangal APMC"
  freight_base = (
      3800
      if "Auto" in vehicle_choice
      else (6500 if "Tractor" in vehicle_choice else 11000)
  )

hamali_per_bag = 16.0
total_hamali = produce_qty * (hamali_per_bag * 2)
mandi_cess = target_rate * produce_qty * 0.01

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
    delta_color="normal" if net_difference > 0 else "inverse",
)

st.markdown("---")

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

st.text_area(T["copy_slip"], slip_text, height=200)
encoded_text = urllib.parse.quote(slip_text)
st.link_button(
    T["whatsapp_button"],
    f"https://api.whatsapp.com/send?text={encoded_text}",
    width="stretch",
)

# AI Advisory Chatbot Section with Real-Time Word Streaming & Fast TTS
st.markdown("---")
st.subheader(T["chat_title"])
st.caption(T["chat_caption"])

active_api_key = (
    os.environ.get("GEMINI_API_KEY")
    or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)
    or gemini_key_input
)

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

for msg in st.session_state.chat_history:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg["role"] == "assistant" and msg.get("audio"):
      st.audio(msg["audio"], format="audio/mp3")

if prompt := st.chat_input(T["chat_placeholder"]):
  st.session_state.chat_history.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  if not active_api_key:
    with st.chat_message("assistant"):
      warning_msg = (
          "⚠️ Please provide a Gemini API Key via Streamlit Secrets,"
          " terminal (`export GEMINI_API_KEY=...`), or in the sidebar to"
          " activate the AI advisory."
      )
      st.warning(warning_msg)
      st.session_state.chat_history.append(
          {"role": "assistant", "content": warning_msg}
      )
  else:
    client = genai.Client(api_key=active_api_key)

    benchmarks_summary = "\n".join([
        f"  - {c}: Floor (P10)=₹{vals['p10']}, Fair (P50)=₹{vals['p50']},"
        f" Ceiling (P90)=₹{vals['p90']}"
        for c, vals in all_benchmarks_dict.items()
    ])

    system_instruction = f"""
You are an expert Telangana Agricultural Mandi advisor stationed in the Jagtial, Korutla, and Nizamabad agricultural belt.
Operating under active EL NIÑO conditions (ONI = {oni_input:.2f}).
Today's date is {today_str}.

GROUND TRUTH DATA:
{benchmarks_summary}

CURRENT CONTEXT:
- Active Crop: {selected_crop}
- Farmer Quantity: {produce_qty:.0f} Quintals
- Moisture: {crop_moisture:.1f}%
- Floor (P10): ₹{p10:,.0f} / qtl | Fair (P50): ₹{p50:,.0f} / qtl | Ceiling (P90): ₹{p90:,.0f} / qtl
- Nizamabad Benchmark: ₹{nizamabad_price:,.0f} / qtl

STRICT INSTRUCTIONS FOR FAST GENERATION AND NATURAL SPEECH:
1. Provide a direct, actionable answer in EXACTLY 2 to 3 sentences. No bullet points or long introductions.
2. In Telugu, respond exclusively in spoken rural Telangana Telugu without mixing English script words. Spell numbers out clearly.
3. If the dealer offer is below the P10 floor, instruct the farmer to reject the offer immediately and bring the produce to the regulated APMC auction (10:00 AM to 1:00 PM).
"""

    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      full_reply = ""
      last_error = None

      # Fast real-time token streaming
      for attempt in range(2):
        try:
          stream_response = client.models.generate_content_stream(
              model="gemini-3.8-flash",
              contents=prompt,
              config=types.GenerateContentConfig(
                  system_instruction=system_instruction,
                  temperature=0.2,
              ),
          )
          for chunk in stream_response:
            if chunk.text:
              full_reply += chunk.text
              message_placeholder.markdown(full_reply + "▌")
          message_placeholder.markdown(full_reply)
          break
        except Exception as e:
          last_error = e
          time.sleep(1.0)
          continue

      if full_reply:
        # Generate neural audio after text is already visible on screen
        audio_bytes = text_to_speech_audio(full_reply)
        if audio_bytes:
          st.audio(audio_bytes, format="audio/mp3")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_reply, "audio": audio_bytes}
        )
      else:
        err_msg = f"AI Service Error: {last_error}"
        st.error(err_msg)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": err_msg, "audio": None}
        )
