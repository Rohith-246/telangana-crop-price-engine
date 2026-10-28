# 🌾 Telangana Mandi Price Intelligence Engine

An end-to-end, climate-conditioned agricultural pricing intelligence engine built for farmers and buyers in the **Jagtial, Korutla, and Nizamabad** agricultural belt of Telangana. 

Designed specifically to tackle **information asymmetry** and **middleman distress pricing** exacerbated by **El Niño climate disruptions**, this platform predicts daily fair settlement prices, dynamic negotiation corridors ($P_{10}$ floor, $P_{50}$ fair price, $P_{90}$ quality ceiling), regional mandi arbitrage spreads, and net take-home realization after logistics deductions.

---

## 🚀 Key Features

* **Climate-Conditioned Quantile Regression ($P_{10}$ / $P_{50}$ / $P_{90}$):**
  * **$P_{10}$ (Protection Floor):** Statistical lower bound. Protects farmers against distressed gate-buying.
  * **$P_{50}$ (Fair Equilibrium):** Median market clearance price reflecting active supply-demand fundamentals.
  * **$P_{90}$ (Quality Ceiling):** Premium target rate for well-cured, low-moisture produce.
  * Incorporates NOAA Oceanic Niño Index (ONI) and rainfall anomaly signals.

* **9 Core Regional Crops:**
  * **Commercial & Grains:** Maize (Corn), Soybean, Paddy (Common), Turmeric, Sesame (Til), Groundnut.
  * **Perishables & Fast-Rotational Produce:** Chilli (Dry Red), Tomato, Brinjal.

* **Private Dealer Bid Loss Analyzer:**
  * Evaluates private trader bids against current APMC trends and calculates the total financial margin surrendered across the farmer's batch size.

* **Transport & Net Take-Home Realization Calculator:**
  * Compares **Jagtial APMC (Local)** vs. **Nizamabad Mandi** and **Warangal APMC**.
  * Factors in vehicle type (Auto-Trolley, Tractor, 10-Ton Truck), distance, freight, diesel, mandi loading/unloading (Hamali), and 1% APMC cess to determine whether traveling to an alternative market is economically viable.

* **Moisture Penalty Modeling:**
  * Adjusts price expectations dynamically when produce moisture exceeds standard limits ($> 17\%$).

* **Bilingual UI & WhatsApp Negotiation Slip:**
  * Full support for **English** and **తెలుగు (Telugu)**.
  * One-click generation of a formatted benchmark receipt shareable directly to WhatsApp or SMS for on-field bargaining.

---

## 🛠️ Architecture & Tech Stack

* **Language:** Python 3.11+
* **Framework:** Streamlit
* **ML Engine:** Scikit-Learn / LightGBM (Multi-Quantile Regression with Pinball Loss)
* **Data Processing & Analytics:** Pandas, NumPy
* **Visualization:** Plotly Graph Objects
* **Deployment & Containerization:** Docker, Streamlit Community Cloud

---

## 📂 Project Structure

```text
telangana-crop-price-engine/
│
├── data/
│   ├── historical_mandi_data.csv   # Historical mandi records conditioned on climate cycles
│   └── engineered_features.csv     # Cyclical, lag, and climate-interaction features
│
├── models/
│   └── *_models.pkl                # Trained multi-quantile model bundles per crop
│
├── src/
│   ├── data_pipeline.py            # Mandi simulation, climate indexing, and feature engineering
│   └── model_trainer.py            # Multi-quantile model training and serialization
│
├── app.py                          # Bilingual Streamlit dashboard & negotiation engine
├── Dockerfile                      # Production container configuration
├── .dockerignore                   # Docker exclusion patterns
├── requirements.txt                # Pinned project dependencies
└── README.md                       # Documentation
