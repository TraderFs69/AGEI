import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

############################################
# PAGE CONFIG
############################################

st.set_page_config(
    page_title="Geopolitical Market Impact Engine",
    layout="wide"
)

st.title("🌍 Geopolitical Market Impact Engine")

############################################
# LOAD AI MODEL
############################################

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

############################################
# HISTORICAL EVENTS
############################################

historical_events = [

{"event":"Russia invades Ukraine","sector":"Oil","asset":"XLE","move":8.4},
{"event":"China tariffs on US semiconductors","sector":"Semiconductors","asset":"SMH","move":-4.1},
{"event":"Iran tensions escalate in Persian Gulf","sector":"Oil","asset":"XLE","move":3.7},
{"event":"Fed emergency rate cut","sector":"Growth","asset":"QQQ","move":5.3},
{"event":"Sanctions on Russian energy exports","sector":"Oil","asset":"XLE","move":6.2}

]

hist_df = pd.DataFrame(historical_events)

############################################
# HISTORICAL EMBEDDINGS
############################################

@st.cache_resource
def compute_embeddings():
    return model.encode(hist_df["event"].tolist())

hist_embeddings = compute_embeddings()

############################################
# SAFE GDELT FETCH
############################################

@st.cache_data(ttl=300)
def get_events():

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": "war OR military OR sanctions OR tariff OR election OR oil",
        "mode": "ArtList",
        "maxrecords": 30,
        "format": "json",
        "sort": "HybridRel"
    }

    headers = {
        "User-Agent": "MacroShockEngine/1.0"
    }

    for attempt in range(3):

        try:

            r = requests.get(url, params=params, headers=headers, timeout=10)

            if r.status_code != 200:
                time.sleep(1)
                continue

            data = r.json()

            if "articles" in data:
                return pd.DataFrame(data["articles"])

            if "artlist" in data:
                return pd.DataFrame(data["artlist"])

            return pd.DataFrame()

        except Exception:
            time.sleep(1)

    return pd.DataFrame()

############################################
# MARKET DATA
############################################

@st.cache_data(ttl=300)
def get_market_move(symbol):

    try:

        data = yf.download(symbol, period="2d", interval="1d")

        if len(data) < 2:
            return 0

        move = (data["Close"][-1] - data["Close"][-2]) / data["Close"][-2] * 100

        return round(move,2)

    except:
        return 0

############################################
# SIMILARITY ENGINE
############################################

def find_similar_event(event_text):

    emb = model.encode([event_text])

    sim = cosine_similarity(emb, hist_embeddings)

    idx = sim.argmax()

    similarity = sim.max()

    return hist_df.iloc[idx], similarity

############################################
# IMPACT ENGINE
############################################

def geopolitical_engine():

    df = get_events()

    st.write("Events retrieved:", len(df))

    if df.empty:
        return pd.DataFrame()

    alerts = []

    for _,row in df.iterrows():

        title = row.get("title","")

        if not title:
            continue

        hist_event, similarity = find_similar_event(title)

        asset = hist_event["asset"]
        sector = hist_event["sector"]
        expected_move = hist_event["move"]

        current_move = get_market_move(asset)

        probability = round(similarity * 100,1)

        alerts.append({

            "Event":title,
            "Similar Historical Event":hist_event["event"],
            "Sector":sector,
            "Asset":asset,
            "Expected Move %":expected_move,
            "Current Move %":current_move,
            "Probability %":probability

        })

    return pd.DataFrame(alerts)

############################################
# RUN ENGINE
############################################

alerts = geopolitical_engine()

############################################
# DISPLAY
############################################

if alerts.empty:

    st.warning("No geopolitical events retrieved.")

else:

    st.subheader("🚨 Market Impact Alerts")

    st.dataframe(alerts, use_container_width=True)
