import requests
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

############################################
# LOAD AI MODEL
############################################

model = SentenceTransformer('all-MiniLM-L6-v2')

############################################
# HISTORICAL EVENTS DATASET
############################################

historical_events = [

{"event":"Russia invades Ukraine",
 "sector":"Oil",
 "asset":"XLE",
 "move":8.4},

{"event":"China tariffs on US semiconductors",
 "sector":"Semiconductors",
 "asset":"SMH",
 "move":-4.1},

{"event":"Iran tensions escalate in Persian Gulf",
 "sector":"Oil",
 "asset":"XLE",
 "move":3.7},

{"event":"Fed emergency rate cut",
 "sector":"Growth",
 "asset":"QQQ",
 "move":5.3},

{"event":"Sanctions on Russian energy exports",
 "sector":"Oil",
 "asset":"XLE",
 "move":6.2}

]

hist_df = pd.DataFrame(historical_events)

############################################
# EMBEDDINGS FOR HISTORICAL EVENTS
############################################

hist_embeddings = model.encode(hist_df["event"].tolist())

############################################
# GDELT FETCH
############################################

def get_events():

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {

        "query":"war OR sanctions OR tariffs OR oil OR military",

        "mode":"ArtList",

        "maxrecords":25,

        "format":"json"

    }

    r = requests.get(url,params=params)

    data = r.json()

    return pd.DataFrame(data["articles"])

############################################
# MARKET MOVE
############################################

def get_market_move(symbol):

    data = yf.download(symbol,period="2d",interval="1d")

    if len(data) < 2:

        return 0

    move = (data["Close"][-1]-data["Close"][-2])/data["Close"][-2]*100

    return round(move,2)

############################################
# SIMILARITY ENGINE
############################################

def find_similar_event(event_text):

    emb = model.encode([event_text])

    sim = cosine_similarity(emb,hist_embeddings)

    idx = sim.argmax()

    similarity = sim.max()

    return hist_df.iloc[idx], similarity

############################################
# IMPACT ENGINE
############################################

def geopolitical_engine():

    df = get_events()

    alerts = []

    for _,row in df.iterrows():

        title = row["title"]

        hist_event, similarity = find_similar_event(title)

        asset = hist_event["asset"]

        sector = hist_event["sector"]

        expected_move = hist_event["move"]

        current_move = get_market_move(asset)

        probability = round(similarity * 100 ,1)

        alerts.append({

            "event":title,

            "similar_event":hist_event["event"],

            "sector":sector,

            "asset":asset,

            "expected_move_%":expected_move,

            "current_move_%":current_move,

            "probability_%":probability

        })

    return pd.DataFrame(alerts)

############################################
# RUN ENGINE
############################################

alerts = geopolitical_engine()

print("\n ADVANCED GEOPOLITICAL MARKET IMPACT ENGINE\n")

print(alerts)
