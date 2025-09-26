from consts import POLYGONURL1, POLYGONURL2
import re
import csv
import json
import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class PolygonAPI:
    def __init__(self, apikey):
        self.apikey = apikey

    def getOptionChart(self, option_symbol, timeframe, fromdate, todate, ncandles, csvname, pngname):
        url = f"{POLYGONURL1}O:{option_symbol}/range/1/{timeframe}/{fromdate}/{todate}{POLYGONURL2}{ncandles}&apiKey={self.apikey}"
        response = requests.get(url)
        data = json.loads(response.text)
        closes = []
        try:
            closes = [entry["c"] for entry in data["results"]]
        except:
            print(f"src/polygon.py :: No option data returned for contract {option_symbol}")
            return
        
        timestamps = [datetime.fromtimestamp(entry["t"] / 1000) for entry in data["results"]]

        writeCandlesCsv(data, csvname)
        print(f"src/polygon.py :: Successfully saved contract {option_symbol} candlestick data to {csvname}")

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, closes, marker="o", linestyle="-")
        plt.title(f"{option_symbol} Close Price History")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pngname, dpi=150)
        plt.close()
        print(f"src/polygon.py :: Successfully created option chart {pngname}")

    def getUnderlyingChart(self, ticker, timeframe, fromdate, todate, ncandles, csvname, pngname):
        url = f"{POLYGONURL1}{ticker}/range/1/{timeframe}/{fromdate}/{todate}{POLYGONURL2}{ncandles}&apiKey={self.apikey}"
        response = requests.get(url)
        data = json.loads(response.text)
        closes = []
        try:
            closes = [entry["c"] for entry in data["results"]]
        except:
            print(f"src/polygon.py :: No underlying data returned for {ticker}")
            return

        timestamps = [datetime.fromtimestamp(entry["t"] / 1000) for entry in data["results"]]

        writeCandlesCsv(data, csvname)
        print(f"src/polygon.py :: Successfully saved underlying {ticker} candlestick data to {csvname}")

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, closes, marker="o", linestyle="-")
        plt.title(f"{ticker} Close Price History")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pngname, dpi=150)
        plt.close()
        print(f"src/polygon.py :: Successfully created underlying chart {pngname}")
    
    def getProjectionChart(self, ticker, ohlc_csvname, stats_csvname, pngname):
        ohlc = pd.read_csv(ohlc_csvname, parse_dates=["Date"])
        stats = pd.read_csv(stats_csvname)

        stats["Expiration"] = stats["Expiration"].str.extract(r"([A-Za-z]{3} \d{2}, \d{4})")
        stats["Expiration"] = pd.to_datetime(stats["Expiration"], format="%b %d, %Y")

        stats["Expected Move"] = stats["Expected Move"].apply(parseExpectedMove)
        stats = stats.dropna(subset=["Expected Move"])

        stats["Max Pain"] = pd.to_numeric(stats["Max Pain"], errors="coerce")

        stats = stats.sort_values("Expiration")

        last_close = ohlc.sort_values("Date").iloc[-1]["Close"]

        projection = stats.copy()
        projection["Base Price"] = last_close
        projection["Upper Band"] = last_close + projection["Expected Move"]
        projection["Lower Band"] = last_close - projection["Expected Move"]
        
        plt.figure(figsize=(12,6))
        plt.plot(ohlc["Date"], ohlc["Close"], label="Close (History)", color="black")
        plt.plot(projection["Expiration"], projection["Upper Band"], label="Upper Band", color="green", linestyle="--", marker="o")
        plt.plot(projection["Expiration"], projection["Lower Band"], label="Lower Band", color="red", linestyle="--", marker="o")
        plt.fill_between(projection["Expiration"], projection["Lower Band"], projection["Upper Band"], color="gray", alpha=0.2)
        plt.plot(projection["Expiration"], projection["Max Pain"], label="Max Pain", color="orange", linestyle="-", marker="x")
        plt.axhline(last_close, color="blue", linestyle="-.", label=f"Last Close {last_close:.2f}")
        plt.title(f"{ticker} History & Projection")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(pngname, dpi=150)
        plt.close()
        print(f"src/polygon.py :: Successfully created projection chart {pngname}")

def writeCandlesCsv(data, csvname):
    if not data.get("results"):
        print(f"src/polygon.py :: No data to save for {csvname}")
        return
    
    with open(csvname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"])
        for entry in data["results"]:
            writer.writerow([
                datetime.fromtimestamp(entry["t"] / 1000),  # convert ms → datetime
                entry.get("o", ""),  # open
                entry.get("h", ""),  # high
                entry.get("l", ""),  # low
                entry.get("c", ""),  # close
                entry.get("v", ""),  # volume
                entry.get("vw", ""), # volume-weighted avg price
                entry.get("n", ""),  # number of transactions
            ])

def parseExpectedMove(val):
    if pd.isna(val) or not isinstance(val, str):
        return 0.0
    match = re.search(r"±([\d\.]+)", val)  # extract digits after ±
    if match:
        return float(match.group(1))
    return 0.0

if __name__ == "__main__":
    pass