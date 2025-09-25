from consts import POLYGONURL1, POLYGONURL2
import json
import requests
from datetime import datetime
import matplotlib.pyplot as plt

class PolygonAPI:
    def __init__(self, apikey):
        self.apikey = apikey

    def getOptionChart(self, option_symbol, timeframe, fromdate, todate, ncandles, pngname):
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

    def getUnderlyingChart(self, ticker, timeframe, fromdate, todate, ncandles, pngname):
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

if __name__ == "__main__":
    pass