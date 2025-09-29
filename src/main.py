from utils import *
from consts import DIRS, MODES, BASEURL, URLP2, FVURL
from polygon import PolygonAPI
from key import POLYGONKEY
import sys
import os

def startupRoutine():
    if len(sys.argv) != 2:
        print("src/main.py :: Only one ticker argument required [EX. python src/main.py SPY, NVDA]")
        sys.exit(1)

    if any(char.isdigit() for char in sys.argv[1]):
        print(f"src/main.py :: No numerical values allowed in tickers, you entered {sys.argv[1]}")
        sys.exit(1)

    if len(sys.argv[1]) > 4:
        print(f"src/main.py :: Must enter at most 4 characters, you entered {sys.argv[1]}")
        sys.exit(1)

    for dir in DIRS:
        if not os.path.exists(dir):
            os.makedirs(dir)

    return sys.argv[1].upper()

if __name__ == "__main__":
    ticker = startupRoutine()
    
    scrapeUnderlyingInfo(ticker, f"{FVURL}{ticker}", f"data/{ticker}info.csv")
    
    scrapeChainStats(ticker, f"{BASEURL}{ticker}", f"data/{ticker}stats.csv")

    plotChainIvCurve(ticker, f"data/{ticker}stats.csv", f"img/{ticker}iv.png")
    
    chain = scrapeEntireChain(ticker, f"{BASEURL}{ticker}{URLP2}", f"data/{ticker}chain.csv")
    symbolc, symbolp = chain.getAtmOptionSymbols()
    call = chain.getOptionContract(symbolc)
    print(f"src/main.py :: Call option {call.symbol} = {symbolc} successfully found")
    put = chain.getOptionContract(symbolp)
    print(f"src/main.py :: Put option {put.symbol} = {symbolp} successfully found")
    
    for mode in MODES:
        plotChainSurface(ticker, mode, f"data/{ticker}chain.csv", f"img/{ticker}c{mode}.png", f"img/{ticker}p{mode}.png")
    
    polygon = PolygonAPI(POLYGONKEY)
    polygon.getOptionChart(symbolc, "day", "2025-01-01", "2025-09-26", 365, f"data/{symbolc}ohlc.csv", f"img/{symbolc}.png")
    polygon.getOptionChart(symbolp, "day", "2025-01-01", "2025-09-26", 365, f"data/{symbolp}ohlc.csv", f"img/{symbolp}.png")
    polygon.getUnderlyingChart(ticker, "day", "2025-01-01", "2025-09-26", 365, f"data/{ticker}ohlc.csv", f"img/{ticker}.png")
    getProjectionChart(ticker, f"data/{ticker}ohlc.csv", f"data/{ticker}stats.csv", f"img/{ticker}projection.png")