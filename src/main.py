from utils import scrapeChainStats, scrapeEntireChain, plotChainIvCurve, plotChainSurface
from consts import DIRS, MODES, BASEURL, URLP2
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

    #scrapeChainStats(ticker, f"{BASEURL}{ticker}", f"data/{ticker}stats.csv")

    #plotChainIvCurve(ticker, f"data/{ticker}stats.csv", f"img/{ticker}iv.png")

    #scrapeEntireChain(ticker, f"{BASEURL}{ticker}{URLP2}", f"data/{ticker}chain.csv")

    #for mode in MODES:
    #    plotChainSurface(ticker, mode, f"data/{ticker}chain.csv", f"img/{ticker}c{mode}.png", f"img/{ticker}p{mode}.png")
    
    #polygon = PolygonAPI(POLYGONKEY)
    #option_symbol = "NVDA250926C00175000"
    #polygon.getOptionChart(option_symbol, "day", "2025-01-01", "2025-09-24", 100, f"img/{option_symbol}.png")
    #polygon.getEquityChart(ticker, "day", "2025-04-01", "2025-09-24", 100, f"img/{ticker}.png")