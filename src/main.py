from utils import scrapeChainInfo, scrapeEntireChain, plotIvCurve, plotIvSurface
from consts import DIRS, BASEURL, URLP2
import sys
import os

if __name__ == "__main__":
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

    ticker = sys.argv[1].upper()

    scrapeChainInfo(f"{BASEURL}{ticker}", ticker, f"data/{ticker}stats.csv")

    plotIvCurve(ticker, f"data/{ticker}stats.csv", f"img/{ticker}chainiv.png")

    scrapeEntireChain(f"{BASEURL}{ticker}{URLP2}", ticker, f"data/{ticker}chain.csv")

    plotIvSurface(ticker, f"data/{ticker}chain.csv", f"img/{ticker}civ.png", f"img/{ticker}piv.png")