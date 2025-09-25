DIRS = ["data", "img"]
FVURL = "https://finviz.com/quote.ashx?t="
BASEURL = "https://optioncharts.io/options/"
URLP2 = "/option-chain"
URLP3 = "?option_type=all&expiration_dates="
URLP4 = "&view=straddle&strike_range=all"
POLYGONURL1 = "https://api.polygon.io/v2/aggs/ticker/"
POLYGONURL2 = "?adjusted=true&sort=arc&limit="
TABLEHEADERS = [
    "Expiration", "Volume Calls", "Volume Puts", "Volume Put-Call Ratio",
    "OI Calls", "OI Puts", "OI Put-Call Ratio",
    "IV", "Expected Move", "Max Pain", "Max Pain vs Current Price"
]
TRADINGDAYS = 252
FEDFUNDS = 0.0408
MODES = ["bs_iv", "baw_iv", "bin_iv", "delta", "gamma", "vega", "theta",
    "rho", "charm", "vanna", "vomma", "veta", "speed",
    "zomma", "color", "ultima", "time_value"]
PORT = 8080
MINIV = 1e-6
MAXIV = 5.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://finviz.com/"
}

if __name__ == "__main__":
    pass