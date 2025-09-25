DIRS = ["data", "img"]
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
MODES = ["iv", "delta", "gamma", "vega", "theta",
    "rho", "charm", "vanna", "vomma", "veta", "speed",
    "zomma", "color", "ultima"]
PORT = 8080

if __name__ == "__main__":
    pass