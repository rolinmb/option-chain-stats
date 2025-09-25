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
MODES = ["bs_iv", "baw_iv", "bin_iv", "delta", "gamma", "vega", "theta",
    "rho", "charm", "vanna", "vomma", "veta", "speed",
    "zomma", "color", "ultima", "time_value"]
PORT = 8080
MINIV = 1e-6
MAXIV = 5.0

if __name__ == "__main__":
    pass