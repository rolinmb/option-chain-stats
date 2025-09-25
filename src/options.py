from consts import TRADINGDAYS, FEDFUNDS, MINIV, MAXIV
import os
import re
import math
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

class OptionContract:
    def __init__(self, ticker, symbol, underlyingprice, strike, yte, lastprice, bidprice, askprice, vol, oi, cp_flag):
        self.underlying = ticker
        self.symbol = symbol
        self.underlying_price = float(underlyingprice)
        self.strike = float(strike.replace(",", ""))
        self.yte = float(yte)
        self.lastprice = float(lastprice.replace(",", ""))
        self.bidprice = float(bidprice.replace(",", ""))
        self.askprice = float(askprice.replace(",", ""))
        self.midprice = float((self.bidprice + self.askprice) / 2)
        self.volume = float(vol.replace(",", ""))
        self.openinterest = float(oi.replace(",", ""))
        self.iscall = cp_flag
        self.bsiv = 0.000001
        self.bawiv = baw_implied_vol(self.midprice, self.strike, self.underlying_price, self.yte, FEDFUNDS, 0.0, self.iscall)
        self.biniv = binomial_implied_vol(self.midprice, self.strike, self.underlying_price, self.yte, FEDFUNDS, 0.0, self.iscall)
        self.intrinsic_value = 0.0
        if self.iscall:
            self.bsiv = bsimplied_volatility_call(self.midprice, self.underlying_price, self.strike, self.yte, FEDFUNDS)
           
            self.intrinsic_value = max(self.underlying_price - self.strike, 0)
        else:
            self.bsiv = bsimplied_volatility_put(self.midprice, self.underlying_price, self.strike, self.yte, FEDFUNDS)
            self.intrinsic_value = max(self.strike - self.underlying_price, 0)
        self.time_value = max(self.midprice - self.intrinsic_value, 0)
        self.delta = getDelta(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS, self.iscall)
        self.gamma = getGamma(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.vega = getVega(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.theta = getTheta(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS, self.iscall)
        self.rho = getRho(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS, self.iscall)
        self.charm = getCharm(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS, self.iscall)
        self.vanna = getVanna(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.vomma = getVomma(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.veta  = getVeta(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.speed = getSpeed(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.zomma = getZomma(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.color = getColor(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)
        self.ultima = getUltima(self.underlying_price, self.strike, self.yte, self.bsiv, FEDFUNDS)        

    def __repr__(self):
        return (
            f"<OptionContract {self.underlying} "
            f"{'Call' if self.iscall else 'Put'} "
            f"Strike={self.strike:.2f} "
            f"Mid={self.midprice:.2f} "
            f"YtE={self.yte:.4f}>\n"
            f"  bsIV={self.bsiv:.4f} | Delta={self.delta:.4f} | Gamma={self.gamma:.6f} | "
            f"Vega={self.vega:.4f} | Theta={self.theta:.4f} | Rho={self.rho:.4f}\n"
            f"  Charm={self.charm:.6f} | Vanna={self.vanna:.6f} | Vomma={self.vomma:.6f} | "
            f"Veta={self.veta:.6f}\n"
            f"  Speed={self.speed:.6f} | Zomma={self.zomma:.6f} | Color={self.color:.6f} | "
            f"Ultima={self.ultima:.6f} | Intrinsic Value={self.bsintrinsic_value:.2f} | Time Value={self.bstime_value:.2f}>"
        )

class OptionExpiry:
    def __init__(self, ticker, date, yte, calls=None, puts=None):
        self.underlying = ticker
        self.date = date
        self.yte = yte
        self.calls = calls if calls is not None else []
        self.puts = puts if puts is not None else []

    def __repr__(self):
        return (
            f"OptionExpiry("
            f"underlying='{self.underlying}', "
            f"date='{self.date}', "
            f"yte={self.yte:.4f}, "
            f"calls={len(self.calls)} contracts, "
            f"puts={len(self.puts)} contracts)"
        )

class UnderlyingAsset:
    def __init__(self, ticker, csvname):
        if not os.path.exists(csvname):
            print(f"src/options.py :: {csvname} does not exist; cannot build Underlying Asset")
            return
        self.underlying = ticker
        self.divyield = 0.0
        tempdf = pd.read_csv(csvname)
        tempdf = tempdf.set_index("Label")
        dividend_ttm = tempdf.loc["Dividend TTM", "Value"]
        match = re.search(r"\((.*?)\)", dividend_ttm)
        divyield_text = match.group(1) if match else "0.0%"
        self.divyield = float(divyield_text.split("%")[0]) / 100
        price = tempdf.loc["Price", "Value"].replace(",", "")
        percent_change = tempdf.loc["Change", "Value"].replace(",", "")
        dollar_change = tempdf.loc["$ Change", "Value"].replace(",",  "")
        self.price = float(price)
        self.pchange = float(percent_change.split("%")[0]) / 100
        self.change = float(dollar_change)

class OptionChain:
    def __init__(self, ticker, expiries=None):
        self.underlying = ticker
        self.expiries = expiries if expiries is not None else []

    def __repr__(self):
        expiry_dates = [exp.date for exp in self.expiries]
        return (
            f"OptionChain(underlying='{self.underlying}', "
            f"expiries={len(self.expiries)}, "
            f"dates={expiry_dates})"
        )

def getChainFromCsv(csvname):
    if not os.path.exists(csvname):
        print(f"src/utils.py :: {csvname} does not exist")
        return None
    
    base = os.path.basename(csvname)
    if "chain" not in base:
        print(f"src/utils.py :: {csvname} is not an option chain csv")
        return None
    
    ticker = base.split("chain")[0]
    df = pd.read_csv(csvname)
    expiries = []

    grouped = df.groupby("expiry")
    for expiry_date, group in grouped:
        yte = group["yte"].iloc[0]
        calls = []
        puts = []
        for _, row in group.iterrows():
            is_call = row["call_or_put"] == "Call"
            contract = OptionContract(
                ticker=row["underlying"],
                symbol=row["symbol"],
                underlyingprice=row["underlying_price"],
                strike=str(row["strike"]),
                yte=row["yte"],
                lastprice=str(row["last"]),
                bidprice=str(row["bid"]),
                askprice=str(row["ask"]),
                vol=str(row["volume"]),
                oi=str(row["open_interest"]),
                cp_flag=is_call
            )
            if is_call:
                calls.append(contract)
            else:
                puts.append(contract)

        expiries.append(OptionExpiry(ticker, expiry_date, yte, calls=calls, puts=puts))

    return OptionChain(ticker, expiries)

# Black-Scholes-Merton Model
def bs_call_price(S, K, T, sigma, r=FEDFUNDS):
    if sigma == 0 or T == 0:
        return max(0.0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

def bsimplied_volatility_call(C_market, S, K, T, r=FEDFUNDS):
    f = lambda sigma: bs_call_price(S, K, T, r, sigma) - C_market
    try:
        iv = brentq(f, MINIV, MAXIV)
    except (ValueError, OverflowError, ZeroDivisionError):
        iv = 0.000001
    return iv

def bs_put_price(S, K, T, sigma, r=FEDFUNDS):
    if sigma == 0 or T == 0:
        return max(0.0, K - S)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bsimplied_volatility_put(P_market, S, K, T, r=FEDFUNDS):
    f = lambda sigma: bs_put_price(S, K, T, r, sigma) - P_market
    try:
        iv = brentq(f, MINIV, MAXIV)
    except (ValueError, OverflowError, ZeroDivisionError):
        iv = 0.000001
    return iv

def bs_d1_d2(S, K, T, sigma, r=FEDFUNDS):
    if sigma <= 0 or T <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

# Barone-Adesi & Whaley (1987) approximation for American options.
def baw_price(S, K, T, r, q, sigma, is_call=True):
    if T < 1e-8:
        return max(0.0, (S - K) if is_call else (K - S))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call and q == 0:  # dividend-free call = European
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    # Critical stock price computation
    M = 2 * r / sigma**2
    N = 2 * (r - q) / sigma**2
    K1 = 1 - math.exp(-r * T) * norm.cdf(d2)
    if is_call:
        q2 = (-(N - 1) + math.sqrt((N - 1)**2 + 4 * M / K1)) / 2
        S_crit = K / (1 - 1/q2)
    else:
        q2 = (-(N - 1) - math.sqrt((N - 1)**2 + 4 * M / K1)) / 2
        S_crit = K / (1 - 1/q2)

    # If not in early exercise region, use European
    euro_price = (S * math.exp(-q * T) * norm.cdf(d1) - 
                  K * math.exp(-r * T) * norm.cdf(d2)) if is_call else \
                 (K * math.exp(-r * T) * norm.cdf(-d2) - 
                  S * math.exp(-q * T) * norm.cdf(-d1))
    if (is_call and S < S_crit) or (not is_call and S > S_crit):
        return euro_price

    # Otherwise add early exercise premium
    A = (S_crit / q2) * (1 - math.exp(-q * T) * norm.cdf(d1))
    return euro_price + A * ((S / S_crit) ** q2)

def baw_implied_vol(market_price, S, K, T, r, q, is_call=True):
    f = lambda sigma: baw_price(S, K, T, r, q, sigma, is_call) - market_price
    try:
        return brentq(f, MINIV, MAXIV)
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0.000001

# Binomial tree pricing for European/American options.
def binomial_price(S, K, T, r, q, sigma, steps=200, is_call=True, american=True):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # stock prices at maturity
    ST = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if is_call:
        values = [max(0, s - K) for s in ST]
    else:
        values = [max(0, K - s) for s in ST]

    # backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            values[j] = disc * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                ST_ij = S * (u ** j) * (d ** (i - j))
                exercise = (max(0, ST_ij - K) if is_call else max(0, K - ST_ij))
                values[j] = max(values[j], exercise)
    return values[0]

def binomial_implied_vol(market_price, S, K, T, r, q, is_call=True, american=True, steps=200):
    f = lambda sigma: binomial_price(S, K, T, r, q, sigma, steps, is_call, american) - market_price
    try:
        return brentq(f, MINIV, MAXIV)
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0.000001

def getDelta(S, K, T, sigma, r=FEDFUNDS, is_call=True):
    d1, _ = bs_d1_d2(S, K, T, sigma, r)
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def getGamma(S, K, T, sigma, r=FEDFUNDS):
    d1, _ = bs_d1_d2(S, K, T, sigma, r)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def getVega(S, K, T, sigma, r=FEDFUNDS):
    d1, _ = bs_d1_d2(S, K, T, sigma, r)
    return S * norm.pdf(d1) * math.sqrt(T) / 100.0  # expressed per 1% change in vol

def getTheta(S, K, T, sigma, r=FEDFUNDS, is_call=True):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if is_call:
        second_term = -r * K * math.exp(-r*T) * norm.cdf(d2)
        return (first_term + second_term) / TRADINGDAYS  # per day
    else:
        second_term = r * K * math.exp(-r*T) * norm.cdf(-d2)
        return (first_term + second_term) / TRADINGDAYS

def getRho(S, K, T, sigma, r=FEDFUNDS, is_call=True):
    _, d2 = bs_d1_d2(S, K, T, sigma, r)
    if is_call:
        return K * T * math.exp(-r*T) * norm.cdf(d2) / 100.0  # per 1% change in rates
    else:
        return -K * T * math.exp(-r*T) * norm.cdf(-d2) / 100.0

def getCharm(S, K, T, sigma, r=FEDFUNDS, is_call=True):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    if T <= 0: return 0.0
    first_term = -norm.pdf(d1) * (2*r*T - d2*sigma*math.sqrt(T)) / (2*T*sigma*math.sqrt(T))
    if is_call:
        return first_term - r*math.exp(-r*T)*norm.cdf(d2)
    else:
        return first_term + r*math.exp(-r*T)*norm.cdf(-d2)

def getVanna(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    return -norm.pdf(d1) * d2 / sigma

def getVomma(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    v = getVega(S, K, T, sigma, r) * 100.0  # undo /100
    return v * d1 * d2 / sigma

def getVeta(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    v = getVega(S, K, T, sigma, r) * 100.0
    return -v * (r + (d1*d2) / (2*T))

def getSpeed(S, K, T, sigma, r=FEDFUNDS):
    d1, _ = bs_d1_d2(S, K, T, sigma, r)
    g = getGamma(S, K, T, sigma, r)
    return -g * (d1 / (S * sigma * math.sqrt(T)))

def getZomma(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    g = getGamma(S, K, T, sigma, r)
    return g * (d1*d2 - 1) / sigma

def getColor(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    g = getGamma(S, K, T, sigma, r)
    return -g * (2*r*T + 1 + d1*d2) / (2*T)

def getUltima(S, K, T, sigma, r=FEDFUNDS):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    v = getVega(S, K, T, sigma, r) * 100.0
    return -v * (d1*d2*(1 - d1*d2) + d1**2 + d2**2) / sigma**2