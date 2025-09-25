from consts import TABLEHEADERS, TRADINGDAYS, BASEURL, URLP2, URLP3, URLP4, FEDFUNDS, MODES, MINIV, MAXIV
import os
import sys
import csv
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument("--headless")  # run without opening browser window

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

def scrapeChainStats(ticker, url, csvname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=OPTIONS)
    driver.get(url)
    # Give time for JavaScript to load (you can also use explicit waits)
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table table-sm table-hover table-bordered table-responsive optioncharts-table-styling")
    if not table:
        print("src/util.py :: No HTML table found")
        driver.quit()
        return
    
    driver.quit()
    data_rows = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        text_vals = [col.get_text(strip=True).replace("\xa0", " ") for col in cols]
        data_rows.append(text_vals)
    
    with open(csvname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TABLEHEADERS)
        writer.writerows(data_rows)
    
    print(f"src/utils.py :: Successfully webscraped {ticker} option data table to {csvname}")
    
def scrapeEntireChain(ticker, url, csvname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=OPTIONS)
    driver.get(url)
    # Give time for JavaScript to load
    time.sleep(3)
    try:
        button = driver.find_element(By.ID, "expiration-dates-form-button-1")
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        driver.execute_script("arguments[0].click();", button)
        wait = WebDriverWait(driver, 1)
        wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "label.tw-ml-3.tw-min-w-0.tw-flex-1.tw-text-gray-600")
            )
        )
    except Exception as e:
        print(f"src/utils.py :: Error clicking expiration button or waiting for labels: {e}")
        driver.quit()
        sys.exit(1)

    spans = driver.find_elements(By.CSS_SELECTOR, ".tw-text-3xl.tw-font-semibold")
    price_string = spans[1].text.strip()
    print(f"src/utils.py :: Scraped Underlying price: {price_string}")
    labels = driver.find_elements(
        By.CSS_SELECTOR, "label.tw-ml-3.tw-min-w-0.tw-flex-1.tw-text-gray-600"
    )
    expirations_text = [lbl.text for lbl in labels if lbl.text.strip()]
    formatted_expiration_dates = []
    exp_in_years = []
    wms = []
    for text in expirations_text:
        parts = text.split("(")

        date_str = parts[0].strip()
        exp_dt = datetime.strptime(date_str, "%b %d, %Y")
        final_dt = exp_dt.strftime("%Y-%m-%d")
        formatted_expiration_dates.append(final_dt)

        days_part = parts[1].split()[0]
        days = int(days_part)
        years = days / TRADINGDAYS
        exp_in_years.append(years)

        if len(formatted_expiration_dates) != len(exp_in_years):
            print(f"src/utils.py :: len(formatted_expiration_dates) != len(exp_in_years)")
            sys.exit(1)

        if "w" in parts[2]:
            wms.append("w")
        elif "m" in parts[2]:
            wms.append("m")

    print(f"src/utils.py :: Successfully parsed expiration dates and calculated yte for each expiry/contract")
    time.sleep(1.0)
    expiries = []
    for i in range(0, len(exp_in_years)):
        url = f"{BASEURL}{ticker}{URLP2}{URLP3}{formatted_expiration_dates[i]}:{wms[i]}{URLP4}"
        driver.get(url)
        time.sleep(3.0)
        shortdate_parts = formatted_expiration_dates[i].split("-")
        shortdate = shortdate_parts[0][2:]+shortdate_parts[1]+shortdate_parts[2]
        calls = []
        puts = []
        table = driver.find_element(By.CSS_SELECTOR, "table.table.table-sm.table-hover")
        rows = table.find_elements(By.TAG_NAME, "tr")[1:]
        for row in rows:
            cbid = cask = cvol = c_oi = strike = 0
            pbid = pask = pvol = p_oi = 0
            cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
            if len(cols) != 11:
                continue
            
            clast = cols[0] if cols[0] != "-" else "0.00"
            cbid = cols[1]
            cask = cols[2]
            cvol = cols[3]
            c_oi = cols[4]
            strike = cols[5]
            plast = cols[6] if cols[6] != "-" else "0.00"
            pbid = cols[7]
            pask = cols[8]
            pvol = cols[9]
            p_oi = cols[10]

            # Make symbols like CHGG251121C00002000 = CHGG 2025-11-21 $2.00 call
            strike_val = float(strike)            # e.g. "2.00" → 2.0
            strike_int = int(strike_val * 1000)   # scale by 1000 → 2000
            strike_str = f"{strike_int:08d}"
            csymbol = f"{ticker}{shortdate}C{strike_str}"
            psymbol = f"{ticker}{shortdate}P{strike_str}"

            calls.append(OptionContract(ticker, csymbol, price_string, strike, exp_in_years[i], clast, cbid, cask, cvol, c_oi, True))
            puts.append(OptionContract(ticker, psymbol, price_string, strike, exp_in_years[i], plast, pbid, pask, pvol, p_oi, False))

        print(f"src/utils.py :: Processed Calls and Puts for expiration {formatted_expiration_dates[i]}")
        expiries.append(OptionExpiry(ticker, formatted_expiration_dates[i], exp_in_years[i], calls, puts))

    option_chain = OptionChain(ticker, expiries)
    with open(csvname, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "expiry", "yte", "underlying", "symbol", "underlying_price", "strike", "call_or_put",
            "last", "bid", "ask", "volume", "open_interest",
            "bs_iv", "baw_iv", "bin_iv", "delta", "gamma", "vega", "theta", "rho",
            "charm", "vanna", "vomma", "veta", "speed", "zomma", "color", "ultima",
            "intrinsic_value", "time_value"
        ])
        for expiry in option_chain.expiries:
            for c in expiry.calls:
                writer.writerow([
                    expiry.date, f"{c.yte:.4f}", c.underlying, c.symbol, c.underlying_price, c.strike, "Call",
                    c.midprice, c.bidprice, c.askprice, c.volume, c.openinterest,
                    c.bsiv, c.bawiv, c.biniv, c.delta, c.gamma, c.vega, c.theta, c.rho,
                    c.charm, c.vanna, c.vomma, c.veta, c.speed, c.zomma, c.color, c.ultima,
                    c.intrinsic_value, c.time_value
                ])
            for p in expiry.puts:
                writer.writerow([
                    expiry.date, f"{p.yte:.4f}", p.underlying, p.symbol, p.underlying_price, p.strike, "Put",
                    p.midprice, p.bidprice, p.askprice, p.volume, p.openinterest,
                    p.bsiv, p.bawiv, p.biniv, p.delta, p.gamma, p.vega, p.theta, p.rho,
                    p.charm, p.vanna, p.vomma, p.veta, p.speed, p.zomma, p.color, p.ultima,
                    p.intrinsic_value, p.time_value
                ])

    print(f"scr/utils.py :: Successfully saved {ticker} option chain to {csvname}")

def plotChainIvCurve(ticker, csvname, pngname):
    if not os.path.exists(csvname):
        print(f"src/utils.py :: {csvname} does not exist")
        return
    
    if "chain" in csvname:
        print(f"src/utils.py :: {csvname} cannot be used because it is an option chain csv")
        return

    df = pd.read_csv(csvname)
    df = df[df["IV"].notna() & (df["IV"] != "")]  # skip missing
    df["IV"] = df["IV"].str.replace("%", "", regex=False).astype(float)
    df["Expiration"] = df["Expiration"].str.extract(r'([A-Za-z]+ \d{2}, \d{4})')[0]
    df["ExpirationDate"] = pd.to_datetime(df["Expiration"], format="%b %d, %Y")

    plt.figure(figsize=(10, 6))
    plt.plot(df["ExpirationDate"], df["IV"], marker="o", linestyle="-", color="blue")
    plt.xlabel("Expiration")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"{ticker} Implied Volatility Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pngname, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created IV curve and saved to {pngname}")

def plotChainSurface(ticker, mode, csvname, pngnamec, pngnamep):
    if not os.path.exists(csvname):
        print(f"src/utils.py :: {csvname} does not exist")
        return
    
    if "chain" not in csvname:
        print(f"src/utils.py :: {csvname} cannot be used because it is not an option chain csv")
        return

    if mode not in MODES:
        print(f"src/utils.py :: Mode {mode} is invalid; valid modes are: {MODES}")
        return
    
    df = pd.read_csv(csvname)
    df = df[df[mode].notna()]
    df["strike"] = df["strike"].astype(float)
    df[mode] = df[mode].astype(float)
    df["yte"] = df["yte"].astype(float)
    
    df_calls = df[df["call_or_put"] == "Call"]
    df_puts = df[df["call_or_put"] == "Put"]

    strikesc = sorted(df_calls["strike"].unique())
    expiriesc = sorted(df_calls["yte"].unique())
    strikesp = sorted(df_puts["strike"].unique())
    expiriesp = sorted(df_puts["yte"].unique())

    XC, YC = np.meshgrid(strikesc, expiriesc)
    XP, YP = np.meshgrid(strikesp, expiriesp)
    ZC = np.zeros_like(XC, dtype=float)
    ZP = np.zeros_like(XP, dtype=float)

    for i, t in enumerate(expiriesc):
        for j, k in enumerate(strikesc):
            data_row = df_calls[(df_calls["yte"] == t) & (df_calls["strike"] == k)]
            if not data_row.empty:
                ZC[i, j] = data_row[mode].values[0]
            else:
                ZC[i, j] = 0.000001  # in case some strike-expiry combinations don't exist

    for i, t in enumerate(expiriesp):
        for j, k in enumerate(strikesp):
            data_row = df_puts[(df_puts["yte"] == t) & (df_puts["strike"] == k)]
            if not data_row.empty:
                ZP[i, j] = data_row[mode].values[0]
            else:
                ZP[i, j] = 0.000001  # in case some strike-expiry combinations don't exist

    figc = plt.figure(figsize=(12,8))
    axc = figc.add_subplot(111, projection="3d")
    surfc = axc.plot_surface(XC, YC, ZC, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)
    axc.set_xlabel("Strike")
    axc.set_ylabel("Time to Expiry (Years)")
    axc.set_title(f"{ticker} {mode} Surface (Calls)")
    figc.colorbar(surfc, shrink=0.5, aspect=10, label=mode)
    plt.tight_layout()
    plt.savefig(pngnamec, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created {mode} call surface and saved to {pngnamec}")

    figp = plt.figure(figsize=(12,8))
    axp = figp.add_subplot(111, projection="3d")
    surfp = axp.plot_surface(XP, YP, ZP, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)
    axp.set_xlabel("Strike")
    axp.set_ylabel("Time to Expiry (Years)")
    axp.set_title(f"{ticker} {mode} Surface (Puts)")
    figp.colorbar(surfp, shrink=0.5, aspect=10, label=mode)
    plt.tight_layout()
    plt.savefig(pngnamep, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created {mode} put surface and saved to {pngnamep}")

if __name__ == "__main__":
    pass