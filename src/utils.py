from consts import TABLEHEADERS, TRADINGDAYS, BASEURL, URLP2, URLP3, URLP4
import os
import sys
import csv
import time
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
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
    def __init__(self, ticker, underlyingprice, strike, yte, lastprice, bidprice, askprice, vol, oi, cp_flag):
        self.underlying = ticker
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

    def __repr__(self):
        return (
            f"OptionContract("
            f"underlying='{self.underlying}', "
            f"strike={self.strike}, "
            f"type='{"Call" if self.iscall else "Put"}', "
            f"midprice={self.midprice}, "
            f"yte={self.yte:.4f})"
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

def scrapeChainInfo(url, ticker, csvname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=OPTIONS)
    driver.get(url)
    # Give time for JavaScript to load (you can also use explicit waits)
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table table-sm table-hover table-bordered table-responsive optioncharts-table-styling")
    if not table:
        print("src/util.py :: No HTML table found")
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
    
def scrapeEntireChain(url, ticker, csvname):
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
    for text in expirations_text:
        date_str = text.split("(")[0].strip()
        exp_dt = datetime.strptime(date_str, "%b %d, %Y")
        final_dt = exp_dt.strtftime("%Y-%m-%d")
        formatted_expiration_dates.append(final_dt)

        days_part = text.split("(")[1].split()[0]
        days = int(days_part)
        years = days / TRADINGDAYS
        exp_in_years.append(years)
        if len(formatted_expiration_dates) != len(exp_in_years):
            print(f"src/utils.py :: len(formatted_expiration_dates) != len(exp_in_years)")
            sys.exit(1)

    print(f"src/utils.py :: Successfully parsed expiration dates and calculated yte for each expiry/contract")
    time.sleep(1.0)
    expiries = []
    for i in range(0, len(exp_in_years)):
        url = f"{url}{URLP3}{formatted_expiration_dates[i]}{URLP4}"
        driver.get(url)
        time.sleep(1.0)
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
            calls.append(OptionContract(ticker, strike, price_string, exp_in_years[i], clast, cbid, cask, cvol, c_oi, True))
            puts.append(OptionContract(ticker, strike, price_string, exp_in_years[i], plast, pbid, pask, pvol, p_oi, False))

        print(f"src/utils.py :: Processed Calls and Puts for expiration {formatted_expiration_dates[i]}")
        expiries.append(OptionExpiry(ticker, formatted_expiration_dates[i], exp_in_years[i], calls, puts))

    option_chain = OptionChain(ticker, expiries)
    with open(csvname, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "expiry", "underlying", "underlying_price", "strike", "call_or_put",
            "last", "bid", "ask", "volume", "open_interest", "yte"
        ])
        for expiry in option_chain.expiries:
            for c in expiry[3]:
                writer.writerow([
                    expiry.date, c.underlying, c.underlying_price, c.strike, "Call",
                    c.midprice, c.bidprice, c.askprice, c.volume, c.openinterest, f"{c.yte:.2f}"
                ])
            for p in expiry[4]:
                writer.writerow([
                    expiry.date, p.underlying, p.underlying_price, p.strike, "Put",
                    p.midprice, p.bidprice, p.askprice, p.volume, p.openinterest, f"{p.yte:.2f}"
                ])

    print(f"scr/utils.py :: Saved {ticker} option chain to {csvname}")

def plotIvCurve(csvname, pngname):
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
    plt.title("Implied Volatility Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pngname, dpi=150)
    plt.close()
    print(f"src/utils.py :: Successsfully created IV curve and saved to {pngname}")