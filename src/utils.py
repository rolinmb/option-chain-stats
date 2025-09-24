from consts import TABLEHEADERS
import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument("--headless")  # run without opening browser window

def webscrape(url, csvname):
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
    data = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        text_vals = [col.get_text(strip=True).replace("\xa0", " ") for col in cols]
        data.append(text_vals)
    
    with open(csvname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TABLEHEADERS)
        writer.writerows(data)
    
    print(f"src/utils.py :: Successfully webscraped option data table to {csvname}")
    