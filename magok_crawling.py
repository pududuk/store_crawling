import csv
import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def main():
    with open(
        "magok_store.json", "r", encoding="utf-8"
    ) as f:
        magok_store_info = json.load(f)

    header = ["store", "type", "lat", "lon", "menu", "price", "img_url"]
    data = [header]

    options = Options()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    for store_info in magok_store_info:
        store_name = store_info["store"]
        lat = store_info["lat"]
        lon = store_info["lon"]
        # store_type = store_info["type"]

        url = f"https://map.naver.com/p/search/{store_name}"
        print(url)
        driver.get(url)
        time.sleep(2)

        try:
            # entryIframe으로 전환
            WebDriverWait(driver, 10).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, "entryIframe"))
            )
        except:
            continue

        try:
            store_name = (
                WebDriverWait(driver, 10)
                .until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.GHAhO")))
                .text
            )
        except:
            continue

        try:
            store_type = (
                WebDriverWait(driver, 10)
                .until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.lnJFt")))
                .text
            )
        except:
            store_type = "N/A"

        try:
            menu_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'a._tab-menu[href*="menu"]')
                )
            )
            menu_tab.click()
        except:
            continue

        time.sleep(2)

        # 메뉴 항목들 수집 (E2jtl 클래스)
        try:
            menu_items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.E2jtL"))
            )
        except:
            continue

        for idx, item in enumerate(menu_items):
            store_info = [store_name, store_type, lat, lon]
            # 메뉴 이름
            try:
                name = item.find_element(By.CSS_SELECTOR, "span.lPzHi").text
                store_info.append(name)
            except:
                continue

            # 가격
            try:
                price = item.find_element(By.CSS_SELECTOR, "div.GXS1X em").text
                store_info.append(price)
            except:
                continue

            # 썸네일 이미지
            try:
                img_url = item.find_element(
                    By.CSS_SELECTOR, "div.place_thumb img"
                ).get_attribute("src")
                store_info.append(img_url)
            except:
                store_info.append("N/A")
            data.append(store_info)

        time.sleep(2)

        with open("crawling_v1.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    driver.quit()


if __name__ == "__main__":
    main()
