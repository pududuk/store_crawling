import csv
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from datetime import datetime, timedelta


def main(img_src_path, csv_dst_path):
    th_1 = 100

    img_pil = Image.open(img_src_path)

    img_y = np.array(img_pil.convert("L"))
    img = np.repeat(img_y[..., None], 3, axis=-1)
    img[np.where(img_y < th_1)] = 0

    ocr = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # extract date
    t = ocr.predict(img[70:150, 270:370])[0]["rec_texts"][1].split(" ")
    month = t[0][:-1]
    day = t[1][:-1]

    datetime_obj = datetime(2025, int(month), int(day))
    date_list = [(datetime_obj + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]

    # extract menu patches
    th_2 = 20
    skip_row = 900
    margin = 4

    top = img[:skip_row, 2, 0]
    bottom = img[skip_row:, 2, 0]

    v = np.unique(bottom)[0].item()

    lower = np.where(bottom == v)[0][0]
    upper = np.where(top == v)[0][-1]

    img_cropped = img[(upper + margin) : (lower + skip_row - margin), 1:-margin]

    temp = img_cropped.mean(axis=0).mean(axis=-1)
    temp[np.where(temp < th_2)] = 0
    temp[np.where(temp > th_2)] = 255
    v_split_idx = np.where(temp == np.unique(temp)[0])[0]
    v_split_img = np.split(img_cropped, v_split_idx, axis=1)[1:]

    temp = v_split_img[0].mean(axis=1).mean(axis=-1)
    temp[np.where(temp < th_2)] = 0
    temp[np.where(temp > th_2)] = 255
    h_split_idx = np.where(temp == np.unique(temp)[0])[0]

    menu_data = []
    for v_img in v_split_img:
        patches = np.split(v_img, h_split_idx, axis=0)[:-1]
        menu_data.append(patches)
    menu_data = menu_data

    # extract corner names
    corner_names = []
    for corner_patch in menu_data[0]:
        corner_names.append(ocr.predict(corner_patch)[0]["rec_texts"][0])

    # extract menu data
    weekly_menu_data = []
    for date_idx, day_menu_data in enumerate(menu_data[1:]):
        t = []
        for loc_idx, menu in enumerate(day_menu_data):
            result_temp = ocr.predict(menu)[0]["rec_texts"][:-1]
            result = []
            for r in result_temp:
                if len(r) == 1:
                    continue
                result.append(r)
            weekly_menu_data.append([date_list[date_idx], corner_names[loc_idx], "/".join(result)])

    with open(csv_dst_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "corner", "menu"])
        for data in weekly_menu_data:
            writer.writerow(data)


if __name__ == "__main__":
    img_src_path = "freshway_weekly_menu.png"
    csv_dst_path = "freshway_weekly_menu_data.csv"
    main(img_src_path, csv_dst_path)
  
