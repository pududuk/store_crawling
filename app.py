backend_url = "http://34.64.137.143:8080/api/menu/ocr-response"
# backend_url = "http://127.0.0.1:8001/test"

##### 이 것을 구현
from PIL import Image
import numpy as np
import skimage as ski
from skimage.morphology import dilation, erosion
from paddleocr import PaddleOCR
import os
import getpass
import threading
import logging

username = "ubuntu"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_ocr():
    return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_dir=f'/home/{username}/.paddlex/official_models/PP-OCRv5_mobile_det',
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_dir=f'/home/{username}/.paddlex/official_models/korean_PP-OCRv5_mobile_rec',
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
        )

def ocr_ourhome(file):
    ocr = get_ocr()
    img_pil = Image.open(file).convert("RGB")
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))

    fit = img_y[:10, :10].mean() < 80

    img_y[np.where(img_y < 240)] = 0
    img_y[np.where(img_y > 0)] = 255

    img_y_splits = np.array_split(img_y, 3, axis=1)
    img_splits = np.array_split(img, 3, axis=1)

    menu_corners_in = []
    for img_y_split in img_y_splits:
        img_y_split = img_y_split[50:250, 200:650, None]
        for _ in range(2):
            img_y_split = erosion(img_y_split)[::2, ::2]
        menu_corners_in.append(img_y_split)

    if fit:
        menu_corners = ["B1", "B1", "B1"]
    else:
        menu_corners = [
            ocr.predict(np.repeat(img_y_split, 3, axis=-1))[0]["rec_texts"][0].replace(".", ",")
            for img_y_split in menu_corners_in
        ]

    menu_names = [
        ocr.predict(np.repeat(dilation(img_y_split[800:1000, :, None])[::2, ::2], 3, axis=-1))[0]["rec_texts"][0]
        for img_y_split in img_y_splits
    ]

    crop_h = 475
    menu_imgs = []

    for i, img_y_split in enumerate(img_y_splits):
        lb_idx_w_l = np.where(img_y_split[:, :400].mean(axis=0) > 150)[0][-1]
        lb_idx_w_r = np.where(img_y_split[:, 400:].mean(axis=0) > 150)[0][0] + 400
        lb_idx_h = np.where(img_y_split[600:,].mean(axis=1) > 150)[0][0] + 600
        menu_img = img_splits[i][(lb_idx_h - crop_h) : lb_idx_h, lb_idx_w_l : lb_idx_w_r]

        buffer = BytesIO()
        Image.fromarray(menu_img).save(buffer, format="PNG")  # 압축 저장
        buffer.seek(0)
        menu_imgs.append(buffer)

    return menu_imgs, menu_corners, menu_names

def ocr_cjfresh(file):
    ocr = get_ocr()
    img_pil = Image.open(file).convert("RGB")
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))

    img_y[np.where(img_y < 240)] = 0
    img_y[np.where(img_y > 0)] = 255

    corner_img_in = img_y[60:250, 120:350, None]
    for _ in range(2):
        corner_img_in = erosion(corner_img_in)[::2, ::2]
    menu_corners = [ocr.predict(np.repeat(corner_img_in, 3, axis=-1))[0]["rec_texts"][0][:2]]

    w_l = np.where(img_y[:, :500].mean(axis=0) > 200)[0][-1].item()
    w_r = np.where(img_y[:, 500:].mean(axis=0) > 200)[0][0].item() + 500
    h_t = np.where(img_y[:500, :].mean(axis=1) > 200)[0][-1].item()
    h_b = np.where(img_y[500:, :].mean(axis=1) > 200)[0][0].item() + 500

    # 메뉴 이미지 잘라내기
    menu_cropped = img[h_t:h_b, w_l:w_r]
    buffer = BytesIO()
    Image.fromarray(menu_cropped).save(buffer, format="PNG")
    buffer.seek(0)
    menu_imgs = [buffer]

    # 메뉴 이름 OCR (G 채널 활용)
    img_g = img[..., 1].copy()
    img_g[np.where(img_g > 150)] = 255
    img_g[np.where(img_g < 130)] = 255

    name_img_in = dilation(img_g[400:1000, 1100:1850, None])[::2, ::2]
    idxs = np.where(name_img_in.mean(axis=1) < 255)[0]
    name_img_in = name_img_in[(idxs[0] - 10) : (idxs[-1] + 10)]
    menu_names = ["".join(ocr.predict(np.repeat(name_img_in, 3, axis=-1))[0]["rec_texts"]).replace(" ", "")]

    return menu_imgs, menu_corners, menu_names


### 아래는 건들지 말기
from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

def post_image(image_bytesio_list, image_meta, url):
    files = [(
                'images',
                (f'menu_{i+1}.png',image_bytesio_list[i].getvalue(), 'image/png')
            ) for i in range(len(image_bytesio_list))]
    form_data = image_meta
    s = requests.Session()
    r = requests.Request('POST', url, files=files, data=form_data)
    p = s.prepare_request(r)
    logging.debug(p.headers)
    logging.debug(p.body[:500])
    r = s.send(p)
    logging.info("POST Success")


def ocr_and_post(menu_urls, ocr_fn):
    # url -> image
    imgs = []
    for menu_url in menu_urls:
        if menu_url:
            r = requests.get(menu_url)
            if r.status_code == requests.codes.ok:
                imgs.append(BytesIO(r.content))
                r.close()
            else:
                r.close()
    # ocr
    menu_images_list = []
    menu_corners_list = []
    menu_names_list = []
    for idx, img in enumerate(imgs):
        menu_images, menu_corners, menu_names = ocr_fn(img)
        menu_images_list.extend(menu_images)
        menu_corners_list.extend(menu_corners)
        menu_names_list.extend(menu_names)

    # POST
    menu_meta = {"corners": ','.join(menu_corners_list), "names": ','.join(menu_names_list)}
    post_image(menu_images_list, menu_meta, backend_url)
    logging.info("POST success")

@app.route('/test', methods=["POST"])
def test():
    logging.debug("data : ", request.form)
    logging.debug("files: ", request.files)
    return {"success": True}

@app.route("/ourhome")
def ourhome():
    args = request.args.to_dict()
    menu_urls = [args.get(f"menu_url_{i}") for i in range(1, 7)]

    thread = threading.Thread(target=ocr_and_post, args=(menu_urls, ocr_ourhome))
    thread.start()

    # return
    return jsonify({'thread_name': str(thread.name),
                    'started': True})

@app.route("/cjfresh")
def cjfresh():
    args = request.args.to_dict()
    menu_urls = [args.get(f"menu_url_{i}") for i in range(1, 7)]

    thread = threading.Thread(target=ocr_and_post, args=(menu_urls, ocr_cjfresh))
    thread.start()

    # return
    return jsonify({'thread_name': str(thread.name),
                    'started': True})


if __name__ == "__main__":
    app.run('0.0.0.0', 8001, debug=False)