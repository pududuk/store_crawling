backend_url = "http://127.0.0.1:8001/test" ###### 변현섭님께 받기

##### 이 것을 구현
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import cv2
import os

ocr_kr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_dir=f'/home/{os.getlogin()}/.paddlex/official_models/PP-OCRv5_server_det',
    text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_dir=f'/home/{os.getlogin()}/.paddlex/official_models/korean_PP-OCRv5_mobile_rec',
    text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
)
ocr_en = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_dir=f'/home/{os.getlogin()}/.paddlex/official_models/PP-OCRv5_server_det',
    text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_dir=f'/home/{os.getlogin()}/.paddlex/official_models/PP-OCRv5_server_rec',
    text_recognition_model_name='PP-OCRv5_server_rec',
)

def ocr_ourhome(file):
    img_pil = Image.open(file).convert("RGB")
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))

    fit = img_y[:10, :10].mean() < 80

    # 이진화
    img_y_bin = np.where(img_y < 240, 0, 255).astype(np.uint8)

    # 세로 방향 3분할
    img_y_splits = np.array_split(img_y_bin, 3, axis=1)
    img_splits = np.array_split(img, 3, axis=1)

    # 메뉴 번호 추출
    if fit:
        menu_corners = ["B1", "B1", "B1"]
    else:
        menu_corners = []
        for img_y_split in img_y_splits:
            crop = img_y_split[:500]
            resized = cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2))
            input_img = np.stack([resized] * 3, axis=-1)
            result = ocr_en.predict(input_img)[0]["rec_texts"][0]
            menu_corners.append(result)

    # 메뉴 이름 추출
    menu_names = []
    for img_y_split in img_y_splits:
        crop = img_y_split[800:1000]
        resized = cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2))
        input_img = np.stack([resized] * 3, axis=-1)
        result = ocr_kr.predict(input_img)[0]["rec_texts"][0]
        menu_names.append(result)

    # 메뉴 이미지 추출
    crop_h = 475
    menu_imgs = []

    for i, img_y_split in enumerate(img_y_splits):
        col_mean = img_y_split.mean(axis=0)
        row_mean = img_y_split.mean(axis=1)

        lb_idx_w_l = np.where(col_mean[:400] > 150)[0][-1]
        lb_idx_w_r = np.where(col_mean[400:] > 150)[0][0] + 400
        lb_idx_h = np.where(row_mean[600:] > 150)[0][0] + 600

        menu_img = img_splits[i][(lb_idx_h - crop_h):lb_idx_h, lb_idx_w_l:lb_idx_w_r]
        buffer = BytesIO()
        Image.fromarray(menu_img).save(buffer, format="PNG")  # 압축 저장
        buffer.seek(0)
        menu_imgs.append(buffer)

    return menu_imgs, menu_corners, menu_names

def ocr_cjfresh(file):
    img_pil = Image.open(file).convert("RGB")
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))

    # 이진화
    img_y_bin = np.where(img_y < 240, 0, 255).astype(np.uint8)

    # 메뉴 번호 추출 (좌측 상단 부분 다운샘플링)
    crop_corner = img_y_bin[:250, :500]
    resized_corner = cv2.resize(crop_corner, (crop_corner.shape[1] // 2, crop_corner.shape[0] // 2))
    corner_rgb = np.stack([resized_corner] * 3, axis=-1)
    menu_corners = [ocr_en.predict(corner_rgb)[0]["rec_texts"][0]]

    # 메뉴 바운딩 좌표 계산
    w_l = np.where(img_y_bin[:, :500].mean(axis=0) > 200)[0][-1].item()
    w_r = np.where(img_y_bin[:, 500:].mean(axis=0) > 200)[0][0].item() + 500
    h_t = np.where(img_y_bin[:500, :].mean(axis=1) > 200)[0][-1].item()
    h_b = np.where(img_y_bin[500:, :].mean(axis=1) > 200)[0][0].item() + 500

    # 메뉴 이미지 잘라내기
    menu_crop = img[h_t:h_b, w_l:w_r]
    buffer = BytesIO()
    Image.fromarray(menu_crop).save(buffer, format="PNG")  # 압축 저장
    buffer.seek(0)
    menu_imgs = [buffer]

    # 메뉴 이름 OCR (G 채널 활용)
    img_g = img[..., 1]
    img_g_bin = np.where((img_g < 130) | (img_g > 150), 255, img_g).astype(np.uint8)

    text_crop = img_g_bin[400:1000, 1100:1850]
    resized_text = cv2.resize(text_crop, (text_crop.shape[1] // 2, text_crop.shape[0] // 2))
    text_rgb = np.stack([resized_text] * 3, axis=-1)
    menu_names = ["".join(ocr_kr.predict(text_rgb)[0]["rec_texts"])]

    return menu_imgs, menu_corners, menu_names


### 아래는 건들지 말기
from flask import Flask, request, jsonify
import threading
import requests
from io import BytesIO

app = Flask(__name__)

def post_image(image_bytesio_list, image_meta, url):
    files = [(
                'images',
                (f'menu_{i+1}.png',image_bytesio_list[i].getvalue(), 'image/png')
            ) for i in range(len(image_bytesio_list))]
    form_data = image_meta
    r = requests.post(url, files=files, data=form_data)

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
    for img in imgs:
        menu_images, menu_corners, menu_names = ocr_fn(img)
        menu_images_list.extend(menu_images)
        menu_corners_list.extend(menu_corners)
        menu_names_list.extend(menu_names)

    # POST
    menu_meta = {"corners": menu_corners_list, "names": menu_names_list}
    post_image(menu_images_list, menu_meta, backend_url)

@app.route('/test', methods=["POST"])
def test():
    print("data : ", request.form)
    print("files: ", request.files)
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