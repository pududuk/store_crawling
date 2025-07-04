backend_url = "http://127.0.0.1:8001/test" ###### 변현섭님께 받기

##### 이 것을 구현
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

ocr_kr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_model_dir='models/kor/det',
    rec_model_dir='models/kor/rec',
    cls_model_dir='models/kor/cls'
)
ocr_en = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_model_dir='models/en/det',
    rec_model_dir='models/en/rec',
    cls_model_dir='models/en/cls'
)

def ocr_ourhome(file):

    img_pil = Image.open(file)
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))

    fit = img_y[:10, :10].mean() < 80

    img_y[np.where(img_y < 240)] = 0
    img_y[np.where(img_y > 0)] = 255

    img_y_splits = np.array_split(img_y, 3, axis=1)
    img_splits = np.array_split(img, 3, axis=1)

    if fit:
        menu_corners = ["B1", "B1", "B1"]
    else:
        menu_corners = [
            ocr_en.predict(np.repeat(img_y_split[:500, :, None], 3, axis=-1))[0]["rec_texts"][0]
            for img_y_split in img_y_splits
        ]

    menu_names = [
        ocr_kr.predict(np.repeat(img_y_split[800:1000, :, None], 3, axis=-1))[0]["rec_texts"][0]
        for img_y_split in img_y_splits
    ]

    crop_h = 475

    menu_imgs = []
    for i, img_y_split in enumerate(img_y_splits):
        lb_idx_w_l = np.where(img_y_split[:, :400].mean(axis=0) > 150)[0][-1]
        lb_idx_w_r = np.where(img_y_split[:, 400:].mean(axis=0) > 150)[0][0] + 400
        lb_idx_h = np.where(img_y_split[600:,].mean(axis=1) > 150)[0][0] + 600
        menu_img = img_splits[i][(lb_idx_h - crop_h) : lb_idx_h, lb_idx_w_l : lb_idx_w_r]
        menu_imgs.append(BytesIO(Image.fromarray(menu_img).tobytes()))
    
    return menu_imgs, menu_corners, menu_names

def ocr_cjfresh(file):
    ocr_kr = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    ocr_en = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )    
    img_pil = Image.open(file)
    img = np.array(img_pil)
    img_y = np.array(img_pil.convert("L"))
    img_y[np.where(img_y < 240)] = 0
    img_y[np.where(img_y > 0)] = 255
    menu_corners = [ocr_en.predict(np.repeat(img_y[:250, :500, None], 3, axis=-1))[0]["rec_texts"][0]]
    w_l = np.where(img_y[:, :500].mean(axis=0) > 200)[0][-1].item()
    w_r = np.where(img_y[:, 500:].mean(axis=0) > 200)[0][0].item() + 500
    h_t = np.where(img_y[:500, :].mean(axis=1) > 200)[0][-1].item()
    h_b = np.where(img_y[500:, :].mean(axis=1) > 200)[0][0].item() + 500
    menu_cropped = img[h_t:h_b, w_l:w_r]
    menu_imgs = [BytesIO(Image.fromarray(menu_cropped).tobytes())]
    img_g = img[..., 1].copy()
    img_g[np.where(img_g > 150)] = 255
    img_g[np.where(img_g < 130)] = 255
    menu_names = ["".join(ocr_kr.predict(np.repeat(img_g[400:1000, 1100:1850, None], 3, axis=-1))[0]["rec_texts"])]
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

@app.route('/test', methods=["POST"])
def test():
    print("data : ", request.form)
    print("files: ", request.files)
    return {"success": True}

@app.route("/ourhome")
def ourhome():
    args = request.args.to_dict()
    menu_urls = [args.get(f"menu_url_{i}") for i in range(1, 7)]

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
        menu_images, menu_corners, menu_names = ocr_ourhome(img)
        menu_images_list.extend(menu_images)
        menu_corners_list.extend(menu_corners)
        menu_names_list.extend(menu_names)

    # POST
    menu_meta = {"corners": menu_corners_list, "names": menu_names_list}
    thread = threading.Thread(target=post_image, args=(menu_images_list, menu_meta, backend_url))
    thread.daemon = True
    thread.start()

    # return
    return jsonify({'thread_name': str(thread.name),
                    'started': True})

@app.route("/cjfresh")
def cjfresh():
    args = request.args.to_dict()
    menu_urls = [args.get(f"menu_url_{i}") for i in range(1, 7)]

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
        menu_images, menu_corners, menu_names = ocr_cjfresh(img)
        menu_images_list.extend(menu_images)
        menu_corners_list.extend(menu_corners)
        menu_names_list.extend(menu_names)

    # POST
    menu_meta = {"corners": menu_corners_list, "names": menu_names_list}
    thread = threading.Thread(target=post_image, args=(menu_images_list, menu_meta, backend_url))
    thread.daemon = True
    thread.start()

    # return
    return jsonify({'thread_name': str(thread.name),
                    'started': True})


if __name__ == "__main__":
    app.run('0.0.0.0', 8001, debug=True)