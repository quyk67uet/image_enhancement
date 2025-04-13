from flask import Flask, render_template, request
import cv2
import os
import enhancement_experiments
import cluster_enhancement_rgb
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Biến toàn cục để lưu ảnh gốc và ảnh đã xử lý
original_image_path = None
processed_image_path = None

@app.route("/", methods=["GET", "POST"])
def index():
    global original_image_path, processed_image_path
    original = None
    processed = None

    if request.method == "POST":
        if request.files.get("image"):
            # Nếu người dùng upload ảnh mới, lưu ảnh gốc
            file = request.files["image"]
            original_image_path = os.path.join(UPLOAD_FOLDER, 'uploaded.png')
            file.save(original_image_path)
            processed_image_path = None  # Reset ảnh đã xử lý

        # Lấy phương pháp đã chọn
        method = request.form.get("method")

        # Nếu đã có ảnh gốc hoặc ảnh đã xử lý, thực hiện xử lý
        if original_image_path:
            # Nếu ảnh chưa được xử lý, dùng ảnh gốc
            img = cv2.imread(original_image_path) if processed_image_path is None else cv2.imread(original_image_path)
            
            def resize_image(image, size):
                size = 256
                h, w = image.shape[:2]
                if h < w:
                    new_h = size
                    new_w = int(size * w / h)
                else:
                    new_h = int(size * h / w)
                    new_w = size
                image = cv2.resize(image, (new_w, new_h))
                return image
            
            img = resize_image(img, size=256)

            # if method == "gray":
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # elif method == "blur":
            #     img = cv2.GaussianBlur(img, (7, 7), 0)
            # elif method == "edge":
            #     img = cv2.Canny(img, 100, 200)

            median_window = 15
            gaussian_sigma = 3.0
            s_factor = 0.5
            p_value = 5

            input_img = img.copy()
            if method == "gaussian":
                img, _ = enhancement_experiments.unsharp_masking_gaussian(
                    input_img, sigma=gaussian_sigma, s=s_factor, p_value=p_value
                )
            elif method == "median":
                img, _ = enhancement_experiments.unsharp_masking_median(
                    input_img, window_size=median_window, s=s_factor, p_value=p_value
                )
            elif method == "AD":
                img, _ = enhancement_experiments.anisotropic_diffusion(
                    cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY), num_iter=15, K=20, g_func='exp', s=s_factor, p_value=p_value
                )
            elif method == "CF":
                img = np.array(cluster_enhancement_rgb.enhance_image(
                    input_img, kernel_size=15, alpha=0.5, k=5, p_value=p_value
                ))

            # Nếu ảnh là grayscale thì cần chuyển lại để lưu đúng màu PNG
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Lưu ảnh đã xử lý  
            processed_image_path = os.path.join(UPLOAD_FOLDER, 'processed.png')
            cv2.imwrite(processed_image_path, img)
            processed = 'processed.png'

    # Trả về các biến cho HTML
    if original_image_path:
        original = 'uploaded.png'
    return render_template("index.html", original=original, processed=processed)

if __name__ == "__main__":
    app.run(debug=True)
