import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from PIL import Image
from CNN_model import resize_image, crop_center, image_size, image_to_grayscale_pixel_values, standardize_pixels, CNN

#UPLOAD_FOLDER = '/Users/Alex/Documents/CNN/uploads/'
UPLOAD_FOLDER = '/home/expressionrecog/mysite3/uploads/'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resize_rsp = resize_image(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
            crop_rsp = crop_center(resize_rsp)
            grayscale_rsp = image_to_grayscale_pixel_values(crop_rsp)
            standardize_rsp = standardize_pixels(grayscale_rsp)
            predicted_label, probabilities = CNN(standardize_rsp)
            total = sum(probabilities[0])
            percentages = [round((x / float(total)) * 100, 2) for x in probabilities[0]]
            labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
            rsp_img = url_for('uploaded_file', filename=filename)
            return render_template('index.html', rsp_img=rsp_img, rsp_label=labels[predicted_label], rsp_dict=zip(labels, percentages))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

"""
if __name__ == '__main__':
    app.run(debug=True)
"""
