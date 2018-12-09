import os

import numpy as np
from PIL import Image
from flask import Flask, request, redirect, flash
from keras.models import load_model
import keras
from werkzeug.utils import secure_filename
import importlib
import this

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = '/Users/yokokawahirotaka/PycharmProjects/number_search/uploads'  # 要変更
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("-------------------------------------------")
            print(request.files)
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # model = load_model('./animal_cnn_aug.h5')
            #
            # image = Image.open(filepath)
            # image = image.convert('RGB')
            # image = image.resize((image_size, image_size))
            # data = np.asarray(image)
            # X = []
            # X.append(data)
            # X = np.array(X)
            #
            # result = model.predict([X])[0]
            # predicted = result.argmax()
            # percentage = int(result[predicted] * 100)
            #
            # return "ラベル： " + classes[predicted] + ", 確率："+ str(percentage) + " %"
            #
            print(filename)

            return redirect('http://127.0.0.1:5000/uploads/' + str(filename))


    importlib.reload(this)
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>数字をアップロードして判定しよう</title></head>
    <body>
    <h1>数字をアップロードして判定しよう！</h1>
    <form method = "post" action="/" enctype = "multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
    </form>
    </body>
    </html>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    importlib.reload(this)
    model = load_model('./number_cnn_aug.h5')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    image = Image.open(filepath)
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = [data]
    X = np.array(X)

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    keras.backend.clear_session()

    return "ラベル： " + classes[predicted] + ", 確率：" + str(percentage) + " %"

    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
