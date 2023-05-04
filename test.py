from flask import Flask, render_template, request
from face_detection import face_detection
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files['file']
        image.save('static/' + image.filename)
        results,name = face_detection('static/' + image.filename)
        nouvelle_taille = (500, 400)
        nouvelle_image = results.resize(nouvelle_taille, resample=Image.LANCZOS)
        nouvelle_image.save('static/r'+image.filename)

        return render_template('home.html', filename='r'+image.filename,name=name)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
