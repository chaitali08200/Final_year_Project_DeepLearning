from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('skin_cancer_detection_model.h5')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/"+ imagefile.filename
        imagefile.save(image_path)

        img = cv2.imread(image_path)
        resize = tf.image.resize(img,(256,256))
        yhat = model.predict(np.expand_dims(resize/255, 0))

        prob = yhat[0][0]
        if prob > 0.5:
            prediction = "Predicted class is Malignant with probability " + str(prob)
        else:
            prediction = "Predicted class is Benign with probability " + str(1-prob)

        return render_template('index3.html', prediction=prediction)

    return render_template('index3.html')

if __name__=='__main__':
    app.run(port=3000, debug=True)
