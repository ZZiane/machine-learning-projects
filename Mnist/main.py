import numpy as np
import cv2
from keras.models import load_model




model = load_model('resources/mnist.h5')
np.random.seed(0)

def predict_img(path):  
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28,28))
        image = image.astype('float32')
        image = 255.0-image
        image /= 255.0
        image=image.reshape(1,-1)
        print(image.shape)
        pred = model.predict(image)
        print(np.argmax(pred))

 


predict_img('images/zero.png')
predict_img('images/two.png')
predict_img('images/one.png')