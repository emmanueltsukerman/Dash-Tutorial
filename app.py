# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



import keras
from keras.datasets import mnist
from keras import backend as K
def load_MNIST_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_MNIST_dataset()

NUM_IMAGES = 50
import numpy as np
from PIL import Image
def save_random_images(num_images):
    np.random.shuffle(x_test)
    for i in range(num_images):
        sample = x_test[i]
        sample = sample.reshape([28,28])
        im = Image.fromarray(sample*255)
        im = im.convert('RGB')
        im.save("assets/test_image_"+str(i)+".png")
save_random_images(NUM_IMAGES)
        

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
def load_MNIST_model(model_path = "MNIST_model"):
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    loaded_model = load_model(r'C:\Users\ETsukerman\Desktop\Dash\MNIST_model')
    return loaded_model


MNIST_model = load_MNIST_model()

title = html.H1(
        children='Deep Neural Network Model for MNIST'
    )
subtitle = html.Div(children='Click button to pick a random image from the MNIST dataset and display the deep neural network\'s prediction on that image.',
            style={'padding-bottom': 10})
button = html.Button(children='Predict Random Image', id='submit-val')
space = html.Br()
sample_image = html.Img(style = {'padding': 10, 'width': '400px', 'height': '400px'}, id='image')
model_prediction = html.Div(id="pred", children=None)
intermediate = html.Div(id='intermediate-operation', style={'display': 'none'})

app.layout = html.Div(style={'textAlign': "center"},children=[
    title,
    subtitle,
    button,
    space,
    sample_image,
    model_prediction,
    intermediate
])


import random
@app.callback(
    dash.dependencies.Output('intermediate-operation', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')]
)
def update_random_image(n_clicks):
    if (n_clicks is None):
        raise PreventUpdate
    else:
        return random.choice(range(NUM_IMAGES))


@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('intermediate-operation', 'children')]
)
def update_figure(img_number):
    return app.get_asset_url("test_image_"+str(img_number)+".png")

@app.callback(
    dash.dependencies.Output('pred', 'children'),
    [dash.dependencies.Input('intermediate-operation', 'children')]
)
def update_prediction(img_number):
    img = x_test[img_number]
    img = img.reshape([1,28,28,1])
    predicted_class = MNIST_model.predict_classes(img)[0]
    return "Prediction: "+str(predicted_class)

if __name__ == '__main__':
    app.run_server(debug=True)