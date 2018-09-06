# AI-Deep-Learning-04-Intro-to-Keras
Udacity Self-Driving Car Engineer Nanodegree: Keras

## Layers

A Keras layer is just like a neural network layer. 

There are fully connected layers, max pool layers, and activation layers.

```python
    # Create the Sequential model
    model = Sequential()

    #1st Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))

    #2nd Layer - Add a fully connected layer
    model.add(Dense(100))

    #3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    #4th Layer - Add a fully connected layer
    model.add(Dense(60))

    #5th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))
```

## Neural Networks in Keras

```python 
# Build the Fully Connected Neural Network in Keras
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
```

1. Set the first layer to a ``Flatten()`` layer with the ``input_shape`` set to (32, 32, 3).
2. Set the second layer to a ``Dense()`` layer with an output width of 128.
3. Use a ReLU activation function after the second layer.
5. Use a softmax activation function after the output layer.

```python 
# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)
```

6. Train the model for 3 epochs. You should be able to get over 50% training accuracy.

## Convolutions in Keras

```python
# Build Convolutional Neural Network in Keras Here
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
```

1. Build from the previous network.
2. Add a convolutional layer with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
3. Add a ReLU activation after the convolutional layer.

## Pooling in Keras

1. Add a 2x2 max pooling layer immediately following your convolutional layer.

```python
model.add(MaxPooling2D((2, 2)))
```

## Dropout

1. Add a dropout layer after the pooling layer. Set the dropout rate to 50%.

```python
model.add(Dropout(0.5))
```

## Test

```python
# evaluate model against the test data
with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))   
```
