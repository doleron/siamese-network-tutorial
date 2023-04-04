# siamese-network-tutorial

This is a tutorial of building a Siamese Network for checking image similarity as described in the 

## Overview

In this tutorial, we build a Siamese Network to check when 2 images are from the same person.

## The data

We use a database containing 128 face images of 16 different celebrites. The 128 are used to generate 16,256 pairs. A subset containing 1/2 of original pairs set is randomically selected to generate the training data (70%), validation data (20%) and test data (10%).

For each pair it is assigned a label value of 0 when the two images come from different persons and 1 otherwise.

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/test_data.png)

## The model

The choosen model is a 2 branch Siamese Network using standard Euclidean Distance. The sub-branches are two headless VGG 19 networks previously trained on ImageNet.

![model](https://raw.githubusercontent.com/doleron/siamese-network-tutorial/raw/main/model.png)

The Euclidean distance is define as usual:

```python
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
```

## The training

The model is trained during 20 EPOCHS using RMSProp with default configuration:

```python
EPOCHS = 20

model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=tf.keras.optimizers.RMSprop())

history = model.fit(train_ds, steps_per_epoch=(training_size // TRAINING_BATCH_SIZE),
                    validation_data=validation_ds,
                    epochs=EPOCHS)
```

The constrative loss is defined as usual:

```python
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss
```


## Results

![test data](https://raw.githubusercontent.com/doleron/siamese-network-tutorial/raw/main/test_results.png)
