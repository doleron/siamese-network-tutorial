# siamese-network-tutorial

This is a tutorial of building a Siamese Network for checking image similarity as described in [Learning a Similarity Metric Discriminatively, with Application to Face Verification (Chopra, Hadsell, & LeCun, 2005)](https://ieeexplore.ieee.org/document/1467314)

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/architecture.png)

## Case study

In this tutorial, we build a Siamese Network to check when 2 images belong to the same type of flower.

## The data

We use the [Oxford Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/test_data.png)

## The model

The choosen model is a two-branch Siamese Network using standard Euclidean Distance. The sub-branches are headless MobileNetV2 networks previously trained on ImageNet.

![model](https://github.com/doleron/siamese-network-tutorial/raw/main/model.png)

The Euclidean distance is define as usual:

```python
def euclidean_distance(x, y):
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
```

## The training

The model is trained during atmost 200 EPOCHS using Adam and default configuration:

```python
EPOCHS = 200

model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=tf.keras.optimizers.Adam(),
              metrics=[Custom_Accuracy(), Custom_Precision(), Custom_Recall()])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30)

history = model.fit(train_ds, 
                    steps_per_epoch=(len(training_pairs) // TRAINING_BATCH_SIZE),
                    validation_data=validation_ds,
                    epochs=EPOCHS, 
                    callbacks=[early_stop])
```

The contrastive loss is defined as usual:

```python
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss
```
## Results

Most of time, the model achives good performance on test data.

For example:

```
714/714 [==============================] - 8s 11ms/step - loss: 0.0909 - accuracy: 0.8859 - custom__precision: 0.8875 - custom__recall: 0.8838
Validation Loss = 0.0909, Validation Accuracy = 0.8859, Validation Precision = 0.8875, Validation Recall = 0.8838
```

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/test_results.png)
