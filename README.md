# siamese-network-tutorial

This is a tutorial of building a Siamese Network for checking image similarity as described in [Learning a Similarity Metric Discriminatively, with Application to Face Verification (Chopra, Hadsell, & LeCun, 2005)](https://ieeexplore.ieee.org/document/1467314)

## Case study

In this tutorial, we build a Siamese Network to check when 2 images belong from the same person.

## The data

We use a database containing 750 face images of 15 different celebrites: katty perry, nicolas cage, chris emsworth, elon musk, angelababy, messi, meryl streep, obama, priyanka chopra, lula, idris elba, mbappe, brad pitt, megan rapinoe, and bia miranda.

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/test_data.png)

## The model

The choosen model is a 2 branch Siamese Network using standard Euclidean Distance. The sub-branches are headless MobileNetV2 networks previously trained on ImageNet.

![model](https://raw.githubusercontent.com/doleron/siamese-network-tutorial/raw/main/model.png)

The Euclidean distance is define as usual:

```python
def euclidean_distance(x, y):
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
```

## The training

The model is trained during atmost 20 EPOCHS using RMSProp with default configuration:

```python
EPOCHS = 100

model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=tf.keras.optimizers.RMSprop(),
              metrics=[Custom_Accuracy(), Custom_Precision(), Custom_Recall()])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20, restore_best_weights = True, start_from_epoch = 10)

history = model.fit(train_ds, 
                    steps_per_epoch=(len(training_pairs) // TRAINING_BATCH_SIZE),
                    validation_data=validation_ds,
                    epochs=EPOCHS, 
                    callbacks=[early_stop])
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

In the most of executions, the model achived good performance even on test data.

For example:

```
135/135 [==============================] - 2s 10ms/step - loss: 0.0895 - accuracy: 0.8963 - custom__precision: 0.9084 - custom__recall: 0.8815
Test Loss = 0.0895, Test Accuracy = 0.8963, Test Precision = 0.9084, Test Recall = 0.8815
```

![test data](https://github.com/doleron/siamese-network-tutorial/raw/main/test_results.png)
