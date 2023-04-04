# siamese-network-tutorial

This is a tutorial of building a Siamese Network for checking image similarity. 

## Overview

In this tutorial, we build a Siamese Network to check when 2 images are from the same person.

## The data

We use a database containing 128 face images of 16 different celebrites. The 128 are used to generate 16,256 pairs. A subset containing 1/4 of pairs is randomically selected to generate the training data (70%), validation data (20%) and test data (10%).

For each pair it is assigned a label value of 0 when the two images come from different persons and 1 otherwise.

![test data](https://raw.githubusercontent.com/doleron/simple-object-detector-from-scratch/main/test_data.png)

## The model

The choosen model is a 2 branch Siamese Network using standard Euclidean Distance. The sub-branches are two headless VGG 19 networks previously trained on ImageNet.

![model](https://raw.githubusercontent.com/doleron/simple-object-detector-from-scratch/main/model.png)

## The training


## Results

![test data](https://raw.githubusercontent.com/doleron/simple-object-detector-from-scratch/main/test_resuults.png)
