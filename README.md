# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Developed by Bardia Hassanzadeh © February 2019

Please find the checkpoint.pth in the project folder which includes the trained vgg16 model with 500 hidden units.

Here are some command line examples to run the code:
  train.py:
  	python train.py flowers/ --save_directory '' --arch vgg16 --learning_rate 0.001 --hidden_units 500  --epochs 15 --gpu
  predict.py:
  	python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 5 --gpu
=============================================================
The result of the training with 15 epochs is as the following:
  Epoch: 1/15...  Loss: 4.3444
  Validation accuracy of the network: 25 %
  Epoch: 2/15...  Loss: 3.1707
  Validation accuracy of the network: 48 %
  Epoch: 3/15...  Loss: 2.2423
  Validation accuracy of the network: 66 %
  Epoch: 4/15...  Loss: 1.7248
  Validation accuracy of the network: 76 %
  Epoch: 5/15...  Loss: 1.3993
  Validation accuracy of the network: 80 %
  Epoch: 6/15...  Loss: 1.1650
  Validation accuracy of the network: 84 %
  Epoch: 7/15...  Loss: 1.0108
  Validation accuracy of the network: 87 %
  Epoch: 8/15...  Loss: 0.8452
  Validation accuracy of the network: 87 %
  Epoch: 9/15...  Loss: 0.7629
  Validation accuracy of the network: 90 %
  Epoch: 10/15...  Loss: 0.6695
  Validation accuracy of the network: 89 %
  Epoch: 11/15...  Loss: 0.6101
  Validation accuracy of the network: 90 %
  Epoch: 12/15...  Loss: 0.5554
  Validation accuracy of the network: 91 %
  Epoch: 13/15...  Loss: 0.4780
  Validation accuracy of the network: 92 %
  Epoch: 14/15...  Loss: 0.4444
  Validation accuracy of the network: 91 %
  Epoch: 15/15...  Loss: 0.4119
  Validation accuracy of the network: 92 %
In addition, the test results show:
  Test accuracy of the network: 89 %

and, here is the command line output for the above example which correctly classifies the image:
  The most probable flower kind is 'pink primrose' with the associated probability of %99.35; and, the next 4 probable flower kinds are:
  - 'tree mallow' with the probability of %0.6280
  - 'morning glory' with the probability of %0.0121
  - 'hibiscus' with the probability of %0.0056
  - 'petunia' with the probability of %0.0028