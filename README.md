# POKEMON
## DISPLAY
### 1.Nice images
![4562](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4562.png "4562")
![4563](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4563.png "4563")
![4686](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4686.png "4686")
![4732](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4732.png "4732")
![4849](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4849.png "4849")
![4947](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4947.png "4947")
![4999](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/4999.png "4999")
### 2.Processing gif
step1

![step1](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/gif/step1.gif "step1")

step2

![step2](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/gif/step2.gif "step2")

step3

![step3](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/gif/step3.gif "step3")
## IDE
I've been studying on a Jupyter notebook, but I find it inconvenient. I decided to use PyCharm to complete this assignment.

## INSPIRATION
After seeing this project Anime-WGAN-GP(https://github.com/luzhixing12345/Anime-WGA), I hope to do a similar project to generate animation pictures. I want to train a model of my own. So the early stage of my project followed the instructions in this tutorial.

## DATASET
https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset

I am a big Pokemon fan, I love this dataset, I want to use this dataset to train models and generate Pokemon ons.

## What I did in this Project
### 1.Download and Test
I downloaded the Anime-WGAN-GP (https://github.com/luzhixing12345/Anime-WGA). And a set of pictures were generated with the Pretrained model_WGAN_ANIME256 provided by it, which was very successful.
![WGAN](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/WGAN.png "WGAN")
### 2.Train
I put the Pokemon dataset into datasets and used the WGAN model it provided to train it, and it started well, but slowly. You can see the resulting image. 

![fail](https://github.com/Yyyoung6699/Coding3_pokemon/raw/main/Nice%20images/fail.png "fail")

But for some reason, whenever we reach Generator iteration: 6499/40000, I get a RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN  tensor and weight is a dense tensor. I looked for a solution(https://www.kaggle.com/questions-and-answers/256188), but it didn't work out. 
### 3.New model
I tried to use the new model to train the Pokemon dataset(https://github.com/HuiiJi/GAN_.py/blob/main/GAN_.py).

I cut the image size all the way down to 96 because 256 is too slow.```ruby img_size = 96```

The first time I set 1000 epochs, it didn't work very well, the second time I changed to 5000 epochs.```ruby max_epoch = 5000```

epoch 7

![7](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/7.png "7")

epoch 177

![177](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/177.png "177")

epoch 493

![493](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/493.png "493")

epoch 1011

![1011](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/1011.png "1011")

epoch 2424

![2424](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/2424.png "2424")
