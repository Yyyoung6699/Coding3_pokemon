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
### 4.Setting
I cut the image size all the way down to 96 because 256 is too slow.```ruby img_size = 96```

The first time I set 1000 epochs, it didn't work very well, the second time I changed to 5000 epochs.```ruby max_epoch = 5000```

This is not my code(https://github.com/HuiiJi/GAN_.py/blob/main/GAN_.py), I changed some Settings on the basis.
```ruby
    for iteration, (img, _) in tqdm(enumerate(dataloader)): # Iterate over the dataset
        real_img = img.to(device)

        if iteration % 1 == 0:

            optimize_d.zero_grad() # Clear discriminator gradients

            output = d_net(real_img) # Real data input to discriminator
            d_real_loss = criterions(output, true_labels) # Discriminator loss for real images with label 1
            fake_image = g_net(noises.detach()).detach() # Generate fake images using the generator
            output = d_net(fake_image) # Fake data input to discriminator
            d_fake_loss = criterions(output, fake_labels) # Discriminator loss for fake images with label 0
            d_loss = (d_fake_loss + d_real_loss) / 2 # Combined discriminator loss

            d_loss.backward() # Backpropagate discriminator loss
            optimize_d.step() # Update discriminator optimizer

        if iteration % 1 == 0:
            optimize_g.zero_grad() # Clear generator gradients
            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1)) # Copy noise data for generator input
            fake_image = g_net(noises) # Generate fake images
            output = d_net(fake_image) # Fake data input to discriminator
            g_loss = criterions(output, true_labels) # Generator loss, aiming to generate images classified as real

            g_loss.backward() # Backpropagate generator loss
            optimize_g.step() # Update generator optimizer
```
I set Train discriminator 5 times, then train generator 1 time. ```ruby if iteration % 5 == 0:```
### 5.Key epochs
epoch 7

![7](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/7.png "7")

epoch 177

![177](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/177.png "177")

epoch 493

![493](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/493.png "493")

epoch 1011

![1011](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/1011.png "1011")

epoch 3024

![3024](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/3024.png "3024")

epoch 4385

![4385](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/4385.png "4385")

epoch 4826

![4826](https://github.com/Yyyoung6699/Coding3_pokemon/blob/main/Processing/4826.png "4826")
### 6.Other code
Generate Gif
```ruby
from PIL import Image
import glob

# GIF path
image_path = r'D:\work\pythonProject1\FR\*.png'
output_gif_path = r'D:\work\pythonProject1\animation.gif'

# image list
image_files = glob.glob(image_path)
image_files.sort()  # 确保图像按照顺序加载

image_sequence = []
for image_file in image_files:
    image = Image.open(image_file)
    image_sequence.append(image)

num_images = len(image_files)
print(f"转换为 GIF 的图像数量: {num_images}")
# save GIF
image_sequence[0].save(output_gif_path, save_all=True, append_images=image_sequence[1:], optimize=False, duration=1, loop=0)
print("GIF 动画生成成功！")
```

images processing

Do some optimization on the generated images.
```ruby
import cv2
import numpy as np

image_path = r"C:\Users\zhand\Desktop\Nice images\4562.png"

image = cv2.imread(image_path)

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 锐化滤波器
sharp_image = cv2.filter2D(image, -1, kernel)

cv2.imshow("Enhanced Image", sharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## END
This project has made me advance a lot in the field of machine learning. What I'm not happy with is that the Pokemon dataset is too small at 841 images, so the resulting images aren't as good as the animated avatars. Gans require a large number of data sets. I put the model here(https://github.com/Yyyoung6699/Coding3_pokemon/tree/main/pokemon). I hope that in the future my project will not only generate images, but even provide ideas for Pokemon design.
