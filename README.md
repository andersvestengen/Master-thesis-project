# Repository for my thesis work <br /> "Improving Image‐sensor performance by developing on‐chip machine‐learning algorithm"

The following repository holds all the code used for testing, prototyping and creating the architecture and datasets for the project.

Images and graphs were created using the following sources.
  - Illustrations: https://github.com/HarisIqbal88/PlotNeuralNet
  - Images used for training and inference were taken from the Celeb-A dataset ref: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (citation provided in thesis)


## For the project an 'encoder-decoder' style fully convolutional network was created and trained in a GAN arrangement.

Generator:
<br />
<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/f36462e9-aa70-4b92-88d1-d5f48af7d8b8" width="900">
<br />
Discriminator:
<br />
<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/d0db1ec2-0adc-4144-8c2f-169fa5d35d67" width="600">

## Objective functions
Two objective functions were created for the project. The first was a local defect loss, which compared and penalized the model around the local area were the defect was located. This was necessary as the defects are small enough for the model to often score better if it learns to ignore them if there is no extra local penalty. The second objective function was a latent loss function, which was meant to make the model retain high frequency detail. The inpainter would be penalized if its latent (bottleneck) layer was different to the same network trained as an autoencoder. This helped reduce the blurring sometimes introduced when using regularization schemes like Spectral or batch normalization.

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/90ed38cb-2963-47f8-bfbc-e84f2c3bd638" width="600">
<br />


## Dataset creation
For the project I developed my own algorithm to add synthetic defect onto almost any image. The intention was to create defects similar to those in image sensors. In that way defects could be white- or black-out, they would have a random gradient to simulate non-linearity, as well as a range of random sizes and and number of defects. the results look like the image below. 

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/03988914-fa90-46d3-8fe0-accf3844310a" width="900">
<br />

## Results
The project used the SSIM and PSNR metric to give a quantitative measurement of model performance. Below is a table showing several model prototypes and their performance, averaged over 500 images. Compared to a more conventional averaging filter.

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/ae9ae889-9a1a-4fc0-9168-a4895d662c59" width="600">
<br />

## Subjective results 
Below are some subjective results of the different models compared to ground truth.

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/04ec310c-6fa2-4d80-9e44-c1aeb9ebe789" width="600">
<br />

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/05bc9414-3f37-4bbe-b700-58c2c2a8d529" width="600">
<br />

<img src="https://github.com/andersvestengen/Master-thesis-project/assets/43368671/b27ddcc6-32b3-42e5-bab4-8d2572625e8e" width="600">
<br />
