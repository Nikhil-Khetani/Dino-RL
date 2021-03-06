# Dino-RL
Using reinforcement learning to play a copy of the Google Chrome "no internet" dinosaur game.

The aim of this project is to use q-learning in order to allow a computer to play the game and gain a higher score than a human playing the same game.

#### Update 24/01/2021
Though both the game and the training of the reinforcement learning network work as intended, further training is required. However, on a more positive note, the model can now be trained on a GPU on Google Colabatory by creating a dummy video device - though there is no visual output, the model can be trained a lot faster. Currently the game outputs rewards for training but hopefully in future versions, training can be done independent of this.

## Prerequisites
This project uses PyGame for the game itself (www.pygame.org)
```
pip install pygame
```
PyTorch is used as the main framework for reinforcement learning (pytorch.org)
```
pip install torch
```
## Built with
+ Visual Studio Code
+ Google Colabatory

## Authors
+ Nikhil Khetani

## License
