import torch
import game

training_game = game.DinoGame(800,300)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")