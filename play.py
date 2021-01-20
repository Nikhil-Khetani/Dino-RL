import pygame
import game

pygame.init()
a= 0
run=True
current_game = game.DinoGame(800,300)
while run:
    current_game.nextframe(a)