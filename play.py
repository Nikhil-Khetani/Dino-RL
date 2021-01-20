import pygame
import game
pygame.init()

my_game = game.DinoGame(800,400)
run=True
while run:
    a=0
    for event in pygame.event.get():
        if event == pygame.QUIT:
            run=False
            pygame.display.quit()
            pygame.quit()
        if event == pygame.MOUSEBUTTONUP:
            a=1
 
    my_game.nextframe(a)
    

