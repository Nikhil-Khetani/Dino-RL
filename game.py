import pygame


class DinoGame(object):

    def __init__(self,display_width, display_height):
        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT =  display_height
        self.DISPLAY = pygame.display.set_mode((display_width,display_height))
        pygame.display.set_caption("Dino Game")
        
        self.dino_xpos