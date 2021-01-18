import pygame
import os



class Dino(object):
    def __init__(self, x, y, floor):
        self.x = x
        self.y = y
        self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','dino.png')),(50,50))
        self.rect = self.image.get_rect()
        self.jump=False
        self.vel_y=0
        self.floor = floor

    def step(self):
        if self.y <= self.floor:
            self.vel_y = 0
        if self.jump and self.y==self.floor:
            self.vel_y = -20
        self.y+=self.vel_y
        self.vel_y+=1

    def render(self, window):
        window.blit(self.image,(self.x,self.y))

class Cactus(object):
    def __init__(self,x,y,size):
        if size=='small':
            self.x = x
            self.y = y
            self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','cactus.png')),(40,80))

    def step(self):
        self.x-=1
        if self.x<0:
            return True
        else:
            return False

    def render(self, window):
        window.blit(self.image,(self.x,self.y))
        


class DinoGame(object):

    def __init__(self,display_width, display_height):
        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT =  display_height
        self.DISPLAY = pygame.display.set_mode((display_width,display_height))
        pygame.display.set_caption("Dino Game")
        self.cacti = []
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','background.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        self.frame =0
    

    def nextframe(self,action):
        self.frame+=1
        
        pass