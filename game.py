import pygame
import os
import random



class Dino(object):
    def __init__(self, x,floor):
        self.x = x
        self.y = floor-50
        self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','dino.png')),(50,50))
        self.rect = self.image.get_rect()
        self.jump=False
        self.vel_y=0
        self.floor = floor
        self.clock = pygame.time.Clock()

    def step(self,action):
        if action == 1:
            self.jump =True
        else: 
            self.jump=False

        self.y+=self.vel_y
        self.vel_y+=1

        if self.jump and self.y<=self.floor:
            self.y = self.floor
            self.vel_y = -20
            
        if self.y <= self.floor and self.jump:
            self.vel_y = 0

    def render(self, window):
        window.blit(self.image,(self.x,self.y))

class Cactus(object):
    def __init__(self,x,floor,size):
        if size=='small':
            self.x = x
            self.y = floor - 80
            self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','cactus.png')),(40,80))
            self.rect = self.image.get_rect()

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
        self.DISPLAY = pygame.display.set_mode((400,300))
        pygame.display.set_caption("Dino Game")
        self.cacti = []
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','background.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        self.frame =0
        self.floor = display_height*7/8
        self.dino = Dino(self.DISPLAY_WIDTH/8,self.floor)
    

    def nextframe(self,action):
        self.frame+=1
        reward = 1
        endgame = 0
        if self.frame%100 ==0:
            self.cacti.append(Cactus(self.DISPLAY_WIDTH,self.floor,'small'))
        i=0
        while i < len(self.cacti):
            if self.cacti[i].step():
                self.cacti.pop(i)
                i-=1
            i+=1
        self.dino.step(action)

        for i in self.cacti:
            if i.rect.colliderect(self.dino.rect):
                reward = 0
                self.reset()
                endgame = 1

        self.render_all()

        state = pygame.surfarray.array2d(self.DISPLAY)
        
        self.clock.tick(20)
        return state, reward, endgame

    def render_all(self):
        print('render')
        self.DISPLAY.blit(self.background_image,(0,0))
        self.dino.render(self.DISPLAY)
        for i in self.cacti:
            i.render(self.DISPLAY)


    def reset(self):
        self.cacti = []
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','background.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        self.frame =0
        self.floor = self.DISPLAY_HEIGHT*7/8
        self.dino = Dino(self.DISPLAY_WIDTH/8,self.floor)




a= 0
run=True
game = DinoGame(400,300)
while run:
    game.nextframe(a)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
