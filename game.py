import pygame
import os
import random



class Dino(object):
    def __init__(self, x, floor):
        self.x = x
        self.y = floor
        self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','dino.png')),(50,50))
        self.rect = self.image.get_rect(topleft=(self.x,self.y))
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

        if self.jump and self.y>=self.floor:
            self.y = self.floor
            self.vel_y = -18
            
        if self.y >= self.floor and not self.jump:
            self.vel_y = 0
        self.rect = self.image.get_rect(topleft = (self.x,self.y))

    def render(self, window):
        window.blit(self.image,(self.x,self.y))

class Cactus(object):
    def __init__(self,x,floor,size):
        if size=='small':
            self.x = x
            self.y = floor-20
            self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','cactus.png')),(20,80))
            self.rect = self.image.get_rect(topleft = (self.x,self.y))

    def step(self):
        self.x-=5
        self.rect = self.image.get_rect(topleft = (self.x,self.y))
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
        
        self.DISPLAY.fill((100,100,100))
        pygame.display.set_caption("Dino Game")
        self.cacti = []
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','background.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        self.frame =0
        self.floor = display_height*5/8
        self.dino = Dino(self.DISPLAY_WIDTH/8,self.floor)
        self.clock = pygame.time.Clock()
    

    def nextframe(self,action):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONUP:
                action = 1

        self.frame+=1
        reward = 1
        endgame = 0


        if self.frame%60 ==0:
            offset =  random.randint(0,20)
            if random.random()>0.6:
                self.cacti.append(Cactus(self.DISPLAY_WIDTH+100+offset,self.floor,'small'))
            if random.random()>0.8:
                self.cacti.append(Cactus(self.DISPLAY_WIDTH+80+offset,self.floor,'small'))
            if random.random()>0.85:
                self.cacti.append(Cactus(self.DISPLAY_WIDTH+60+offset,self.floor,'small'))    
            if random.random()>0.89:
                self.cacti.append(Cactus(self.DISPLAY_WIDTH+60+offset,self.floor,'small'))    

  
            
        i=0
        while i < len(self.cacti):
            if self.cacti[i].step():
                self.cacti.pop(i)
                i-=1
            i+=1
        self.dino.step(action)

        

        self.render_all()

        state = pygame.surfarray.array3d(self.DISPLAY)[:400,:400,:]
        
        self.clock.tick(60)
        for i in self.cacti:
            if i.rect.colliderect(self.dino.rect):
                print("Game Over!")
                reward = 0
                self.reset()
                endgame = 1
        return state, reward, endgame


    def render_all(self):
        self.DISPLAY.blit(self.background_image,(0,0))
        for i in range(len(self.cacti)):
            self.cacti[i].render(self.DISPLAY)
        self.dino.render(self.DISPLAY)
        pygame.display.update()


    def reset(self):
        self.cacti = []
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','background.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        self.frame =0
        self.floor = self.DISPLAY_HEIGHT*5/8
        self.dino = Dino(self.DISPLAY_WIDTH/8,self.floor)





