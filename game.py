import pygame



class Dino(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','dino.png')),(50,50))
        self.rect = self.image.get_rect()

    def step(self):

        pass
    
    def render(self, window):
        window.blit(self.image,(self.x,self.y))

class Cactus(object):
    def __init__(self,x,y,size):
        if size=='small':
            self.x = x
            self.y = y

    def step(self):
        self.x-=1
        


class DinoGame(object):

    def __init__(self,display_width, display_height):
        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT =  display_height
        self.DISPLAY = pygame.display.set_mode((display_width,display_height))
        pygame.display.set_caption("Dino Game")
        
        self.dino_xpos = self.DISPLAY_WIDTH/8
        self.dino_ypos = self.DISPLAY_HEIGHT*(1-1/8)
        self.dino_image = 
        self.background_image = pygame.transform.scale(pygame.image.load(os.path.join('Assets','dino.png')),(self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
    

    def nextframe(action):
        pass