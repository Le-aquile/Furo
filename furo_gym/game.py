import pygame as pg
from typing import NoReturn
class Game:
    def __init__(self, width=800, height=600, fps=60, training=True, agent=None) -> NoReturn:
        self.width = width
        self.height = height
        self.fps = fps
        self.training = training
        self.agent = agent

        pg.init()
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption("Reinforcement Learning Game")

        self.clock = pg.time.Clock()
        self.game_over = False
        self.points = 0

    def handle_events(self):
        raise NotImplementedError("handle_events method must be implemented in subclasses.")

    def update(self):
        raise NotImplementedError("update method must be implemented in subclasses.")

    def draw(self):
        raise NotImplementedError("draw method must be implemented in subclasses.")

    def mainloop(self):
        while not self.game_over:
            self.handle_events()
            self.update()
            self.draw()
            pg.display.update()
            self.clock.tick(self.fps)
        pg.quit()

