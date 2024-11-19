import pygame as pg
import random

network_parameters = {}

class FallingGame:
    def __init__(self, width=800, height=600, fps=60, training=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.training = training
        

        pg.init()
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption("Simple Game")

        self.player_size = 50
        self.player_pos = [width // 2, height - 2 * self.player_size]

        self.enemy_size = 50
        self.enemy_pos = (random.randint(0, width - self.enemy_size), 0)

        self.SPEED = 20

        self.clock = pg.time.Clock()

        self.game_over = False
        self.points = 0

    def draw_player(self, color):
        pg.draw.rect(self.screen, color, (self.player_pos[0], self.player_pos[1], self.player_size, self.player_size))

    def draw_enemy(self, color):
        pg.draw.rect(self.screen, color, (self.enemy_pos[0], self.enemy_pos[1], self.enemy_size, self.enemy_size))

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_over = True

        keys = pg.key.get_pressed()

        if keys[pg.K_LEFT] and self.player_pos[0] > 0:
            self.player_pos[0] -= self.SPEED
        if keys[pg.K_RIGHT] and self.player_pos[0] < self.width - self.player_size:
            self.player_pos[0] += self.SPEED

    def update(self):
        if self.enemy_pos[1] >= 0 and self.enemy_pos[1] < self.height:
            self.enemy_pos = (self.enemy_pos[0], self.enemy_pos[1] + self.SPEED + 15)
        else:
            self.enemy_pos = (random.randint(0, self.width - self.enemy_size), 0)
            self.points += 10

        if (self.enemy_pos[0] in range(self.player_pos[0], self.player_pos[0] + self.player_size) or
            self.player_pos[0] in range(self.enemy_pos[0], self.enemy_pos[0] + self.enemy_size)):
            if (self.enemy_pos[1] in range(self.player_pos[1], self.player_pos[1] + self.player_size) or
                self.player_pos[1] in range(self.enemy_pos[1], self.enemy_pos[1] + self.enemy_size)):
                if self.training:
                    self.points -= 1000
                else:
                    self.game_over = True

    def mainloop(self):
        while not self.game_over:
            if self.training:
                network_parameters['enemy_x'] = self.enemy_pos[0]
                network_parameters['enemy_y'] = self.enemy_pos[1]
                network_parameters['player_x'] = self.player_pos[0]
                network_parameters['player_y'] = self.player_pos[1]
            self.points += 1

            self.handle_events()
            self.update()

            self.screen.fill((0, 0, 0))
            self.draw_player((255, 0, 0))
            self.draw_enemy((255, 255, 255))

            font = pg.font.SysFont(None, 36)
            text = font.render(f'Points: {self.points}', True, (0, 255, 0))
            self.screen.blit(text, (10, 10))

            pg.display.update()

            self.clock.tick(self.fps)

        pg.quit()


if __name__ == "__main__":
    # Manual testing
    game = FallingGame(width=800, height=600, fps=60, training=False)
    game.mainloop()

