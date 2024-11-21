# File: furogym.py
import pygame as pg
import random
from .game import Game
from typing import NoReturn


class FallingGame(Game):
    def __init__(self, speed=10,width=800, height=600, fps=60, training=True, agent=None, max_points=None,save_model=False,load_model=None) -> NoReturn:
        super().__init__(width, height, fps, training, agent)
        self.player_size = 50
        self.player_pos = [width // 2, height - 2 * self.player_size]
        self.enemy_size = 50
        self.enemy_pos = (random.randint(0, width - self.enemy_size), 0)
        self.SPEED = speed
        self.max_points = max_points
        self.save_model = save_model

        if load_model:
            self.agent.load_model(load_model)

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_over = True

        if self.agent and self.training:
            state = {
                'enemy_x': self.enemy_pos[0],
                'enemy_y': self.enemy_pos[1],
                'player_x': self.player_pos[0],
                'player_y': self.player_pos[1],
            }
            action = self.agent.act(state)
            if action == "LEFT" and self.player_pos[0] > 0:
                self.player_pos[0] -= self.SPEED
            elif action == "RIGHT" and self.player_pos[0] < self.width - self.player_size:
                self.player_pos[0] += self.SPEED
        else:
            keys = pg.key.get_pressed()
            if keys[pg.K_LEFT] and self.player_pos[0] > 0:
                self.player_pos[0] -= self.SPEED
            if keys[pg.K_RIGHT] and self.player_pos[0] < self.width - self.player_size:
                self.player_pos[0] += self.SPEED

    def update(self):
        if self.max_points is not None and self.points >= self.max_points:
            if self.save_model:
                self.agent.save_model("q_table.npy")
            self.game_over = True
            
        prev_state = {
            'enemy_x': self.enemy_pos[0],
            'enemy_y': self.enemy_pos[1],
            'player_x': self.player_pos[0],
            'player_y': self.player_pos[1],
        }

        # Update enemy position
        if self.enemy_pos[1] < self.height:
            self.enemy_pos = (self.enemy_pos[0], self.enemy_pos[1] + self.SPEED)
        else:
            self.enemy_pos = (random.randint(0, self.width - self.enemy_size), 0)

        # Collision detection
        collision = (
            self.player_pos[0] < self.enemy_pos[0] + self.enemy_size and
            self.player_pos[0] + self.player_size > self.enemy_pos[0] and
            self.player_pos[1] < self.enemy_pos[1] + self.enemy_size and
            self.player_pos[1] + self.player_size > self.enemy_pos[1]
        )

        # Reward calculation based on points
        reward = -1000 if collision else 10
        self.points += reward

        if collision and not self.training:
            self.game_over = True

        # Update RL agent
        if self.agent and self.training:
            action = self.agent.act(prev_state)
            if action == "LEFT" and self.player_pos[0] > 0:
                self.player_pos[0] -= self.SPEED
            elif action == "RIGHT" and self.player_pos[0] < self.width - self.player_size:
                self.player_pos[0] += self.SPEED

            # Next state after taking action
            next_state = {
                'enemy_x': self.enemy_pos[0],
                'enemy_y': self.enemy_pos[1],
                'player_x': self.player_pos[0],
                'player_y': self.player_pos[1],
            }

            # Update the agent with the transition
            self.agent.update(prev_state, action, reward, next_state)

    def draw(self):
        self.screen.fill((0, 0, 0))
        pg.draw.rect(self.screen, (255, 0, 0), (self.player_pos[0], self.player_pos[1], self.player_size, self.player_size))
        pg.draw.rect(self.screen, (255, 255, 255), (self.enemy_pos[0], self.enemy_pos[1], self.enemy_size, self.enemy_size))
        font = pg.font.SysFont(None, 36)
        text = font.render(f'Points: {self.points}', True, (0, 255, 0))
        self.screen.blit(text, (10, 10))


