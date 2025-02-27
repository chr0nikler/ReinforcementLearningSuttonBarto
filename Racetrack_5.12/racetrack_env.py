import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame

class RacetrackEnv(gym.Env):
    """
    Custom racetrack environment for reinforcement learning.
    """

    metadata = {"render_modes": ["human"], "render_fps": 144}

    def __init__(self, track_number, render_mode=None):
        super(RacetrackEnv, self).__init__()

        # Define the racetrack grid (1: track, 0: boundary)

        if track_number == 0:
            self.track = np.array([
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

            # Start and finish lines
            self.start_line = [(3, 31), (4, 31), (5, 31), (6, 31), (7, 31), (8, 31)]  # Leftmost track cells
            self.finish_line = [(16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5)]  # Rightmost track cells
        elif track_number == 1:
            """ Toy Example """
            self.track = np.array([
                [0,0,1,0,0],
                [0,1,1,1,1],
                [0,1,1,1,1],
                [1,1,1,1,0],
                [1,1,0,0,0]
            ])

            self.start_line = [(0,4),(1,4)]
            self.finish_line = [(4,1),(4,2)]
        elif track_number == 2:
            self.track = np.array([
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
            ])

            self.start_line = [(x,self.track.shape[0]-1) for x in range(0,23)]
            self.finish_line = [(self.track.shape[1]-1,y) for y in range(0,9)]

        # Track size
        self.height, self.width = self.track.shape


        # Define action space (acceleration in x and y)
        self.action_space = spaces.Discrete(9)  # (dx, dy) with (-1, 0, +1) changes
        self.actions = np.array([(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]])

        # Define observation space: (x, y, velocity_x, velocity_y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -4, -4]),
            high=np.array([self.height - 1, self.width - 1, 4, 4]),
            dtype=np.int32
        )

        self.render_mode = render_mode
        self.cell_size = 20  # Size of each grid cell
        self.window_size = (
            self.cell_size * self.width,
            self.cell_size * self.height
        )
        self.screen = None
        self.clock = None

        self.state = None

    def reset(self, seed=None, render=True):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)

        # Start at a random position on the start line
        start_pos = self.start_line[np.random.choice(len(self.start_line))]
        self.state = np.array([start_pos[0], start_pos[1], 0, 0])  # (x, y, v_x, v_y)

        if self.render_mode == "human" and not not render:
            self._render_frame()
        
        return self.state, {}

    def step(self, action, noise_enabled=True, render=True,  fps=144):
        """
        Executes one step in the environment.
        Action: An integer 0-8 corresponding to acceleration change in (x, y).
        """

        done = False
        reward = -1  # Default reward per step
        x, y, v_x, v_y = self.state
        dx, dy = self.actions[action]  # Get acceleration change
        dy *= -1 # up is negative, so ensure an action of +1 in y actually goes up
    
        # Introduce 10% chance that velocity increments are ignored (dx, dy = 0)
        if noise_enabled and random.random() < 0.1:
            dx, dy = 0, 0  
    
        # Update velocity (bounded by 0 to 4)
        v_x = np.clip(v_x + dx, 0, 4)
        v_y = np.clip(-1 * v_y + dy, -4, 0)

        if v_x == 0 and v_y == 0:
            # If car stalls, reset
            new_x, new_y = random.choice(self.start_line)  # Move to a random start position
            v_x, v_y = 0, 0  # Reset velocity
            # Update state
            self.state = np.array([new_x, new_y, v_x,-1 * v_y])
            return self.state, reward, done, False, {}

    
        # Update position
        new_x = x + v_x
        new_y = y + v_y

        clip_x = np.clip(new_x, 0, self.width - 1)
        clip_y = np.clip(new_y, 0, self.height - 1)

        if (clip_x, clip_y) in self.finish_line:
            done = True
        # If off track, reset to random start position (NO PENALTY APPLIED)
        elif new_y >= self.height or new_y < 0 or new_x >= self.width or new_x < 0 or self.track[new_y, new_x] == 0:
            new_x, new_y = random.choice(self.start_line)  # Move to a random start position
            v_x, v_y = 0, 0  # Reset velocity

        # Update state
        self.state = np.array([new_x, new_y, v_x,-1 * v_y])

        if self.render_mode == "human" and not not render:
            self._render_frame(fps)

        return self.state, reward, done, False, {}

    def _render_frame(self, fps=None):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("runner")
            self.screen = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid cells
        for y in range(self.track.shape[0]):
            for x in range(self.track.shape[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                

                if self.track[y,x] == 0:
                    # Track color (grey)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)


        # Draw start line (blue)
        for x, y in self.start_line:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)

        # Draw finish line (red)
        for x, y in self.finish_line:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw agent position (green)
        x, y, v_x, v_y = self.state
        agent_rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)

        # Draw grid lines
        for x in range(0, self.window_size[0], self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.window_size[1]))
        for y in range(0, self.window_size[1], self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.window_size[0], y))

        pygame.event.pump()
        pygame.display.flip()
        if fps == None:
            self.clock.tick(self.metadata["render_fps"])  # 30 FPS for smooth updates
        else:
            self.clock.tick(fps)  # 30 FPS for smooth updates

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
