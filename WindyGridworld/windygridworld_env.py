"""

Windy Gridworld Exercise 6.9


"""

import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces

class WindyGridworldEnv(gym.Env):
    """
    Windy Gridworld environment with Pygame visualization.
    """

    def __init__(self, render_mode=None, cell_size=60):
        super().__init__()

        # Gridworld dimensions
        self.grid_height = 7
        self.grid_width = 10

        # Start and goal states
        self.start_state = (3, 0)  # (row, col)
        self.goal_state = (3, 7)   # (row, col)

        # Wind strength (number of cells pushed upward)
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

        # Action space: 0 = left, 1 = down, 2 = right, 3 = up
        self.action_space = spaces.Discrete(4)

        # Observation space: (row, col) position of agent
        self.observation_space = spaces.MultiDiscrete([self.grid_height, self.grid_width])

        # Rendering variables
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.window_size = (self.grid_width * self.cell_size, self.grid_height * self.cell_size)
        self.window = None

        # Reset environment
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset agent to the start position.
        """
        super().reset(seed=seed)
        self.agent_pos = list(self.start_state)

        if self.render_mode == 'human':
            self.render()

        return tuple(self.agent_pos), {}

    def step(self, action, fps=None):
        """
        Apply an action and move the agent in the grid.
        """
        row, col = self.agent_pos

        # Apply wind effect first (push upward based on wind strength in that column)
        row = max(row - self.wind[col], 0)

        # Apply action (left, down, right, up)
        if action == 0:  # Left
            col = max(col - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.grid_height - 1)
        elif action == 2:  # Right
            col = min(col + 1, self.grid_width - 1)
        elif action == 3:  # Up
            row = max(row - 1, 0)


        # Update agent position
        self.agent_pos = [row, col]

        # Reward system: -1 per step to encourage shortest path
        reward = -1

        # Check if goal is reached
        done = (self.agent_pos == list(self.goal_state))

        if self.render_mode == 'human':
            self.render(fps)

        return tuple(self.agent_pos), reward, done, False, {}

    def render(self, fps=None):
        """
        Render the environment using Pygame.
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Windy Gridworld")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))  # White background
        font = pygame.font.Font(None, 24)

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # Color grid cells based on wind strength (darker = stronger wind)
                wind_strength = self.wind[col]
                color_intensity = 255 - (wind_strength * 50)  # Darker for stronger wind
                color = (color_intensity, color_intensity, 255)  # Blue shades

                # Draw cell
                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )

                # Draw gridlines
                pygame.draw.rect(
                    self.window,
                    (0, 0, 0),
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size),
                    1  # Line thickness
                )

                # Display wind strength as a number
                if wind_strength > 0:
                    text = font.render(str(wind_strength), True, (0, 0, 0))
                    self.window.blit(text, (col * self.cell_size + 20, row * self.cell_size + 15))

        # Draw the agent
        pygame.draw.circle(
            self.window,
            (255, 0, 0),  # Red for agent
            (self.agent_pos[1] * self.cell_size + self.cell_size // 2, self.agent_pos[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )

        # Draw the goal state
        pygame.draw.rect(
            self.window,
            (0, 255, 0),  # Green for goal
            pygame.Rect(self.goal_state[1] * self.cell_size, self.goal_state[0] * self.cell_size, self.cell_size, self.cell_size)
        )

        pygame.event.pump()
        pygame.display.flip()
        if fps == None:
            self.clock.tick(300)  # 30 FPS for smooth updates
        else:
            self.clock.tick(fps)  # 30 FPS for smooth updates

    def close(self):
        """
        Close Pygame window.
        """
        if self.window is not None:
            pygame.quit()
            self.window = None




