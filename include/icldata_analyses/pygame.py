import pygame
import random

# Initialize pygame
pygame.init()

# Define screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Point Movement in Search Space")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Define point properties
point_x = WIDTH // 2
point_y = HEIGHT // 2
point_radius = 5
speed = 5

# Define movement directions
directions = [(speed, 0), (-speed, 0), (0, speed), (0, -speed)]

# Main loop
running = True
while running:
    pygame.time.delay(100)  # Control the speed of movement

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the point randomly
    dx, dy = random.choice(directions)
    point_x += dx
    point_y += dy

    # Ensure the point stays within the search space
    point_x = max(point_radius, min(WIDTH - point_radius, point_x))
    point_y = max(point_radius, min(HEIGHT - point_radius, point_y))

    # Drawing
    screen.fill(WHITE)  # Clear screen
    pygame.draw.circle(screen, RED, (point_x, point_y), point_radius)
    pygame.display.update()

# Quit pygame
pygame.quit()
