import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Signal Simulation")

# Traffic signal configuration
signal_colors = [GREEN, YELLOW, RED]  # Green, Yellow, Red
current_lane = 0

# YOLO NAS output
yolo_output = [27, 20, 51, 0]

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw the traffic signal
    signal_x = WIDTH // 2 - 20
    signal_y = HEIGHT // 2 - 100
    signal_radius = 50

    for i in range(len(signal_colors)):
        color = signal_colors[i]
        pygame.draw.circle(screen, color, (signal_x, signal_y + i * 100), signal_radius)

    # Update the current lane based on YOLO NAS output
    current_lane = yolo_output.index(max(yolo_output))

    # Highlight the current lane
    pygame.draw.circle(screen, WHITE, (signal_x, signal_y + current_lane * 100), signal_radius, width=5)

    pygame.display.flip()
    clock.tick(1)  # Control the frame rate

# Quit Pygame
pygame.quit()
sys.exit()