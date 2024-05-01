import pygame
import random

# Initialize Pygame
pygame.init()

# Set up window dimensions
WIN_WIDTH, WIN_HEIGHT = 800, 600
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Simple Button Example")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (0, 128, 255)
HOVER_COLOR = (255, 100, 70)

# Font setup
font = pygame.font.Font(None, 40)
text = font.render('Click Me!', True, WHITE)

# Button dimensions and properties
button_width, button_height = 200, 50
button_x, button_y = (WIN_WIDTH - button_width) // 2, (WIN_HEIGHT - button_height) // 2
button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

# Images
images = [pygame.image.load(f'image_{i}.png') for i in range(1, 11)]

# Function to generate and display images above the button
def generate_images():
    # Generate random images
    random_images = random.sample(images, 3)
    print(random_images)
    # Display images above the button
    for i, img in enumerate(random_images):
        img_rect = img.get_rect()
        img_rect.center = (WIN_WIDTH // 4 * (i + 1), WIN_HEIGHT // 4)
        img_rect.y = button_rect.top - img_rect.height - 20  # Position above the button
        win.blit(img, img_rect)
    
    # Check for victory or jackpot
    if len(set(random_images)) == 1:
        print("Jackpot!")
    elif len(set(random_images)) == 2:
        print("Victory!")

# Main loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if mouse click is inside the button
            if button_rect.collidepoint(event.pos):
                print("Button was clicked!")
                generate_images()

    # Check for mouse hover on button
    mouse_pos = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_pos):
        current_button_color = HOVER_COLOR
    else:
        current_button_color = BUTTON_COLOR

    # Drawing to the screen
    win.fill(WHITE)
    pygame.draw.rect(win, current_button_color, button_rect)
    text_rect = text.get_rect(center=button_rect.center)
    win.blit(text, text_rect)
    pygame.display.flip()

# Quit Pygame
pygame.quit()
