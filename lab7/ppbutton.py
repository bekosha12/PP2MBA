import pygame

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

    # Check for mouse hover on button
    mouse_pos = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_pos):
        current_button_color = HOVER_COLOR
    else:
        current_button_color = BUTTON_COLOR

    # Drawing to the screen
    win.fill(WHITE)
    pygame.draw.rect(win, current_button_color, button_rect)
    text_rect = text.get_rect(center = button_rect.center)
    win.blit(text, text_rect)
    pygame.display.flip()

# Quit Pygame
pygame.quit()
