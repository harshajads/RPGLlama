import pygame
from sprites import *
from config import *
from Level_generator import get_level
import sys
import os
from sample_generator import Sample

# Set your OpenAI API key
os.environ[
    "OPENAI_API_KEY"] = "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"


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(".venv/8bitoperator_jve.ttf", 32)
        self.running = True
        self.won = False
        self.current_level = 1
        self.max_level = 10
        self.screen_width =  WIN_WIDTH  # Assuming WIDTH is defined in config
        self.screen_height = WIN_HEIGHT  # Assuming HEIGHT is defined in config

        self.character_spritesheet = Spritesheet("img/character.png")
        self.terrain_spritesheet = Spritesheet("img/terrain.png")
        self.enemy_spritesheet = Spritesheet("img/enemy.png")
        self.attack_spritesheet = Spritesheet("img/attack.png")
        self.intro_background = pygame.image.load("img/introbackground.png")
        self.go_background = pygame.image.load("img/gameover.png")

        # Loading screen
        self.loading = False

    def createTilemap(self, tilemap):
        # Clear existing sprites
        for sprite in self.all_sprites:
            sprite.kill()

        # Create new tilemap
        for i, row in enumerate(tilemap):
            for j, column in enumerate(row):
                Ground(self, j, i)
                if column == "B":
                    Block(self, j, i)
                if column == "E":
                    Enemy(self, j, i)
                if column == "P":
                    #print("Player found from CREATE TILE MAP")
                    self.player = Player(self, j, i)

    def show_loading_screen(self):
        """Display a loading screen while generating levels"""
        self.screen.fill(BLACK)
        loading_text = self.font.render(f"Generating Level {self.current_level}...", True, WHITE)
        self.screen.blit(loading_text, (WIN_WIDTH // 2 - 150, WIN_HEIGHT // 2 - 16))
        pygame.display.update()

    def new(self):
        self.playing = True
        self.won = False

        self.all_sprites = pygame.sprite.LayeredUpdates()
        self.blocks = pygame.sprite.LayeredUpdates()
        self.enemies = pygame.sprite.LayeredUpdates()
        self.attacks = pygame.sprite.LayeredUpdates()

        # Show loading screen while generating level
        self.show_loading_screen()

        # Generate the map for the current level
        prompt, tilemap = get_level(self.current_level) #Map is already filtered
        self.createTilemap(tilemap)
        s = Sample(prompt, tilemap)
        s.create_sample()
        s.save_sample()
        # batch -> sample will be saved on dataset.json and used later
        # 'realtime' -> sample will be saved on sample_instance.json and will be used realtime to train.
        #print(tilemap)
        #print(prompt)
        print("New map created.")

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.playing = False
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.player.facing == "up":
                        Attack(self, self.player.rect.x, self.player.rect.y - TILESIZE)
                    if self.player.facing == "down":
                        Attack(self, self.player.rect.x, self.player.rect.y + TILESIZE)
                    if self.player.facing == "right":
                        Attack(self, self.player.rect.x + TILESIZE, self.player.rect.y)
                    if self.player.facing == "left":
                        Attack(self, self.player.rect.x - TILESIZE, self.player.rect.y)

    def update(self):
        self.all_sprites.update()

    def draw(self):
        self.screen.fill(BLACK)

        # Update camera to follow player (center player on screen)
        self.camera_offset_x = self.player.rect.x - (self.screen_width // 2) + (TILESIZE // 2)
        self.camera_offset_y = self.player.rect.y - (self.screen_height // 2) + (TILESIZE // 2)

        # Draw all sprites with camera offset
        for sprite in self.all_sprites:
            # Only draw sprites that are visible on screen
            draw_x = sprite.rect.x - self.camera_offset_x
            draw_y = sprite.rect.y - self.camera_offset_y

            # Check if the sprite is within the screen bounds
            if (-sprite.rect.width <= draw_x <= self.screen_width and
                    -sprite.rect.height <= draw_y <= self.screen_height):
                self.screen.blit(sprite.image, (draw_x, draw_y))

        # Display current level
        level_text = self.font.render(f"Level: {self.current_level}", True, WHITE)
        self.screen.blit(level_text, (10, 10))

        # Display enemies remaining
        enemies_text = self.font.render(f"Enemies: {len(self.enemies.sprites())}", True, WHITE)
        self.screen.blit(enemies_text, (10, 50))

        self.clock.tick(FPS)
        pygame.display.update()

    def main(self):
        while self.playing:
            self.events()
            self.update()
            self.draw()
            if len(self.enemies.sprites()) == 0:
                self.won = True
                self.playing = False
        if self.won == False:
            self.playing = False

    def game_over(self):
        text = self.font.render("GaMe Over", True, RED)
        text_rect = text.get_rect(center=(WIN_WIDTH / 2, WIN_HEIGHT / 2))

        restart_button = Button(
            10, WIN_HEIGHT - 135, 120, 125, WHITE, BLACK, "Restart", 32
        )

        for sprite in self.all_sprites:
            sprite.kill()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()

            if restart_button.is_pressed(mouse_pos, mouse_pressed):
                self.current_level = 1  # Reset to level 1
                self.won = False
                self.new()
                break

            self.screen.blit(self.go_background, (0, 0))
            self.screen.blit(text, text_rect)
            self.screen.blit(restart_button.image, restart_button.rect)
            self.clock.tick(FPS)
            pygame.display.update()

    def intro_screen(self):
        intro = True

        title = self.font.render("Spud-nik : SOLO", True, BLUE)
        title_rect = title.get_rect(x=WIN_WIDTH / 2 - 100, y=100)

        play_button = Button(
            WIN_WIDTH / 2 - 50, 200, 100, 100, WHITE, BLACK, "Play", 32
        )

        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    intro = False
                    self.running = False

            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()

            if play_button.is_pressed(mouse_pos, mouse_pressed):
                intro = False

            self.screen.blit(self.intro_background, (0, 0))
            self.screen.blit(title, title_rect)
            self.screen.blit(play_button.image, play_button.rect)
            self.clock.tick(FPS)
            pygame.display.update()

    def game_won(self):
        if self.current_level < self.max_level:
            # Level completed, advance to next level
            self.current_level += 1
            self.new()
            return

        # Game completed (all levels finished)
        text = self.font.render("YOU WON THE GAME!", True, BLUE)
        text_rect = text.get_rect(center=(WIN_WIDTH / 2, WIN_HEIGHT / 2))

        restart_button = Button(
            10, WIN_HEIGHT - 135, 120, 125, WHITE, BLACK, "Restart", 32
        )

        for sprite in self.all_sprites:
            sprite.kill()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()

            if restart_button.is_pressed(mouse_pos, mouse_pressed):
                self.current_level = 1  # Reset to level 1
                self.new()
                break

            self.screen.blit(self.intro_background, (0, 0))
            self.screen.blit(text, text_rect)
            self.screen.blit(restart_button.image, restart_button.rect)
            self.clock.tick(FPS)
            pygame.display.update()


"""g = Game()
#g.intro_screen()
g.new()

while g.running:
    g.main()
    if g.won == True:
        g.game_won()
    else:
        g.game_over()

pygame.quit()
sys.exit()"""