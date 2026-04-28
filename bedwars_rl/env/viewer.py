"""
Визуализатор для Bedwars с возможностью:
- Наблюдения за игрой в реальном времени
- Свободного перемещения камеры
- Игры против агента человеком
"""
import numpy as np
import pygame
from pygame.locals import *
from typing import Optional, Dict, List, Tuple
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.bedwars_env import BedwarsEnv, Block
from config import ENV_CONFIG


class BedwarsViewer:
    """
    3D визуализатор для Bedwars
    Поддерживает:
    - Изометрическую проекцию
    - Свободное перемещение камеры
    - Управление от человека
    - Отображение статистики
    """
    
    def __init__(
        self,
        env: BedwarsEnv,
        screen_size: Tuple[int, int] = (1200, 800),
        block_size: int = 20,
    ):
        self.env = env
        self.screen_size = screen_size
        self.block_size = block_size
        
        # Инициализация Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Bedwars RL Viewer")
        self.clock = pygame.time.Clock()
        
        # Камера
        self.camera_pos = np.array([self.env.map_size // 2, 30, self.env.map_size // 2], dtype=np.float32)
        self.camera_rotation = np.array([60, 0])  # pitch, yaw
        self.camera_speed = 1.0
        self.rotation_speed = 3.0
        
        # Цвета блоков
        self.block_colors = {
            Block.AIR: (0, 0, 0, 0),
            Block.WOOL: (255, 255, 255),
            Block.WOOD: (139, 90, 43),
            Block.STONE: (128, 128, 128),
            Block.BEDROCK: (64, 64, 64),
            Block.BED: (255, 0, 0),
            Block.OBSIDIAN: (32, 0, 64),
            Block.GLASS: (200, 220, 255, 128),
        }
        
        # Цвета команд
        self.team_colors = [
            (255, 0, 0),      # Красный
            (0, 0, 255),      # Синий
            (0, 255, 0),      # Зеленый
            (255, 255, 0),    # Желтый
        ]
        
        # Шрифты
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Режимы
        self.show_grid = True
        self.pause = False
        self.follow_player = True
        self.selected_player = 0
        
        # Для управления человеком
        self.human_controlled = False
        self.human_player_id = 0
        
    def isometric_project(self, point: np.ndarray) -> Tuple[int, int]:
        """Проекция 3D точки в 2D изометрическую"""
        # Относительно камеры
        rel_point = point - self.camera_pos
        
        # Вращение вокруг оси Y (yaw)
        yaw_rad = np.radians(self.camera_rotation[1])
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        x = rel_point[0] * cos_yaw - rel_point[2] * sin_yaw
        z = rel_point[0] * sin_yaw + rel_point[2] * cos_yaw
        y = rel_point[1]
        
        # Изометрическая проекция
        iso_x = (x - z) * self.block_size
        iso_y = (x + z) * self.block_size * 0.5 - y * self.block_size * 1.2
        
        # Центрирование на экране
        screen_x = self.screen_size[0] // 2 + iso_x
        screen_y = self.screen_size[1] // 3 + iso_y
        
        return int(screen_x), int(screen_y)
    
    def draw_block(self, x: int, y: int, z: int, color: Tuple[int, int, int]):
        """Рисование блока"""
        # Вершины блока в изометрии
        points_3d = [
            np.array([x, y, z]),
            np.array([x + 1, y, z]),
            np.array([x + 1, y, z + 1]),
            np.array([x, y, z + 1]),
            np.array([x, y + 1, z]),
            np.array([x + 1, y + 1, z]),
            np.array([x + 1, y + 1, z + 1]),
            np.array([x, y + 1, z + 1]),
        ]
        
        points_2d = [self.isometric_project(p) for p in points_3d]
        
        # Рисуем грани (видимые)
        # Верхняя грань
        if y > self.camera_pos[1] - 10:  # Только если не слишком низко
            pygame.draw.polygon(self.screen, self._darken_color(color, 0.8), [
                points_2d[4], points_2d[5], points_2d[6], points_2d[7]
            ])
        
        # Передняя грань
        pygame.draw.polygon(self.screen, color, [
            points_2d[0], points_2d[1], points_2d[5], points_2d[4]
        ])
        
        # Правая грань
        pygame.draw.polygon(self.screen, self._darken_color(color, 0.6), [
            points_2d[1], points_2d[2], points_2d[6], points_2d[5]
        ])
        
        # Контур
        pygame.draw.lines(self.screen, (0, 0, 0), True, [
            points_2d[0], points_2d[1], points_2d[2], points_2d[3]
        ], 1)
        
    def _darken_color(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Затемнение цвета"""
        return tuple(int(c * factor) for c in color[:3])
    
    def draw_players(self):
        """Рисование игроков"""
        for i, player in enumerate(self.env.players):
            if not player.is_alive:
                continue
                
            # Позиция игрока
            pos = player.position
            screen_pos = self.isometric_project(pos)
            
            # Цвет команды
            team_color = self.team_colors[player.team_id % len(self.team_colors)]
            
            # Рисуем игрока как круг
            pygame.draw.circle(self.screen, team_color, screen_pos, 8)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 8, 2)
            
            # Полоска здоровья
            health_width = 20
            health_height = 4
            health_x = screen_pos[0] - health_width // 2
            health_y = screen_pos[1] - 15
            
            # Фон полоски
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (health_x, health_y, health_width, health_height))
            
            # Здоровье
            health_percent = player.health / player.max_health
            health_color = (0, 255, 0) if health_percent > 0.5 else \
                          (255, 255, 0) if health_percent > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color,
                           (health_x, health_y, int(health_width * health_percent), health_height))
            
            # Индикатор выбранного игрока
            if i == self.selected_player:
                pygame.draw.circle(self.screen, (255, 255, 0), screen_pos, 12, 2)
                
            # Имя игрока
            name_text = f"P{i}"
            name_surface = self.small_font.render(name_text, True, (255, 255, 255))
            name_rect = name_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 25))
            self.screen.blit(name_surface, name_rect)
    
    def draw_beds(self):
        """Рисование кроватей"""
        for bed in self.env.beds:
            if not bed.exists:
                continue
                
            pos = bed.position
            screen_pos = self.isometric_project(pos)
            
            # Рисуем кровать как красный прямоугольник
            bed_width = 16
            bed_height = 10
            bed_rect = pygame.Rect(
                screen_pos[0] - bed_width // 2,
                screen_pos[1] - bed_height // 2,
                bed_width,
                bed_height
            )
            pygame.draw.rect(self.screen, (255, 0, 0), bed_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), bed_rect, 2)
            
            # Индикатор здоровья кровати
            health_text = f"{bed.health}%"
            health_surface = self.small_font.render(health_text, True, (255, 255, 255))
            health_rect = health_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 15))
            self.screen.blit(health_surface, health_rect)
    
    def draw_ui(self):
        """Рисование пользовательского интерфейса"""
        # Фон для UI
        ui_bg = pygame.Surface((300, 200), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 128))
        self.screen.blit(ui_bg, (10, 10))
        
        # Информация об эпизоде
        y_offset = 20
        texts = [
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            f"Episode Over: {self.env.episode_over}",
            f"Selected Player: {self.selected_player}",
            f"Camera: ({self.camera_pos[0]:.1f}, {self.camera_pos[1]:.1f}, {self.camera_pos[2]:.1f})",
            f"Rotation: ({self.camera_rotation[0]:.0f}, {self.camera_rotation[1]:.0f})",
        ]
        
        for text in texts:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 25
            
        # Информация об игроках
        y_offset += 20
        for i, player in enumerate(self.env.players):
            status = "ALIVE" if player.is_alive else "DEAD"
            player_text = f"P{i} (Team {player.team_id}): {status} | HP: {player.health} | K: {player.kills}"
            color = self.team_colors[player.team_id % len(self.team_colors)]
            surface = self.small_font.render(player_text, True, color)
            self.screen.blit(surface, (20, y_offset))
            y_offset += 20
            
        # Информация о кроватях
        y_offset += 20
        for i, bed in enumerate(self.env.beds):
            status = "EXISTS" if bed.exists else "DESTROYED"
            bed_text = f"Bed {i} (Team {bed.team_id}): {status}"
            color = (255, 0, 0) if bed.exists else (100, 100, 100)
            surface = self.small_font.render(bed_text, True, color)
            self.screen.blit(surface, (20, y_offset))
            y_offset += 20
            
        # Управление
        y_offset += 20
        controls = [
            "Controls:",
            "WASD - Move camera",
            "Q/E - Up/Down",
            "Arrow keys - Rotate",
            "1-9 - Select player",
            "F - Follow player",
            "H - Human control",
            "P - Pause",
            "ESC - Quit",
        ]
        
        for text in controls:
            surface = self.small_font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 18
    
    def handle_input(self, action_dict: Optional[Dict] = None) -> Optional[Dict]:
        """Обработка ввода пользователя"""
        keys = pygame.key.get_pressed()
        
        # Движение камеры
        move_speed = self.camera_speed
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            move_speed *= 2
            
        # WASD для движения камеры
        if keys[K_w]:
            self.camera_pos[0] -= move_speed
        if keys[K_s]:
            self.camera_pos[0] += move_speed
        if keys[K_a]:
            self.camera_pos[2] += move_speed
        if keys[K_d]:
            self.camera_pos[2] -= move_speed
            
        # Q/E для вверх/вниз
        if keys[K_q]:
            self.camera_pos[1] += move_speed
        if keys[K_e]:
            self.camera_pos[1] -= move_speed
            
        # Стрелки для вращения
        if keys[K_LEFT]:
            self.camera_rotation[1] -= self.rotation_speed
        if keys[K_RIGHT]:
            self.camera_rotation[1] += self.rotation_speed
        if keys[K_UP]:
            self.camera_rotation[0] = min(90, self.camera_rotation[0] + self.rotation_speed)
        if keys[K_DOWN]:
            self.camera_rotation[0] = max(0, self.camera_rotation[0] - self.rotation_speed)
            
        # Выбор игрока
        for i in range(1, 10):
            if keys[getattr(pygame, f'K_{i}')]:
                self.selected_player = i - 1
                if self.selected_player < len(self.env.players):
                    break
                    
        # Следование за игроком
        if keys[K_f]:
            self.follow_player = not self.follow_player
            
        # Человеческое управление
        if keys[K_h]:
            self.human_controlled = not self.human_controlled
            
        # Пауза
        if keys[K_p]:
            self.pause = not self.pause
            
        # Выход
        if keys[K_ESCAPE]:
            return "quit"
            
        # Если человек управляет, возвращаем действие
        if self.human_controlled and not self.pause:
            return self._get_human_action()
            
        return None
    
    def _get_human_action(self) -> Dict:
        """Получение действия от человека"""
        keys = pygame.key.get_pressed()
        
        # Движение
        movement = 0  # 0 = нет, 1 = вперед, 2 = назад, 3 = влево, 4 = вправо
        if keys[K_UP]:
            movement = 1
        elif keys[K_DOWN]:
            movement = 2
        elif keys[K_LEFT]:
            movement = 3
        elif keys[K_RIGHT]:
            movement = 4
            
        # Прыжок
        jump = 1 if keys[K_SPACE] else 0
        
        # Спринт
        sprint = 1 if keys[K_LSHIFT] or keys[K_RSHIFT] else 0
        
        # Атака
        attack = 1 if keys[K_j] or keys[K_z] else 0
        
        # Строительство
        place_block = 1 if keys[K_k] or keys[K_x] else 0
        
        # Разрушение
        break_block = 1 if keys[K_l] or keys[K_c] else 0
        
        # Выбор предмета
        inventory = 0
        for i in range(1, 10):
            if keys[getattr(pygame, f'K_{i}')]:
                inventory = i - 1
                break
                
        # Покупка
        buy_menu = 0
        
        # Взгляд
        look = 0
        
        return {
            "movement": movement,
            "jump": jump,
            "sprint": sprint,
            "attack": attack,
            "place_block": place_block,
            "break_block": break_block,
            "look": look,
            "inventory": inventory,
            "buy_menu": buy_menu,
        }
    
    def run(self, agent=None, fps: int = 30):
        """Запуск визуализатора"""
        running = True
        episode_reward = 0
        
        while running:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    
            # Обновление камеры если следование включено
            if self.follow_player and self.selected_player < len(self.env.players):
                player = self.env.players[self.selected_player]
                if player.is_alive:
                    self.camera_pos[0] = player.position[0]
                    self.camera_pos[1] = player.position[1] + 15
                    self.camera_pos[2] = player.position[2]
                    
            # Получение действия
            human_action = self.handle_input()
            
            if human_action == "quit":
                running = False
                continue
                
            if not self.pause:
                if self.human_controlled and human_action:
                    action = human_action
                elif agent:
                    obs = self.env._get_observation(self.selected_player)
                    action, _ = agent.select_action(obs, deterministic=True)
                else:
                    action = self.env._get_bot_action(self.env.players[self.selected_player])
                    
                # Шаг среды
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    print(f"Episode finished! Reward: {episode_reward:.2f}")
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    
            # Отрисовка
            self.screen.fill((30, 30, 50))  # Темно-синий фон
            
            # Рисуем блоки
            if self.show_grid:
                # Оптимизация: рисуем только видимые блоки
                render_distance = 20
                cx, cy, cz = int(self.camera_pos[0]), int(self.camera_pos[1]), int(self.camera_pos[2])
                
                for x in range(max(0, cx - render_distance), 
                              min(self.env.grid.shape[0], cx + render_distance)):
                    for y in range(max(0, cy - render_distance), 
                                  min(self.env.grid.shape[1], cy + 10)):
                        for z in range(max(0, cz - render_distance), 
                                      min(self.env.grid.shape[2], cz + render_distance)):
                            block_type = self.env.grid[x, y, z]
                            if block_type != Block.AIR:
                                color = self.block_colors.get(block_type, (255, 0, 255))
                                self.draw_block(x, y, z, color)
            
            # Рисуем игроков и кровати
            self.draw_players()
            self.draw_beds()
            
            # Рисуем UI
            self.draw_ui()
            
            # Обновление экрана
            pygame.display.flip()
            self.clock.tick(fps)
            
        pygame.quit()


def main():
    """Главная функция для запуска визуализатора"""
    # Создание среды
    env = BedwarsEnv(
        num_players=2,
        num_teams=2,
        render_mode="human"
    )
    
    # Создание визуализатора
    viewer = BedwarsViewer(env)
    
    # Запуск (без агента, можно играть человеком)
    print("Starting Bedwars Viewer!")
    print("Press H to toggle human control")
    print("Press F to toggle follow mode")
    print("Press ESC to quit")
    
    viewer.run(agent=None, fps=30)


if __name__ == "__main__":
    main()
