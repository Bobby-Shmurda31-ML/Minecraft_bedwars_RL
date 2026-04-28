"""
Базовый класс среды для Bedwars
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import copy

from config import ENV_CONFIG, GAME_CONFIG, ITEM_PRICES, REWARD_CONFIG


class Player:
    """Класс игрока"""
    
    def __init__(self, player_id: int, spawn_pos: np.ndarray, team_id: int):
        self.player_id = player_id
        self.team_id = team_id
        self.position = spawn_pos.copy().astype(np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.health = GAME_CONFIG["player_health"]
        self.max_health = GAME_CONFIG["player_health"]
        self.is_alive = True
        self.inventory = {
            "wool": 16,
            "wood": 0,
            "stone_sword": 0,
            "iron_sword": 0,
            "diamond_sword": 0,
            "chainmail_armor": 0,
            "iron_armor": 0,
            "diamond_armor": 0,
            "golden_apple": 0,
            "fireball": 0,
            "tnt": 0,
            "obsidian": 0,
            "end_pearl": 0,
            "shield": 0,
        }
        self.resources = {
            "iron": 0,
            "gold": 0,
            "emerald": 0,
            "diamond": 0,
        }
        self.selected_item = 0  # Индекс выбранного предмета
        self.on_ground = False
        self.is_sprinting = False
        self.kills = 0
        self.deaths = 0
        
    def reset(self, spawn_pos: np.ndarray):
        """Сброс игрока"""
        self.position = spawn_pos.copy()
        self.velocity = np.zeros(3)
        self.health = GAME_CONFIG["player_health"]
        self.is_alive = True
        self.on_ground = False
        self.is_sprinting = False
        
    def get_inventory_array(self) -> np.ndarray:
        """Получить инвентарь как массив"""
        return np.array([
            self.inventory.get("wool", 0),
            self.inventory.get("wood", 0),
            self.inventory.get("stone_sword", 0),
            self.inventory.get("iron_sword", 0),
            self.inventory.get("diamond_sword", 0),
            self.inventory.get("chainmail_armor", 0),
            self.inventory.get("iron_armor", 0),
            self.inventory.get("diamond_armor", 0),
            self.inventory.get("golden_apple", 0),
            self.inventory.get("fireball", 0),
        ], dtype=np.float32)
    
    def get_resources_array(self) -> np.ndarray:
        """Получить ресурсы как массив"""
        return np.array([
            self.resources["iron"],
            self.resources["gold"],
            self.resources["emerald"],
            self.resources["diamond"],
        ], dtype=np.float32)


class Bed:
    """Класс кровати"""
    
    def __init__(self, position: np.ndarray, team_id: int):
        self.position = position.copy()
        self.team_id = team_id
        self.health = GAME_CONFIG["bed_health"]
        self.exists = True
        self.protection_blocks = []  # Блоки защиты вокруг кровати
        
    def damage(self, amount: int):
        """Нанести урон кровати"""
        if self.exists:
            self.health -= amount
            if self.health <= 0:
                self.exists = False
                return True  # Кровать разрушена
        return False


class Block:
    """Класс блока"""
    
    AIR = 0
    WOOL = 1
    WOOD = 2
    STONE = 3
    BEDROCK = 4
    BED = 5
    OBSIDIAN = 6
    GLASS = 7
    
    def __init__(self, block_type: int = 0):
        self.type = block_type
        self.solid = block_type != self.AIR and block_type != self.BED


class BedwarsEnv(gym.Env):
    """
    Базовая среда для Bedwars
    
    Поддерживает:
    - 3D пространство
    - Физику (гравитация, прыжки, движение)
    - Строительство и разрушение блоков
    - Систему ресурсов и покупок
    - Боевую систему
    - Разрушение кроватей
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        num_players: int = 2,
        num_teams: int = 2,
        map_size: int = None,
        render_mode: Optional[str] = None,
        curriculum_stage: int = 0,
    ):
        super().__init__()
        
        self.num_players = num_players
        self.num_teams = num_teams
        self.map_size = map_size or ENV_CONFIG["map_size"]
        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        
        # Конфигурация
        self.gravity = ENV_CONFIG["gravity"]
        self.jump_force = ENV_CONFIG["jump_force"]
        self.move_speed = ENV_CONFIG["move_speed"]
        self.sprint_multiplier = ENV_CONFIG["sprint_multiplier"]
        self.max_steps = ENV_CONFIG["max_steps"]
        
        # Пространство наблюдений
        from config import TOTAL_OBS_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_OBS_DIM,),
            dtype=np.float32
        )
        
        # Пространство действий (дискретное для каждого типа действия)
        from config import ACTION_SPACE
        self.action_space = spaces.Dict({
            key: spaces.Discrete(value)
            for key, value in ACTION_SPACE.items()
        })
        
        # Состояние среды
        self.grid = None  # 3D сетка блоков
        self.players: List[Player] = []
        self.beds: List[Bed] = []
        self.resource_generators = []  # Генераторы ресурсов
        self.current_step = 0
        self.episode_over = False
        
        # Для рендеринга
        self.camera_position = None
        self.camera_rotation = None
        
        # Статистика
        self.total_rewards = 0
        
    def _init_grid(self):
        """Инициализация сетки блоков"""
        # Создаем пустую сетку (увеличиваем размер для запасов)
        # Увеличиваем размер чтобы избежать IndexError при генерации островов по краям
        padding = 20  # Увеличенный запас
        self.grid = np.zeros(
            (self.map_size + padding, self.map_size * 2 + padding, self.map_size + padding),
            dtype=np.int8
        )
        
        # Добавляем бедрок на дне (пустота)
        self.grid[:, 0, :] = Block.BEDROCK
        
    def _generate_map(self):
        """Генерация карты для Bedwars"""
        self._init_grid()
        
        island_height = ENV_CONFIG["island_height"]
        center = self.map_size // 2
        
        # Позиции островов команд
        team_positions = []
        if self.num_teams == 2:
            team_positions = [
                np.array([center - 12, island_height, center]),
                np.array([center + 12, island_height, center]),
            ]
        elif self.num_teams == 4:
            team_positions = [
                np.array([center - 12, island_height, center]),
                np.array([center + 12, island_height, center]),
                np.array([center, island_height, center - 12]),
                np.array([center, island_height, center + 12]),
            ]
        else:
            # По умолчанию 2 команды
            team_positions = [
                np.array([center - 12, island_height, center]),
                np.array([center + 12, island_height, center]),
            ]
        
        # Создаем острова
        for i, pos in enumerate(team_positions[:self.num_teams]):
            self._create_island(pos, i)
            
        # Создаем центральный остров с генераторами
        center_island_pos = np.array([center, island_height - 2, center])
        self._create_center_island(center_island_pos)
        
        return team_positions[:self.num_teams]
    
    def _create_island(self, pos: np.ndarray, team_id: int):
        """Создание острова команды"""
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        
        # Основная платформа
        for dx in range(-5, 6):
            for dz in range(-5, 6):
                if abs(dx) + abs(dz) <= 6:  # Ромбовидная форма
                    self.grid[x + dx, y, z + dz] = Block.STONE
                    if abs(dx) + abs(dz) <= 4:
                        self.grid[x + dx, y + 1, z + dz] = Block.STONE
                        
        # Кровать в центре
        bed_pos = np.array([x, y + 2, z], dtype=np.float32)
        self.grid[x, y + 1, z] = Block.BED
        self.grid[x, y + 2, z] = Block.AIR  # Воздух над кроватью
        
        bed = Bed(bed_pos, team_id)
        # Добавляем защиту из бедрока (в начале игры)
        bed.protection_blocks = [
            (x - 1, y + 1, z),
            (x + 1, y + 1, z),
            (x, y + 1, z - 1),
            (x, y + 1, z + 1),
        ]
        self.beds.append(bed)
        
        # Генератор ресурсов на острове
        self.resource_generators.append({
            "position": np.array([x + 3, y + 1, z], dtype=np.float32),
            "type": "iron",
            "team_id": team_id,
            "tick_counter": 0,
            "interval": 60,  # Тиков между генерацией
        })
        
    def _create_center_island(self, pos: np.ndarray):
        """Создание центрального острова"""
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        
        # Платформа
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                if abs(dx) + abs(dz) <= 5:
                    self.grid[x + dx, y, z + dz] = Block.STONE
                    
        # Генераторы ресурсов
        self.resource_generators.extend([
            {
                "position": np.array([x - 2, y + 1, z], dtype=np.float32),
                "type": "gold",
                "team_id": -1,  # Нейтральный
                "tick_counter": 0,
                "interval": 120,
            },
            {
                "position": np.array([x + 2, y + 1, z], dtype=np.float32),
                "type": "gold",
                "team_id": -1,
                "tick_counter": 0,
                "interval": 120,
            },
        ])
        
    def _spawn_players(self, team_positions: List[np.ndarray]):
        """Спавн игроков"""
        self.players = []
        
        players_per_team = max(1, self.num_players // max(1, len(team_positions)))
        
        for i in range(self.num_players):
            team_id = i % len(team_positions)
            spawn_pos = team_positions[team_id].copy()
            spawn_pos[1] += 3  # Над кроватью
            
            player = Player(i, spawn_pos, team_id)
            self.players.append(player)
            
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Сброс среды"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_over = False
        self.total_rewards = 0
        self.beds = []
        self.resource_generators = []
        
        # Генерация карты и спавн игроков
        team_positions = self._generate_map()
        self._spawn_players(team_positions)
        
        # Начальная камера
        if self.render_mode == "human":
            self.camera_position = self.players[0].position.copy() + np.array([0, 5, 10])
            self.camera_rotation = np.array([-30, 0])  # pitch, yaw
            
        return self._get_observation(0), {}
    
    def _get_observation(self, player_idx: int) -> np.ndarray:
        """Получение наблюдения для игрока"""
        player = self.players[player_idx]
        
        if not player.is_alive:
            # Если игрок мертв, возвращаем нулевое наблюдение
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs_parts = []
        
        # Позиция (нормализованная)
        obs_parts.append(player.position / self.map_size)
        
        # Скорость
        obs_parts.append(player.velocity * 0.5)
        
        # Здоровье (нормализованное)
        obs_parts.append(np.array([player.health / player.max_health]))
        
        # Здоровье кровати своей команды
        own_bed_health = 0.0
        for bed in self.beds:
            if bed.team_id == player.team_id and bed.exists:
                own_bed_health = bed.health / GAME_CONFIG["bed_health"]
                break
        obs_parts.append(np.array([own_bed_health]))
        
        # Инвентарь
        obs_parts.append(player.get_inventory_array() / 64.0)
        
        # Ресурсы
        obs_parts.append(player.get_resources_array() / 100.0)
        
        # Блоки вокруг (3x3x3)
        nearby_blocks = self._get_nearby_blocks(player.position)
        obs_parts.append(nearby_blocks.flatten() / 7.0)
        
        # Информация о враге
        enemy_info = self._get_nearest_enemy_info(player)
        obs_parts.append(enemy_info)
        
        # Существует ли своя кровать
        own_bed_exists = 0.0
        for bed in self.beds:
            if bed.team_id == player.team_id and bed.exists:
                own_bed_exists = 1.0
                break
        obs_parts.append(np.array([own_bed_exists]))
        
        # Существует ли кровать врага
        enemy_bed_exists = 0.0
        for bed in self.beds:
            if bed.team_id != player.team_id and bed.exists:
                enemy_bed_exists = 1.0
                break
        obs_parts.append(np.array([enemy_bed_exists]))
        
        # Расстояние до ближайшей кровати врага
        dist_to_enemy_bed = self._get_distance_to_enemy_bed(player)
        obs_parts.append(dist_to_enemy_bed / self.map_size)
        
        # На земле ли
        obs_parts.append(np.array([1.0 if player.on_ground else 0.0]))
        
        # Падает ли
        obs_parts.append(np.array([1.0 if player.velocity[1] < 0 and not player.on_ground else 0.0]))
        
        observation = np.concatenate(obs_parts).astype(np.float32)
        return observation
    
    def _get_nearby_blocks(self, position: np.ndarray) -> np.ndarray:
        """Получить блоки вокруг позиции (3x3x3)"""
        blocks = np.zeros((3, 3, 3), dtype=np.float32)
        cx, cy, cz = int(position[0]), int(position[1]), int(position[2])
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    if 0 <= x < self.grid.shape[0] and \
                       0 <= y < self.grid.shape[1] and \
                       0 <= z < self.grid.shape[2]:
                        blocks[dx + 1, dy + 1, dz + 1] = self.grid[x, y, z]
                    else:
                        blocks[dx + 1, dy + 1, dz + 1] = Block.BEDROCK
                        
        return blocks
    
    def _get_nearest_enemy_info(self, player: Player) -> np.ndarray:
        """Получить информацию о ближайшем враге"""
        enemy_visible = 0.0
        enemy_pos = np.zeros(3, dtype=np.float32)
        enemy_health = 0.0
        enemy_distance = 1.0
        
        min_dist = float('inf')
        nearest_enemy = None
        
        for other in self.players:
            if other.team_id != player.team_id and other.is_alive:
                dist = np.linalg.norm(other.position - player.position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy = other
                    
        if nearest_enemy is not None:
            # Проверка видимости (raycasting можно добавить позже)
            enemy_visible = 1.0
            enemy_pos = nearest_enemy.position / self.map_size
            enemy_health = nearest_enemy.health / nearest_enemy.max_health
            enemy_distance = min_dist / self.map_size
            
        return np.array([enemy_visible, *enemy_pos, enemy_health, enemy_distance], dtype=np.float32)[:4]
    
    def _get_distance_to_enemy_bed(self, player: Player) -> np.ndarray:
        """Получить расстояние до ближайшей кровати врага"""
        min_dist = float('inf')
        target_bed = None
        
        for bed in self.beds:
            if bed.team_id != player.team_id and bed.exists:
                dist = np.linalg.norm(bed.position - player.position)
                if dist < min_dist:
                    min_dist = dist
                    target_bed = bed
                    
        if target_bed is not None:
            return (target_bed.position - player.position) / self.map_size
        else:
            return np.zeros(3, dtype=np.float32)
    
    def step(self, action: Dict[str, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение шага среды"""
        if self.episode_over:
            return self._get_observation(0), 0.0, True, False, {}
        
        reward = 0.0
        info = {}
        
        # Обновление генераторов ресурсов
        self._update_resource_generators()
        
        # Обработка действий для всех живых игроков
        for player_idx, player in enumerate(self.players):
            if not player.is_alive:
                continue
                
            # Получаем действие для этого игрока
            # В multi-agent среде действия должны приходить для каждого игрока
            # Но для simplicity используем одно действие для первого игрока
            if player_idx == 0:
                player_action = action
            else:
                # Для других игроков можно добавить ботов или использовать то же действие
                player_action = self._get_bot_action(player)
            
            # Применение физики
            self._apply_physics(player)
            
            # Обработка движения
            self._handle_movement(player, player_action)
            
            # Обработка строительства/разрушения
            block_change = self._handle_building(player, player_action)
            if block_change:
                reward += REWARD_CONFIG["bridge_built"]
            
            # Обработка атаки
            attack_reward = self._handle_attack(player, player_action)
            reward += attack_reward
            
            # Обработка покупок
            buy_reward = self._handle_buying(player, player_action)
            reward += buy_reward
            
            # Проверка смерти от падения в пустоту
            if player.position[1] < 1:
                player.health = 0
                player.is_alive = False
                player.deaths += 1
                reward += REWARD_CONFIG["death"]
                
            # Проверка взаимодействия с кроватью врага
            bed_reward = self._handle_bed_interaction(player)
            reward += bed_reward
            
            # Штраф за шаг
            reward += REWARD_CONFIG["step_penalty"]
        
        # Обновление шага
        self.current_step += 1
        
        # Проверка окончания эпизода
        self._check_episode_end()
        
        # Получение наблюдения для первого игрока
        obs = self._get_observation(0)
        
        # Для multi-agent можно возвращать наблюдения для всех
        info["all_observations"] = [self._get_observation(i) for i in range(len(self.players))]
        
        return obs, reward, self.episode_over, False, info
    
    def _apply_physics(self, player: Player):
        """Применение физики к игроку"""
        # Гравитация
        player.velocity[1] -= self.gravity
        
        # Применение скорости
        player.position += player.velocity
        
        # Коллизия с блоками
        self._handle_collisions(player)
        
        # Проверка на земле
        player.on_ground = False
        check_pos = player.position.copy()
        check_pos[1] -= 0.5
        if self._is_block_solid(check_pos):
            player.on_ground = True
            if player.velocity[1] < 0:
                player.velocity[1] = 0
                
    def _handle_collisions(self, player: Player):
        """Обработка коллизий с блоками"""
        # Упрощенная коллизия (можно улучшить)
        px, py, pz = player.position
        player_width = 0.3
        player_height = 1.8
        
        # Проверка границ карты
        if px < 0.5:
            player.position[0] = 0.5
            player.velocity[0] = 0
        if px > self.map_size - 0.5:
            player.position[0] = self.map_size - 0.5
            player.velocity[0] = 0
        if pz < 0.5:
            player.position[2] = 0.5
            player.velocity[2] = 0
        if pz > self.map_size - 0.5:
            player.position[2] = self.map_size - 0.5
            player.velocity[2] = 0
            
        # Проверка столкновений с блоками
        for dx in [-player_width, player_width]:
            for dz in [-player_width, player_width]:
                for dy in [0, player_height - 0.1]:
                    check_x = int(px + dx)
                    check_y = int(py + dy)
                    check_z = int(pz + dz)
                    
                    if self._is_block_solid(np.array([check_x, check_y, check_z])):
                        # Отталкивание
                        if dx != 0:
                            player.position[0] = px - dx * 2
                            player.velocity[0] = 0
                        if dz != 0:
                            player.position[2] = pz - dz * 2
                            player.velocity[2] = 0
                            
    def _is_block_solid(self, pos: np.ndarray) -> bool:
        """Проверка, является ли блок твердым"""
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        
        if x < 0 or x >= self.grid.shape[0]:
            return True
        if y < 0 or y >= self.grid.shape[1]:
            return True
        if z < 0 or z >= self.grid.shape[2]:
            return True
            
        block_type = self.grid[x, y, z]
        return block_type != Block.AIR
    
    def _handle_movement(self, player: Player, action: Dict):
        """Обработка движения"""
        move_dir = np.zeros(3)
        
        # Направление движения
        movement = action.get("movement", 0)
        if movement == 1:  # Вперед
            move_dir[0] = 1
        elif movement == 2:  # Назад
            move_dir[0] = -1
        elif movement == 3:  # Влево
            move_dir[2] = 1
        elif movement == 4:  # Вправо
            move_dir[2] = -1
            
        # Спринт
        sprint = action.get("sprint", 0)
        player.is_sprinting = (sprint == 1)
        speed = self.move_speed
        if player.is_sprinting and player.on_ground:
            speed *= self.sprint_multiplier
            
        # Применение движения
        if np.any(move_dir != 0):
            player.velocity[0] = move_dir[0] * speed
            player.velocity[2] = move_dir[2] * speed
            
        # Прыжок
        jump = action.get("jump", 0)
        if jump == 1 and player.on_ground:
            player.velocity[1] = self.jump_force
            player.on_ground = False
            
    def _handle_building(self, player: Player, action: Dict) -> bool:
        """Обработка строительства/разрушения блоков"""
        place = action.get("place_block", 0)
        break_block = action.get("break_block", 0)
        
        if place == 1:
            # Попытка поставить блок
            item_idx = action.get("inventory", 0)
            items = ["wool", "wood", "obsidian", "glass"]
            if item_idx < len(items):
                item_name = items[item_idx]
                if player.inventory.get(item_name, 0) > 0:
                    # Определяем позицию для установки (перед игроком)
                    build_pos = self._get_build_position(player)
                    if build_pos is not None and not self._is_block_solid(build_pos):
                        bx, by, bz = int(build_pos[0]), int(build_pos[1]), int(build_pos[2])
                        if 0 <= bx < self.grid.shape[0] and \
                           0 <= by < self.grid.shape[1] and \
                           0 <= bz < self.grid.shape[2]:
                            block_type = Block.WOOL if item_name == "wool" else \
                                        Block.WOOD if item_name == "wood" else \
                                        Block.OBSIDIAN if item_name == "obsidian" else \
                                        Block.GLASS
                            self.grid[bx, by, bz] = block_type
                            player.inventory[item_name] -= 1
                            return True
                            
        elif break_block == 1:
            # Попытка сломать блок
            break_pos = self._get_break_position(player)
            if break_pos is not None:
                bx, by, bz = int(break_pos[0]), int(break_pos[1]), int(break_pos[2])
                if 0 <= bx < self.grid.shape[0] and \
                   0 <= by < self.grid.shape[1] and \
                   0 <= bz < self.grid.shape[2]:
                    if self.grid[bx, by, bz] not in [Block.BEDROCK, Block.BED]:
                        self.grid[bx, by, bz] = Block.AIR
                        return True
                        
        return False
    
    def _get_build_position(self, player: Player) -> Optional[np.ndarray]:
        """Получить позицию для установки блока"""
        # Упрощенно: блок перед игроком на уровне ног
        direction = np.zeros(3)
        # Можно добавить направление взгляда
        direction[0] = 1  # По умолчанию вперед
        
        build_pos = player.position.copy() + direction * 1.5
        build_pos[1] = int(build_pos[1])  # На уровне ног
        return build_pos
    
    def _get_break_position(self, player: Player) -> Optional[np.ndarray]:
        """Получить позицию для разрушения блока"""
        # Упрощенно: блок перед игроком
        direction = np.zeros(3)
        direction[0] = 1
        
        break_pos = player.position.copy() + direction * 1.5
        return break_pos
    
    def _handle_attack(self, player: Player, action: Dict) -> float:
        """Обработка атаки"""
        reward = 0.0
        attack = action.get("attack", 0)
        
        if attack == 1:
            # Проверка наличия меча
            has_sword = (player.inventory.get("stone_sword", 0) > 0 or
                        player.inventory.get("iron_sword", 0) > 0 or
                        player.inventory.get("diamond_sword", 0) > 0)
            
            if has_sword:
                # Поиск врагов в радиусе атаки
                attack_range = 2.0
                for enemy in self.players:
                    if enemy.team_id != player.team_id and enemy.is_alive:
                        dist = np.linalg.norm(enemy.position - player.position)
                        if dist <= attack_range:
                            # Нанесение урона
                            damage = 3  # Базовый урон
                            if player.inventory.get("iron_sword", 0) > 0:
                                damage = 5
                            elif player.inventory.get("diamond_sword", 0) > 0:
                                damage = 7
                                
                            enemy.health -= damage
                            reward += REWARD_CONFIG["damage_dealt"]
                            
                            if enemy.health <= 0:
                                enemy.is_alive = False
                                enemy.deaths += 1
                                player.kills += 1
                                reward += REWARD_CONFIG["kill"]
                                
        return reward
    
    def _handle_buying(self, player: Player, action: Dict) -> float:
        """Обработка покупок"""
        reward = 0.0
        buy_action = action.get("buy_menu", 0)
        
        if buy_action > 0:
            # buy_action - 1 это индекс предмета в магазине
            items_list = list(ITEM_PRICES.keys())
            if buy_action - 1 < len(items_list):
                item_name = items_list[buy_action - 1]
                price = ITEM_PRICES[item_name]
                
                # Проверка наличия ресурсов
                can_buy = True
                for resource, amount in price.items():
                    if player.resources.get(resource, 0) < amount:
                        can_buy = False
                        break
                        
                if can_buy:
                    # Покупка
                    for resource, amount in price.items():
                        player.resources[resource] -= amount
                    player.inventory[item_name] = player.inventory.get(item_name, 0) + 1
                    reward += REWARD_CONFIG["item_purchased"]
                    
        return reward
    
    def _handle_bed_interaction(self, player: Player) -> float:
        """Обработка взаимодействия с кроватью"""
        reward = 0.0
        
        for bed in self.beds:
            if bed.team_id != player.team_id and bed.exists:
                dist = np.linalg.norm(bed.position - player.position)
                if dist < 2.0:
                    # Игрок рядом с кроватью врага
                    # Автоматическое разрушение если нет защиты
                    has_protection = False
                    for px, py, pz in bed.protection_blocks:
                        if 0 <= px < self.grid.shape[0] and \
                           0 <= py < self.grid.shape[1] and \
                           0 <= pz < self.grid.shape[2]:
                            if self.grid[px, py, pz] != Block.AIR:
                                has_protection = True
                                break
                                
                    if not has_protection:
                        # Разрушение кровати
                        if bed.damage(10):  # 10 урона за тик
                            reward += REWARD_CONFIG["bed_destroyed"]
                    else:
                        # Нужно сначала разрушить защиту
                        pass
                        
        return reward
    
    def _update_resource_generators(self):
        """Обновление генераторов ресурсов"""
        for gen in self.resource_generators:
            gen["tick_counter"] += 1
            if gen["tick_counter"] >= gen["interval"]:
                gen["tick_counter"] = 0
                
                # Найти ближайшего игрока и дать ресурс
                resource_type = gen["type"]
                for player in self.players:
                    if player.is_alive:
                        dist = np.linalg.norm(gen["position"] - player.position)
                        if dist < 3.0:
                            player.resources[resource_type] = player.resources.get(resource_type, 0) + 1
                            break
                            
    def _get_bot_action(self, player: Player) -> Dict:
        """Получить действие для бота (упрощенный ИИ)"""
        # Простейшая эвристика
        return {
            "movement": np.random.randint(0, 5),
            "jump": np.random.randint(0, 2),
            "sprint": np.random.randint(0, 2),
            "attack": np.random.randint(0, 2),
            "place_block": np.random.randint(0, 2),
            "break_block": np.random.randint(0, 2),
            "look": np.random.randint(0, 9),
            "inventory": np.random.randint(0, 10),
            "buy_menu": np.random.randint(0, 12),
        }
    
    def _check_episode_end(self):
        """Проверка окончания эпизода"""
        # Проверка по времени
        if self.current_step >= self.max_steps:
            self.episode_over = True
            return
        
        # Для curriculum stages 0-2 (обучение) не заканчиваем эпизод по условию победы
        # чтобы агент мог учиться дольше
        if self.curriculum_stage in [0, 1, 2]:
            # Только проверка по времени для ранних этапов
            return
        
        # Проверка победы (все враги мертвы или все кровати разрушены)
        # Нужно минимум 2 команды для проверки победы
        if self.num_teams < 2:
            return
            
        teams_with_beds = set()
        teams_with_players = set()
        
        for bed in self.beds:
            if bed.exists:
                teams_with_beds.add(bed.team_id)
                
        for player in self.players:
            if player.is_alive:
                teams_with_players.add(player.team_id)
                
        # Победа если осталась одна команда с кроватью или игроками
        # и есть как минимум 2 команды в игре
        if len(teams_with_beds) <= 1 and len(teams_with_players) <= 1 and len(teams_with_beds) > 0:
            # Проверяем что это действительно конец игры (есть победитель и проигравшие)
            all_team_ids = set(range(self.num_teams))
            eliminated_teams = all_team_ids - teams_with_beds
            if len(eliminated_teams) > 0:
                self.episode_over = True
            
    def render(self):
        """Рендеринг среды"""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None
    
    def _render_human(self):
        """Рендеринг для человека (консольный или простой графический)"""
        # Для простоты выводим статистику
        print(f"\n=== Step {self.current_step} ===")
        for i, player in enumerate(self.players):
            status = "ALIVE" if player.is_alive else "DEAD"
            print(f"Player {i} (Team {player.team_id}): {status} | HP: {player.health} | "
                  f"Pos: {player.position.astype(int)} | Kills: {player.kills}")
                  
        for i, bed in enumerate(self.beds):
            status = "EXISTS" if bed.exists else "DESTROYED"
            print(f"Bed {i} (Team {bed.team_id}): {status} | HP: {bed.health}")
            
    def _render_rgb(self):
        """Рендеринг в RGB массив (для записи видео)"""
        # Упрощенная визуализация сверху
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Отрисовка блоков
        mid_y = self.map_size // 2 + ENV_CONFIG["island_height"]
        for x in range(self.map_size):
            for z in range(self.map_size):
                if self.grid[x, mid_y, z] != Block.AIR:
                    img[x, z] = [128, 128, 128]  # Серый для блоков
                    
        # Отрисовка игроков
        for player in self.players:
            if player.is_alive:
                px, pz = int(player.position[0]), int(player.position[2])
                if 0 <= px < self.map_size and 0 <= pz < self.map_size:
                    color = [255, 0, 0] if player.team_id == 0 else [0, 0, 255]
                    img[px, pz] = color
                    
        return img
    
    def close(self):
        """Закрытие среды"""
        pass


# Регистрация среды
def register_envs():
    """Регистрация сред в gym"""
    from gymnasium.envs.registration import register
    
    register(
        id="Bedwars-v0",
        entry_point="bedwars_env:BedwarsEnv",
        max_episode_steps=2000,
    )
