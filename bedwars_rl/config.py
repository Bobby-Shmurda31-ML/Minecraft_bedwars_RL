"""
Конфигурация для RL агента Bedwars
"""
import numpy as np

# Параметры среды
ENV_CONFIG = {
    "map_size": 32,  # Размер карты (блоки)
    "island_height": 10,  # Высота островов
    "bridge_width": 3,  # Ширина мостов
    "max_steps": 2000,  # Максимальное количество шагов в эпизоде
    "tick_rate": 20,  # Тиков в секунду
    "gravity": 0.08,  # Гравитация
    "jump_force": 0.42,  # Сила прыжка
    "move_speed": 0.15,  # Скорость движения
    "sprint_multiplier": 1.3,  # Множитель спринта
}

# Параметры игры
GAME_CONFIG = {
    "starting_iron": 0,
    "starting_gold": 0,
    "starting_emerald": 0,
    "starting_diamond": 0,
    "bed_health": 100,
    "player_health": 100,
    "respawn_time": 0,  # 0 = без возрождения (как в финале)
    "void_damage": 10,  # Урон от падения в пустоту
    "fall_damage_threshold": 3.0,  # Высота для получения урона от падения
}

# Цены предметов
ITEM_PRICES = {
    "wool": {"iron": 4},
    "wood": {"iron": 1},
    "stone_sword": {"iron": 7},
    "iron_sword": {"iron": 9, "gold": 1},
    "diamond_sword": {"emerald": 4},
    "chainmail_armor": {"iron": 12},
    "iron_armor": {"iron": 24},
    "diamond_armor": {"emerald": 6},
    "golden_apple": {"gold": 3},
    "fireball": {"gold": 1},
    "tnt": {"gold": 4},
    "obsidian": {"gold": 4},
    "end_pearl": {"emerald": 4},
    "shield": {"iron": 5},
}

# Параметры обучения
TRAINING_CONFIG = {
    "total_timesteps": 10000000,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 8,
    "num_minibatches": 4,
    "update_epochs": 4,
}

# Награды
REWARD_CONFIG = {
    "step_penalty": -0.001,  # Штраф за каждый шаг (для ускорения игры)
    "damage_dealt": 0.1,  # Награда за нанесение урона
    "damage_taken": -0.1,  # Штраф за получение урона
    "kill": 1.0,  # Награда за убийство
    "death": -1.0,  # Штраф за смерть
    "bed_destroyed": 5.0,  # Награда за разрушение кровати
    "bed_defended": 2.0,  # Награда за защиту кровати
    "bridge_built": 0.01,  # Награда за каждый блок моста
    "resource_collected": 0.001,  # Награда за сбор ресурса
    "item_purchased": 0.01,  # Награда за покупку предмета
    "win": 10.0,  # Награда за победу
    "lose": -5.0,  # Штраф за поражение
}

# Пространства действий
ACTION_SPACE = {
    "movement": 5,  # вперед, назад, влево, вправо, нет движения
    "jump": 2,  # прыжок / нет прыжка
    "sprint": 2,  # спринт / нет спринта
    "attack": 2,  # атака / нет атаки
    "place_block": 2,  # поставить блок / нет
    "break_block": 2,  # сломать блок / нет
    "look": 9,  # направление взгляда (8 направлений + нет изменения)
    "inventory": 10,  # выбор предмета в инвентаре (0-9)
    "buy_menu": 12,  # открытие меню покупки и выбор предмета
}

# Размеры наблюдений
OBSERVATION_SPACE = {
    "position": 3,  # x, y, z
    "velocity": 3,  # vx, vy, vz
    "health": 1,  # здоровье игрока
    "bed_health": 1,  # здоровье кровати
    "inventory": 10,  # количество предметов каждого типа
    "resources": 4,  # железо, золото, изумруды, алмазы
    "nearby_blocks": 27,  # блоки вокруг (3x3x3)
    "enemy_visible": 4,  # виден ли враг, его позиция, здоровье, расстояние
    "own_bed_exists": 1,  # существует ли своя кровать
    "enemy_bed_exists": 1,  # существует ли кровать врага
    "distance_to_enemy_bed": 3,  # расстояние до кровати врага (dx, dy, dz)
    "on_ground": 1,  # на земле ли игрок
    "is_falling": 1,  # падает ли игрок
}

TOTAL_OBS_DIM = sum(OBSERVATION_SPACE.values())
TOTAL_ACT_DIM = sum(ACTION_SPACE.values())

print(f"Total observation dimension: {TOTAL_OBS_DIM}")
print(f"Total action dimension: {TOTAL_ACT_DIM}")
