"""
Curriculum Learning среды для Bedwars

Постепенное усложнение задач:
1. Простое движение по мосту
2. Построение моста
3. Сбор ресурсов и покупка
4. Бой с врагом
5. Разрушение кровати
6. Полный цикл атаки
7. Полная игра
"""
import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.bedwars_env import BedwarsEnv, Player, Block
from config import ENV_CONFIG, GAME_CONFIG, REWARD_CONFIG


class CurriculumBedwarsEnv(BedwarsEnv):
    """
    Среда с поддержкой Curriculum Learning
    """
    
    def __init__(
        self,
        curriculum_stage: int = 0,
        **kwargs
    ):
        super().__init__(curriculum_stage=curriculum_stage, **kwargs)
        
        # Специфичные настройки для каждого этапа
        self.stage_config = self._get_stage_config(curriculum_stage)
        
    def _get_stage_config(self, stage: int) -> Dict:
        """Конфигурация для каждого этапа обучения"""
        configs = {
            0: {  # Движение по прямой
                "name": "movement_basic",
                "description": "Научиться двигаться вперед",
                "spawn_distance": 5,
                "target_distance": 20,
                "enemies": False,
                "beds_protected": True,
                "starting_items": ["wool"],
                "reward_focus": ["movement"],
            },
            1: {  # Построение моста
                "name": "bridge_building",
                "description": "Построить мост до вражеского острова",
                "spawn_distance": 10,
                "target_distance": 25,
                "enemies": False,
                "beds_protected": True,
                "starting_items": ["wool"] * 5,
                "reward_focus": ["building", "movement"],
            },
            2: {  # Сбор ресурсов
                "name": "resource_collection",
                "description": "Собрать ресурсы и купить предмет",
                "spawn_distance": 5,
                "target_distance": 15,
                "enemies": False,
                "beds_protected": True,
                "starting_items": [],
                "reward_focus": ["resources", "buying"],
            },
            3: {  # Бой
                "name": "combat_basic",
                "description": "Победить врага в бою",
                "spawn_distance": 8,
                "target_distance": 10,
                "enemies": True,
                "beds_protected": True,
                "starting_items": ["stone_sword"],
                "reward_focus": ["combat"],
            },
            4: {  # Разрушение кровати
                "name": "bed_destruction",
                "description": "Добраться до кровати и разрушить её",
                "spawn_distance": 12,
                "target_distance": 25,
                "enemies": True,
                "beds_protected": False,
                "starting_items": ["wool"] * 3,
                "reward_focus": ["bed_destruction"],
            },
            5: {  # Полный цикл атаки
                "name": "full_attack",
                "description": "Собрать ресурсы, построить мост, разрушить кровать",
                "spawn_distance": 15,
                "target_distance": 30,
                "enemies": True,
                "beds_protected": False,
                "starting_items": ["wool"] * 2,
                "reward_focus": ["full_cycle"],
            },
            6: {  # Полная игра
                "name": "full_game",
                "description": "Полноценная игра Bedwars",
                "spawn_distance": 20,
                "target_distance": 35,
                "enemies": True,
                "beds_protected": True,
                "starting_items": ["wool"] * 2,
                "reward_focus": ["win"],
            },
        }
        return configs.get(stage, configs[0])
    
    def reset(self, seed=None, options=None):
        """Сброс с учетом этапа curriculum"""
        # Переопределяем генерацию карты для текущего этапа
        return super().reset(seed=seed, options=options)
    
    def _generate_map(self):
        """Генерация карты с учетом этапа"""
        self._init_grid()
        
        island_height = ENV_CONFIG["island_height"]
        center = self.map_size // 2
        
        stage_name = self.stage_config["name"]
        
        # Для ранних этапов упрощаем карту
        if stage_name in ["movement_basic", "bridge_building"]:
            # Один остров с целевой платформой
            team_positions = [
                np.array([center - 10, island_height, center]),
            ]
            
            # Создаем только свой остров
            self._create_island(team_positions[0], 0)
            
            # Добавляем целевую платформу
            target_x = center + self.stage_config["target_distance"]
            for dx in range(-3, 4):
                for dz in range(-3, 4):
                    self.grid[target_x + dx, island_height, center + dz] = Block.STONE
                    
            # Для этапа строительства моста оставляем пустоту между островами
            if stage_name == "bridge_building":
                # Очищаем блоки между островами
                for x in range(center - 5, target_x + 5):
                    for z in range(center - 2, center + 3):
                        self.grid[x, island_height, z] = Block.AIR
                        
        else:
            # Полноценная генерация для поздних этапов
            return super()._generate_map()
            
        return team_positions
    
    def step(self, action: Dict[str, int]):
        """Шаг с модифицированными наградами для текущего этапа"""
        obs, reward, done, truncated, info = super().step(action)
        
        # Модификация наград в зависимости от этапа
        stage_name = self.stage_config["name"]
        
        if stage_name == "movement_basic":
            # Награда за движение вперед
            player = self.players[0]
            if action.get("movement", 0) == 1:  # Вперед
                reward += 0.1
                
        elif stage_name == "bridge_building":
            # Увеличенная награда за строительство
            reward *= 2.0
            
        elif stage_name == "resource_collection":
            # Награда за сбор ресурсов
            player = self.players[0]
            for resource, amount in player.resources.items():
                if amount > 0:
                    reward += 0.05 * amount
                    
        elif stage_name == "combat_basic":
            # Увеличенная награда за бой
            pass  # Уже есть в базовой среде
            
        elif stage_name == "bed_destruction":
            # Увеличенная награда за разрушение кровати
            pass
            
        elif stage_name == "full_attack":
            # Комбинированные награды
            pass
            
        # Штраф за бездействие на ранних этапах
        if stage_name in ["movement_basic", "bridge_building"]:
            if action.get("movement", 0) == 0 and action.get("jump", 0) == 0:
                reward -= 0.005
                
        return obs, reward, done, truncated, info


def create_curriculum_env(stage: int = 0, **kwargs):
    """Фабрика для создания сред curriculum learning"""
    return CurriculumBedwarsEnv(curriculum_stage=stage, **kwargs)


# Пример использования
if __name__ == "__main__":
    # Тестирование разных этапов
    for stage in range(7):
        print(f"\n=== Testing Stage {stage} ===")
        env = create_curriculum_env(stage=stage, num_players=2, num_teams=1)
        obs, _ = env.reset()
        print(f"Stage: {env.stage_config['name']}")
        print(f"Description: {env.stage_config['description']}")
        print(f"Observation shape: {obs.shape}")
        
        # Несколько шагов
        for _ in range(10):
            action = {
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
            obs, reward, done, truncated, info = env.step(action)
            
        env.close()
        
    print("\nAll stages tested successfully!")
