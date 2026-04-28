# Bedwars RL Agent

RL-агент для игры в Bedwars (мини-игра в Minecraft) с использованием Curriculum Learning и PPO.

## Структура проекта

```
bedwars_rl/
├── config.py              # Конфигурация (параметры среды, обучения, награды)
├── main.py                # Главный скрипт для обучения и тестирования
├── env/
│   ├── bedwars_env.py     # Основная среда Bedwars
│   ├── curriculum_envs.py # Среды для Curriculum Learning
│   └── viewer.py          # 3D визуализатор с возможностью игры человеком
├── agents/
│   └── ppo_agent.py       # PPO агент
├── models/                # Сохраненные модели
└── logs/                  # Логи обучения
```

## Установка

```bash
pip install numpy torch gymnasium pygame
```

## Использование

### Обучение агента

```bash
# Базовое обучение
python main.py train --timesteps 1000000 --output models/bedwars_final.pt

# С использованием Curriculum Learning (рекомендуется)
python main.py train --curriculum --curriculum-stage 0 --timesteps 500000 --output models/stage_0.pt
python main.py train --curriculum --curriculum-stage 1 --load models/stage_0.pt --timesteps 500000 --output models/stage_1.pt
# ... продолжить для всех этапов (0-6)

# Обучение с более высокой скоростью обучения
python main.py train --lr 1e-3 --timesteps 2000000
```

### Оценка агента

```bash
# Оценка без визуализации
python main.py evaluate --model models/bedwars_final.pt --episodes 20

# Оценка с визуализацией
python main.py evaluate --model models/bedwars_final.pt --episodes 5 --render
```

### Визуализация и игра против агента

```bash
# Просмотр среды без агента
python -m env.viewer

# Игра против обученного агента
python main.py view --model models/bedwars_final.pt

# Запуск с другим FPS
python main.py view --model models/bedwars_final.pt --fps 60
```

## Этапы Curriculum Learning

| Этап | Название | Описание |
|------|----------|----------|
| 0 | movement_basic | Научиться двигаться вперед |
| 1 | bridge_building | Построить мост до вражеского острова |
| 2 | resource_collection | Собрать ресурсы и купить предмет |
| 3 | combat_basic | Победить врага в бою |
| 4 | bed_destruction | Добраться до кровати и разрушить её |
| 5 | full_attack | Полный цикл атаки |
| 6 | full_game | Полноценная игра Bedwars |

## Управление в режиме просмотра

| Клавиша | Действие |
|---------|----------|
| WASD | Перемещение камеры |
| Q/E | Вверх/Вниз |
| Стрелки | Вращение камеры |
| 1-9 | Выбор игрока |
| F | Следование за игроком |
| H | Включить управление человеком |
| P | Пауза |
| ESC | Выход |

### Управление человеком (после нажатия H)

| Клавиша | Действие |
|---------|----------|
| Стрелки | Движение |
| Пробел | Прыжок |
| Shift | Спринт |
| J/Z | Атака |
| K/X | Построить блок |
| L/C | Сломать блок |
| 1-9 | Выбор предмета |

## Архитектура агента

Агент использует PPO (Proximal Policy Optimization) с Actor-Critic архитектурой:

- **Observation**: Вектор состояния (~70 измерений) включающий:
  - Позиция и скорость игрока
  - Здоровье (игрока и кровати)
  - Инвентарь и ресурсы
  - Блоки вокруг (3x3x3)
  - Информация о врагах
  - Расстояние до кровати врага

- **Action Space**: Дискретное пространство из 9 типов действий:
  - Movement (5): вперед, назад, влево, вправо, нет движения
  - Jump (2): прыжок / нет
  - Sprint (2): спринт / нет
  - Attack (2): атака / нет
  - Place Block (2): поставить / нет
  - Break Block (2): сломать / нет
  - Look (9): направление взгляда
  - Inventory (10): выбор предмета
  - Buy Menu (12): покупка предметов

## Настройка параметров

Основные параметры можно изменить в `config.py`:

- `ENV_CONFIG` - параметры среды (размер карты, гравитация, скорость)
- `GAME_CONFIG` - параметры игры (здоровье, урон, время возрождения)
- `ITEM_PRICES` - цены предметов
- `REWARD_CONFIG` - коэффициенты наград
- `TRAINING_CONFIG` - параметры обучения PPO

## Рекомендации по обучению

1. **Начните с Curriculum Learning**: Обучайте поэтапно от простого к сложному
2. **Используйте достаточно шагов**: Минимум 1M шагов для базового обучения
3. **Сохраняйте чекпоинты**: Регулярно сохраняйте модель (--save-freq)
4. **Мониторьте энтропию**: Если слишком низкая - агент застрял в локальном оптимуме
5. **Экспериментируйте с наградами**: Баланс наград критичен для хорошего поведения

## Расширение функциональности

### Добавление новых режимов игры

Для добавления режимов типа 1x1x1x1, 2x2x2x2 и т.д.:

```python
env = BedwarsEnv(
    num_players=4,  # 1x1x1x1
    num_teams=4,
)
```

### Добавление новых предметов

В `config.py` добавьте в `ITEM_PRICES`:

```python
"new_item": {"iron": 5, "gold": 2},
```

### Изменение карт

Переопределите метод `_generate_map()` в `bedwars_env.py`.

## Лицензия

MIT License
