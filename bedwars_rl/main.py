"""
Главный скрипт для обучения и тестирования агента Bedwars
"""
import argparse
import sys
import os
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TOTAL_OBS_DIM, ACTION_SPACE, TRAINING_CONFIG
from env.bedwars_env import BedwarsEnv
from env.curriculum_envs import create_curriculum_env
from agents.ppo_agent import PPOAgent


def train(args):
    """Обучение агента"""
    print(f"Starting training with curriculum stage {args.curriculum_stage}")
    
    # Создание среды
    if args.curriculum:
        env = create_curriculum_env(
            stage=args.curriculum_stage,
            num_players=args.num_players,
            num_teams=args.num_teams,
        )
    else:
        env = BedwarsEnv(
            num_players=args.num_players,
            num_teams=args.num_teams,
        )
    
    # Создание агента
    agent = PPOAgent(
        obs_dim=TOTAL_OBS_DIM,
        action_dims=ACTION_SPACE,
        device=args.device,
        learning_rate=args.lr,
    )
    
    # Загрузка если указана
    if args.load:
        agent.load(args.load)
    
    # Обучение
    agent.train(
        env,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        verbose=not args.quiet,
    )
    
    # Сохранение финальной модели
    if args.output:
        agent.save(args.output)
    
    env.close()
    print("Training completed!")


def evaluate(args):
    """Оценка обученного агента"""
    print(f"Evaluating agent from {args.model}")
    
    # Создание среды
    env = BedwarsEnv(
        num_players=args.num_players,
        num_teams=args.num_teams,
        render_mode="human" if args.render else None,
    )
    
    # Создание и загрузка агента
    agent = PPOAgent(
        obs_dim=TOTAL_OBS_DIM,
        action_dims=ACTION_SPACE,
        device=args.device,
    )
    agent.load(args.model)
    
    # Оценка
    num_episodes = args.episodes
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if args.render:
                env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    # Статистика
    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f}")
    print(f"Std Reward: {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    
    env.close()


def view(args):
    """Запуск визуализатора"""
    print("Starting Bedwars Viewer")
    
    from env.viewer import BedwarsViewer
    
    # Создание среды
    env = BedwarsEnv(
        num_players=args.num_players,
        num_teams=args.num_teams,
        render_mode="human",
    )
    
    # Создание визуализатора
    viewer = BedwarsViewer(env)
    
    # Загрузка агента если указан
    agent = None
    if args.model:
        agent = PPOAgent(
            obs_dim=TOTAL_OBS_DIM,
            action_dims=ACTION_SPACE,
            device=args.device,
        )
        agent.load(args.model)
        print(f"Loaded agent from {args.model}")
    
    # Запуск
    viewer.run(agent=agent, fps=args.fps)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Bedwars RL Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    train_parser.add_argument("--curriculum-stage", type=int, default=0, help="Curriculum stage (0-6)")
    train_parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    train_parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    train_parser.add_argument("--num-teams", type=int, default=2, help="Number of teams")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    train_parser.add_argument("--save-freq", type=int, default=100000, help="Save frequency")
    train_parser.add_argument("--load", type=str, help="Load model from path")
    train_parser.add_argument("--output", type=str, help="Output model path")
    train_parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument("--model", type=str, required=True, help="Model path")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render environment")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")
    eval_parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    eval_parser.add_argument("--num-teams", type=int, default=2, help="Number of teams")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View the environment")
    view_parser.add_argument("--model", type=str, help="Model path (optional)")
    view_parser.add_argument("--fps", type=int, default=30, help="FPS")
    view_parser.add_argument("--device", type=str, default="auto", help="Device")
    view_parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    view_parser.add_argument("--num-teams", type=int, default=2, help="Number of teams")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "view":
        view(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
