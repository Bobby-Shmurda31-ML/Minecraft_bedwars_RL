"""
PPO Агент для Bedwars
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import copy

from config import TRAINING_CONFIG, ACTION_SPACE


class ActorCritic(nn.Module):
    """
    Актер-Критик сеть для PPO
    Использует общую основу для извлечения признаков
    """
    
    def __init__(self, obs_dim: int, action_dims: Dict[str, int], hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Общая основа
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Критик (значение состояния)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Актеры для каждого типа действия
        self.actors = nn.ModuleDict()
        for action_name, action_dim in action_dims.items():
            self.actors[action_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
            
        self.action_dims = action_dims
        
    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Прямой проход"""
        features = self.base(obs)
        
        # Значение состояния
        value = self.critic(features)
        
        # Логиты действий
        action_logits = {}
        for action_name, actor in self.actors.items():
            action_logits[action_name] = actor(features)
            
        return action_logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Получить значение состояния"""
        features = self.base(obs)
        return self.critic(features)
    
    def get_action_probs(self, obs: torch.Tensor) -> Dict[str, Categorical]:
        """Получить распределения вероятностей действий"""
        action_logits, _ = self.forward(obs)
        action_probs = {}
        for action_name, logits in action_logits.items():
            action_probs[action_name] = Categorical(logits=logits)
        return action_probs


class RolloutBuffer:
    """Буфер для хранения траекторий"""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
    def add(self, obs: np.ndarray, actions: Dict[str, int], reward: float, 
            done: bool, value: float, log_probs: Dict[str, float]):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_probs)
        
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
    def get(self) -> Dict:
        """Получить все данные"""
        return {
            'obs': np.array(self.obs),
            'actions': {k: np.array([a[k] for a in self.actions]) for k in self.actions[0].keys()},
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
            'log_probs': {k: np.array([lp[k] for lp in self.log_probs]) for k in self.log_probs[0].keys()},
        }


class PPOAgent:
    """
    PPO агент для обучения в среде Bedwars
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dims: Dict[str, int],
        device: str = "auto",
        **kwargs
    ):
        # Автовыбор устройства
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Параметры
        self.lr = kwargs.get("learning_rate", TRAINING_CONFIG["learning_rate"])
        self.gamma = kwargs.get("gamma", TRAINING_CONFIG["gamma"])
        self.gae_lambda = kwargs.get("gae_lambda", TRAINING_CONFIG["gae_lambda"])
        self.clip_epsilon = kwargs.get("clip_epsilon", TRAINING_CONFIG["clip_epsilon"])
        self.entropy_coef = kwargs.get("entropy_coef", TRAINING_CONFIG["entropy_coef"])
        self.value_loss_coef = kwargs.get("value_loss_coef", TRAINING_CONFIG["value_loss_coef"])
        self.max_grad_norm = kwargs.get("max_grad_norm", TRAINING_CONFIG["max_grad_norm"])
        self.num_minibatches = kwargs.get("num_minibatches", TRAINING_CONFIG["num_minibatches"])
        self.update_epochs = kwargs.get("update_epochs", TRAINING_CONFIG["update_epochs"])
        
        # Сеть
        self.policy = ActorCritic(obs_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Буфер
        self.buffer = RolloutBuffer()
        
        # Статистика
        self.total_steps = 0
        self.episode_rewards = []
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Выбор действия"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy.get_action_probs(obs_tensor)
            
            actions = {}
            log_probs = {}
            
            for action_name, prob_dist in action_probs.items():
                if deterministic:
                    action = prob_dist.probs.argmax(dim=-1).item()
                else:
                    action = prob_dist.sample().item()
                    
                actions[action_name] = action
                log_probs[action_name] = prob_dist.log_prob(torch.tensor(action)).item()
                
        return actions, log_probs
    
    def get_value(self, obs: np.ndarray) -> float:
        """Получить значение состояния"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.policy.get_value(obs_tensor).item()
        return value
    
    def store_transition(
        self,
        obs: np.ndarray,
        actions: Dict[str, int],
        reward: float,
        done: bool,
        value: float,
        log_probs: Dict[str, float]
    ):
        """Сохранение перехода в буфер"""
        self.buffer.add(obs, actions, reward, done, value, log_probs)
        self.total_steps += 1
        
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = last_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
            next_value = values[t]
            
        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Обновление политики"""
        data = self.buffer.get()
        
        obs_tensor = torch.FloatTensor(data['obs']).to(self.device)
        actions_tensor = {k: torch.LongTensor(v).to(self.device) for k, v in data['actions'].items()}
        old_log_probs = {k: torch.FloatTensor(v).to(self.device) for k, v in data['log_probs'].items()}
        values = np.array(data['values'])
        rewards = np.array(data['rewards'])
        dones = np.array(data['dones'])
        
        # Вычисление last_value для последнего состояния
        with torch.no_grad():
            last_obs = obs_tensor[-1].unsqueeze(0)
            last_value = self.policy.get_value(last_obs).cpu().numpy().flatten()[0]
            
        # GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        returns_tensor = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        
        # Разбиение на мини-батчи
        num_samples = len(data['obs'])
        batch_size = num_samples // self.num_minibatches
        
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
        }
        
        # Обучение
        for _ in range(self.update_epochs):
            # Перемешивание
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                minibatch_indices = indices[start:end]
                
                # Получение данных мини-батча
                mb_obs = obs_tensor[minibatch_indices]
                mb_actions = {k: v[minibatch_indices] for k, v in actions_tensor.items()}
                mb_old_log_probs = {k: v[minibatch_indices] for k, v in old_log_probs.items()}
                mb_returns = returns_tensor[minibatch_indices]
                mb_advantages = advantages_tensor[minibatch_indices]
                
                # Прямой проход
                new_action_logits, new_values = self.policy(mb_obs)
                
                # Вычисление логарифмов вероятностей
                new_log_probs = {}
                entropy_sum = 0
                for action_name, logits in new_action_logits.items():
                    dist = Categorical(logits=logits)
                    new_log_probs[action_name] = dist.log_prob(mb_actions[action_name])
                    entropy_sum += dist.entropy().mean()
                    
                # Объединение лог-вероятностей всех действий
                new_log_prob_sum = sum(new_log_probs.values())
                old_log_prob_sum = sum(mb_old_log_probs.values())
                
                # Отношение вероятностей
                ratio = torch.exp(new_log_prob_sum - old_log_prob_sum)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Loss для значения
                value_loss = nn.MSELoss()(new_values, mb_returns)
                
                # Энтропия для исследования
                entropy_loss = -entropy_sum
                
                # Общая потеря
                loss = policy_loss + \
                       self.value_loss_coef * value_loss + \
                       self.entropy_coef * entropy_loss
                
                # Градиентный шаг
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Сбор статистики
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy_sum.item()
                
        # Очистка буфера
        self.buffer.clear()
        
        # Усреднение статистики
        num_updates = self.update_epochs * (num_samples // batch_size)
        for key in stats:
            stats[key] /= num_updates
            
        return stats
    
    def save(self, path: str):
        """Сохранение модели"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"Model loaded from {path}")
        
    def train(
        self,
        env,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 100000,
        verbose: bool = True
    ):
        """Обучение агента"""
        print(f"Starting training for {total_timesteps} timesteps")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        global_step = 0
        
        for step in range(total_timesteps):
            # Выбор действия
            actions, log_probs = self.select_action(obs)
            value = self.get_value(obs)
            
            # Шаг среды
            next_obs, reward, done, truncated, info = env.step(actions)
            
            # Сохранение перехода
            self.store_transition(obs, actions, reward, done or truncated, value, log_probs)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            # Обновление
            if step > 0 and step % TRAINING_CONFIG["batch_size"] == 0:
                stats = self.update()
                
                if verbose and step % (eval_freq // 10) == 0:
                    print(f"Step {step}: policy_loss={stats['policy_loss']:.4f}, "
                          f"value_loss={stats['value_loss']:.4f}, "
                          f"entropy={stats['entropy']:.4f}")
            
            # Конец эпизода
            if done or truncated:
                self.episode_rewards.append(episode_reward)
                
                if verbose and len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"Episode {len(self.episode_rewards)}: reward={episode_reward:.2f}, "
                          f"avg_last_10={avg_reward:.2f}, ep_steps={episode_length}, "
                          f"total_steps={global_step}")
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
            # Сохранение
            if step > 0 and step % save_freq == 0:
                self.save(f"models/ppo_bedwars_{step}.pt")
                
        print("Training completed!")
        return self.episode_rewards
