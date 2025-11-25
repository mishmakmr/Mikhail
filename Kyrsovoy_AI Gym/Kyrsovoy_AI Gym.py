# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import numpy as np
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

print("=== ФИНАЛЬНАЯ ВЕРСИЯ DQN ДЛЯ MOUNTAINCAR-V0 ===")

# Улучшенная нейронная сеть с нормализацией
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        
        # Инициализация весов
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.00025
        self.batch_size = 128
        self.update_target_every = 100
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.update_target_network()
        self.train_step = 0
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Double DQN
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)  # Huber loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_mountain_car_final():
    """Финальная версия обучения с улучшенной стратегией"""
    try:
        env = gym.make('MountainCar-v0')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        print(f"✓ Среда: MountainCar-v0")
        print(f"Состояние: {state_size} (позиция, скорость)")
        print(f"Действия: {action_size} (0-влево, 1-стоп, 2-вправо)")
        print(f"Цель: достичь позиции ≥ 0.5")
        
        agent = DQNAgent(state_size, action_size)
        episodes = 400000
        scores = []
        steps_history = []
        max_positions = []
        success_history = []
        
        print("\n🚀 Начинаем финальное обучение...")
        print("Эпизод\tШаги\tМакс.Поз\tУспехи\tEpsilon")
        print("-" * 55)
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_position = -1.2
            positions = []
            
            while True:
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                position = next_state[0]
                velocity = next_state[1]
                max_position = max(max_position, position)
                positions.append(position)
                
                # УЛУЧШЕННАЯ СИСТЕМА НАГРАД
                if position >= 0.5:
                    reward = 100.0  # Награда за успех
                    done = True
                else:
                    # Основная награда за прогресс
                    reward = position * 10  # Чем выше позиция, тем лучше
                    
                    # Бонус за скорость в правильном направлении
                    if (position < -0.2 and velocity < 0) or (position > -0.2 and velocity > 0):
                        reward += abs(velocity) * 5
                    
                    # Штраф за время
                    reward -= 0.1
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or steps >= 1000:
                    break
            
            # Обучение на нескольких батчах
            if len(agent.memory) > agent.batch_size:
                for _ in range(4):
                    agent.replay()
            
            # Отслеживание прогресса
            success = max_position >= 0.5
            scores.append(total_reward)
            steps_history.append(steps)
            max_positions.append(max_position)
            success_history.append(1 if success else 0)
            
            # Вывод прогресса
            if episode % 100 == 0 or success:
                recent_success = np.mean(success_history[-100:]) * 100 if len(success_history) >= 100 else 0
                recent_avg_pos = np.mean(max_positions[-100:]) if len(max_positions) >= 100 else max_position
                status = " 🎉" if success else ""
                print(f"{episode}\t{steps}\t{max_position:.3f}\t{recent_success:.1f}%\t{agent.epsilon:.3f}{status}")
            
            # Ранняя остановка при стабильном успехе
            if len(success_history) >= 100:
                recent_success_rate = np.mean(success_history[-100:]) * 100
                if recent_success_rate >= 90:
                    print(f"\n🎉 ДОСТИГНУТА ЦЕЛЬ на эпизоде {episode}!")
                    print(f"Финальная успешность: {recent_success_rate:.1f}%")
                    break
        
        env.close()
        torch.save(agent.model.state_dict(), 'mountain_car_final.pth')
        print("✓ Финальная модель сохранена")
        
        return agent, scores, steps_history, max_positions, success_history
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None, None, None, None

def analyze_final_results(scores, steps_history, max_positions, success_history):
    """Детальный анализ финальных результатов"""
    total_episodes = len(scores)
    success_count = sum(success_history)
    
    print(f"\n{'='*60}")
    print("📊 ФИНАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    print(f"Всего эпизодов обучения: {total_episodes}")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Общая успешность: {success_count/total_episodes*100:.1f}%")
    print(f"Лучшая позиция: {np.max(max_positions):.3f}")
    print(f"Средняя позиция: {np.mean(max_positions):.3f}")
    print(f"Средние шаги: {np.mean(steps_history):.1f}")
    
    # Анализ по фазам обучения
    if total_episodes >= 300:
        quarter = total_episodes // 4
        phases = [
            ("1-я четверть", 0, quarter),
            ("2-я четверть", quarter, quarter*2),
            ("3-я четверть", quarter*2, quarter*3),
            ("4-я четверть", quarter*3, total_episodes)
        ]
        
        print(f"\n📈 ЭФФЕКТИВНОСТЬ ПО ФАЗАМ:")
        for phase_name, start, end in phases:
            if start < total_episodes:
                phase_success = np.mean(success_history[start:end]) * 100
                phase_avg_pos = np.mean(max_positions[start:end])
                phase_avg_steps = np.mean(steps_history[start:end])
                print(f"{phase_name}: {phase_success:.1f}% успеха, позиция {phase_avg_pos:.3f}, шаги {phase_avg_steps:.1f}")

def create_comprehensive_plots(scores, steps_history, max_positions, success_history):
    """Создание комплексных графиков"""
    plt.figure(figsize=(18, 12))
    
    # График 1: Общий прогресс
    plt.subplot(2, 3, 1)
    x = range(len(max_positions))
    plt.scatter(x, max_positions, alpha=0.3, s=1, color='blue', label='Макс. позиция')
    
    # Скользящее среднее
    if len(max_positions) >= 50:
        window = 50
        moving_avg = [np.mean(max_positions[i:i+window]) for i in range(len(max_positions)-window+1)]
        plt.plot(range(window-1, len(max_positions)), moving_avg, 'red', linewidth=2, label='Среднее за 50 эп.')
    
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Цель')
    plt.xlabel('Эпизод')
    plt.ylabel('Максимальная позиция')
    plt.title('Прогресс обучения')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Успешность
    plt.subplot(2, 3, 2)
    if len(success_history) >= 100:
        window = 100
        success_rates = [np.mean(success_history[i:i+window]) * 100 for i in range(len(success_history)-window+1)]
        plt.plot(range(window-1, len(success_history)), success_rates, 'purple', linewidth=2)
        plt.axhline(y=90, color='orange', linestyle='--', label='Цель 90%')
        plt.xlabel('Эпизод')
        plt.ylabel('Успешность (%)')
        plt.title('Успешность (скользящее среднее)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # График 3: Распределение позиций
    plt.subplot(2, 3, 3)
    successful_positions = [pos for pos, success in zip(max_positions, success_history) if success]
    unsuccessful_positions = [pos for pos, success in zip(max_positions, success_history) if not success]
    
    plt.hist(successful_positions, bins=20, alpha=0.7, color='green', label='Успешные', edgecolor='black')
    plt.hist(unsuccessful_positions, bins=20, alpha=0.7, color='red', label='Неуспешные', edgecolor='black')
    plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Цель')
    plt.xlabel('Максимальная позиция')
    plt.ylabel('Частота')
    plt.title('Распределение позиций')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 4: Шаги по эпизодам
    plt.subplot(2, 3, 4)
    successful_steps = [steps for steps, success in zip(steps_history, success_history) if success]
    unsuccessful_steps = [steps for steps, success in zip(steps_history, success_history) if not success]
    
    plt.scatter(range(len(successful_steps)), successful_steps, alpha=0.6, color='green', s=10, label='Успешные')
    plt.scatter(range(len(unsuccessful_steps)), unsuccessful_steps, alpha=0.3, color='red', s=10, label='Неуспешные')
    plt.xlabel('Эпизод')
    plt.ylabel('Шаги')
    plt.title('Шаги по эпизодам')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 5: Награды
    plt.subplot(2, 3, 5)
    successful_scores = [score for score, success in zip(scores, success_history) if success]
    unsuccessful_scores = [score for score, success in zip(scores, success_history) if not success]
    
    plt.scatter(range(len(successful_scores)), successful_scores, alpha=0.6, color='green', s=10, label='Успешные')
    plt.scatter(range(len(unsuccessful_scores)), unsuccessful_scores, alpha=0.3, color='red', s=10, label='Неуспешные')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.title('Награды по эпизодам')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 6: Финальный прогресс (последние 200 эпизодов)
    plt.subplot(2, 3, 6)
    recent_episodes = min(200, len(max_positions))
    recent_positions = max_positions[-recent_episodes:]
    recent_success = success_history[-recent_episodes:]
    
    colors = ['green' if success else 'red' for success in recent_success]
    plt.scatter(range(recent_episodes), recent_positions, c=colors, alpha=0.6, s=20)
    plt.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Цель')
    plt.xlabel('Эпизод (последние)')
    plt.ylabel('Макс. позиция')
    plt.title('Финальный прогресс')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mountain_car_final_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Комплексные графики сохранены")

def run_final_evaluation(agent, num_episodes=50):
    """Финальное тестирование агента"""
    print(f"\n🎯 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ({num_episodes} эпизодов)")
    
    env = gym.make('MountainCar-v0')
    results = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        steps = 0
        positions = []
        
        while True:
            action = agent.act(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            
            positions.append(next_state[0])
            state = next_state
            steps += 1
            
            if done or steps >= 1000:
                max_pos = max(positions)
                success = max_pos >= 0.5
                results.append({
                    'steps': steps,
                    'max_position': max_pos,
                    'success': success
                })
                break
    
    env.close()
    
    # Статистика
    success_rate = np.mean([r['success'] for r in results]) * 100
    avg_steps = np.mean([r['steps'] for r in results])
    avg_position = np.mean([r['max_position'] for r in results])
    
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"Успешность: {success_rate:.1f}%")
    print(f"Средние шаги: {avg_steps:.1f}")
    print(f"Средняя позиция: {avg_position:.3f}")
    
    # Анализ эффективности
    if success_rate >= 90:
        print("🎉 ПРЕВОСХОДНО! Агент надежно решает задачу")
    elif success_rate >= 70:
        print("✅ ХОРОШО! Агент успешно решает задачу")
    elif success_rate >= 50:
        print("⚠️ УДОВЛЕТВОРИТЕЛЬНО! Агент решает задачу, но нестабильно")
    else:
        print("❌ ТРЕБУЕТСЯ ДООБУЧЕНИЕ!")
    
    return results

if __name__ == "__main__":
    # Запуск финального обучения
    agent, scores, steps_history, max_positions, success_history = train_mountain_car_final()
    
    if agent and scores:
        # Анализ результатов
        analyze_final_results(scores, steps_history, max_positions, success_history)
        
        # Создание графиков
        create_comprehensive_plots(scores, steps_history, max_positions, success_history)
        
        # Финальное тестирование
        test_results = run_final_evaluation(agent, 50)
        
        # Итоговый вывод
        final_success_rate = np.mean([r['success'] for r in test_results]) * 100
        print(f"\n{'='*60}")
        print("🎓 ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'='*60}")
        print(f"Алгоритм: Deep Q-Network (DQN) с улучшениями")
        print(f"Среда: MountainCar-v0")
        print(f"Результат: {final_success_rate:.1f}% успешности")
        
        if final_success_rate >= 70:
            print("✅ ЗАДАЧА УСПЕШНО РЕШЕНА!")
            print("Агент освоил стратегию раскачки для достижения цели")
        else:
            print("⚠️ Задача решена частично")
            print("Рекомендуется увеличить количество эпизодов обучения")
    
    else:
        print("❌ Обучение не удалось")

print("\n🏁 ПРОГРАММА ЗАВЕРШЕНА")