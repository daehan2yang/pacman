import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning 에이전트 클래스
    - 상태(state)를 키로, 행동(action)을 인덱스로 가지는 Q-테이블을 관리
    - ε-탐욕 정책(epsilon-greedy)으로 행동 선택
    """
    def __init__(self,
                 actions,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05):
        """
        Args:
            actions (list): 가능한 행동(action) 리스트 (예: ['UP','DOWN','LEFT','RIGHT'])
            alpha (float): 학습률
            gamma (float): 할인율
            epsilon (float): 탐험 비율 초기값
            epsilon_decay (float): 매 에피소드마다 epsilon 감소 계수
            epsilon_min (float): epsilon 하한값
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-테이블: key=state(tuple), value=list of Q-values for each action
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

    def choose_action(self, state):
        """
        ε-탐욕 정책에 따라 행동 선택
        Args:
            state (tuple): 현재 상태를 나타내는 튜플
        Returns:
            action (str): 선택한 행동
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        max_indices = np.where(q_values == np.max(q_values))[0]
        chosen_index = random.choice(max_indices)
        return self.actions[chosen_index]

    def learn(self, state, action, reward, next_state, done):
        """
        Q-러닝 업데이트
        Args:
            state (tuple): 이전 상태
            action (str): 수행한 행동
            reward (float): 받은 보상
            next_state (tuple): 다음 상태
            done (bool): 에피소드 종료 여부
        """
        action_index = self.actions.index(action)
        current_q = self.q_table[state][action_index]
        if done:
            target = reward
        else:
            next_max = np.max(self.q_table[next_state])
            target = reward + self.gamma * next_max
        # Q-값 업데이트
        self.q_table[state][action_index] += self.alpha * (target - current_q)

    def update_epsilon(self):
        """
        에피소드가 끝난 후 ε 감소
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min