import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 점수 변화 (원본+이동평균)
df = pd.read_csv("training_results_Q.csv")

df['score_ma'] = df['score'].rolling(window=20).mean()

plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['score'], label='Score per Episode', alpha=0.4)
plt.plot(df['episode'], df['score_ma'], label='Moving Average (20)', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Episode vs. Score (Raw & Moving Avg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 평균 생존 시간(막대그래프)
mean_time = df['elapsed_time'].mean()
plt.figure(figsize=(6, 4))
plt.bar(['Mean Elapsed Time'], [mean_time], color='skyblue', width=0.4)
plt.ylabel("Average survival time (sec)")
plt.title("Average survival time")
plt.ylim(0, mean_time * 1.3)
plt.tight_layout()
plt.text(0, mean_time + mean_time * 0.05, f"{mean_time:.2f}", ha='center', va='bottom', fontsize=12)
plt.show()

# 3. 행동 경로 밀도 히트맵 (Q_Position 폴더)
POSITION_DIR = "Q_Position"
GRID_SIZE = 19
visit_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

for fname in os.listdir(POSITION_DIR):
    if fname.startswith("positions_episode_") and fname.endswith(".csv"):
        df_pos = pd.read_csv(os.path.join(POSITION_DIR, fname))
        for x, y in zip(df_pos['x'], df_pos['y']):
            x_idx = int(x) // 30
            y_idx = int(y) // 30
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                visit_grid[y_idx, x_idx] += 1

plt.figure(figsize=(7,7))
sns.heatmap(visit_grid, cmap='hot', square=True, linewidths=0.3)
plt.title("Behavioral path density hit맵")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()

# 4-1. 최고 점수 에피소드 번호 찾기
max_epi = int(df.loc[df['score'].idxmax()]['episode'])
max_score = df.loc[df['score'].idxmax()]['score']
pos_df = pd.read_csv(os.path.join(POSITION_DIR, f"positions_episode_{max_epi}.csv"))
x, y = pos_df['x'].values, pos_df['y'].values

steps = np.arange(len(x))
plt.figure(figsize=(7,7))
sc = plt.scatter(x, y, c=steps, cmap='viridis', s=30)
plt.plot(x, y, alpha=0.2, color='gray')
plt.scatter(x[0], y[0], color='lime', s=120, label='Start', edgecolor='k')
plt.scatter(x[-1], y[-1], color='red', s=120, label='End', edgecolor='k')
plt.title(f"Best Score Path (Episode {max_epi}, Score: {max_score})")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.colorbar(sc, label='Step (time order)')
plt.legend()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.grid(True)
plt.show()

#가장 넓게 탐색
max_unique = 0
best_epi = None
best_file = None

# Step 1: Find the episode with the most unique positions
for fname in os.listdir(POSITION_DIR):
    if fname.startswith("positions_episode_") and fname.endswith(".csv"):
        df_pos = pd.read_csv(os.path.join(POSITION_DIR, fname))
        coords = set(zip(df_pos['x'], df_pos['y']))
        if len(coords) > max_unique:
            max_unique = len(coords)
            best_epi = int(fname.split("_")[-1].replace(".csv",""))
            best_file = fname

print(f"Episode with widest exploration: {best_epi} ({max_unique} unique positions)")

# Step 2: Plot the path
df_best = pd.read_csv(os.path.join(POSITION_DIR, best_file))
x, y = df_best['x'].values, df_best['y'].values
steps = np.arange(len(x))

plt.figure(figsize=(7,7))
sc = plt.scatter(x, y, c=steps, cmap='plasma', s=30)
plt.plot(x, y, alpha=0.2, color='gray')
plt.scatter(x[0], y[0], color='lime', s=120, label='Start', edgecolor='k')
plt.scatter(x[-1], y[-1], color='red', s=120, label='End', edgecolor='k')
plt.title(f"Widest Exploration Path (Episode {best_epi}, Unique positions: {max_unique})")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.colorbar(sc, label='Step (time order)')
plt.legend()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.grid(True)
plt.show()