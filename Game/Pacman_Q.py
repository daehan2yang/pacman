import os
import time
import csv
import numpy as np
from Agents.Q_learning_agent import QLearningAgent
from Game import Config, Setup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
QTABLE_PATH = os.path.join(BASE_DIR, "q_table.npz")


def get_state(pacman, ghost_group):
    grid_size = 30
    pac_x_idx = pacman.rect.left // grid_size
    pac_y_idx = pacman.rect.top // grid_size
    if len(ghost_group) > 0:
        closest_ghost = min(
            ghost_group.values(),
            key=lambda g: abs(g.rect.left - pacman.rect.left) + abs(g.rect.top - pacman.rect.top)
        )
        dx = closest_ghost.rect.left - pacman.rect.left
        dy = closest_ghost.rect.top - pacman.rect.top
        dx_sign = int(np.sign(dx))
        dy_sign = int(np.sign(dy))
    else:
        dx_sign, dy_sign = 0, 0
    return (pac_x_idx, pac_y_idx, dx_sign, dy_sign)


def decode_action_to_speed(action):
    speed = Config.PACMAN_SPEED if hasattr(Config, 'PACMAN_SPEED') else 15
    if action == 'UP':
        return 0, -speed
    elif action == 'DOWN':
        return 0, speed
    elif action == 'LEFT':
        return -speed, 0
    elif action == 'RIGHT':
        return speed, 0
    else:
        return 0, 0


def train_q_learning(episodes=1000, max_steps_per_episode=1000):
    import pygame
    from Game import Sprites

    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((606, 606))
    pygame.display.set_caption("Pacman Q-Learning Train")
    clock = pygame.time.Clock()
    font = pygame.font.Font("freesansbold.ttf", 24)

    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    agent = QLearningAgent(actions)

    # ─── 유령 및 turns_steps(에피소드 간 누적) ───
    ghosts = {
        "Blinky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, 'Blinky.png')),
        "Pinky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, 'Pinky.png')),
        "Inky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, 'Inky.png')),
        "Clyde": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, 'Clyde.png'))
    }
    # ─ Ghost 이미지 및 스케일 적용 (한 번만)
    for name, ghost in ghosts.items():
        img_path = os.path.join(IMAGES_DIR, f"{name}.png")
        ghost.image = pygame.image.load(img_path).convert_alpha()
        ghost.image = pygame.transform.scale(ghost.image, (30, 30))
        ghost.rect = ghost.image.get_rect(topleft=(287, 199))

    # ─ Ghost들을 담을 그룹
    ghost_group_master = pygame.sprite.Group()
    for ghost in ghosts.values():
        ghost_group_master.add(ghost)

    # ─ 유령 이동 경로 Directions (Config에서 정의된 방향 패턴)
    directions = {
        "Blinky": Config.Blinky_directions,
        "Pinky": Config.Pinky_directions,
        "Inky": Config.Inky_directions,
        "Clyde": Config.Clyde_directions
    }

    # ─ “turns_steps” 딕셔너리: 한 번만 생성 → 이후 에피소드마다 초기화하지 않음
    turns_steps = {name: [0, 0] for name in ghosts}  # 에피소드 간 누적

    # ─── 에피소드별 결과를 저장할 리스트 추가 ───
    #   형태: [(episode, score, elapsed_time), ...]
    results = []
    total_start = time.time()

    POSITION_DIR = os.path.join(BASE_DIR, "Q_Position")
    if not os.path.exists(POSITION_DIR):
        os.makedirs(POSITION_DIR)

    for episode in range(episodes):
        episode_start = time.time()
        pacman_positions = []
        # --------------------------------------------
        # 1) 에피소드마다 “게임 환경”만 재구성 (Ghost 인스턴스 재사용)
        # --------------------------------------------
        all_sprites = pygame.sprite.RenderPlain()
        block_list = pygame.sprite.RenderPlain()
        ghost_group = pygame.sprite.Group(ghosts.values())
        all_sprites.add(ghosts.values())  # Ghost들 한 번만 add

        # 벽과 게이트 생성
        wall_list = Setup.setup_room_one(all_sprites)
        gate = Setup.setup_gate(all_sprites)

        # ── 팩맨 생성 및 초기 위치 설정 (매 에피소드마다)
        pacman_path = os.path.join(IMAGES_DIR, 'pacman.png')
        pacman = Sprites.Player(287, 439, pacman_path)
        pacman.image = pygame.image.load(pacman_path).convert_alpha()
        pacman.image = pygame.transform.scale(pacman.image, (30, 30))
        pacman.rect = pacman.image.get_rect(topleft=(287, 439))
        all_sprites.add(pacman)

        # ── Ghost 위치 초기화 (turns_steps는 초기화하지 않음)
        for name, ghost in ghosts.items():
            ghost.rect.topleft = (287, 199)

        # ── 블록(코인) 생성
        for row in range(19):
            for column in range(19):
                if (row, column) in [(7, 8), (7, 9), (7, 10), (8, 8), (8, 9), (8, 10)]:
                    continue
                block = Sprites.Block(Config.YELLOW, 4, 4)
                block.rect.x = (30 * column + 6) + 26
                block.rect.y = (30 * row + 6) + 26
                if not pygame.sprite.spritecollide(block, wall_list, False):
                    block_list.add(block)
                    all_sprites.add(block)

        state = get_state(pacman, ghosts)
        done = False
        score = 0

        # **60초 제한용 타이머 초기화**
        game_start_time = time.time()

        for step in range(max_steps_per_episode):
            pacman_positions.append((pacman.rect.left, pacman.rect.top))
            # ─── 60초 초과 시 즉시 에피소드 종료 ───
            if time.time() - game_start_time >= 60.0:
                done = True
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # ─── (1) 팩맨: 블록 단위로 이동 ───
            action = agent.choose_action(state)  # 'UP','DOWN','LEFT','RIGHT'
            pacman.move_block(action, wall_list, gate)  # 한 블록(30px) 이동

            # ─── (2) 유령 이동 & 유령-블록 충돌 처리 ───
            for name, ghost in ghosts.items():
                turn, steps = turns_steps[name]
                length = len(directions[name]) - 1
                turn, steps = ghost.changespeed(
                    directions[name], name, turn, steps, length
                )
                ghost.update(wall_list, gate)
                turns_steps[name] = [turn, steps]

                collided_blocks = pygame.sprite.spritecollide(ghost, block_list, False)
                for block in collided_blocks:
                    block.make_orange()

            # ─── (3) 팩맨-블록 충돌 검사 → 블록 삭제 + 점수 ───
            blocks_hit_by_pacman = pygame.sprite.spritecollide(pacman, block_list, True)
            for block in blocks_hit_by_pacman:
                if getattr(block, 'is_orange', False):
                    score += 100
                else:
                    score += 50

            # ─── (4) 팩맨-유령 충돌 검사 ───
            reward = len(blocks_hit_by_pacman)
            if pygame.sprite.spritecollideany(pacman, ghost_group):
                reward -= 10
                done = True

            next_state = get_state(pacman, ghosts)
            if len(block_list) == 0:
                reward += 50
                done = True

            agent.learn(state, action, reward, next_state, done)
            state = next_state

            # ─── (5) 화면 그리기 ───
            screen.fill(Config.BLACK)
            all_sprites.draw(screen)

            # 왼쪽 상단: Score, Time
            info_text = font.render(
                f"Score: {score}  |  Time: {time.time() - episode_start:.2f}s",
                True, Config.RED
            )
            screen.blit(info_text, (10, 10))

            # 오른쪽 상단: Episode 번호
            epi_text = font.render(f"Episode: {episode + 1}/{episodes}", True, Config.RED)
            epi_rect = epi_text.get_rect(topright=(596, 10))
            screen.blit(epi_text, epi_rect)

            pygame.display.flip()
            clock.tick(15)

            if done:
                break

        # ─── 에피소드가 끝난 뒤 결과를 리스트에 저장 ───
        elapsed_time = time.time() - episode_start
        results.append((episode + 1, score, round(elapsed_time, 2)))

        agent.update_epsilon()
        print(f"[Episode {episode + 1}/{episodes}] Score: {score}, Elapsed: {elapsed_time:.2f}s")

        # (에피소드 끝나면 위치 기록 저장!)
        pos_path = os.path.join(POSITION_DIR, f"positions_episode_{episode + 1}.csv")
        with open(pos_path, "w", newline="") as pf:
            writer = csv.writer(pf)
            writer.writerow(["x", "y"])
            for xy in pacman_positions:
                writer.writerow(xy)

    print(f"Total Training Time: {time.time() - total_start:.2f}s")

    # ─── 학습이 모두 끝난 뒤, CSV 파일로 저장 ───
    csv_path = os.path.join(BASE_DIR, "training_results_Q.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "elapsed_time"])
        for row in results:
            writer.writerow(row)
    print(f"Training results saved to {csv_path}")

    # ─── Q-테이블 저장 ───
    states = np.array(list(agent.q_table.keys()), dtype=object)
    qvalues = np.array(list(agent.q_table.values()), dtype=object)
    np.savez_compressed(QTABLE_PATH, states=states, qvalues=qvalues)
    print(f"Q-table saved to {QTABLE_PATH}")

    pygame.quit()


def play_q_learning():
    """
    학습된 Q-테이블을 불러와 게임 플레이
    """
    if not os.path.exists(QTABLE_PATH):
        print("Q-table 파일을 찾을 수 없습니다. 먼저 학습을 수행하세요.")
        return
    data = np.load(QTABLE_PATH, allow_pickle=True)
    states = data['states']
    qvalues = data['qvalues']
    # 딕셔너리 형태로 복원
    q_table = {tuple(states[i]): qvalues[i] for i in range(len(states))}

    import pygame
    from Game import Sprites

    pygame.init()
    screen = pygame.display.set_mode([606, 606])
    pygame.display.set_caption('Pacman Q-Learning Play')
    background = pygame.Surface(screen.get_size()).convert()
    background.fill(Config.BLACK)
    clock = pygame.time.Clock()
    font = pygame.font.Font("freesansbold.ttf", 24)

    all_sprites = pygame.sprite.RenderPlain()
    block_list = pygame.sprite.RenderPlain()

    # ─── Ghost 인스턴스와 turns_steps도 train_q_learning과 똑같이 에피소드 바깥에 한 번만 생성 ───
    ghosts = {
        "Blinky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, "Blinky.png")),
        "Pinky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, "Pinky.png")),
        "Inky": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, "Inky.png")),
        "Clyde": Sprites.Ghost(287, 199, os.path.join(IMAGES_DIR, "Clyde.png"))
    }
    for name, ghost in ghosts.items():
        ghost.image = pygame.image.load(os.path.join(IMAGES_DIR, f"{name}.png")).convert_alpha()
        ghost.image = pygame.transform.scale(ghost.image, (30, 30))
        ghost.rect = ghost.image.get_rect(topleft=(287, 199))

    ghost_group = pygame.sprite.Group(ghosts.values())
    all_sprites.add(ghosts.values())

    directions = {
        "Blinky": Config.Blinky_directions,
        "Pinky": Config.Pinky_directions,
        "Inky": Config.Inky_directions,
        "Clyde": Config.Clyde_directions
    }
    # ─── “turns_steps”도 에피소드 밖에서 한 번만 생성 → 플레이 중 같은 인스턴스로 누적됨
    turns_steps = {name: [0, 0] for name in ghosts}

    # 한 번만 블록·벽 등 세팅
    wall_list = Setup.setup_room_one(all_sprites)
    gate = Setup.setup_gate(all_sprites)

    pacman_path = os.path.join(IMAGES_DIR, 'pacman.png')
    pacman = Sprites.Player(287, 439, pacman_path)
    pacman.image = pygame.image.load(pacman_path).convert_alpha()
    pacman.image = pygame.transform.scale(pacman.image, (30, 30))
    pacman.rect = pacman.image.get_rect(topleft=(287, 439))
    all_sprites.add(pacman)

    # 블록 생성
    for row in range(19):
        for column in range(19):
            if (row, column) in [(7, 8), (7, 9), (7, 10), (8, 8), (8, 9), (8, 10)]:
                continue
            block = Sprites.Block(Config.YELLOW, 4, 4)
            block.rect.x = (30 * column + 6) + 26
            block.rect.y = (30 * row + 6) + 26
            if not pygame.sprite.spritecollide(block, wall_list, False):
                block_list.add(block)
                all_sprites.add(block)

    # ─── 플레이 루프
    state = get_state(pacman, ghosts)
    score = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # ─── (1) Q-테이블 기반으로 팩맨 한 블록 이동 ───
        q_vals = q_table.get(state, np.zeros(len(actions)))
        action = actions[int(np.argmax(q_vals))]  # 'UP','DOWN','LEFT','RIGHT'
        pacman.move_block(action, wall_list, gate)

        # ─── (2) 유령 이동 & 유령-블록 충돌 처리 ───
        for name, ghost in ghosts.items():
            turn, steps = turns_steps[name]
            length = len(directions[name]) - 1
            turn, steps = ghost.changespeed(
                directions[name], name, turn, steps, length
            )
            ghost.update(wall_list, gate)
            turns_steps[name] = [turn, steps]

            collided_blocks = pygame.sprite.spritecollide(ghost, block_list, False)
            for block in collided_blocks:
                block.make_orange()

        # ─── (3) 팩맨-블록 충돌 & 점수 ───
        blocks_hit = pygame.sprite.spritecollide(pacman, block_list, True)
        for block in blocks_hit:
            if getattr(block, 'is_orange', False):
                score += 100
            else:
                score += 50

        # ─── (4) 팩맨-유령 충돌 검사 ───
        if pygame.sprite.spritecollideany(pacman, ghost_group):
            done = True

        # ─── (5) 화면 그리기 ───
        screen.blit(background, (0, 0))
        all_sprites.draw(screen)
        score_text = font.render(f"Score: {score}", True, Config.RED)
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        clock.tick(15)

    pygame.quit()
    print(f"Final Score: {score}")


def start_game(mode="train", episodes=500):
    if mode == "train":
        train_q_learning(episodes=episodes)
    else:
        play_q_learning()


if __name__ == "__main__":
    start_game(mode="train", episodes=100)