import pygame
import numpy as np
import time
from Game import Config, Sprites, Setup
from Agents.GeneticAlgorithm import GeneticAlgorithm
from Agents.NeuralNetwork import NeuralNetwork


def decode_output_to_move(output):
    """
    신경망 출력값을 움직임 방향으로 변환
    - output: 신경망의 출력값 리스트
    """
    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    print(f"Decoded output: {output}")  # 출력값 디버깅 로그
    return moves[np.argmax(output)]  # 가장 큰 값의 인덱스를 움직임으로 매핑

def start_game():
    """
    팩맨 게임 시작 함수
    - 게임 초기화, 스프라이트 생성 및 유전 알고리즘 학습 실행
    - 수동 또는 자동 모드로 팩맨을 조작
    """
    # pygame 초기화 및 화면 설정
    pygame.init()
    screen = pygame.display.set_mode([606, 606])  # 게임 화면 크기 설정
    pygame.display.set_caption('Pacman')
    background = pygame.Surface(screen.get_size()).convert()
    background.fill(Config.BLACK)
    clock = pygame.time.Clock()  # 게임 루프를 제어하는 시계 객체
    font = pygame.font.Font("freesansbold.ttf", 24)  # 점수 표시를 위한 폰트 설정

    # 스프라이트 그룹 생성
    all_sprites_list = pygame.sprite.RenderPlain()  # 모든 스프라이트 관리 그룹
    block_list = pygame.sprite.RenderPlain()         # 블록(코인) 스프라이트 그룹
    monsta_list = pygame.sprite.RenderPlain()        # 유령 스프라이트 그룹
    pacman_collide = pygame.sprite.RenderPlain()     # 팩맨 충돌 검사용 그룹

    # 방(벽) 및 게이트 생성
    wall_list = Setup.setup_room_one(all_sprites_list)  # 첫 번째 방 생성
    gate = Setup.setup_gate(all_sprites_list)           # 게이트 생성

    # 팩맨 생성 및 초기 위치 설정
    pacman = Sprites.Player(287, 439, "images/pacman.png")  # 팩맨 객체 생성
    all_sprites_list.add(pacman)                            # 팩맨을 모든 스프라이트 리스트에 추가
    pacman_collide.add(pacman)                              # 충돌 그룹에 팩맨 추가

    # 유령 생성 및 초기 위치 설정
    ghosts = {
        "Blinky": Sprites.Ghost(287, 199, "images/Blinky.png"),
        "Pinky":  Sprites.Ghost(287, 199, "images/Pinky.png"),
        "Inky":   Sprites.Ghost(287, 199, "images/Inky.png"),
        "Clyde":  Sprites.Ghost(287, 199, "images/Clyde.png")
    }

    ghost_group = pygame.sprite.Group()  # 유령 스프라이트 그룹 생성
    for ghost in ghosts.values():
        ghost_group.add(ghost)            # 유령 그룹에 추가

    directions = {  # 유령 이동 경로 설정 (Config에 정의된 방향 리스트)
        "Pinky":  Config.Pinky_directions,
        "Blinky": Config.Blinky_directions,
        "Inky":   Config.Inky_directions,
        "Clyde":  Config.Clyde_directions
    }
    turns_steps = {ghost: [0, 0] for ghost in ghosts}  # 유령 이동 상태 초기화

    for ghost in ghosts.values():
        monsta_list.add(ghost)           # 유령 그룹에 유령 추가
        all_sprites_list.add(ghost)      # 모든 스프라이트 리스트에 유령 추가

    # 블록(코인) 생성 및 초기화
    for row in range(19):
        for column in range(19):
            # 중앙 영역 제외
            if (row, column) in [(7, 8), (7, 9), (7, 10), (8, 8), (8, 9), (8, 10)]:
                continue
            block = Sprites.Block(Config.YELLOW, 4, 4)
            block.rect.x = (30 * column + 6) + 26  # 블록의 x 좌표 설정
            block.rect.y = (30 * row + 6) + 26     # 블록의 y 좌표 설정
            if not pygame.sprite.spritecollide(block, wall_list, False):
                block_list.add(block)
                all_sprites_list.add(block)

    # 신경망 및 유전 알고리즘 생성
    input_size = 8   # 입력: 팩맨 좌표, 가장 가까운 유령 좌표
    output_size = 4  # 출력: UP, DOWN, LEFT, RIGHT
    network = NeuralNetwork(input_size, output_size)  # 신경망 생성
    ga = GeneticAlgorithm(                           # 유전 알고리즘 생성
        population_size=16,
        mutation_rate=0.2,
        generations=100,
        network=network
    )

    # 유전 알고리즘 실행 → best_genes 반환
    best_genes = ga.run(
        wall_list, block_list,
        pacman, ghost_group,
        screen, all_sprites_list,
        gate, font, directions
    )
    print("Best Genes:", best_genes)  # 학습된 최적 유전자 출력

    # 학습된 유전자 저장
    np.save(Config.filename, np.array(best_genes))
    print("학습된 모델이 'GA.npy' 파일로 저장")

    # → 만약 저장된 GA.npy 로 플레이할 때는 아래 두 줄 활성화
    # best_genes = np.load(Config.filename)
    # network.set_weights(best_genes)

    # ─────────────────────────────────────────────────────────────────────
    #  아래부터 “게임 루프” (최적 유전자 베이스로 팩맨을 자동 실행)
    #  그리고 “유령이 지나간 블록 주황색 처리” + “팩맨이 먹으면 2배 점수” 로직 추가
    # ─────────────────────────────────────────────────────────────────────


    game_start_time = time.time()
    score = 0
    total_blocks = len(block_list)  # 블록의 총 개수
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 창 종료 이벤트 처리
                done = True

            # → (수동 조작 코드 추가 시 여기에 arrow-key 입력 처리)

        # 1) 팩맨 움직여서 벽/게이트 충돌 처리
        pacman.update(wall_list, gate)

        # 2) 유령 이동 & “유령-블록 충돌 → 블록을 주황색으로 바꾸기”
        for ghost_name, ghost in ghosts.items():
            turn, steps = turns_steps[ghost_name]
            length = len(directions[ghost_name]) - 1
            turn, steps = ghost.changespeed(directions[ghost_name], ghost_name, turn, steps, length)
            ghost.update(wall_list, gate)
            turns_steps[ghost_name] = [turn, steps]

            # ── 유령이 충돌한 블록(coin)을 탐지 → 주황(make_orange) 처리
            collided_blocks = pygame.sprite.spritecollide(ghost, block_list, False)
            for block in collided_blocks:
                block.make_orange()
                # Block 내부에서 is_orange=True로 바뀌고, 이미지가 주황으로 다시 그려집니다.

        # 3) 팩맨-블록 충돌 검사 → 블록 삭제 + 점수(주황이면 2배)
        blocks_hit_list = pygame.sprite.spritecollide(pacman, block_list, True)
        for block in blocks_hit_list:
            if getattr(block, 'is_orange', False):
                score += 100  # 주황 블록을 먹으면 2배 점수
            else:
                score += 50   # 기본 노란 블록

        # 4) (원래 코드) pacman_x, pacman_y, ghost_x, ghost_y 계산용
        #    → 네트워크 입력으로 쓰거나 디버깅용일 수도 있습니다.
        if ghost_group:
            closest_ghost = min(
                ghost_group,
                key=lambda g: abs(g.rect.left - pacman.rect.left) + abs(g.rect.top - pacman.rect.top)
            )
            ghost_x = (closest_ghost.rect.left - pacman.rect.left)
            ghost_y = (closest_ghost.rect.top - pacman.rect.top)
        else:
            ghost_x, ghost_y = 0, 0

        # 5) 화면 업데이트
        screen.blit(background, (0, 0))
        all_sprites_list.draw(screen)
        # Score, 남은 블록 수 등을 화면 상단에 표시
        info_text = font.render(
            f"Score: {score}  |  Time: {int(60 - (time.time() - game_start_time))}s  |  Blocks left: {len(block_list)}",
            True, Config.RED)
        screen.blit(info_text, (10, 10))

        

        pygame.display.flip()
        clock.tick(15)

        # 6) 게임 종료 조건: 블록이 모두 사라지면 종료
        if len(block_list) == 0:
            done = True
        if time.time() - game_start_time > 60:
            done = True 

    pygame.quit()  # 게임 종료