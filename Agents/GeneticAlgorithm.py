import random
import numpy as np
import pygame
from Game import Config, Sprites
import time


def is_wall(x, y, wall_list):
    dummy = Sprites.Player(x, y, "images/pacman.png")
    return bool(pygame.sprite.spritecollide(dummy, wall_list, False))


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, network):
        """
        유전 알고리즘 초기화
        - population_size: 개체군의 크기
        - mutation_rate: 돌연변이 확률
        - generations: 세대 수
        - network: 신경망 객체
        """
        self.population_size = population_size
        self.gene_length = network.total_weights()  # 신경망의 총 가중치 개수를 기준으로 유전자 길이 설정
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []  # 개체군
        self.network = network

    def initialize_population(self):
        """
        개체군 초기화
        - 랜덤 값으로 유전자(신경망 가중치) 생성
        """
        self.population = [
            [random.uniform(-1, 1) for _ in range(self.network.total_weights())]
            for _ in range(self.population_size)
        ]

    def apply_weights_to_network(self, genes):
        """
        신경망에 유전자를 가중치로 설정
        - genes: 신경망의 가중치 리스트
        """
        self.network.set_weights(genes)

    def reset_game_state(self, pacman, ghost_names, ghost_group, block_list, initial_positions, all_sprites_list):
        """
        게임 상태 초기화
        - 팩맨, 유령, 블록 위치를 초기 상태로 되돌림
        """
        # 팩맨 위치 초기화
        pacman.rect.left, pacman.rect.top = initial_positions['pacman']

        # 유령 위치 초기화
        for ghost_name, ghost in zip(ghost_names, ghost_group):
            ghost.rect.left, ghost.rect.top = initial_positions['ghosts'][ghost_name]

        # 블록 초기화
        for block in block_list:
            all_sprites_list.remove(block)  # 렌더링 리스트에서 제거
        block_list.empty()  # 블록 그룹 비우기

        # 초기 블록 위치에서 새 블록 생성
        for block_position in initial_positions['blocks']:
            new_block = Sprites.Block(Config.YELLOW, 4, 4)
            new_block.rect.x, new_block.rect.y = block_position
            block_list.add(new_block)  # 블록 그룹에 추가
            all_sprites_list.add(new_block)  # 렌더링 리스트에 추가

        print(f"Blocks reset: {len(block_list)}")  # 디버깅 로그

    def move_pacman(self, pacman, move):
        """
        팩맨의 이동 처리
        - move: 움직임 방향 ('UP', 'DOWN', 'LEFT', 'RIGHT')
        """
        if move == 'UP':
            pacman.rect.top -= 30
        elif move == 'DOWN':
            pacman.rect.top += 30
        elif move == 'LEFT':
            pacman.rect.left -= 30
        elif move == 'RIGHT':
            pacman.rect.left += 30

    def decode_output_to_move(self, output):
        """
        신경망 출력값을 움직임 방향으로 변환
        - output: 신경망의 출력값 리스트
        """
        moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Decoded output: {output}")  # 출력값 디버깅 로그
        return moves[np.argmax(output)]  # 가장 큰 값의 인덱스를 움직임으로 매핑
    
    
    
    def fitness(self, genes, wall_list, block_list, pacman, ghost_group, directions, turns_steps, screen, all_sprites_list, gate, font, generation, individual):
        """
        적합도 평가 함수
        - genes: 유전자 (팩맨의 신경망 가중치)
        - wall_list: 벽 스프라이트 그룹
        - block_list: 블록(코인) 스프라이트 그룹
        - pacman: 팩맨 객체
        - ghost_group: 유령 스프라이트 그룹
        - directions: 유령의 이동 방향 설정
        - turns_steps: 유령의 현재 회전 및 이동 상태
        - screen: pygame 화면 객체
        - all_sprites_list: 모든 스프라이트 리스트
        - gate: 문(게이트) 스프라이트
        - font: 폰트 객체
        """
        # 유전자를 신경망에 적용
        self.network.set_weights(genes)

        visited = set()

        # 게임 초기 상태 설정
        initial_positions = {
            'pacman': (287, 439),
            'ghosts': {
                "Blinky": (287, 199),
                "Pinky": (287, 199),
                "Inky": (287, 199),
                "Clyde": (287, 199)
            },
            'blocks': [(block.rect.x, block.rect.y) for block in block_list]
        }
        pacman.rect.left, pacman.rect.top = initial_positions['pacman']
        score, survival_time = 0, 0
        ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
        start_time = pygame.time.get_ticks()

        game_start_time = time.time() 

        for _ in range(self.gene_length):
            # 팩맨의 이전 위치 저장

            old_x, old_y = pacman.rect.left, pacman.rect.top

            # 가장 가까운 유령과의 상대적 좌표 계산
            if ghost_group:
                closest_ghost = min(
                    ghost_group,
                    key=lambda ghost: abs(ghost.rect.left - pacman.rect.left) + abs(ghost.rect.top - pacman.rect.top)
                )
                ghost_x = closest_ghost.rect.left - pacman.rect.left
                ghost_y = closest_ghost.rect.top - pacman.rect.top
            else:
                ghost_x, ghost_y = 0, 0  # 유령이 없을 경우 0으로 설정

            # 신경망 입력값
            # 현재 팩맨 위치
            px, py = pacman.rect.left, pacman.rect.top
            tile_size = 30  # 블록 셀 크기

            # 각 방향(상, 하, 좌, 우) 좌표
            up    = (px, py - tile_size)
            down  = (px, py + tile_size)
            left  = (px - tile_size, py)
            right = (px + tile_size, py)

            # 각 방향에 벽이 있으면 1, 없으면 0
            up_blocked    = 1 if is_wall(*up, wall_list) else 0
            down_blocked  = 1 if is_wall(*down, wall_list) else 0
            left_blocked  = 1 if is_wall(*left, wall_list) else 0
            right_blocked = 1 if is_wall(*right, wall_list) else 0

            # 기존 상대좌표(팩맨, 유령) + 4방향 벽 정보로 입력값 구성!
            inputs = [px, py, ghost_x, ghost_y, up_blocked, down_blocked, left_blocked, right_blocked]


            # 신경망 출력값으로 다음 이동 결정
            output = self.network.predict(inputs)
            moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            block_flags = [up_blocked, down_blocked, left_blocked, right_blocked]
            masked_output = [o if b == 0 else -1e9 for o, b in zip(output, block_flags)]
            move = moves[np.argmax(masked_output)]
            self.move_pacman(pacman, move)

            # 벽 충돌 시 점수 페널티
            if pygame.sprite.spritecollide(pacman, wall_list, False):
                pacman.rect.left, pacman.rect.top = old_x, old_y
                score -= 5

            # 유령 충돌 시 게임 종료
            if pygame.sprite.spritecollide(pacman, ghost_group, False):
                score -= 100
                break
            
            pos = (pacman.rect.left, pacman.rect.top)

            if pos in visited:
                score -= 5    # 재방문 감점
            else:
                visited.add(pos)

            # 블록(코인) 충돌 시 점수 증가
            blocks_hit = pygame.sprite.spritecollide(pacman, block_list, True)
            for block in blocks_hit:
                if block.is_orange:
                    score += 100  # 주황블록
                else:
                    score += 50   # 노란블록

            # 유령 업데이트
            for ghost, ghost_name in zip(ghost_group, ghost_names):
                turn, steps = turns_steps[ghost_name]
                length = len(directions[ghost_name]) - 1
                turn, steps = ghost.changespeed(directions[ghost_name], ghost_name, turn, steps, length)
                ghost.update(wall_list, gate)
                turns_steps[ghost_name] = [turn, steps]

                # ***** [여기] 유령이 블록 위를 지나가면 주황색으로! *****
                collided_blocks = pygame.sprite.spritecollide(ghost, block_list, False)
                for block in collided_blocks:
                    if not block.is_orange:
                        block.make_orange()

            # 화면 업데이트
            screen.fill(Config.BLACK)
            wall_list.draw(screen)
            gate.draw(screen)
            all_sprites_list.draw(screen)

            survival_time = (pygame.time.get_ticks() - start_time) / 1000
            text = font.render(f"Score: {score} | Time: {survival_time:.2f}", True, Config.RED)
            screen.blit(text, [10, 10])
            if generation is not None and individual is not None:
                epi_text = font.render(
                    f"Gene: {generation+1}/{self.generations} | Indiv: {individual+1}/{self.population_size}",
                    True, Config.RED)
                epi_rect = epi_text.get_rect(topright=(596, 10))
                screen.blit(epi_text, epi_rect)
            pygame.display.flip()
            pygame.time.delay(1)

            if len(block_list) == 0 or time.time() - game_start_time > 60:
                break

        # 게임 상태 초기화
        self.reset_game_state(
            pacman, ghost_names, ghost_group, block_list,
            initial_positions, all_sprites_list
        )

        print(f"Survival Time: {survival_time:.2f} seconds, Score: {score}")
        return score

    def select_parents(self, fitness_scores):
        """
        적합도가 높은 부모 개체 선택
        - fitness_scores: 각 개체의 적합도 리스트
        """
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), reverse=True)]
        return sorted_population[:2]  # 적합도가 높은 상위 2개 개체 선택

    def crossover(self, parent1, parent2):
        """
        부모 개체를 기반으로 자식 유전자 생성 (교차)
        """
        point = random.randint(1, self.gene_length - 1)  # 교차 지점 설정
        return (
            parent1[:point] + parent2[point:],  # 첫 번째
            parent2[:point] + parent1[point:]   # 두 번째
        )

    def mutate(self, genes):
        """
        돌연변이 적용
        """
        return [
            gene + random.uniform(-0.1, 0.1) if random.random() < self.mutation_rate else gene
            for gene in genes
        ]

    def evolve(self, fitness_scores):
        """
        새로운 세대 생성
        """
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents(fitness_scores)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.extend([self.mutate(child1), self.mutate(child2)])
        self.population = new_population

    def run(self, wall_list, block_list, pacman, ghost_group, screen, all_sprites_list, gate, font, directions):
        """
        유전 알고리즘 실행
        - 각 세대를 반복하며 최적의 유전자 탐색
        """
        self.initialize_population()
        ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
        turns_steps = {name: [0, 0] for name in ghost_names}

        for generation in range(self.generations):
            print(f"Generation {generation + 1}")

            # 각 개체의 적합도 평가
            fitness_scores = []
            for i, genes in enumerate(self.population):
                fitness = self.fitness(
                    genes, wall_list, block_list, pacman, ghost_group, directions,
                    turns_steps, screen, all_sprites_list, gate, font,
                    generation, i
                )
                fitness_scores.append(fitness)
            # 평균 적합도 출력
            avg_survival = sum(fitness_scores) / len(fitness_scores)
            print(f"Generation {generation + 1}: Avg Fitness = {avg_survival:.2f}")

            # 다음 세대 생성
            self.evolve(fitness_scores)

        # 최적의 유전자 선택
        best_genes = max(
            self.population,
            key=lambda genes: self.fitness(
                genes, wall_list, block_list, pacman, ghost_group, directions,
                turns_steps, screen, all_sprites_list, gate, font,
                self.generations-1,   # 마지막 세대 번호 (혹은 적절히 0 등)
                0                     # 개체 인덱스 (여기선 의미 없음, 0)
            )
        )
        return best_genes
