import pygame
from Game import Config

class Block(pygame.sprite.Sprite):
    """
    코인(Block) 클래스
    - 노란색(기본) 또는 주황색(유령이 지나간 후) 상태를 가질 수 있음
    """
    def __init__(self, color, width, height):
        super().__init__()
        self.width = width
        self.height = height

        # 기본 색상(대개 Config.YELLOW)
        self.base_color = color

        # 주황색으로 바꿀 색상 (Config.ORANGE가 정의되어 있다고 가정)
        # 만약 Config에 ORANGE가 없다면, (255,165,0) 같은 튜플을 직접 써도 됩니다.
        self.orange_color = getattr(Config, 'ORANGE', (255, 165, 0))

        # 현재 색상이 노랑인가 주황인가 구분하기 위한 플래그
        self.is_orange = False

        # 이미지 생성
        self.image = pygame.Surface([width, height])
        self.image.fill(Config.WHITE)
        self.image.set_colorkey(Config.WHITE)
        # 노란색 타원 그리기
        pygame.draw.ellipse(self.image, self.base_color, [0, 0, width, height])

        self.rect = self.image.get_rect()

    def make_orange(self):
        """
        유령이 이 블록 위를 지나갔을 때 호출
        - 블록 색상을 주황색으로 바꾸고, is_orange=True로 설정
        - 이후 팩맨이 이 블록을 먹으면 2배 점수를 주기 위함
        """
        if not self.is_orange:
            self.is_orange = True
            # 이미지 다시 그리기: 주황색 타원
            self.image.fill(Config.WHITE)
            self.image.set_colorkey(Config.WHITE)
            pygame.draw.ellipse(self.image, self.orange_color, [0, 0, self.width, self.height])

class Wall(pygame.sprite.Sprite):
    """
    벽(Wall) 클래스
    """
    def __init__(self, x, y, width, height, color=Config.BLUE):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Player(pygame.sprite.Sprite):
    """
    팩맨(Player) 클래스
    - 한 블록(30px)씩 격자 단위로 이동하도록 구현
    """
    def __init__(self, x, y, image_path):
        super().__init__()
        raw_image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(raw_image, (30, 30))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.block_size = 30

    def move_block(self, direction, walls, gate):
        """
        한 번 호출 시 30px만큼 해당 방향으로 이동 후 충돌 처리
        Args:
            direction (str): 'UP','DOWN','LEFT','RIGHT'
            walls (Group): 벽 스프라이트 그룹
            gate  (Group or None): 게이트 스프라이트 그룹 (없으면 None)
        """
        old_x, old_y = self.rect.x, self.rect.y

        if direction == 'UP':
            self.rect.y -= self.block_size
        elif direction == 'DOWN':
            self.rect.y += self.block_size
        elif direction == 'LEFT':
            self.rect.x -= self.block_size
        elif direction == 'RIGHT':
            self.rect.x += self.block_size

        # 벽 충돌 시 원위치
        if pygame.sprite.spritecollideany(self, walls):
            self.rect.x, self.rect.y = old_x, old_y

        # 게이트 충돌 시 원위치
        if gate is not None and pygame.sprite.spritecollideany(self, gate):
            self.rect.x, self.rect.y = old_x, old_y

class Ghost(pygame.sprite.Sprite):
    """
    유령(Ghost) 클래스
    """
    def __init__(self, x, y, image_path):
        super().__init__()
        raw_image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(raw_image, (30, 30))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.speed_x = 0
        self.speed_y = 0

    def changespeed(self, directions_list, ghost_name, turn, steps, length):
        # directions_list: [(dx, dy, step_count), ...]
        dx, dy, step_count = directions_list[turn]
        self.speed_x = dx
        self.speed_y = dy
        steps += 1
        if steps > step_count:
            turn = (turn + 1) % (length + 1)
            steps = 0
        return turn, steps

    def update(self, walls, gate):
        # (1) 이동 전 위치 보관
        old_x, old_y = self.rect.x, self.rect.y

        # (2) X 방향 이동 및 벽 충돌 처리
        self.rect.x += self.speed_x
        collided_walls = pygame.sprite.spritecollide(self, walls, False)
        if collided_walls:
            # 벽과 충돌 → 원위치로 복원
            self.rect.x, self.rect.y = old_x, old_y
            self.speed_x = 0
            # Y 이동은 하지 않도록 return 해도 됩니다.
            return

        # (3) Y 방향 이동 및 벽 충돌 처리
        self.rect.y += self.speed_y
        collided_walls = pygame.sprite.spritecollide(self, walls, False)
        if collided_walls:
            # 벽과 충돌 → 원위치로 복원
            self.rect.x, self.rect.y = old_x, old_y
            self.speed_y = 0
            return

        # (4) 게이트와 충돌하면 무조건 원위치로 복원
        #     → 펜 안으로 재진입하지 못하게 막는다
        if gate is not None and pygame.sprite.spritecollideany(self, gate):
            self.rect.x, self.rect.y = old_x, old_y
            # 속도도 0으로 만들면 게이트 앞에서 멈추게 됩니다.
            self.speed_x, self.speed_y = 0, 0
            return