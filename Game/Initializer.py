from Game.Sprites import Block
import pygame

def initialize_blocks(
    all_sprites_list: pygame.sprite.RenderPlain,
    wall_list:        pygame.sprite.RenderPlain
) -> pygame.sprite.RenderPlain:
    """
    블록(코인)을 초기화하는 함수

    Args:
        all_sprites_list: 모든 스프라이트를 관리하는 그룹
        wall_list:        벽 스프라이트가 담긴 그룹

    Returns:
        pygame.sprite.RenderPlain: 새로 생성한 블록(코인) 그룹
    """
    block_list = pygame.sprite.RenderPlain()

    GRID_SIZE = 19
    CELL_SIZE = 30
    MARGIN    = 6
    OFFSET    = 26

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # 중앙 영역 제외
            if (row, col) in [(7,8), (7,9), (7,10), (8,8), (8,9), (8,10)]:
                continue

            block = Block((255, 255, 0), 4, 4)
            block.rect.x = (CELL_SIZE * col + MARGIN) + OFFSET
            block.rect.y = (CELL_SIZE * row + MARGIN) + OFFSET

            if not pygame.sprite.spritecollide(block, wall_list, False):
                block_list.add(block)
                all_sprites_list.add(block)

    return block_list