from Game import Pacman
from Game import Pacman_Q
import pygame


def main():
    print("=== Pacman AI 실행 ===")
    print("1. 유전 알고리즘 (GA) 실행")
    print("2. Q-러닝 실행")
    choice = input("실행할 방식을 선택하세요 (1, 2): ")

    if choice == "1":
        Pacman.start_game()
    elif choice == "2":
        Pacman_Q.start_game()
    else:
        print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")

if __name__ == "__main__":
    main()
