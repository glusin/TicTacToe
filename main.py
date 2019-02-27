from collections import deque

from game import Game
from agent import Agent


if __name__ == '__main__':
    BOARD_SIZE = (3, 3)  # dimensions of the tic-tac-toe board
    IN_ROW_TO_WIN = 3  # how many xs or os in a row (horizontally, laterally or diagonally) player need to win
    game = Game(BOARD_SIZE, IN_ROW_TO_WIN)  # we create the game
    player = Agent(game.human_player_number, BOARD_SIZE)  # and initialize our Agent
    scores = deque(maxlen=1000)  # keeping track of 1000 last scores

    for i in range(1, 10_000_000):
        # training every 100 games
        if i % 100 == 0:
            print(i, '\t', player.epsilon, '\t', round(sum(scores) / len(scores) * 100, 2))
            player.learn()
        game.restart()  # we need to start new game
        done_ = False
        while not done_:
            state_ = game.state()
            action_ = player.action(state_)
            state_, next_state_, reward_, done_ = game.action(*action_)
            action_ = action_[1] * BOARD_SIZE[0] + action_[0]
            player.remember(state_, action_, next_state_, reward_, done_)

        if game.winner == player.number:
            scores.append(1)
        else:
            scores.append(0)
