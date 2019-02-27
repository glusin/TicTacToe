import random

import numpy as np
import tabulate


class IllegalMoveError(Exception):
    def __init__(self, message):
        super().__init__(message)


class GameOverError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Board:
    def __init__(self, board_size, in_row_to_win):
        self.in_row_to_win = in_row_to_win
        self.board_size = board_size  # board_size = (x, y)
        self.map = self.generate_map()
        self.current_player = 0
        self._winner = None
        self.has_game_ended = False

    def generate_map(self):
        return np.full(list(reversed(self.board_size)), None)

    def state(self):
        return [[element for element in row] for row in self.map]  # list comprehension in order to pass copy of the map

    def is_legal_move(self, x, y):
        return self.map[y][x] is None

    def move(self, player, x, y):
        if self.has_game_ended:
            raise GameOverError('The game has ended')

        if self.is_legal_move(x, y):
            self.map[y][x] = player
        else:
            raise IllegalMoveError(f'Player {player} tried to make illegal move - x: {x}, y: {y}')

    def print(self):
        print(tabulate.tabulate(self.map, tablefmt='grid'))

    def is_game_over(self):
        if self._check_lateral() or self._check_horizontal() or self._check_diagonal() or self._no_moves_left():
            self.has_game_ended = True
            return True
        else:
            return False

    def _check_sequence_for_winner(self, sequence):
        in_a_row = 0
        current_player = None
        for field_owner in sequence:
            if field_owner is None:
                in_a_row = 0
                current_player = None
                continue

            if field_owner == current_player:
                in_a_row += 1
                if in_a_row >= self.in_row_to_win:
                    return current_player
            else:
                current_player = field_owner
                in_a_row = 1
        else:
            return None

    def _check_lateral(self):
        for column in self.map.transpose():
            winner = self._check_sequence_for_winner(column)
            if winner is not None:
                self._winner = winner
                return True
        else:
            return False

    def _check_horizontal(self):
        for row in self.map:
            winner = self._check_sequence_for_winner(row)
            if winner is not None:
                self._winner = winner
                return True
        else:
            return False

    def _check_diagonal(self):
        diags = [self.map[::-1, :].diagonal(i) for i in range(-self.map.shape[0] + 1, self.map.shape[1])]
        diags.extend(self.map.diagonal(i) for i in range(self.map.shape[1] - 1, -self.map.shape[0], -1))
        for diagonal in diags:
            winner = self._check_sequence_for_winner(diagonal)
            if winner is not None:
                self._winner = winner
                return True
        else:
            return False

    def _no_moves_left(self):
        return None not in self.map.reshape(-1)

    @property
    def winner(self):
        if self._winner is None:
            return 'draw'
        else:
            return self._winner


class RandomPlayer:
    def __init__(self, number):
        self.number = number

    @staticmethod
    def action(state):
        legal_moves = []
        for row_num in range(len(state)):
            for col_num in range(len(state[0])):
                if state[row_num][col_num] is None:
                    legal_moves.append((col_num, row_num))

        return random.choice(legal_moves)


class Game:
    random_player = RandomPlayer(0)

    def __init__(self, board_size, in_row_to_win):
        self.in_row_to_win = in_row_to_win
        self.current_player = 0
        self.human_player_number = 1
        self.board_size = board_size
        self.board = Board(board_size, in_row_to_win)
        # Random player makes move
        self._make_move(*self.random_player.action(self.board.state()))

    def state(self):
        return self.board.state()

    def restart(self):
        self.current_player = 0
        self.board = Board(self.board_size, self.in_row_to_win)
        # Random player makes move
        self._make_move(*self.random_player.action(self.board.state()))

    def _make_move(self, x, y):
        self.board.move(self.current_player, x, y)
        self.current_player = abs(self.current_player - 1)
        return self.state()

    def is_legal_move(self, x, y):
        return self.board.is_legal_move(x, y)

    def reward(self):
        if self.board.is_game_over():
            if self.board.winner == abs(self.human_player_number - 1):
                return -10
            elif self.board.winner == self.human_player_number:
                return 10
            else:
                # draw
                return 0
        else:
            return 1

    @property
    def winner(self):
        return self.board.winner

    def action(self, x, y):
        done = False
        state = self.board.state()
        self._make_move(x, y)
        if not self.board.is_game_over():
            # Random player makes move
            self._make_move(*self.random_player.action(self.board.state()))

        if self.board.is_game_over():
            next_state = None
            done = True
        else:
            next_state = self.board.state()

        return state, next_state, self.reward(), done









