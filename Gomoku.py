import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import multiprocessing
import time
import math
import random
from collections import namedtuple
import matplotlib.pyplot as plt
#from sys import exit
random.seed(0)
torch.manual_seed(0)

Transition = namedtuple('Transition', ('prevState', 'prevAction', 'state', 'prevReward'))

def main():
    g = Gomoku(visualize=0)
    g.train()
#    g.display()

class Gomoku:
    def __init__(self, visualize=0):
        self.batchSize = 1024
        self.gamma = 0.999
        self.blackView, self.whiteView = torch.zeros(15, 15), torch.zeros(15, 15)
        self.thresStart, self.thresEnd, self.thresDecay = 0.99, 0.01, 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        if visualize:
#            self.show = None
#            if self.show:
#                self.stop()
#            self.show = multiprocessing.Process(target=self.display)
#            self.show.start()

    def display(self):
        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("五子棋")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(r"C:\Windows\Fonts\consola.ttf", 24)
        self.going = True

        self.chessboard = Chessboard()
        self.chessboard.grid = (self.blackView - self.whiteView).tolist()
        self.loopDisplay()
    
    def train(self):
        self.net = Net().to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
        self.memory = replayMemory(20000)
        start = time.time()
        losses = []
        for episode in range(10000):
            self.moveThres = self.thresEnd + (self.thresStart - self.thresEnd) * math.exp(-episode/self.thresDecay)
            blackView, whiteView = self.playGame()
            loss = self.optimize(iterate=1)
            if loss is not None:
                losses.append(loss)
            if (episode+1) % 10 == 0:
                print('Episode %d' % (episode+1))
                if loss is not None:
                    print('Loss: %.4f' % loss)
        end = time.time()
        print('Time: %.f s' % (end-start))
        self.blackView = blackView.to('cpu')
        self.whiteView = whiteView.to('cpu')
        torch.save(self.net.state_dict(), 'gomoku.pt')
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(losses, '-')
        ax.set_xlabel('Optimize Step')
        ax.set_ylabel('Loss')

    def playGame(self):
        blackView = torch.zeros(15,15).to(self.device)
        whiteView = torch.zeros(15,15).to(self.device)
        episodeEnd = False
        blackMove = True
        moveCount = 0
        prevBlackState, prevBlackAction, prevBlackReward = None, None, None
        prevWhiteState, prevWhiteAction, prevWhiteReward = None, None, None
        while not episodeEnd:
            state = torch.stack([torch.unsqueeze(blackView, 0), torch.unsqueeze(whiteView, 0)] \
                                 if blackMove else [torch.unsqueeze(whiteView, 0),  torch.unsqueeze(blackView, 0)], dim=1)
            sample = random.random()
            if sample > self.moveThres: # use network to select move.
                with torch.no_grad():
                    values = self.net.forward(state)
                values = values.view(-1, 15, 15) * (1 - blackView - whiteView) - (blackView + whiteView)
                values = torch.squeeze(values)
                index = torch.argmax(values).item()
                moveRow, moveCol = index // 15, index % 15
                if blackView[moveRow, moveCol] == 1 or whiteView[moveRow, moveCol] == 1:
                    print('Conflict!')
            else: # random move.
                exist = blackView + whiteView
                possible = (exist==0).nonzero()
                moveRow, moveCol = possible[torch.randint(0, possible.size(0), (1, 1)).item()]
                moveRow, moveCol = moveRow.item(), moveCol.item()
            action = torch.tensor([[moveRow * 15 + moveCol]], device=self.device)
            reward = 0
            if blackMove:
                blackView[moveRow, moveCol] = 1
                blackMove = False
                if self.checkWin(blackView, moveRow, moveCol):
                    reward = 1
                    episodeEnd = True
            else:
                whiteView[moveRow, moveCol] = 1
                blackMove = True
                if self.checkWin(whiteView, moveRow, moveCol):
                    reward = 1
                    episodeEnd = True
            moveCount += 1
            if moveCount >= 15**2 - 1 and not episodeEnd:
                reward = 0.1 # both color get reward if they survive to this point.
                if moveCount == 15**2:
                    print('Game draw.')
                    episodeEnd = True
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            if blackMove:
                if prevBlackState is not None:
                    self.memory.push(prevBlackState, prevBlackAction, state, prevBlackReward)
                prevBlackState = state
                prevBlackAction = action
                prevBlackReward = reward
            else:
                if prevWhiteState is not None:
                    self.memory.push(prevWhiteState, prevWhiteAction, state, prevWhiteReward)
                prevWhiteState = state
                prevWhiteAction = action
                prevWhiteReward = reward
        return blackView, whiteView

    def checkWin(self, selfView, r, c):
        n_count = self.getContinuousNum(selfView, r, c, -1, 0)
        s_count = self.getContinuousNum(selfView, r, c, 1, 0)

        e_count = self.getContinuousNum(selfView, r, c, 0, 1)
        w_count = self.getContinuousNum(selfView, r, c, 0, -1)

        se_count = self.getContinuousNum(selfView, r, c, 1, 1)
        nw_count = self.getContinuousNum(selfView, r, c, -1, -1)

        ne_count = self.getContinuousNum(selfView, r, c, -1, 1)
        sw_count = self.getContinuousNum(selfView, r, c, 1, -1)

        if (n_count + s_count + 1 >= 5) or (e_count + w_count + 1 >= 5) or \
                (se_count + nw_count + 1 >= 5) or (ne_count + sw_count + 1 >= 5):
            return True
        else:
            return False

    def getContinuousNum(self, selfView, r, c, dr, dc):
        piece = selfView[r, c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < 15 and 0 <= new_c < 15:
                if selfView[new_r, new_c] == piece:
                    result += 1
                else:
                    break
            else:
                break
            i += 1
        return result

    def optimize(self, iterate=1):
        for i in range(iterate):
            if len(self.memory) < self.batchSize:
                return
            transitions = self.memory.sample(self.batchSize)
            batch = Transition(*zip(*transitions))
            prevStateBatch = torch.cat(batch.prevState)
            prevActionBatch = torch.cat(batch.prevAction)
            stateBatch = torch.cat(batch.state)
            prevRewardBatch = torch.cat(batch.prevReward)
            prevValues = self.net(prevStateBatch).gather(1, prevActionBatch)
            with torch.no_grad():
                stateValues = self.net(stateBatch).max(1)[0]
            expectedValues = (stateValues * self.gamma) + prevRewardBatch
            loss = F.smooth_l1_loss(prevValues, expectedValues.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        return loss

    def loopDisplay(self):
        while self.going:
            self.update()
            self.draw()
            self.clock.tick(24)

        pygame.quit()
        sys.exit(0)

    def update(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.chessboard.handle_key_event(e)

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.font.render("FPS: {0:.2F}".format(self.clock.get_fps()), True, (0, 0, 0)), (10, 10))

        self.chessboard.draw(self.screen)
        if self.chessboard.game_over:
            self.screen.blit(self.font.render("{0} Win".format("Black" if self.chessboard.winner == 'b' else "White"), True, (0, 0, 0)), (500, 10))

        pygame.display.update()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 16, 1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(5*5*16, 15**2)
        
#        self.conv4 = nn.Conv2d(64, 4, 1)
#        self.bn3 = torch.nn.BatchNorm2d(4)
#        self.fc2 = nn.Linear(5*5*4, 128)
#        self.bn4 = torch.nn.BatchNorm1d(128)
#        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = x.view(-1, 5*5*16)
        x = F.relu(self.fc1(x))
#        value = F.relu(self.bn3(self.conv4(x)))
#        value = value.view(-1, 5*5*4)
#        value = F.relu(self.fc2(value))
#        value = (torch.tanh(self.fc3(value)) + 1) / 2
        return x

class replayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Chessboard:

    def __init__(self):
        self.grid_size = 26
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.grid_count = 15
        self.piece = 1
        self.winner = None
        self.game_over = False

        self.grid = []
        for i in range(self.grid_count): # represents empty slot.
            self.grid.append([0] * self.grid_count)

    def handle_key_event(self, e):
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.grid_count - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.game_over:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                if self.set_piece(r, c):
                    self.check_win(r, c)

    def set_piece(self, r, c):
        if self.grid[r][c] == 0:
            self.grid[r][c] = self.piece

            if self.piece == 1:
                self.piece = -1
            else:
                self.piece = 1

            return True
        return False

    def check_win(self, r, c):
        n_count = self.get_continuous_count(r, c, -1, 0)
        s_count = self.get_continuous_count(r, c, 1, 0)

        e_count = self.get_continuous_count(r, c, 0, 1)
        w_count = self.get_continuous_count(r, c, 0, -1)

        se_count = self.get_continuous_count(r, c, 1, 1)
        nw_count = self.get_continuous_count(r, c, -1, -1)

        ne_count = self.get_continuous_count(r, c, -1, 1)
        sw_count = self.get_continuous_count(r, c, 1, -1)

        if (n_count + s_count + 1 >= 5) or (e_count + w_count + 1 >= 5) or \
                (se_count + nw_count + 1 >= 5) or (ne_count + sw_count + 1 >= 5):
            self.winner = self.grid[r][c]
            self.game_over = True

    def get_continuous_count(self, r, c, dr, dc):
        piece = self.grid[r][c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < self.grid_count and 0 <= new_c < self.grid_count:
                if self.grid[new_r][new_c] == piece:
                    result += 1
                else:
                    break
            else:
                break
            i += 1
        return result

    def draw(self, screen):
        # 棋盤底色
        pygame.draw.rect(screen, (185, 122, 87),
                         [self.start_x - self.edge_size, self.start_y - self.edge_size,
                          (self.grid_count - 1) * self.grid_size + self.edge_size * 2, (self.grid_count - 1) * self.grid_size + self.edge_size * 2], 0)

        for r in range(self.grid_count):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y], [self.start_x + self.grid_size * (self.grid_count - 1), y], 2)

        for c in range(self.grid_count):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y], [x, self.start_y + self.grid_size * (self.grid_count - 1)], 2)

        for r in range(self.grid_count):
            for c in range(self.grid_count):
                piece = self.grid[r][c]
                if piece != 0:
                    if piece == 1:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)

                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(screen, color, [x, y], self.grid_size // 2)

if __name__ == '__main__':
    main()
    
