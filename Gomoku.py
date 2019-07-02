'''
To do:
    1 use adam or rmsprop.
'''

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
from sys import exit
random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device('cuda')
Transition = namedtuple('Transition', ('prevState', 'prevAction', 'state', 'prevReward'))

def main():
    g = Gomoku(visualize=0, saveModel=0, loadModel=1)
    g.train()
#    g.display()
#    g.test(selfplay=0, chooseBlack=1)
    del g

class Gomoku:
    def __init__(self, visualize, saveModel, loadModel):
        self.episodeNum = 150
        self.trainPerEpisode = 10
        self.batchSize = 256
        self.learningRate = 0.01
        self.gamma = 0.999
        self.memorySize = 5000
        self.thresStart, self.thresEnd, self.thresDecay = 1, 0.05, 500
        
        self.device = DEVICE
        self.blackView, self.whiteView = torch.zeros(15, 15), torch.zeros(15, 15)
        self.saveModel = saveModel
        self.loadModel = loadModel
#        if visualize:
#            self.show = None
#            if self.show:
#                self.stop()
#            self.show = multiprocessing.Process(target=self.display)
#            self.show.start()
        
    def __del__(self):
        torch.cuda.empty_cache()

    def display(self, ai=0, selfplay=0, chooseBlack=1):
        import pygame
        pygame.init()

        self.screen = pygame.display.set_mode((700, 500))
        pygame.display.set_caption("屁坨1.0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(r"C:\Windows\Fonts\consola.ttf", 24)
        self.going = True

        self.chessboard = Chessboard(chooseBlack, self.net)
        self.chessboard.grid = (self.blackView - self.whiteView).tolist()
        self.loopDisplay(ai, selfplay, chooseBlack)
    
    def train(self):
        self.net = Net().to(self.device)
        self.targetNet = Net().to(self.device)
        self.targetNet.load_state_dict(self.net.state_dict())
        self.targetNet.eval()
        if self.loadModel:
            self.net.load_state_dict(torch.load('gomoku.pt'))
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=0.01)
        self.memory = replayMemory(self.memorySize)
        self.memoryEnding = replayMemory(self.memorySize)
        start = time.time()
        losses = []
        lastTen = []
        for episode in range(self.episodeNum):
            self.moveThres = self.thresEnd + (self.thresStart - self.thresEnd) * math.exp(-episode/self.thresDecay)
            blackView, whiteView = self.playGame()
            loss = self.optimize(iterate=self.trainPerEpisode)
            if loss is not None:
                losses.append(loss)
                lastTen.append(loss)
            if (episode+1) % 10 == 0:
                print('Episode %d' % (episode+1))
            if len(lastTen) == 10:
                print('10-avg Loss: %.6f' % (sum(lastTen)/len(lastTen)))
                lastTen = []
            if self.saveModel and (episode+1) % 100 == 0:
                torch.save(self.net.state_dict(), 'gomoku.pt')
            self.targetNet.load_state_dict(self.net.state_dict())
        end = time.time()
        print('Time: %.f s' % (end-start))
        self.blackView = blackView.to('cpu')
        self.whiteView = whiteView.to('cpu')
        
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
        prevBlackState, prevBlackAction = None, None
        prevWhiteState, prevWhiteAction = None, None
        while not episodeEnd:
            state = torch.stack([torch.unsqueeze(blackView, 0), torch.unsqueeze(whiteView, 0)] \
                                 if blackMove else [torch.unsqueeze(whiteView, 0),  torch.unsqueeze(blackView, 0)], dim=1)
            sample = random.random()
            if sample > self.moveThres: # use network to select move.
                with torch.no_grad():
                    values = self.net.forward(state)
                exist = blackView + whiteView
                values = values.view(-1, 15, 15) * (1 - exist) - 100 * exist
                values = torch.squeeze(values)
                index = torch.argmax(values).item()
                moveRow, moveCol = index // 15, index % 15
            else: # random move.
                exist = blackView + whiteView
                possible = (exist==0).nonzero()
                moveRow, moveCol = possible[torch.randint(0, possible.size(0), (1, 1)).item()]
                moveRow, moveCol = moveRow.item(), moveCol.item()
            action = torch.tensor([[moveRow * 15 + moveCol]], device=self.device)
            if blackMove:
                blackView[moveRow, moveCol] = 1
                if self.checkWin(blackView, moveRow, moveCol):
                    reward = torch.tensor([1], device=self.device, dtype=torch.float)
                    self.memoryEnding.push(state, action, None, reward)
                    episodeEnd = True
                else:
                    if prevBlackState is not None:
                        prevBlackReward = torch.tensor([0], device=self.device, dtype=torch.float)
                        self.memory.push(prevBlackState, prevBlackAction, state, prevBlackReward)
                    prevBlackState = state
                    prevBlackAction = action
            else:
                whiteView[moveRow, moveCol] = 1
                if self.checkWin(whiteView, moveRow, moveCol):
                    reward = torch.tensor([1], device=self.device, dtype=torch.float)
                    self.memoryEnding.push(state, action, None, reward)
                    episodeEnd = True
                else:
                    if prevWhiteState is not None:
                        prevWhiteReward = torch.tensor([0], device=self.device, dtype=torch.float)
                        self.memory.push(prevWhiteState, prevWhiteAction, state, prevWhiteReward)
                    prevWhiteState = state
                    prevWhiteAction = action
            moveCount += 1
            if moveCount == 15**2:
                print('Game draw.')
                episodeEnd = True
            blackMove = False if blackMove else True
        return blackView, whiteView

    def optimize(self, iterate=1):
        for i in range(iterate):
            if len(self.memory) < self.batchSize//2 or len(self.memoryEnding) < self.batchSize//2:
                return
            transitions = self.memory.sample(self.batchSize//2) + self.memoryEnding.sample(self.batchSize//2)
            batch = Transition(*zip(*transitions))
            prevStateBatch = torch.cat(batch.prevState)
            prevActionBatch = torch.cat(batch.prevAction)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.state)), device=self.device, dtype=torch.uint8)
            non_final_states = torch.cat([s for s in batch.state
                                                if s is not None])
            prevRewardBatch = torch.cat(batch.prevReward)
            
            prevValues = self.net(prevStateBatch).gather(1, prevActionBatch)
            stateValues = torch.zeros(self.batchSize, device=self.device)
            stateValues[non_final_mask] = self.targetNet(non_final_states).max(1)[0]
            expectedValues = (stateValues * self.gamma) + prevRewardBatch
            loss = F.smooth_l1_loss(prevValues, expectedValues.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        return loss

    def test(self, selfplay=0, chooseBlack=1):
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load('gomoku.pt'))
        self.display(ai=1, selfplay=selfplay, chooseBlack=chooseBlack)

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

    def loopDisplay(self, ai=0, selfplay=0, chooseBlack=1):
        while self.going:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.going = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.chessboard.handle_key_event(e, ai, selfplay, chooseBlack)
            self.screen.fill((255, 255, 255))
            self.screen.blit(self.font.render("FPS: {0:.2F}".format(self.clock.get_fps()), True, (0, 0, 0)), (10, 10))
    
            self.chessboard.draw(self.screen)
            if self.chessboard.game_over:
                self.screen.blit(self.font.render("{0} Win".format("Black" if self.chessboard.winner == 1 else "White"), True, (0, 0, 0)), (500, 10))
    
            pygame.display.update()
            self.clock.tick(20)

        pygame.quit()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 32, 1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(5*5*32, 5*5*32)
        self.fc2 = nn.Linear(5*5*32, 15**2)
        
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
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        y = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(y)) + x)
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = x.view(-1, 5*5*32)
        x = F.relu(self.fc1(x))
        x = (torch.tanh(self.fc2(x)) + 1) / 2
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

    def __init__(self, chooseBlack=1, model=None):
        self.grid_size = 26
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.grid_count = 15
        self.piece = 1 if chooseBlack else -1
        self.winner = None
        self.game_over = False
        self.model = model

        self.grid = []
        for i in range(self.grid_count): # represents empty slot.
            self.grid.append([0] * self.grid_count)

    def AImove(self):
        blackView = torch.tensor(self.grid, device=DEVICE)
        blackView[blackView!=1] = 0
        whiteView = torch.tensor(self.grid, device=DEVICE)
        whiteView[whiteView!=-1] = 0
        whiteView[whiteView==-1] = 1
        state = torch.stack([torch.unsqueeze(blackView, 0), torch.unsqueeze(whiteView, 0)] \
                             if self.piece == 1 else [torch.unsqueeze(whiteView, 0),  torch.unsqueeze(blackView, 0)], dim=1)
        with torch.no_grad():
            values = self.model.forward(state)
#                        print(values)
        values = values.view(-1, 15, 15) * (1 - blackView - whiteView) - (blackView + whiteView)
        values = torch.squeeze(values)
        index = torch.argmax(values).item()
        moveRow, moveCol = index // 15, index % 15
        self.grid[moveRow][moveCol] = self.piece 
            # cannot handle draw !!
        self.check_win(moveRow, moveCol)
        self.piece = 1 if self.piece == -1 else -1

    def handle_key_event(self, e, ai, selfplay, chooseBlack):
        if selfplay:
            self.AImove()
        else:
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
                    if self.set_piece(r, c):# valid move.
                        self.check_win(r, c)
                        if ai:
                            self.AImove()
                            
    def set_piece(self, r, c):
        if self.grid[r][c] == 0:
            self.grid[r][c] = self.piece
            self.piece = 1 if self.piece == -1 else -1
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
    
