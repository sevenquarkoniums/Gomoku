'''
Gomoku.
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
import os
if os.name == 'nt':
    import pygame
#random.seed(0)
#torch.manual_seed(0)

DEVICE = torch.device('cuda')
METHOD = 'predict' # RL, predict.
if METHOD == 'RL':
    Transition = namedtuple('Transition', ('prevState', 'prevAction', 'state', 'prevReward'))
elif METHOD == 'predict':
    Transition = namedtuple('Transition', ('state', 'action'))

def main():
    g = Gomoku(visualize=0, saveModel=1, loadModel=0)
    g.train()
#    g.display()
#    g.test(selfplay=0, chooseBlack=1)
    del g

class Gomoku:
    def __init__(self, visualize, saveModel, loadModel):
        self.episodeNum = 10000
        self.trainPerEpisode = 10
        self.batchSize = 1024
        self.learningRate = 0.01
        self.gamma = 0.999
        self.memorySize = 1000
        self.thresStart, self.thresEnd, self.thresDecay = 1, 0.05, 5000
        
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
        if METHOD == 'RL':
            self.net = RLNet().to(self.device)
            self.targetNet = RLNet().to(self.device)
            self.targetNet.load_state_dict(self.net.state_dict())
            self.targetNet.eval()
        elif METHOD == 'predict':
            self.net = predictNet().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
        if self.loadModel:
            self.net.load_state_dict(torch.load('gomoku.pt'))
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=0.01)
        self.memory = replayMemory(self.memorySize)
        self.memoryWin = replayMemory(self.memorySize)
        self.memoryLose = replayMemory(self.memorySize)
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
            if METHOD == 'RL':
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
        draw = False
        moveCount = 0
        prevBlackMoveState, prevBlackAction = None, None
        prevWhiteMoveState, prevWhiteAction = None, None
        blackTrans, whiteTrans = [], []
        while not episodeEnd:
            state = torch.stack([torch.unsqueeze(blackView, 0), torch.unsqueeze(whiteView, 0)] \
                                 if blackMove else [torch.unsqueeze(whiteView, 0),  torch.unsqueeze(blackView, 0)], dim=1)
            sample = random.random()
            exist = blackView + whiteView
            if sample > self.moveThres: # use network to select move.
                with torch.no_grad():
                    values = self.net.forward(state)
                values = torch.squeeze(values)
                index = torch.argmax(values).item()
                moveRow, moveCol = index // 15, index % 15
            else: # random move.
                possible = (exist==0).nonzero()
                moveRow, moveCol = possible[torch.randint(0, possible.size(0), (1, 1)).item()]
                moveRow, moveCol = moveRow.item(), moveCol.item()
            if METHOD == 'RL':
                action = torch.tensor([[moveRow * 15 + moveCol]], device=self.device)
            elif METHOD == 'predict':
                action = torch.tensor([moveRow * 15 + moveCol], device=self.device)
            if blackMove:
                if exist[moveRow, moveCol] == 0:
                    blackView[moveRow, moveCol] = 1
                    if self.checkWin(blackView, moveRow, moveCol):
                        if METHOD == 'RL':
                            blackTrans.append((state, action, None))
                        elif METHOD == 'predict':
                            self.memoryWin.push(state, action)
                        episodeEnd = True
                        blackWin = True
                if METHOD == 'RL':
                    blackTrans.append((prevBlackMoveState, prevBlackAction, state))
                    prevBlackMoveState = state
                    prevBlackAction = action
            else:
                if exist[moveRow, moveCol] == 0:
                    whiteView[moveRow, moveCol] = 1
                    if self.checkWin(whiteView, moveRow, moveCol):
                        if METHOD == 'RL':
                            whiteTrans.append((state, action, None))
                        elif METHOD == 'predict':
                            self.memoryWin.push(state, action)
                        episodeEnd = True
                        blackWin = False
                if METHOD == 'RL':
                    whiteTrans.append((prevWhiteMoveState, prevWhiteAction, state))
                    prevWhiteMoveState = state
                    prevWhiteAction = action            
            moveCount += 1
            if moveCount == 15**2:
                print('Game draw.')
                episodeEnd = True
                draw = True
            blackMove = False if blackMove else True
            # this episode finished.
            
        if METHOD == 'RL':
            for tran in blackTrans[1:-1] + whiteTrans[1:-1]:
                reward0 = torch.tensor([0], device=self.device, dtype=torch.float)
                self.memory.push(tran[0], tran[1], tran[2], reward0)
            if not draw:
                if blackWin:
                    reward1 = torch.tensor([1], device=self.device, dtype=torch.float)
                    rewardm1 = torch.tensor([-1], device=self.device, dtype=torch.float)
                    self.memoryWin.push(blackTrans[-1][0], blackTrans[-1][1], blackTrans[-1][2], reward1)
                    self.memoryLose.push(whiteTrans[-1][0], whiteTrans[-1][1], whiteTrans[-1][2], rewardm1)
                if not blackWin:
                    reward1 = torch.tensor([1], device=self.device, dtype=torch.float)
                    rewardm1 = torch.tensor([-1], device=self.device, dtype=torch.float)
                    self.memoryLose.push(blackTrans[-1][0], blackTrans[-1][1], blackTrans[-1][2], reward1)
                    self.memoryWin.push(whiteTrans[-1][0], whiteTrans[-1][1], whiteTrans[-1][2], rewardm1)
        return blackView, whiteView

    def optimize(self, iterate=1):
        for i in range(iterate):
            if METHOD == 'RL':
                if len(self.memoryLose) < self.batchSize//3 or len(self.memoryWin) < self.batchSize//3:
                    return
                transitions = self.memory.sample(self.batchSize - self.batchSize//3 * 2) + self.memoryWin.sample(self.batchSize//3) \
                                      + self.memoryLose.sample(self.batchSize//3)
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
            elif METHOD == 'predict':
                if len(self.memoryWin) < self.batchSize:
                    return
                transitions = self.memoryWin.sample(self.batchSize)
                batch = Transition(*zip(*transitions))
                stateBatch = torch.cat(batch.state)
                expectedActionBatch = torch.cat(batch.action)
                actionBatch = self.net(stateBatch)
                loss = self.criterion(actionBatch, expectedActionBatch)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        return loss

    def test(self, selfplay=0, chooseBlack=1):
        if METHOD == 'RL':
            self.net = RLNet().to(self.device)
        elif METHOD == 'predict':
            self.net = predictNet().to(self.device)
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

class RLNet(nn.Module):
    def __init__(self):
        super(RLNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)
#        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.bn16(self.conv1(x)))
        y = F.relu(self.bn32(self.conv2(x)))
        x = F.relu(self.bn32(self.conv2_2(y)))
        y = F.relu(self.bn64(self.conv3(x)))
        x = F.relu(self.bn64(self.conv3_2(y)))
        x = self.conv4(x)
        return x.view(-1, 15*15)

class predictNet(nn.Module):
    def __init__(self):
        super(predictNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 1, 1)
        self.bn2 = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(15*15*2, 15*15)
        self.sm = nn.Softmax(1)

    def forward(self, x):
        x = F.relu(self.bn16(self.conv1(x)))
        x = F.relu(self.bn32(self.conv2(x)))
        y = F.relu(self.bn32(self.conv2_2(x)))
        x = F.relu(self.bn32(self.conv2_2(y)) + x)
        x = F.relu(self.bn64(self.conv3(x)))
        y = F.relu(self.bn64(self.conv3_2(x)))
        x = F.relu(self.bn64(self.conv3_2(y)) + x)
        policy = F.relu(self.bn2(self.conv4(x))).view(-1, 15*15*2)
        policy = self.sm(self.fc1(policy))
        return policy

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
        print(values)
#        values = values.view(-1, 15, 15) * (1 - blackView - whiteView) - (blackView + whiteView)
        values = torch.squeeze(values)
        index = torch.argmax(values).item()
        moveRow, moveCol = index // 15, index % 15
        print('%d  %d' % (moveRow, moveCol))
        if self.grid[moveRow][moveCol] == 0:
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
    
