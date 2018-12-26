#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2015 Ankit Aggarwal <ankitaggarwal011@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Name: 2048 Console v1.0
# Author Name: Ankit Aggarwal
# Author Email: ankitaggarwal011@gmail.com

import math,random,time
import tensorflow as tf
import numpy as np

class Memory:
    def __init__(self, size):
        self.size = size
        self.index = 0
        self.is_full = False
        self.s = np.empty([size, 4, 4],dtype=float)
        self.a = np.empty([size], dtype=int)
        self.r = np.empty([size])
        self.sp = np.empty([size, 4, 4],dtype=float)
        self.is_done = np.empty([size], dtype=bool)

    def push(self, s, a, r, sp, is_done):
        self.s[self.index] = s
        self.a[self.index] = a
        self.r[self.index] = r
        self.sp[self.index] = sp
        self.is_done[self.index] = is_done
        self.index = self.index + 1
        if self.index == self.size:
            self.is_full = True
            self.index = 0

    def isFull(self):
        return self.is_full

    def getBatch(self, batch_size):
        idx = np.random.choice(np.arange(self.size), batch_size)
        return np.take(self.s, idx, 0), np.take(self.a, idx), np.take(self.r, idx), np.take(self.sp, idx, 0), np.take(self.is_done, idx)
# Rotates a 2D list clockwise
def rotate(grid):
    return list(map(list, zip(*grid[::-1])))

# Implements game logic 
# Generalized for all four directions using rotation logic
def move(grid, dir):
    for i in range(dir): grid = rotate(grid)
    for i in range(4): #len(grid)):
        temp = []
        k=0
        for j in grid[i]:
            if j != 0: temp.append(j)
            else: k+=1
        temp += [0] * k
        for j in range(3): #len(temp) - 1):
            if temp[j] == temp[j + 1] and temp[j] != 0:
                tmp=temp[j]+1
                temp[j] = tmp
                move.score += 2**tmp
                temp[j + 1] = 0
        grid[i] = []
        k=0
        for j in temp:
            if j != 0: grid[i].append(j)
            else: k+=1
        grid[i] += [0] * k
    for i in range(4 - dir): grid = rotate(grid)
    return grid

# Finds empty slot in the game grid
def findEmptySlotC(grid):
    c=0
    for i in range(4):
        for j in range(4):
            if grid[i][j] == 0:
                c+=1
    return c
def findEmptySlotN(grid,c):
    a=1
    for i in range(4):
        for j in range(4):
            if grid[i][j] == 0:
                if a!=c: a+=1
                else: return (i, j)
    return (-1, -1)


# Adds a random number to the grid
def addNumber(grid):
    tmp = random.randint(1, 10) 
    if tmp==10: num=2
    else: num=1
    n=findEmptySlotC(grid)
    if n==0: return (grid,1)
    x,y=findEmptySlotN(grid,random.randint(1,n))
    grid[x][y] = num
    return (grid, 0)

# Prints the current game state
def printGrid(grid):
    print("\n")
    for i in range(len(grid)):
        res = "\t\t"
        for j in range(len(grid[i])):
            for _ in range(5 - len(str(2**grid[i][j]))): res += " "
            if grid[i][j]!=0: res += str(2**grid[i][j]) + " "
            else: res+=". "
        print(res)
        print("\n")
    return 0

mem=Memory(262144)
batch=128
eps_append=0.99
eps_decay=0.9997 #0.99999
eps_base=0.01
dis=0.9
lr=0.0001
qt_update_int=64
double_dqn=False
sep_eps=False
# exp
outer_eps_base=0.05
outer_eps_app=0.85
outer_eps_dec=0.998
# linear
inner_eps_init_frac=0.1
inner_eps_inc_dur=1
inner_eps_inc_dur_int=10

train_counter=0
print_counter=0

# Q-value
x=tf.placeholder(tf.float32,[None,4,4])
xp=tf.reshape(x,[-1,4,4,1])
cb=tf.Variable(tf.zeros([128]))
cw=tf.Variable(tf.truncated_normal([2,2,1,128],stddev=math.sqrt(2.0/(2*2*1+128))))
co=tf.nn.relu(tf.nn.conv2d(xp,cw,[1,1,1,1],'VALID')+cb)
cb2=tf.Variable(tf.zeros([64]))
cw2=tf.Variable(tf.truncated_normal([2,2,128,64],stddev=math.sqrt(2.0/(2*2*128+64))))
co2=tf.nn.relu(tf.nn.conv2d(co,cw2,[1,1,1,1],'VALID')+cb2)
xf=tf.reshape(co2,[-1,4*64])
b=tf.Variable(tf.zeros([128]))
w=tf.Variable(tf.truncated_normal([256,128],stddev=math.sqrt(2.0/(256+128))))
o=tf.nn.relu(tf.matmul(xf,w)+b)
#b1=tf.Variable(tf.zeros([32]))
#w1=tf.Variable(tf.truncated_normal([128,32],stddev=math.sqrt(2.0/(128+32))))
#o1=tf.nn.relu(tf.matmul(o,w1)+b1)
b2=tf.Variable(tf.zeros([4]))
w2=tf.Variable(tf.truncated_normal([128,4],stddev=math.sqrt(2.0/(128+4))))
y=tf.matmul(o,w2)+b2
yp=tf.placeholder(tf.float32,[None,4])
ymax=tf.reduce_max(y,1)
argmax=tf.argmax(y,1)
loss=tf.losses.mean_squared_error(yp,y)
step=tf.train.RMSPropOptimizer(lr).minimize(loss)

# Q-target
qt_x=tf.placeholder(tf.float32,[None,4,4])
qt_xp=tf.reshape(qt_x,[-1,4,4,1])
qt_cb=tf.Variable(cb.initialized_value())
qt_cw=tf.Variable(cw.initialized_value())
qt_co=tf.nn.relu(tf.nn.conv2d(qt_xp,qt_cw,[1,1,1,1],'VALID')+qt_cb)
qt_cb2=tf.Variable(cb2.initialized_value())
qt_cw2=tf.Variable(cw2.initialized_value())
qt_co2=tf.nn.relu(tf.nn.conv2d(qt_co,qt_cw2,[1,1,1,1],'VALID')+qt_cb2)
qt_xf=tf.reshape(qt_co2,[-1,4*64])
qt_b=tf.Variable(b.initialized_value())
qt_w=tf.Variable(w.initialized_value())
qt_o=tf.nn.relu(tf.matmul(qt_xf,qt_w)+qt_b)
#qt_b1=tf.Variable(b1.initialized_value())
#qt_w1=tf.Variable(w1.initialized_value())
#qt_o1=tf.nn.relu(tf.matmul(qt_o,qt_w1)+qt_b1)
qt_b2=tf.Variable(b2.initialized_value())
qt_w2=tf.Variable(w2.initialized_value())
qt_y=tf.matmul(qt_o,qt_w2)+qt_b2
qt_ymax=tf.reduce_max(qt_y,1)
qt_update=[qt_cb.assign(cb),qt_cw.assign(cw),
        qt_cb2.assign(cb2),qt_cw2.assign(cw2),
        qt_b.assign(b),qt_w.assign(w),
#        qt_b1.assign(b1),qt_w1.assign(w1),
        qt_b2.assign(b2),qt_w2.assign(w2)]

sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))
saver=tf.train.Saver(max_to_keep=0)
tf.global_variables_initializer().run()

def maxt(grid):
    return 2**max(max(a) for a in grid)

def train():
    global mem,sess,train_counter,print_counter
    if not mem.isFull(): return
    for j in range(1):
        s,a,r,sp,is_done=mem.getBatch(batch)
        q_=sess.run(y,feed_dict={x:s})
        if double_dqn:
            ya=sess.run(argmax,feed_dict={x:sp})
            ynxt=sess.run(qt_y,feed_dict={qt_x:sp})
        else:
            ym=sess.run(qt_ymax,feed_dict={qt_x:sp})
        for i in range(batch):
            if is_done[i]:
                q_[i][a[i]]=r[i]
            else:
                if double_dqn:
                    q_[i][a[i]]=r[i]+dis*ynxt[i][ya[i]]
                else:
                    q_[i][a[i]]=r[i]+dis*ym[i]
        sess.run([step],feed_dict={x:s,yp:q_})
        train_counter+=1
        if train_counter==qt_update_int:
            train_counter=0
            sess.run(qt_update)
        print_counter+=1
        if print_counter==128:
            print_counter=0
            oy,lo=sess.run([y,loss],feed_dict={x:s,yp:q_})
            if not double_dqn: print(lo,q_[0],oy[0],ym[0],r[0],s[0])
            else: print(lo,q_[0],oy[0],ynxt[0][ya[0]],r[0],s[0])
    #time.sleep(1)

def fillGame():
    # Create the game grid 
    # The game should work for square grid of any size though
    grid = [[0, 1, 0, 0],
            [0, 2, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 2]]

    direction = {'L': 0, 'B': 1, 'R': 2, 'T': 3, 'X': 4}

    #printGrid(grid)
    loseStatus = 0
    move.score = 0 # Score of the user
    while True:
        #if random.random()<eps_append+eps_base:
        dir=random.randint(0,3)
        #else:
        #dir=sess.run(argmax,feed_dict={x:[cookg(grid)]})[0]
       # print(grid)
        so=grid
        ro=move.score
        grid = move(grid, dir)
        grid, loseStatus = addNumber(grid)
                #printGrid(grid)
        r=move.score-ro
        if r==0: r=-1
        else: r=math.log(r,2)
        #eps_append*=eps_decay
        if loseStatus:
            #r=0
            mem.push(so,dir,r,grid,True)
            #train()
            #print "\nGame Over"
            print("Filling score: " + str(move.score)+" max: "+str(maxt(grid)))
            #printGrid(grid)
            #time.sleep(1)
            return move.score
        else:
            mem.push(so,dir,r,grid,False)
            #train()


def testGame():
    # Create the game grid 
    # The game should work for square grid of any size though
    grid = [[0, 1, 0, 0],
            [0, 2, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 2]]
    e_step=0

    direction = {'L': 0, 'B': 1, 'R': 2, 'T': 3, 'X': 4}

    #printGrid(grid)
    loseStatus = 0
    move.score = 0 # Score of the user
    while True:
        #if random.random()<eps_append+eps_base:
        #    dir=random.randint(0,3)
        #else:
        dir=sess.run(argmax,feed_dict={x:[grid]})[0]
        e_step+=1
       # print(grid)
        #so=grid
        #ro=move.score
        grid = move(grid, dir)
        grid, loseStatus = addNumber(grid)
                #printGrid(grid)
        #eps_append*=eps_decay
        if loseStatus:
            #mem.push(s,dir,move.score-ro,grid,True)
            #train()
            #print "\nGame Over"
            print("Testing score: " + str(move.score)+ " max: "+str(maxt(grid))+" step: "+str(e_step))
            #printGrid(grid)
            #time.sleep(1)
            return (move.score,e_step)
        #else:
            #mem.push(s,dir,move.score-ro,grid,False)
            #train()

def trainGame():
    global eps_append
    # Create the game grid 
    # The game should work for square grid of any size though
    grid = [[0, 1, 0, 0],
            [0, 2, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 2]]
    e_step=0
    direction = {'L': 0, 'B': 1, 'R': 2, 'T': 3, 'X': 4}
    outer_eps=outer_eps_base+outer_eps_app
    inner_eps=outer_eps*inner_eps_init_frac
    if not sep_eps: inner_eps=eps_append+eps_base
    #print(cookg(grid))
    #printGrid(grid)
    loseStatus = 0
    move.score = 0 # Score of the user
    while True:
        if random.random()<inner_eps:
            dir=random.randint(0,3)
        else:
            dir=sess.run(argmax,feed_dict={x:[grid]})[0]
        e_step+=1
        #print(dir)
        so=grid
        ro=move.score
        grid = move(grid, dir)
        grid, loseStatus = addNumber(grid)
                #printGrid(grid)
        if not sep_eps:
            #eps_append*=eps_decay
            inner_eps=eps_append+eps_base
        else:
            if inner_eps<outer_eps: inner_eps+=outer_eps*(1-inner_eps_init_frac)/inner_eps_inc_dur
        r=move.score-ro
        if r==0: r=-1
        else: r=math.log(r,2)
        if loseStatus:
            #r=0
            mem.push(so,dir,r,grid,True)
            #train()
            #print "\nGame Over"
            print("Training score: " + str(move.score) +" max: "+str(maxt(grid))+ " eps: "+str(inner_eps)+" step: "+str(e_step))
            #printGrid(grid)
            #time.sleep(1)
            return (move.score,e_step)
        else:
            mem.push(so,dir,r,grid,False)
            #train()

'''                
# Starts the game
def startGame():
    print "\nWelcome to the 2048 Console world. Let's play!"
    print "Combine given numbers to get a maximum score.\nYou can move numbers to left, right, top or bottom direction.\n"
    
    # Create the game grid 
    # The game should work for square grid of any size though
    grid = [['.', '2', '.', '.'],
            ['.', '4', '.', '2'],
            ['.', '.', '.', '.'],
            ['2', '.', '2', '4']]

    direction = {'L': 0, 'B': 1, 'R': 2, 'T': 3, 'X': 4}

    printGrid(grid)
    loseStatus = 0
    move.score = 0 # Score of the user
    while True:
        tmp = raw_input("\nTo continue, Press L for left, R for right, T for top, B for bottom or\nPress X to end the game.\n")
        if tmp in ["R", "r", "L", "l", "T", "t", "B", "b", "X", "x"]:
            dir = direction[tmp.upper()]
            if dir == 4:
                print "\nFinal score: " + str(move.score)
                break
            else:
                grid = move(grid, dir)
                grid, loseStatus = addNumber(grid)
                printGrid(grid)
                if loseStatus:
                    print "\nGame Over"
                    print "Final score: " + str(move.score)
                    break
                print "\nCurrent score: " + str(move.score)
        else:
            print "\nInvalid direction, please provide valid movement direction (L, B, R, T)."
    return 0
'''
# Program starts here
#startGame()
while not mem.isFull(): fillGame()
print("Filled")
import sys
for i in range(200):
    t=0
    s=0
    for j in range(100):
        (tmp1,tmp2)=trainGame()
        t+=tmp1
        s+=tmp2
        for k in range(128):
            train()
        eps_append*=eps_decay
        outer_eps_app*=outer_eps_dec
        if (i*100+j)%inner_eps_inc_dur_int==0: inner_eps_inc_dur+=1
    print("train "+str(i)+" avg: "+str(t/100)+" step: "+str(s/100))
    sys.stdout.flush()
    #time.sleep(3)
    if i%2==0: continue
    t=0
    s=0
    for j in range(100):
        (tmp1,tmp2)=testGame()
        t+=tmp1
        s+=tmp2
    fname=str(int(time.time()))+'.'+str(i)
    #fname=str(i)
    print("test "+fname+" avg: "+str(t/100)+" step: "+str(s/100))
    if t/100>16000: saver.save(sess, './model.'+fname+'.ckpt')
    sys.stdout.flush()
    #time.sleep(3)


