##divisor func.
def div(n):
    res=[]
    i=int(1)
    while i*i<=n:
        if n%i==0:
            res.append(i)
            if n!=i**2:
                res.append(n//i)
        i+=1
    res.sort()
    return res

##prime factors
def fac(n):
    res=set()
    i=int(2)
    now=n
    while i*i<=n:
        while now%i==0:
            now//=i
            res.add(i)
        i+=1
    if now!=1:
        res.add(now)
    return sorted(list(res))

##create array
def make_vec(*dims):
    n = len(dims) - 1
    initial = dims[-1]
    code = "[" * n + "{}] * {}" + " for _ in range({})]" * (n - 1)
    return eval(code.format(initial, *reversed(dims[:n])))


##create graph
def graph(n):
    res=[[]for _ in range(n)]
    return res

##least common multiple
import math
def lcm(a,b):
    return a//math.gcd(a,b)*b

#tousa suretsu sum
def tousa_sum(a,d,n):
    return (a+((n-1)*d+a))*n//2

##neighbor search
for i in range(h):
    for j in range(w):
        for x in range(max(0,i-1),min(h,(i+1)+1)):
            for y in range(max(0,j-1),min(w,(j+1)+1)):

##float to int
def toint(s):
    n=len(s)
    pos=0
    for i in range(n):
        if s[i]=='.':
            pos=-(n-1-i)
    s=s.replace('.','')
    ans=s+' '+str(pos)
    return ans

##input func
import sys
input = sys.stdin.readline
LI = lambda:list(map(int,input().split()))
LI0 = lambda:list(map((lambda x:int(x)-1),input().split()))
MI = lambda:map(int, input().split())



##random number generator
import random
rnd=random.randrange()

##string easy sort
def SORT(s):
    return ''.join(sorted(s))
def REV(s):
    return ''.join(reversed(s))


##Prime checker
def isprime(n):
    if n>=10**6:
        if len(div(n))==2:
            return True
        else:
            return False
    else:
        if n==1 or sieve[n]!=n:
            return False
        else:
            return True

##BFS on Grid
from collections import deque 

def grid_bfs(field,sx,sy):
    Q=deque()
    Q.append((sx,sy))

    dist=[[10**20]*w for _ in range(h)]
    dist[sx][sy]=0

    check=[[0]*w for _ in range(h)]
    check[sx][sy]=1

    while len(Q):
        now=Q.popleft()
        x,y=now[0],now[1]

        for dx,dy in ((-1,0),(0,1),(1,0),(0,-1)):
            if not(0<=x+dx<h) or not(0<=y+dy<w):
                continue
            
            if field[x+dx][y+dy]=='#':
                continue
            
            if check[x+dx][y+dy]:
                continue
            
            dist[x+dx][y+dy]=dist[x][y]+1
            Q.append((x+dx,y+dy))
            check[x+dx][y+dy]=1
    
    return dist

#BFS on Graph
from collections import deque 

def graph_bfs(graph,start):
    sz=len(graph)

    Q=deque()
    Q.append(start)
    
    dist=[10**20]*sz
    dist[start]=0

    check=set()
    check.add(start)

    while len(Q):
        now=Q.popleft()
        for i in range(len(graph[now])):
            nex=graph[now][i]
            if nex in check:
                continue
            
            dist[nex]=dist[now]+1
            Q.append(nex)
            check.add(nex)

    return dist

#DFS
import sys
sys.setrecursionlimit(300000)

def dfs(cur,pre):
    #来たときの処理
    check.add(cur)
    for nex in g[cur]:
        if nex not in check and nex!=pre:
            dfs(nex,cur)
            #帰ってきたときの処理

##Ternary Search
##Submit to PyPy!!!
cnt=100
l,r=0,
while cnt:
    x,y=(2*l+r)//3,(l+2*r)//3
    d=(r-l)
    if ##function:
        r=r-d//3
    else:
       l=l+d//3
    cnt-=1
