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

#BFS on Graph
from collections import deque 

def graph_bfs(graph,start=0):
    size=len(graph)

    Q=deque()
    Q.append(start)
    
    dist=[10**20]*size
    dist[start]=0

    while len(Q):
        now=Q.popleft()
        for nex in graph[now]:            
            if dist[nex]<=dist[now]+1:
                continue

            dist[nex]=dist[now]+1
            Q.append(nex)

    return dist

##BFS on Grid
from collections import deque 

def grid_bfs(field,sx,sy):
    Q=deque()
    Q.append((sx,sy))

    dist=[[10**20]*w for _ in range(h)]
    dist[sx][sy]=0

    while len(Q):
        now=Q.popleft()
        x,y=now[0],now[1]

        for dx,dy in ((-1,0),(0,1),(1,0),(0,-1)):
            if not(0<=x+dx<h) or not(0<=y+dy<w):
                continue
            
            if field[x+dx][y+dy]=='#':
                continue
            
            if dist[x+dx][y+dy]<=dist[x][y]+1:
                continue

            dist[x+dx][y+dy]=dist[x][y]+1
            Q.append((x+dx,y+dy))
    
    return dist

#DFS
import sys
import pypyjit
sys.setrecursionlimit(300000)
pypyjit.set_param('max_unroll_recursion=-1')

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
    d=max(3,(r-l))

    if ##value of x <= value of y:
        r=r-d//3
    else:
       l=l+d//3
    cnt-=1

#matrix power
def matrix_pow(A,x,mod=998244353):
    n=len(A)
    res=make_vec(n,n,0)
    for i in range(n):
        res[i][i]=1
    
    while x>0:
        cur=make_vec(n,n,0)
        if x&1:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        cur[i][j]+=res[i][k]*A[k][j]
                        cur[i][j]%=mod
            
            for i in range(n):
                for j in range(n):
                    res[i][j]=cur[i][j]
                    cur[i][j]=0
        
        x//=2
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cur[i][j]+=A[i][k]*A[k][j]
                    cur[i][j]%=mod

        for i in range(n):
            for j in range(n):
                A[i][j]=cur[i][j]
    
    return res

##topological sort 
##O(N), not lexicographic
def topological_sort(G):
    size=len(G)
    ind=[0]*size

    for i in range(size):
        for x in G[i]:
            ind[x]+=1
    
    Q=deque()
    for i in range(size):
        if ind[i]==0:
            Q.append(i)
    
    heapq.heapify(Q)
    res=[]
    while len(Q)>0:
        now=Q.popleft()
        res.append(now)
        for nex in G[now]:
            ind[nex]-=1
            if ind[nex]==0:
                Q.append(nex)
    
    return res


##topological sort
##O(NlogN), lexicographic
import heapq

def ordered_topological_sort(G,asc=True):
    size=len(G)
    ind=[0]*size

    for i in range(size):
        for x in G[i]:
            ind[x]+=1
    
    Q=[]
    for i in range(size):
        if ind[i]==0:
            Q.append(i)
    
    heapq.heapify(Q)
    res=[]
    while len(Q)>0:
        now=-heapq.heappop(Q)
        now*=(-1)**asc
        res.append(now)
        for nex in G[now]:
            ind[nex]-=1
            if ind[nex]==0:
                heapq.heappush(Q,-nex*(-1)**asc)
    
    return res