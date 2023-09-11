'''
g[i]:=iから辺が出ている頂点の集合

constructor
引数 g:有向グラフ, root:親, 

method
lca
引数 a,b:頂点番号
返り値 aとbの最近共通祖先
'''
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

class LowestCommonAncestor():
    # n=5*10**5
    n=1
    bit=64
    # bit=4
    doubling=[]
    G=[]
    p=-1
    distance=[]

    def __init__(self,g,root):
        self.n=len(g)
        self.p=root
        self.G=g
        self.distance=graph_bfs(g,root)

        for _ in range(self.bit):
            self.doubling.append([root]*self.n)
        
        for i in range(self.n):
            for x in g[i]:
                self.doubling[0][x]=i;
    
        for j in range(1,self.bit):
            for i in range(self.n):
                now=self.doubling[j-1][i]
                
                self.doubling[j][i]=self.doubling[j-1][now]
        
    def lca(self,x,y):
        posx,posy=x,y
        distx,disty=self.distance[x],self.distance[y]

        additionalx,additionaly=distx-min(distx,disty),disty-min(distx,disty)

        for j in range(self.bit):
            if additionalx & (1<<j):
                posx=self.doubling[j][posx]
            
            if additionaly & (1<<j):
                posy=self.doubling[j][posy]

        if posx==posy:
            return posx

        dist=0
        for j in range(self.bit-1,-1,-1):
            if self.doubling[j][posx]!=self.doubling[j][posy]:
                dist+=(1<<j)
        
        return self.doubling[0][posx]

    def dist(self,x,y):
        p=self.lca(x,y)
        return self.distance(x)+self.distance(y)-2*self.distance(x)

