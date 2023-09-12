from collections import deque 

class LowestCommonAncestor():
    n=1
    bit=64
    doubling=[]
    G=[]
    p=-1
    distance=[]

    def __init__(self,g,root):
        self.n=len(g)
        self.p=root
        self.G=[[] for _ in range(n)]
        self.distance=[0]*n

        Q=deque()
        Q.append(self.p)

        check=set()
        check.add(self.p)
        
        while len(Q):
            now=Q.popleft()
            for nex in g[now]:
                if nex in check:
                    continue
                
                self.distance[nex]=self.distance[now]+1
                Q.append(nex)
                check.add(nex)
                self.G[now].append(nex)


        for _ in range(self.bit):
            self.doubling.append([root]*self.n)
        
        for i in range(self.n):
            for x in self.G[i]:
                self.doubling[0][x]=i
    
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

        for j in range(self.bit-1,-1,-1):
            if self.doubling[j][posx]!=self.doubling[j][posy]:
                posx=self.doubling[j][posx]
                posy=self.doubling[j][posy]

        return self.doubling[0][posx]

    def dist(self,x,y):
        p=self.lca(x,y)
        return self.distance[x]+self.distance[y]-2*self.distance[p]

    def is_on_path(self,x,y,a):
        if self.dist(x,y)==self.dist(x,a)+self.dist(a,y):
            return True
        else:
            return False