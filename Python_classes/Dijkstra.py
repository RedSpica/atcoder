import heapq

class Dijkstra():
    def __init__(self,N,G,S=0):
        self.n=N
        self.dist=[float('inf')]*N

        self.dist[S]=0
        self.Q=[]
        heapq.heapify(self.Q)
        heapq.heappush(self.Q,(0,S))

        self.vis=set()

        while len(self.Q):
            nex=heapq.heappop(self.Q)
            cost,u=nex[0],nex[1]
            if u in self.vis:
                continue

            self.vis.add(u)

            for g in G[u]:
                x,cost=g[0],g[1]
                if self.dist[u]+cost<self.dist[x]:
                    self.dist[x]=self.dist[u]+cost
                    heapq.heappush(self.Q,(self.dist[x],x))
    
    def getdist(self,x):
        return self.dist[x]
