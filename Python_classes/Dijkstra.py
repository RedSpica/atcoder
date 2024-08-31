import heapq

class Dijkstra():
    def __init__(self,N,G,S=0,restoreflg=False):
        self.n=N
        self.dist=[float('inf')]*N
        self.prev=[-1]*N
        self.prev_num=[-1]*N

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

                    if restoreflg:
                        self.prev[x]=u

                        egdenum=g[2]
                        self.prev_num[x]=egdenum

    def getdist(self,x):
        return self.dist[x]

    def getrestore(self,goal):
        idx=goal
        ans=[]
        while idx!=-1:
            ans.append(self.prev_num[idx])
            idx=self.prev[idx]

        ans.reverse()

        return ans[1:]