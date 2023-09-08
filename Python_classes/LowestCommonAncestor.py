'''
g[i]:=iから辺が出ている頂点の集合

constructor
引数 g:有向グラフ, root:親, 

method
lca
引数 a,b:頂点番号
返り値 aとbの最近共通祖先
'''

class LowestCommonAncestor():
    # n=5*10**5
    n=1
    bit=64
    # bit=4
    doubling=[]
    p=-1

    def __init__(self,g,root):
        self.n=len(g)
        self.p=root

        for _ in range(self.bit):
            self.doubling.append([-1]*self.n)
        
        for i in range(self.n):
            for x in g[i]:
                self.doubling[0][x]=i;
    
        for j in range(1,self.bit):
            for i in range(self.n):
                now=self.doubling[j-1][i]
                
                if now==-1:
                    now=root
                
                self.doubling[j][i]=self.doubling[j-1][now]
        
    def lca(x,y):
        dist=0
        for j in range(self.bit-1,-1,-1):
            if self.doubling[j][x]!=self.doubling[j][y]:
                dist+=(1<<j)
        
        dist+=1
        
        for j in range(self.bit):            
            if dist & (1<<j):
