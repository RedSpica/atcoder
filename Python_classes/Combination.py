class Combination():
    n=1
    fac,finv,inv=[1],[1],[1]

    def __init__(self,N,mod):
        N+=1
        self.n=N
        self.fac=[0]*N;
        self.finv=[0]*N;
        self.inv=[0]*N;

        self.fac[0]=self.fac[1]=1
        self.finv[0]=self.finv[1]=1
        self.inv[1]=1

        for x in range(2,N):
            self.fac[x]=(self.fac[x-1]*x)%mod
            self.inv[x]=(mod-self.inv[mod%x]*(mod//x))%mod
            self.finv[x]=(self.finv[x-1]*self.inv[x])%mod
        
    def combi(self,n,k):
        if n<k:
            return 0
        
        if (n<0) or (k<0):
            return 0
        
        return self.fac[n]*(self.finv[k]*self.finv[n-k]%mod)%mod