class Combination():
    def __init__(self,N,mod):
        N+=1
        self.n=N
        self.fact=[0]*N;
        self.factinv=[0]*N;
        self.modinv=[0]*N;

        self.fact[0]=self.fact[1]=1
        self.factinv[0]=self.factinv[1]=1
        self.modinv[1]=1

        for x in range(2,N):
            self.fact[x]=(self.fact[x-1]*x)%mod
            self.modinv[x]=(mod-self.modinv[mod%x]*(mod//x))%mod
            self.factinv[x]=(self.factinv[x-1]*self.modinv[x])%mod
        
    def combi(self,n,k):
        if n<k:
            return 0
        
        if (n<0) or (k<0):
            return 0
        
        return self.fact[n]*(self.factinv[k]*self.factinv[n-k]%mod)%mod
    
    def fac(self,n):
        return self.fact[n]
    
    def finv(self,n):
        return self.factinv[n]
    
    def inv(self,n):
        # return self.factinv[N]*self.fact[N-1]%mod
        return self.modinv[n]