class Eratosthenes_sieve():
    def __init__(self,N):
        self.n=N
        self.sieve=[]
        
        for x in range(N+1):
            if x%2==0:
                self.sieve.append(2)
            elif x%3==0:
                self.sieve.append(3)
            else:
                self.sieve.append(x)
        
        for x in range(5,self.sqrt_N(N)+1):
            if self.sieve[x]!=x:
                continue
            for y in range(x*x,N+1,x):
                self.sieve[y]=min(self.sieve[y],x)
    
    def div(self,x):
        return self.sieve[x]
    
    def factors(self,x):
        D=dict()
        D[1]=1
        while x>1:
            d=self.sieve[x]
            D[d]=D.get(d,0)+1
            x//=d
        
        return D
    
    def isprime(self,x):
        return self.sieve[x]==x and x!=1
    
    def sqrt_N(self,N):
        return int(N**0.5+0.1)

