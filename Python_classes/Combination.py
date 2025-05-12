def init_combi(N,mod):
    N+=1
    global fact,factinv,modinv
    fact=[0]*N
    factinv=[0]*N
    modinv=[0]*N

    fact[0]=fact[1]=1
    factinv[0]=factinv[1]=1
    modinv[1]=1

    for x in range(2,N):
        fact[x]=(fact[x-1]*x)%mod
        modinv[x]=(mod-modinv[mod%x]*(mod//x))%mod
        factinv[x]=(factinv[x-1]*modinv[x])%mod

def combi(n,k):
    if n<k:
        return 0
    
    if (n<0) or (k<0):
        return 0
    
    return fact[n]*(factinv[k]*factinv[n-k]%mod)%mod

def fac(n):
    return fact[n]

def finv(n):
    return factinv[n]

def inv(n):
    return modinv[n]