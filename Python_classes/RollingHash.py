class RollingHash():
    n=1
    Hash1=[0]
    # Hash2=[0]
    Bases1=[1]
    InvBase1=[1]

    S=''
    mod1=998244353
    modinv1=pow(mod1,mod1-2,mod)
    # mod2=3

    def __init__(self,s):
        self.S=s
        self.n=len(s)
        Base1=1
        Inv1=1
        now1=0
        for i in range(n):
            now1=(now1+(ord(S[i])-96)*Base1)%mod1
            Hash1.append(now1)

            Base1=(Base1*27)%mod1
            Bases1.append(Base1)

            Inv1=(Inv1*modinv1)%mod1
            InvBase1.append(Inv1)
             