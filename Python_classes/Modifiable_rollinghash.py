import typing

class FenwickTree:
    def __init__(self, n: int = 0) -> None:
        self._n = n
        self.data = [0] * n

    def add(self, p: int, x: typing.Any) -> None:
        assert 0 <= p < self._n

        p += 1
        while p <= self._n:
            self.data[p - 1] += x
            p += p & -p

    def sum(self, left: int, right: int) -> typing.Any:
        assert 0 <= left <= right <= self._n

        return self._sum(right) - self._sum(left)

    def _sum(self, r: int) -> typing.Any:
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r

        return s

class Modifiable_RollingHash():
    mod1=998244353
    mod2=10**8-11

    Base1=1007
    Base2=2009

    BaseInv1=pow(Base1,mod1-2,mod1)
    BaseInv2=pow(Base2,mod2-2,mod2)

    def __init__(self,s):
        self.S=s
        self.N=len(s)

        self.Hash1=FenwickTree(self.N+1)
        self.Hash2=FenwickTree(self.N+1)

        self.Bases1=[1]
        self.Bases2=[1]

        self.BaseInvs1=[1]
        self.BaseInvs2=[1]

        for i in range(self.N):
            # 1
            now1=(self.Bases1[i]*ord(self.S[i]))%self.mod1
            self.Hash1.add(i+1,now1)

            add1=(self.Bases1[i]*self.Base1)%self.mod1
            self.Bases1.append(add1)

            inv1=(self.BaseInvs1[i]*self.BaseInv1)%self.mod1
            self.BaseInvs1.append(inv1)

            # 2
            now2=(self.Bases2[i]*ord(self.S[i]))%self.mod2
            self.Hash2.add(i+1,now2)

            add2=(self.Bases2[i]*self.Base2)%self.mod2
            self.Bases2.append(add2)

            inv2=(self.BaseInvs2[i]*self.BaseInv2)%self.mod2
            self.BaseInvs2.append(inv2)

    def get1(self,pos,size):
        res=self.Hash1.sum(pos,pos+size)*self.BaseInvs1[pos-1]%self.mod1
        return res

    def get2(self,pos,size):
        res=self.Hash2.sum(pos,pos+size)*self.BaseInvs2[pos-1]%self.mod2
        return res
    
    def all_get(self,pos,size):
        return [self.get1(pos,size),self.get2(pos,size)]
    
    def setval(self,pos,c):
        now=ord(self.S[pos-1])
        nex=ord(c)
        val1=(nex*self.Bases1[pos-1]%self.mod1-now*self.Bases1[pos-1]%self.mod1)%self.mod1
        val2=(nex*self.Bases2[pos-1]%self.mod2-now*self.Bases2[pos-1]%self.mod2)%self.mod2

        self.Hash1.add(pos,val1)
        self.Hash2.add(pos,val2)

        self.S[pos-1]=c