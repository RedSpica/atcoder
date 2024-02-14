# lower case
class RollingHash():
    mod1=998244353
    modinv1=pow(27,mod1-2,mod1)

    mod2=10**8-11
    modinv2=pow(27,mod2-2,mod2)

    def __init__(self,s):
        self.S=s
        self.N=len(s)

        self.Hash1=[0]
        self.Hash2=[0]

        self.Bases1=[1]
        self.Bases2=[1]

        self.InvBase1=[1]
        self.InvBase2=[1]

        Base1=1
        Base2=1

        Inv1=1
        Inv2=1

        now1=0
        now2=0

        for i in range(self.N):
            # 1
            now1=(now1+(ord(self.S[i])-96)*Base1)%self.mod1
            self.Hash1.append(now1)

            Base1=(Base1*27)%self.mod1
            self.Bases1.append(Base1)

            Inv1=(Inv1*self.modinv1)%self.mod1
            self.InvBase1.append(Inv1)

            # 2
            now2=(now2+(ord(self.S[i])-96)*Base2)%self.mod2
            self.Hash2.append(now2)

            Base2=(Base2*27)%self.mod2
            self.Bases2.append(Base2)

            Inv2=(Inv2*self.modinv2)%self.mod2
            self.InvBase2.append(Inv2)

    def get1(self,pos,size):
        return (self.Hash1[pos+size]-self.Hash1[pos])*self.InvBase1[pos]%self.mod1
    
    def get2(self,pos,size):
        return (self.Hash2[pos+size]-self.Hash2[pos])*self.InvBase2[pos]%self.mod2
    
    def LongestCommonPrefix(self,pos1,pos2):
        l,r=0,self.N-max(pos1,pos2)+1
        while r-l>1:
            cen=(r+l)//2
            if pos1+cen>self.N or pos2+cen>self.N:
                r=cen
                continue

            cur1=self.get1(pos1,cen)
            cur2=self.get1(pos2,cen)

            if cur1!=cur2:
                r=cen
                continue
            
            cur1=self.get2(pos1,cen)
            cur2=self.get2(pos2,cen)

            if cur1!=cur2:
                r=cen
            
            else:
                l=cen
            
        return l
    
    #1つ目のほうが大きいなら0, 2つ目のほうが大きいなら1, 等しいなら2を返す
    #引数の順番に注意！
    def LexicographicalOrder(self,pos1,pos2,end1=-1,end2=-1):
        if end1==-1:
            end1=self.N-1
        if end2==-1:
            end2=self.N-1
        
        now=self.LongestCommonPrefix(pos1,pos2)
        
        if pos1+now>end1 and pos2+now>end2:
            len1,len2=end1-pos1+1,end2-pos2+1
            if len1>len2:
                return 0
            if len1<len2:
                return 1
            if len1==len2:
                return 2
        elif pos1+now>end1:
            return 1
        elif pos2+now>end2:
            return 0
        else:
            c1,c2='?','?'
            if pos1+now<self.N:
                c1=self.S[pos1+now]
            if pos2+now<self.N:
                c2=self.S[pos2+now]
            
            if c1>c2:
                return 0
            if c1<c2:
                return 1