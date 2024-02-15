# lower case
class RollingHash():
    mod1=998244353
    mod2=10**8-11

    def __init__(self,s):
        self.S=s
        self.N=len(s)

        self.Hash1=[0]
        self.Hash2=[0]

        self.Bases1=[1]
        self.Bases2=[1]

        Base1=1007
        Base2=2009

        for i in range(self.N):
            # 1
            now1=(self.Hash1[i]*Base1+ord(self.S[i])-97)%self.mod1
            self.Hash1.append(now1)

            add1=(self.Bases1[i]*Base1)%self.mod1
            self.Bases1.append(add1)

            # 2
            now2=(self.Hash2[i]*Base2+ord(self.S[i])-97)%self.mod2
            self.Hash2.append(now2)

            add2=(self.Bases2[i]*Base2)%self.mod2
            self.Bases2.append(add2)

    def get1(self,pos,size):
        res=self.Hash1[pos+size]-self.Hash1[pos]*self.Bases1[size]%self.mod1
        return res%self.mod1

    def get2(self,pos,size):
        res=self.Hash2[pos+size]-self.Hash2[pos]*self.Bases2[size]%self.mod2
        return res%self.mod2

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