#include<iostream>
#include<algorithm>
#include<math.h>
#include<string>
#include<tuple>
#include<vector>
#include<cassert>
#include<cstdlib>
#include<cstdint>
#include<stdio.h>
#include<cmath>
#include<limits>
#include<iomanip> 
#include<ctime>
#include<climits>
#include<random>
#include<queue>
#include<deque>
#include<map>
#include<time.h>
#include<set>
#include <cassert>
#include <utility>

using namespace std;


namespace internal {
  template <class T> struct simple_queue {
    std::vector<T> payload;
    int pos = 0;
    void reserve(int n) { payload.reserve(n); }
    int size() const { return int(payload.size()) - pos; }
    bool empty() const { return pos == int(payload.size()); }
    void push(const T& t) { payload.push_back(t); }
    T& front() { return payload[pos]; }
    void clear() {
      payload.clear();
      pos = 0;
    }
    void pop() { pos++; }
  };
}

template <class T> using V = vector<T>;

template<class T> inline bool chmin(T& a, T b) {
  if (a > b) {
    a = b;
    return true;
  }
  return false;
}
template<class T> inline bool chmax(T& a, T b) {
  if (a < b) {
    a = b;
    return true;
  }
  return false;
}


const long long INF = 1LL << 60;
const double pi=acos(-1);

using ll = long long;
using vll = V<ll>;
using vvll = V<V<ll>>;
using vpll =V<pair<ll,ll>>;
using graph = V<V<ll>>;
using mp = map<ll,ll>;


#define FOR(i,a,b) for(ll i=(a);i<(b);i++)
#define bgn begin()
#define en end()
#define SORT(a) sort((a).begin(),(a).end())
#define REV(a) reverse((a).bgn,(a).en)
#define gcd(a,b) __gcd(a,b)
#define ALL(a)  (a).begin(),(a).end()


template<typename T>
std::vector<T> make_vec(size_t n){
  return std::vector<T>(n);
}

template<typename T, class... Args>
auto make_vec(size_t n, Args... args){
  return std::vector<decltype(make_vec<T>(args...))>(n,make_vec<T>(args...));
}

map<ll,ll> compress(vector<ll> A){
  vector<ll> B=A;
  sort(B.begin(), B.end());
  B.erase(unique(B.begin(), B.end()), B.end());
  
  map<ll,ll> res;
  for(ll i=0;i<B.size();i++){
    res[B[i]]=i;
  }

  return res;
}





ll modpow(ll a,ll n,ll mod){
  ll ans=1;
  while(n>0){
    if(n&1){
      ans=ans*a%mod;
    }

    a=a*a%mod;
    n/=2;
  }

  return ans;
}

ll POW(ll a,ll n){
  ll ans=1;
  while(n>0){
    if(n&1){
      ans=ans*a;
    }

    a=a*a;
    n/=2;
  }

  return ans;
}

ll modinv(ll a, ll mod) {
    return modpow(a, mod - 2, mod);
}

ll modcombi(int n,int k,int mod){
  ll ans=1;
  for(ll i=n;i>n-k;i--){
    ans*=i;
    ans%=mod;
  }
  for(ll i=1;i<=k;i++){
    ans*=modinv(i,mod);
    ans%=mod;
  }

  return ans;
}


vll div(ll n){
  vll ret(0);
  for(ll i=1;i*i<=n;i++){
    if(n%i==0){
      ret.push_back(i);
      if(i*i!=n){
        ret.push_back(n/i);
      }
    }
  }

  SORT(ret);
  return (ret);
}

vector<ll> graph_bfs(vector<vector<ll>> G, ll start){
  queue<ll> Q;
  Q.push(start);

  ll n=G.size();
  
  vll check(n);
  check[start]=1;
  
  vll dist(n);
  FOR(i,0,n){
    dist[i]=INF;
  }
  dist[start]=0;
  
  while(Q.size()){
    ll now=Q.front();
    Q.pop();
    FOR(i,0,G[now].size()){
      ll nex=G[now][i];
      if(check[nex]){
        continue;
      }
      
      dist[nex]=dist[now]+1;
      check[nex]=1;
      Q.push(nex);
    }
  }
  
  return dist;
}
 
vector<vector<ll>> grid_bfs(vector<string> field, ll sx,ll sy){
  ll h=field.size();
  ll w=field[0].size();
  queue<pair<ll,ll>> Q;
  Q.push({sx,sy});
  
  auto check=make_vec<ll>(h,w);
  check[sx][sy]=1;
  
  auto dist=make_vec<ll>(h,w);
  FOR(i,0,h){
    FOR(j,0,w){
      dist[i][j]=INF;
    }
  }
  dist[sx][sy]=0;
  
  while(Q.size()){
    ll x=Q.front().first;
    ll y=Q.front().second;
    Q.pop();
    
    FOR(dx,-1,2){
      FOR(dy,-1,2){
        if(abs(dx)+abs(dy)!=1){
          continue;
        }
        if(x+dx<0 or h<=x+dx or y+dy<0 or w<=y+dy){
          continue;
        }
        if(field[x+dx][y+dy]=='#'){
          continue;
        }
        if(check[x+dx][y+dy]){
          continue;
        }
        
        dist[x+dx][y+dy]=dist[x][y]+1;
        check[x+dx][y+dy]=1;
        Q.push({x+dx,y+dy});
      }
    }
  }
  
  return dist;
}

ll calc_LIS(vll lis){
  ll n=lis.size();
  vll res(n+1);
  FOR(i,0,n+1){
    res[i]=INF;
  }
  res[0]=-1;

  ll cen;
  ll l,r;
  ll x;
  FOR(i,0,n){
    l=0;
    r=n+1;
    x=lis[i];
    while(r-l>1){
      cen=(r+l)/2;
      if(res[cen]<=x){
        l=cen;
      }
      else{
        r=cen;
      }
    }
    chmin(res[r],x);
  }

  ll ans=0;

  FOR(i,0,n+1){
    if(res[i]==INF){
      break;
    }
    else{
      ans=i;
    }
  }

  return ans;
}

/*
ll calc_inversion(vll A){
  ll n=A.size();
  fenwick_tree<ll> ft(n+1);
  ll ans=0;
  FOR(i,0,n){
    ans+=ft.sum(A[i]+1,n+1);
    ft.add(A[i],1);
  }

  return ans;
}
*/

vector<vector<ll>> matrix_pow(vector<vector<ll>> A, ll x, ll mod){
  ll n=A.size();
  auto res=make_vec<ll>(n,n);
  FOR(i,0,n){
    res[i][i]=1;
  }

  auto cur=make_vec<ll>(n,n);
  while(x>0){
    if(x&1){
      FOR(i,0,n){
        FOR(j,0,n){
          FOR(k,0,n){
            cur[i][j]+=res[i][k]*A[k][j];
            cur[i][j]%=mod;
          }
        }
      }

      FOR(i,0,n){
        FOR(j,0,n){
          res[i][j]=cur[i][j]%mod;
          cur[i][j]=0;
        }
      }
    }

    x>>=1;

    FOR(i,0,n){
      FOR(j,0,n){
        FOR(k,0,n){
          cur[i][j]+=A[i][k]*A[k][j];
          cur[i][j]%=mod;
        }
      }
    }

    FOR(i,0,n){
      FOR(j,0,n){
        A[i][j]=cur[i][j];
        cur[i][j]=0;
      }
    }
  }

  return res;
}

bool compare_by_sec(pair<ll,ll> A,pair<ll,ll> B){
  if(A.first==B.first){
    return A.second>B.second;
  }
  else{
    return A.first<B.first;
  }
}



ll lcm(ll a,ll b){
  return a/gcd(a,b)*b;
}

void bf(ll n,string s){
  for(ll i=0;i<n;i++){
    cout<<s;
  }
  cout<<"\n";

  return;
}


template <class Cap> struct mf_graph {
  public:
  mf_graph() : _n(0) {}
  explicit mf_graph(int n) : _n(n), g(n) {}

  int add_edge(int from, int to, Cap cap) {
    assert(0 <= from && from < _n);
    assert(0 <= to && to < _n);
    assert(0 <= cap);
    int m = int(pos.size());
    pos.push_back({from, int(g[from].size())});
    int from_id = int(g[from].size());
    int to_id = int(g[to].size());
    if (from == to) to_id++;
    g[from].push_back(_edge{to, to_id, cap});
    g[to].push_back(_edge{from, from_id, 0});
    return m;
  }

  struct edge {
    int from, to;
    Cap cap, flow;
  };

  edge get_edge(int i) {
    int m = int(pos.size());
    assert(0 <= i && i < m);
    auto _e = g[pos[i].first][pos[i].second];
    auto _re = g[_e.to][_e.rev];
    return edge{pos[i].first, _e.to, _e.cap + _re.cap, _re.cap};
  }
  std::vector<edge> edges() {
    int m = int(pos.size());
    std::vector<edge> result;
    for (int i = 0; i < m; i++) {
      result.push_back(get_edge(i));
    }
    return result;
  }
  void change_edge(int i, Cap new_cap, Cap new_flow) {
    int m = int(pos.size());
    assert(0 <= i && i < m);
    assert(0 <= new_flow && new_flow <= new_cap);
    auto& _e = g[pos[i].first][pos[i].second];
    auto& _re = g[_e.to][_e.rev];
    _e.cap = new_cap - new_flow;
    _re.cap = new_flow;
  }

  Cap flow(int s, int t) {
    return flow(s, t, std::numeric_limits<Cap>::max());
  }
  Cap flow(int s, int t, Cap flow_limit) {
    assert(0 <= s && s < _n);
    assert(0 <= t && t < _n);
    assert(s != t);

    std::vector<int> level(_n), iter(_n);
    internal::simple_queue<int> que;

    auto bfs = [&]() {
      std::fill(level.begin(), level.end(), -1);
      level[s] = 0;
      que.clear();
      que.push(s);
      while (!que.empty()) {
        int v = que.front();
        que.pop();
        for (auto e : g[v]) {
          if (e.cap == 0 || level[e.to] >= 0) continue;
          level[e.to] = level[v] + 1;
          if (e.to == t) return;
          que.push(e.to);
          }
        }
      };
      auto dfs = [&](auto self, int v, Cap up) {
        if (v == s) return up;
        Cap res = 0;
        int level_v = level[v];
        for (int& i = iter[v]; i < int(g[v].size()); i++) {
          _edge& e = g[v][i];
          if (level_v <= level[e.to] || g[e.to][e.rev].cap == 0) continue;
          Cap d =
            self(self, e.to, std::min(up - res, g[e.to][e.rev].cap));
          if (d <= 0) continue;
          g[v][i].cap += d;
          g[e.to][e.rev].cap -= d;
          res += d;
          if (res == up) return res;
        }
        level[v] = _n;
        return res;
      };

      Cap flow = 0;
      while (flow < flow_limit) {
        bfs();
        if (level[t] == -1) break;
        std::fill(iter.begin(), iter.end(), 0);
        Cap f = dfs(dfs, t, flow_limit - flow);
        if (!f) break;
        flow += f;
      }
      return flow;
    }

    std::vector<bool> min_cut(int s) {
      std::vector<bool> visited(_n);
      internal::simple_queue<int> que;
      que.push(s);
      while (!que.empty()) {
        int p = que.front();
        que.pop();
        visited[p] = true;
        for (auto e : g[p]) {
          if (e.cap && !visited[e.to]) {
            visited[e.to] = true;
            que.push(e.to);
          }
        }
      }
      return visited;
    }

  private:
    int _n;
    struct _edge {
      int to, rev;
      Cap cap;
    };
  std::vector<std::pair<int, int>> pos;
  std::vector<std::vector<_edge>> g;
};

struct rollinghash{
  public:
    rollinghash(string s,int len):n(len),Hash1(len+1),Hash2(len+1),Bases1(len+1),Bases2(len+1){
      ll Base1=1007,Base2=2009;
      ll now1=0,now2=0;
      ll modinv1=modpow(Base1,mod1-2,mod1),modinv2=modpow(Base2,mod2-2,mod2);
      Bases1[0]=1;
      Bases2[0]=1;
      for(int i=0;i<n;i++){ 
        Hash1[i+1]=(Hash1[i]*Base1+(s[i]-'a'))%mod1;
        Bases1[i+1]=(Bases1[i]*Base1)%mod1;

        Hash2[i+1]=(Hash2[i]*Base2+(s[i]-'a'))%mod2;
        Bases2[i+1]=(Bases2[i]*Base2)%mod2;
      }
    }

    ll get1(int pos,int len){
      ll res=Hash1[pos+len]-Hash1[pos]*Bases1[len]%mod1;
      if(res<0){
        res+=mod1;
      }

      return res;
    }

    ll get2(int pos,int len){
      ll res=Hash2[pos+len]-Hash2[pos]*Bases2[len]%mod2;
      if(res<0){
        res+=mod2;
      }

      return res;
    }

    ll longestcommonprefix(int pos1,int pos2){
      ll l=0,r=n+1-max(pos1,pos2);
      while(r-l>1){
        ll cen=(r+l)/2;

        ll cur1=get1(pos1,cen);
        ll cur2=get1(pos2,cen);

        if(cur1!=cur2){
          r=cen;
          continue;
        }

        cur1=get2(pos1,cen);
        cur2=get2(pos2,cen);

        if(cur1!=cur2){
          r=cen;
        }
        else{
          l=cen;
        }
      }

      return l;
    }
  
  private:
    int n;
    ll mod1=998244353;
    ll mod2=100000000-11;
    vector<ll> Hash1;
    vector<ll> Hash2;
    vector<ll> Bases1;
    vector<ll> Bases2;
};

struct dsu {
  public:
    dsu() : _n(0) {}
    explicit dsu(int n) : _n(n), parent_or_size(n, -1) {}

    int merge(int a, int b) {
      assert(0 <= a && a < _n);
      assert(0 <= b && b < _n);
      int x = leader(a), y = leader(b);
      if (x == y) return x;
      if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
      parent_or_size[x] += parent_or_size[y];
      parent_or_size[y] = x;
      return x;
    }

    bool same(int a, int b) {
      assert(0 <= a && a < _n);
      assert(0 <= b && b < _n);
      return leader(a) == leader(b);
    }

    int leader(int a) {
      assert(0 <= a && a < _n);
      if (parent_or_size[a] < 0) return a;
      return parent_or_size[a] = leader(parent_or_size[a]);
    }

    int size(int a) {
      assert(0 <= a && a < _n);
      return -parent_or_size[leader(a)];
    }

    std::vector<std::vector<int>> groups() {
      std::vector<int> leader_buf(_n), group_size(_n);
      for (int i = 0; i < _n; i++) {
        leader_buf[i] = leader(i);
        group_size[leader_buf[i]]++;
      }
      std::vector<std::vector<int>> result(_n);
      for (int i = 0; i < _n; i++) {
        result[i].reserve(group_size[i]);
      }
      for (int i = 0; i < _n; i++) {
        result[leader_buf[i]].push_back(i);
      }
      result.erase(
        std::remove_if(result.begin(), result.end(),
        [&](const std::vector<int>& v) { return v.empty(); }),
        result.end());
      return result;
    }

  private:
    int _n;
    // root node: -1 * component size
    // otherwise: parent
    std::vector<int> parent_or_size;
};


template <class T> struct fenwick_tree {
    using U = T;

  public:
    fenwick_tree() : _n(0) {}
    explicit fenwick_tree(int n) : _n(n), data(n) {}

    void add(int p, T x) {
      assert(0 <= p && p < _n);
      p++;
      while (p <= _n) {
        data[p - 1] += U(x);
        p += p & -p;
      }
    }

    T sum(int l, int r) {
      assert(0 <= l && l <= r && r <= _n);
      return sum(r) - sum(l);
    }

  private:
  int _n;
  std::vector<U> data;

  U sum(int r) {
    U s = 0;
    while (r > 0) {
      s += data[r - 1];
      r -= r & -r;
    }
    return s;
  }
};



int ceil_pow2(int n) {
  int x = 0;
  while ((1U << x) < (unsigned int)(n)) x++;
  return x;
}



template <class S, S (*op)(S, S), S (*e)()> struct segtree {
  public:
    segtree() : segtree(0) {}
    explicit segtree(int n) : segtree(std::vector<S>(n, e())) {}
    explicit segtree(const std::vector<S>& v) : _n(int(v.size())) {
      log = ceil_pow2(_n);
      size = 1 << log;
      d = std::vector<S>(2 * size, e());
      for (int i = 0; i < _n; i++) d[size + i] = v[i];
      for (int i = size - 1; i >= 1; i--) {
        update(i);
      }
    }

    void set(int p, S x) {
      assert(0 <= p && p < _n);
      p += size;
      d[p] = x;
      for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) const {
      assert(0 <= p && p < _n);
      return d[p + size];
    }

    S prod(int l, int r) const {
      assert(0 <= l && l <= r && r <= _n);
      S sml = e(), smr = e();
      l += size;
      r += size;

      while (l < r) {
        if (l & 1) sml = op(sml, d[l++]);
        if (r & 1) smr = op(d[--r], smr);
        l >>= 1;
        r >>= 1;
      }
      return op(sml, smr);
    }

    S all_prod() const { return d[1]; }

    template <bool (*f)(S)> int max_right(int l) const {
      return max_right(l, [](S x) { return f(x); });
    }
    template <class F> int max_right(int l, F f) const {
      assert(0 <= l && l <= _n);
      assert(f(e()));
      if (l == _n) return _n;
      l += size;
      S sm = e();
      do {
        while (l % 2 == 0) l >>= 1;
        if (!f(op(sm, d[l]))) {
          while (l < size) {
            l = (2 * l);
            if (f(op(sm, d[l]))) {
              sm = op(sm, d[l]);
              l++;
            }
          }
          return l - size;
        }
        sm = op(sm, d[l]);
        l++;
      } while ((l & -l) != l);
      return _n;
    }

    template <bool (*f)(S)> int min_left(int r) const {
      return min_left(r, [](S x) { return f(x); });
    }
    template <class F> int min_left(int r, F f) const {
      assert(0 <= r && r <= _n);
      assert(f(e()));
      if (r == 0) return 0;
      r += size;
      S sm = e();
      do {
        r--;
        while (r > 1 && (r % 2)) r >>= 1;
        if (!f(op(d[r], sm))) {
          while (r < size) {
            r = (2 * r + 1);
            if (f(op(d[r], sm))) {
              sm = op(d[r], sm);
              r--;
            }
          }
          return r + 1 - size;
        }
        sm = op(d[r], sm);
      } while ((r & -r) != r);
      return 0;
    }

  private:
    int _n, size, log;
    std::vector<S> d;

  void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
};


// csr
template <class E> struct csr {
  std::vector<int> start;
  std::vector<E> elist;
  explicit csr(int n, const std::vector<std::pair<int, E>>& edges)
    : start(n + 1), elist(edges.size()) {
    for (auto e : edges) {
      start[e.first + 1]++;
    }
    for (int i = 1; i <= n; i++) {
      start[i] += start[i - 1];
    }
    auto counter = start;
    for (auto e : edges) {
      elist[counter[e.first]++] = e.second;
    }
  }
};

// internal scc
struct scc_graph {
  public:
    explicit scc_graph(int n) : _n(n) {}

    int num_vertices() { return _n; }

    // assertどうしよう
    void add_edge(int from, int to) { 
      assert(0 <= from && from < _n);
      assert(0 <= to && to < _n);
      edges.push_back({from, {to}}); 
    }

    // @return pair of (# of scc, scc id)
    std::pair<int, std::vector<int>> scc_ids() {
      auto g = csr<edge>(_n, edges);
      int now_ord = 0, group_num = 0;
      std::vector<int> visited, low(_n), ord(_n, -1), ids(_n);
      visited.reserve(_n);
      auto dfs = [&](auto self, int v) -> void {
        low[v] = ord[v] = now_ord++;
        visited.push_back(v);
        for (int i = g.start[v]; i < g.start[v + 1]; i++) {
          auto to = g.elist[i].to;
          if (ord[to] == -1) {
            self(self, to);
            low[v] = std::min(low[v], low[to]);
          } else {
            low[v] = std::min(low[v], ord[to]);
          }
        }
        if (low[v] == ord[v]) {
          while (true) {
            int u = visited.back();
            visited.pop_back();
            ord[u] = _n;
            ids[u] = group_num;
            if (u == v) break;
          }
          group_num++;
        }
      };
      for (int i = 0; i < _n; i++) {
        if (ord[i] == -1) dfs(dfs, i);
      }
      for (auto& x : ids) {
        x = group_num - 1 - x;
      }
      return {group_num, ids};
    }

    std::vector<std::vector<int>> scc() {
      auto ids = scc_ids();
      int group_num = ids.first;
      std::vector<int> counts(group_num);
      for (auto x : ids.second) counts[x]++;
      std::vector<std::vector<int>> groups(ids.first);
      for (int i = 0; i < group_num; i++) {
        groups[i].reserve(counts[i]);
      }
      for (int i = 0; i < _n; i++) {
        groups[ids.second[i]].push_back(i);
      }
      return groups;
    }

  private:
    int _n;
    struct edge {
      int to;
    };
  std::vector<std::pair<int, edge>> edges;
};


// change
//const int MOD=1000000007;
const int MOD=998244353;


/*
const int MAX = 5100000;
long long fac[MAX], finv[MAX], inv[MAX];


void comuse() {
  fac[0] = fac[1] = 1;
  finv[0] = finv[1] = 1;
  inv[1] = 1;
  for (int i = 2; i < MAX; i++){
    fac[i] = fac[i - 1] * i % MOD;
    inv[i] = MOD - inv[MOD%i] * (MOD / i) % MOD;
    finv[i] = finv[i - 1] * inv[i] % MOD;
  }
}

ll combi(int n, int k){
  if (n < k) return 0;
  if (n < 0 || k < 0) return 0;
  return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD;
}

ll perm(int n,int k){
  if(n < k) return 0;
  if(n < 0 || k < 0) return 0;
  return fac[n] * (finv[k] % MOD) % MOD;
}

*/

/*
const int era=2000000;
long long sieve[era];

void Sieveuse(){
  for(ll i=1;i<era;i++){
    sieve[i]=i;
  }

  for(ll i=2;i<era;i++){
    if(sieve[i]!=i){
      continue;
    }
    for(ll j=i*i;j<era;j+=i){
      chmin(sieve[j],i);
    }
  }
}

bool isprime(int p){
  if(sieve[p]==p){
    return true;
  }
  else{
    return false;
  }
}
*/


//for segment tree
ll op1(ll a,ll b){
  return max(a,b);
}


ll e1(){
  return 0;
}

ll op2(ll a,ll b){
  return max(a,b);
}

ll e2(){
  return 0LL;
}




void Solve();


signed main(){
  cin.tie(0);
  ios::sync_with_stdio(false);
  cout<<setprecision(20)<<fixed;
  
  Solve();
}


/**************************************\
| Thank you for viewing my code:)       |
| Author is RedSpica a.k.a. RanseMirage |
| Twitter:@asakaakasaka                 | 
\**************************************/
//segtreeの葉の先頭の添え字はN-1

void Solve(){

  return;
}