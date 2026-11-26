#include <iterator>
#include "xenithium/arch/qwen35/trained_qsa_device.hpp"
#include "xenithium/kernels/attention/trained_qsa_pages.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/util/env.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace xenithium::arch::qwen35::trained_qsa {
namespace k = kernels;
namespace {
constexpr int width=(heads+1)*dim, workers=32, pf_score_tile=64, pf_fa_min_rows=16;
constexpr size_t allocation_bytes(size_t n) { return std::max(size_t(4096),n); }
constexpr size_t align64(size_t n) { return (n+63)&~size_t(63); }
// A kernel with 256 registers per item and 32-item sub-groups: the properties ride on the kernel object (the
// parallel_for overload that takes them separately is deprecated).
template <class F>
struct Grf256Sg32 {
  F f;
  void operator()(sycl::nd_item<2> it) const { f(it); }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                       sycl::ext::oneapi::experimental::sub_group_size<32>};
  }
};
bool enabled(const char *name) { const char *p=std::getenv(name); return p && p[0]=='1' && !p[1]; }
struct ScratchLayout {
  size_t proj,score,nc,candidate;
  ScratchLayout(int pf,int st,int eligible)
    :proj(allocation_bytes(size_t(std::max(pf,k::row_shape(st)))*width*sizeof(float))),
     score(allocation_bytes(size_t(st)*eligible*sizeof(uint32_t))),
     nc(eligible>k::qsa_select_chunk(eligible)?allocation_bytes(size_t(st)*k::kQsaMaxChunks*sizeof(int32_t)):0),
     candidate(nc?allocation_bytes(size_t(st)*k::kQsaMaxChunks*k::kQsaTopBlocks*sizeof(int32_t)):0) {}
  size_t selection() const { return align64(score)+align64(nc)+2*align64(candidate); }
  size_t peak() const { return std::max(3*align64(proj),selection()); }
};
struct History {
  float *tail, *pool, *staged, *closed, *query, *pf_raw;
  sycl::half *pool16;
  const float *norm, *rope;
  int cap;
  // M-RoPE (image input). thw: this prefill call's own (t, h, w) per row, else null: a row at position p sits at
  // p + its slot's delta on all three (text; the delta is 0 until an image). open: per slot the first row of the
  // open 4-row block when a prefill wrote it (position, t, h, w; position -1: none). hmask / wmask: the rotary
  // frequencies that take h / w (Mrope::pick); zero leaves every frequency on t, the text rope.
  const int32_t *thw=nullptr;
  const int32_t *delta=nullptr;
  int32_t *open=nullptr;
  uint32_t hmask=0, wmask=0;
};
struct Thw { int t, h, w; };
inline Thw text_thw(const History &h, int slot, int p) { const int q=p+h.delta[slot]; return {q,q,q}; }
inline Thw row_thw(const History &h, int r, int slot, int p) {
  if(h.thw) { const int32_t *t=h.thw+size_t(r)*3; return {t[0],t[1],t[2]}; }
  return text_thw(h,slot,p);
}
// rotate the pair (lane, lane + 32) by frequency lane at its section's position: the table's rows are cos | sin
inline void rotate(const History &h, const Thw &p, int lane, float &a0, float &a1) {
  const int at=(h.hmask>>lane&1u)?p.h:(h.wmask>>lane&1u)?p.w:p.t;
  const float *rot=h.rope+size_t(sycl::clamp(at,0,h.cap-1))*64;
  const float aa=a0, bb=a1;
  a0=aa*rot[lane]-bb*rot[lane+32];
  a1=bb*rot[lane]+aa*rot[lane+32];
}
// Position-indexed rotary table (cos then sin); rotate() composes a row's
// M-RoPE from it, text rows being broadcast text MRoPE. We cannot use stock QSA's fused BF16 prep: the trained bank
// rounds neither projected input nor normalized keys to BF16 and has a third norm.
template<bool PfQuery=false>
sycl::event norm_index(sycl::queue &q, const History &h, const std::array<float*,3> &x, float eps,
                       int rows, int slot, int pos, k::GqaStep step, bool draft,uint32_t track=~uint32_t(0), bool tree_queries=false) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows;
  // A tree still stages the index keys, but does not score or use sparse attention.
  const bool queries=draft && (!step.tree || tree_queries);
  const int nhead=PfQuery?heads:queries?heads+1:1;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*nhead*workers,workers), [=](sycl::nd_item<1> it) {
    const int w=it.get_group(0), lane=it.get_local_id(0), r=PfQuery?w/heads:queries?w/(heads+1):w, head=PfQuery?w%heads:queries?w%(heads+1):heads;
    if(draft && !(track&(uint32_t(1)<<r))) return;
    const auto sr=draft?k::seg_row(rec,stride,seg_rows,r):k::SegRow{0,r,pos+r,slot};
    const size_t at=size_t(r)*width+head*dim;
    float a[dim/workers], ss=0;
    for(int i=0;i<dim/workers;++i) {
      const size_t d=at+lane+i*workers;
      a[i]=x[0][d]+(x[1][d]+x[2][d]);
      ss+=a[i]*a[i];
    }
    const float inv=sycl::rsqrt(sycl::reduce_over_group(it.get_group(),ss,sycl::plus<float>())/dim+eps);
    const float *norm=h.norm+(head==heads?dim:0);
    for(int i=0;i<dim/workers;++i) a[i]=(a[i]*inv)*norm[lane+i*workers];
    if (head==heads) {
      float *out=(draft?h.staged:h.pf_raw)+size_t(r)*dim;
      for(int i=0;i<dim/workers;++i) out[lane+i*workers]=a[i];
    } else {
      const int p=tree_queries?k::seg_vpos(rec,stride,sr):sr.pos;
      rotate(h,draft?text_thw(h,sr.slot,p):row_thw(h,r,sr.slot,p),lane,a[0],a[1]);
      float *out=h.query+(size_t(r)*heads+head)*dim;
      for(int i=0;i<dim/workers;++i) out[lane+i*workers]=a[i];
    }
  });
}
sycl::event index_blocks(sycl::queue &q, const History &h, float eps, int rows, int slot, int pos,
                         k::GqaStep step, int32_t *eligible, bool draft,uint32_t track=~uint32_t(0)) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*workers,workers), [=](sycl::nd_item<1> it) {
    const int r=it.get_group(0), lane=it.get_local_id(0);
    const auto sr=draft?k::seg_row(rec,stride,seg_rows,r):k::SegRow{0,r,pos+r,slot};
    // every row's block count, tracked or not: the selection runs over all rows and reads it as the row's bound
    if (draft && lane==0) eligible[r]=(sr.pos+1)/4;
    if(draft && !(track&(uint32_t(1)<<r))) return;
    if (sr.pos%4!=3) return;
    const int start=draft?rec[sr.seg*stride+k::kSegPos]:pos;
    const float *tail=h.tail+size_t(sr.slot)*4*dim;
    const float *raw=draft?h.staged:h.pf_raw;
    float a[dim/workers], ss=0;
    for(int i=0;i<dim/workers;++i) {
      const int d=lane+i*workers;
      float sum=0;
      for(int t=sr.pos-3;t<=sr.pos;++t)
        sum+=t>=start?raw[size_t(r+t-sr.pos)*dim+d]:tail[size_t(t%4)*dim+d];
      a[i]=sum*.25f;
      ss+=a[i]*a[i];
    }
    const float inv=sycl::rsqrt(sycl::reduce_over_group(it.get_group(),ss,sycl::plus<float>())/dim+eps);
    for(int i=0;i<dim/workers;++i) a[i]=(a[i]*inv)*h.norm[2*dim+lane+i*workers];
    // the block's first row: in this call, else the slot's open entry when a prefill wrote that row (an image row
    // can end a chunk); decode rows are text
    const int t0=sr.pos-3;
    Thw p;
    if(t0>=start) p=draft?text_thw(h,sr.slot,t0):row_thw(h,r-3,sr.slot,t0);
    else {
      const int32_t *o=h.open+size_t(sr.slot)*4;
      p=o[0]==t0?Thw{o[1],o[2],o[3]}:text_thw(h,sr.slot,t0);
    }
    rotate(h,p,lane,a[0],a[1]);
    const size_t at=(size_t(sr.slot)*(h.cap/4)+sr.pos/4)*dim;
    for(int i=0;i<dim/workers;++i) {
      const int d=lane+i*workers;
      if(draft) h.closed[size_t(r)*dim+d]=a[i];
      else {
        if(h.pool) h.pool[at+d]=a[i];
        if(h.pool16) h.pool16[at+d]=sycl::half(a[i]);
      }
    }
  });
}
template<bool Half=false> sycl::event index_scores(sycl::queue &q, const History &h, int rows, int bound, k::GqaStep step, uint32_t *scores) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows, pitch=h.cap/4;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*bound*workers,workers), [=](sycl::nd_item<1> it) {
    const int wg=it.get_group(0), r=wg/bound, b=wg%bound, lane=it.get_local_id(0);
    const auto sr=k::seg_row(rec,stride,seg_rows,r);
    const int last=4*b+3;
    if (last>sr.pos) return;
    const int start=rec[sr.seg*stride+k::kSegPos];
    const size_t at=(size_t(sr.slot)*pitch+b)*dim;
    float score=0;
    for(int j=0;j<heads;++j) {
      const float *query=h.query+(size_t(r)*heads+j)*dim;
      float dot=0;
      for(int i=lane;i<dim;i+=workers) {
        const float key=last>=start?h.closed[size_t(r+last-sr.pos)*dim+i]:
          (Half?static_cast<float>(h.pool16[at+i]):h.pool[at+i]);
        dot+=query[i]*key;
      }
      score+=sycl::fmax(0.f,sycl::reduce_over_group(it.get_group(),dot,sycl::plus<float>()));
    }
    if (!lane) scores[size_t(r)*pitch+b]=sycl::bit_cast<uint32_t>(score);
  });
}
// 00aadde stock FP32 arithmetic, with independent <=4-row tiles within
// each verify segment. DSpine can supply chains up to 16 rows per segment.
template<bool Half=false,bool Tree=false,bool Tail=false>  // Tail (chain steps with the DPAS scores): only blocks [start / 4, ...)
sycl::event index_scores_tile(sycl::queue &q,const History &h,int rows,int bound,k::GqaStep step,uint32_t *scores,int32_t *eligible=nullptr) {
  constexpr int R=4,B=8;
  static_assert(workers==32 && dim==128);
  const auto *rec=step.pos;
  const int stride=step.stride,pitch=h.cap/4,tiles=Tail?1:(bound+B-1)/B,layout=step.seg_rows;
  const int segments=k::seg_count(rows,layout),groups=(k::seg_size(rows,layout)+R-1)/R;
  return q.parallel_for(sycl::nd_range<2>({size_t(segments*groups),size_t(tiles)*workers},{1,workers}),Grf256Sg32{
      [=](sycl::nd_item<2> it) {
    const int g=it.get_group(0),s=g/groups,j0=(g%groups)*R,lane=it.get_local_id(1);
    int rseg,nseg; k::seg_span(layout,rows,s,rseg,nseg);
    const int n=sycl::min(R,nseg-j0);
    if(n<1) return;
    const int start=rec[s*stride+k::kSegPos],slot=rec[s*stride+k::kSegSlot];
    const int b0=Tail?start/4:int(it.get_group(1))*B;
    const int r0=rseg+j0,nmax=Tree?start/4:(start+j0+n)/4;
    if constexpr (Tree) if(!b0 && !lane) for(int r=0;r<n;++r) eligible[r0+r]=nmax;
    if(b0>=nmax) return;
    const auto sg=it.get_sub_group();
    float qr[R][heads][dim/workers];
#pragma unroll
    for(int r=0;r<R;++r)
#pragma unroll
      for(int head=0;head<heads;++head)
#pragma unroll
        for(int i=0;i<dim/workers;++i) qr[r][head][i]=r<n?h.query[(size_t(r0+r)*heads+head)*dim+lane+i*workers]:0.f;
    for(int b=b0;b<sycl::min(b0+B,nmax);++b) {
      const int last=4*b+3;
      const size_t at=(size_t(slot)*pitch+b)*dim;
      float kr[dim/workers],score[R]{};
#pragma unroll
      for(int i=0;i<dim/workers;++i) {
        const int d=lane+i*workers;
        kr[i]=!Tree && last>=start?h.closed[size_t(rseg+last-start)*dim+d]:
          (Half?static_cast<float>(h.pool16[at+d]):h.pool[at+d]);
      }
#pragma unroll
      for(int head=0;head<heads;++head)
#pragma unroll
        for(int r=0;r<R;++r) {
          float dot=qr[r][head][0]*kr[0];
#pragma unroll
          for(int i=1;i<dim/workers;++i) dot=sycl::fma(qr[r][head][i],kr[i],dot);
          dot+=sycl::permute_group_by_xor(sg,dot,16);
          dot+=sycl::permute_group_by_xor(sg,dot,8);
          dot+=sycl::permute_group_by_xor(sg,dot,4);
          dot+=sycl::permute_group_by_xor(sg,dot,2);
          dot+=sycl::permute_group_by_xor(sg,dot,1);
          score[r]+=sycl::fmax(0.f,dot);
        }
      if(!lane)
#pragma unroll
        for(int r=0;r<R;++r) if(r<n && (Tree || last<=start+j0+r)) scores[size_t(r0+r)*pitch+b]=sycl::bit_cast<uint32_t>(score[r]);
    }
  }});
}
}

struct Device::Impl {
  struct BankLayer {
    History h{};
    std::array<k::DenseWeights,3> w{};
    bool sparse_logged=false, long_logged=false, tree_logged=false;
  };
  struct Card {
    std::vector<BankLayer> layer;
    std::array<float*,3> x{};
    uint16_t *upload=nullptr;
    int32_t *thw=nullptr,*delta=nullptr,*open=nullptr;  // History::thw rows staged by set_rows_thw, delta, open
    int thw_rows=0;
    const int32_t *thw_now=nullptr;  // the running prefill's rows, for its plan
    size_t allocated=0;
    k::QsaScratch sc{};
    int32_t *entries=nullptr,*counts=nullptr,*status=nullptr;
    struct Prefill {
      k::QsaScratch sc{};
      float *query=nullptr;
      uint32_t *subtiles=nullptr;
      int32_t *pages=nullptr,*counts=nullptr,*entries=nullptr;
      uint16_t *masks=nullptr;
      k::gqa_bf16 *xmx_queries=nullptr,*xmx_keys=nullptr;
    } pf;
    int scored_rows=0;
  };
  Fabric &fab;
  std::vector<Card> cards;
  std::vector<void*> held;
  const int slots, cap, prefill_rows, step_rows;
  const float eps;
  const bool subtile=enabled("XEN_PF_SUBTILE_SKIP");
  const bool tile=enabled("XEN_TRAINED_QSA_TILED_SCORE");
  const bool tree=enabled("XEN_TRAINED_QSA_TREE");
  const bool compact=enabled("XEN_TRAINED_QSA_SHARE_SCRATCH");
  const bool stream=enabled("XEN_TRAINED_QSA_STREAM_PROJECTION");
  const bool half=enabled("XEN_TRAINED_QSA_POOL_FP16");
  size_t projection_words=0,host_bytes=0;
  uint16_t *host_projection=nullptr;
  // Projection, selector and host-to-device uploads run on the same in-order
  // compute queue per rank. The next upload cannot overwrite an operand until
  // its GEMM finishes. The host weights remain immutable until destruction.
  k::DenseWeights projection(Card &d,const BankLayer &l,sycl::queue &q,int layer,int i) const {
    if(!stream) return l.w[i];
    q.memcpy(d.upload,host_projection+(size_t(layer)*3+i)*projection_words,projection_words*sizeof(uint16_t));
    return {d.upload,nullptr,k::DenseKind::bf16,width,l.w[i].k};
  }
  Impl(Fabric &f,int s,int c,int pf,int st,float e,size_t layers):fab(f),cards(f.size()),slots(s),cap(c),prefill_rows(pf),step_rows(st),eps(e) { held.reserve((16+8*layers)*f.size()); }
  ~Impl() { for(auto *p:held) fab.release(p); }
  template<class T> T *alloc(int card,size_t count) {
    if (!count || count>std::numeric_limits<size_t>::max()/sizeof(T))
      throw std::overflow_error("trained QSA layer3: allocation size overflow");
    const size_t bytes=allocation_bytes(count*sizeof(T));
    try {
      auto *p=static_cast<T*>(fab.alloc(card,bytes));
      held.push_back(p); cards[card].allocated+=bytes; return p;
    } catch (const std::exception &e) {
      throw std::runtime_error("trained QSA allocation failed: rank="+std::to_string(card)+" bytes="+std::to_string(bytes)+": "+e.what());
    }
  }
};
size_t Device::bytes_per_rank(size_t layers,int hidden,int slots,int cap,int pf,int st,bool xmx_prefill) {
  if (!layers || layers>64 || hidden!=5120 || slots<1 || slots>k::kSegs16 || cap<64 || cap>kMaxContext || cap%k::kQsaPage ||
      pf<1 || pf>cap || st<1 || st>k::kStepRows32)
    throw std::invalid_argument("trained QSA: invalid layer, hidden, context, slot or row capacity");
  const size_t eligible=cap/k::kQsaCompress;
  const int chunk=k::qsa_select_chunk(static_cast<int>(eligible));
  if ((eligible+chunk-1)/chunk>k::kQsaMaxChunks) throw std::invalid_argument("trained QSA: selector chunk capacity exceeded");
  const auto b=[](size_t n,size_t sz=4) { return std::max(size_t(4096),n*sz); };  // extents bounded above
  const bool half=enabled("XEN_TRAINED_QSA_POOL_FP16");
  if(enabled("XEN_TRAINED_QSA_TREE") && !enabled("XEN_TRAINED_QSA_TILED_SCORE"))
    throw std::invalid_argument("trained QSA tree path requires tiled scoring");
  const size_t words=size_t(slots)*eligible*dim;
  const size_t layer=b(size_t(slots)*4*dim)+b(words,half?sizeof(sycl::half):sizeof(float))+2*b(size_t(st)*dim)+b(3*dim);
  const size_t one_projection=b(k::dense_packed_elems(width,hidden),2);
  const size_t weights=(enabled("XEN_TRAINED_QSA_STREAM_PROJECTION")?1:3*layers)*one_projection;
  const ScratchLayout sc(pf,st,static_cast<int>(eligible));
  const size_t scratch=enabled("XEN_TRAINED_QSA_SHARE_SCRATCH")?sc.peak():3*sc.proj+sc.score+sc.nc+2*sc.candidate;
  const size_t pages=cap/k::kQsaPage, groups=(pf+pf_fa_min_rows-1)/pf_fa_min_rows;
  const size_t pc=size_t(pf_score_tile)*k::kQsaMaxChunks;
  const size_t prefill=xmx_prefill ?
      b(size_t(pf_score_tile)*heads*dim)+b(groups*pages)+b(groups)+b(size_t(pf)*pages,2)+
      b(size_t(pf_score_tile)*heads*dim,2)+b(eligible*dim,2)+
      2*b(pf_score_tile)+b(size_t(pf_score_tile)*k::kTrainedQsaTopBlocks)+b(size_t(pf_score_tile)*eligible)+
      b(size_t(pf_score_tile)*k::kTrainedQsaPages)+(eligible>size_t(chunk)?b(pc)+2*b(pc*k::kQsaTopBlocks):0):0;
  const bool subtile=enabled("XEN_PF_SUBTILE_SKIP");
  if(subtile && !xmx_prefill) throw std::invalid_argument("prefill subtiles require serving XMX prefill");
  const size_t subtile_bytes=subtile?b(size_t((pf+39)/40)*pages):0;
  return subtile_bytes+prefill+layers*layer+weights+scratch+b(size_t(cap)*64)+b(size_t(pf)*dim)+b(size_t(st)*heads*dim)+
      3*b(st)+b(size_t(st)*k::kTrainedQsaTopBlocks)+b(size_t(st)*k::kTrainedQsaPages)+b(size_t(st)*4)+
      b(size_t(pf)*3)+b(slots)+b(size_t(slots)*4);  // M-RoPE rows, deltas, open entries
}
Device::Device(Fabric &fab,std::span<const Layer> weights,int slots,int cap,int pf,int st,bool xmx_prefill) {
  const size_t planned=bytes_per_rank(weights.size(),weights.empty()?0:weights.front().hidden(),slots,cap,pf,st,xmx_prefill);
  for (const auto &w:weights) if(w.hidden()!=weights.front().hidden() || w.epsilon()!=weights.front().epsilon() || w.theta()!=weights.front().theta())
    throw std::invalid_argument("trained QSA: incompatible layer geometry");
  p_=std::make_unique<Impl>(fab,slots,cap,pf,st,weights.front().epsilon(),weights.size());
  const int eligible=cap/k::kQsaCompress;
  const int chunk=k::qsa_select_chunk(eligible);
  const int nch=(eligible-1)/chunk+1;
  const size_t candidate_pitch=size_t(st)*k::kQsaMaxChunks;
  const ScratchLayout sc(pf,st,eligible);
  p_->projection_words=k::dense_packed_elems(width,weights.front().hidden());
  if(p_->stream) {
    p_->host_bytes=weights.size()*3*p_->projection_words*sizeof(uint16_t);
    p_->host_projection=static_cast<uint16_t*>(fab.alloc_host(p_->host_bytes));
    p_->held.push_back(p_->host_projection);
  }
  std::vector<float> rope(size_t(cap)*64);
  const double base=std::pow(weights.front().theta(),-1.0/32);
  for(int p=0;p<cap;++p) for(int j=0;j<32;++j) {
    const double phase=p*std::pow(base,j);
    rope[p*64+j]=std::cos(phase); rope[p*64+j+32]=std::sin(phase);
  }
  std::vector<History> shared(fab.size());
  for(int c=0;c<int(fab.size());++c) {
    auto &d=p_->cards[c]; auto &q=fab.compute(c);
    auto &h=shared[c]; h.cap=cap;
    h.query=p_->alloc<float>(c,size_t(st)*heads*dim);
    h.pf_raw=p_->alloc<float>(c,size_t(pf)*dim);
    auto *rt=p_->alloc<float>(c,rope.size()); h.rope=rt;
    q.memcpy(rt,rope.data(),rope.size()*sizeof(float)).wait_and_throw();
    d.thw=p_->alloc<int32_t>(c,size_t(pf)*3);
    d.delta=p_->alloc<int32_t>(c,slots); h.delta=d.delta;
    d.open=p_->alloc<int32_t>(c,size_t(slots)*4); h.open=d.open;
    q.fill(d.delta,int32_t(0),size_t(slots));
    q.fill(d.open,int32_t(-1),size_t(slots)*4).wait_and_throw();
    d.layer.resize(weights.size());
    d.sc.nblk=p_->alloc<int32_t>(c,st);
    d.sc.sel=p_->alloc<int32_t>(c,size_t(st)*k::kTrainedQsaTopBlocks);
    d.sc.sel_n=p_->alloc<int32_t>(c,st);
    if(p_->stream) d.upload=p_->alloc<uint16_t>(c,p_->projection_words);
    if(p_->compact) {
      // A prefill or a step projects into three live FP32 planes. Norm consumes
      // them before score writes the same storage. Scores and chunk candidates
      // *are* live together, so give those separate panes of the second phase.
      auto *mem=p_->alloc<std::byte>(c,sc.peak());
      for(int i=0;i<3;++i) d.x[i]=reinterpret_cast<float*>(mem+i*align64(sc.proj));
      d.sc.score=reinterpret_cast<uint32_t*>(mem);
      if(nch>1) {
        d.sc.cand_n=reinterpret_cast<int32_t*>(mem+align64(sc.score));
        d.sc.cand_id=reinterpret_cast<int32_t*>(mem+align64(sc.score)+align64(sc.nc));
        d.sc.cand_key=reinterpret_cast<uint32_t*>(mem+align64(sc.score)+align64(sc.nc)+align64(sc.candidate));
      }
    } else {
      for(auto &x:d.x) x=p_->alloc<float>(c,size_t(std::max(pf,k::row_shape(st)))*width);
      d.sc.score=p_->alloc<uint32_t>(c,size_t(st)*eligible);
      if(nch>1) {
        d.sc.cand_n=p_->alloc<int32_t>(c,candidate_pitch);
        d.sc.cand_id=p_->alloc<int32_t>(c,candidate_pitch*k::kQsaTopBlocks);
        d.sc.cand_key=p_->alloc<uint32_t>(c,candidate_pitch*k::kQsaTopBlocks);
      }
    }
    d.entries=p_->alloc<int32_t>(c,size_t(st)*k::kTrainedQsaPages);
    d.counts=p_->alloc<int32_t>(c,size_t(st)*4);
    d.status=p_->alloc<int32_t>(c,st);
    if(xmx_prefill) {
      auto &w=d.pf;
      const size_t pages=cap/k::kQsaPage, groups=(pf+pf_fa_min_rows-1)/pf_fa_min_rows;
      if(p_->subtile) w.subtiles=p_->alloc<uint32_t>(c,size_t((pf+39)/40)*pages);
      w.query=p_->alloc<float>(c,size_t(pf_score_tile)*heads*dim);
      w.pages=p_->alloc<int32_t>(c,groups*pages); w.counts=p_->alloc<int32_t>(c,groups);
      w.masks=p_->alloc<uint16_t>(c,size_t(pf)*pages);
      w.xmx_queries=p_->alloc<k::gqa_bf16>(c,size_t(pf_score_tile)*heads*dim);
      w.xmx_keys=p_->alloc<k::gqa_bf16>(c,size_t(eligible)*dim);
      w.sc.nblk=p_->alloc<int32_t>(c,pf_score_tile); w.sc.sel_n=p_->alloc<int32_t>(c,pf_score_tile);
      w.sc.sel=p_->alloc<int32_t>(c,size_t(pf_score_tile)*k::kTrainedQsaTopBlocks);
      w.sc.score=p_->alloc<uint32_t>(c,size_t(pf_score_tile)*eligible);
      w.entries=p_->alloc<int32_t>(c,size_t(pf_score_tile)*k::kTrainedQsaPages);
      if(nch>1) {
        const size_t pitch=size_t(pf_score_tile)*k::kQsaMaxChunks;
        w.sc.cand_n=p_->alloc<int32_t>(c,pitch);
        w.sc.cand_id=p_->alloc<int32_t>(c,pitch*k::kQsaTopBlocks);
        w.sc.cand_key=p_->alloc<uint32_t>(c,pitch*k::kQsaTopBlocks);
      }
    }
  }
  for(size_t l=0;l<weights.size();++l) {
    const auto tiles=weights[l].device_tiles();
    if(p_->stream) for(int i=0;i<3;++i) {
      if(tiles[i].size()!=p_->projection_words) throw std::logic_error("trained QSA projection extent mismatch");
      std::copy(tiles[i].begin(),tiles[i].end(),p_->host_projection+(l*3+i)*p_->projection_words);
    }
    const auto norms=weights[l].multipliers();
    for(int c=0;c<int(fab.size());++c) {
      auto &d=p_->cards[c].layer[l]; auto &q=fab.compute(c);
      d.h=shared[c];
      d.h.tail=p_->alloc<float>(c,size_t(slots)*4*dim);
      if(p_->half) d.h.pool16=p_->alloc<sycl::half>(c,size_t(slots)*eligible*dim);
      else d.h.pool=p_->alloc<float>(c,size_t(slots)*eligible*dim);
      d.h.staged=p_->alloc<float>(c,size_t(st)*dim);
      d.h.closed=p_->alloc<float>(c,size_t(st)*dim);
      auto *nr=p_->alloc<float>(c,3*dim); d.h.norm=nr;
      q.memcpy(nr,norms.data(),sizeof(norms)).wait_and_throw();
      for(int i=0;i<3;++i) {
        uint16_t *w=nullptr;
        if(!p_->stream) {
          w=p_->alloc<uint16_t>(c,tiles[i].size());
          q.memcpy(w,tiles[i].data(),tiles[i].size()*sizeof(uint16_t)).wait_and_throw();
        }
        d.w[i]={w,nullptr,k::DenseKind::bf16,width,weights[l].hidden()};
      }
    }
  }
  for(size_t c=0;c<p_->cards.size();++c) {
    const size_t actual=p_->cards[c].allocated;
    if(actual!=planned) throw std::logic_error("trained QSA allocation ledger differs from preflight");
    std::fprintf(stderr,"trained QSA allocation: rank=%zu layers=%zu allocated_device_bytes=%zu shared_host_projection_bytes=%zu compact=%d stream=%d pool_bits=%d\n",
                 c,weights.size(),actual,p_->host_bytes,p_->compact,p_->stream,p_->half?16:32);
  }
}
Device::~Device()=default;
namespace {
void publish_tail(sycl::queue &q,const History &h,int slot,int pos,int rows,const float *raw) {
  const int end=pos+rows, from=std::max(pos,end-end%4), n=end-from;
  if (n>0) q.memcpy(h.tail+(size_t(slot)*4+from%4)*dim,raw+size_t(from-pos)*dim,size_t(n)*dim*sizeof(float));
}
}
k::GqaTrainedPrefill Device::prefill(int card,const k::gqa_bf16 *x,int rows,int slot,int pos,int bank_layer,int fa_rows,int thw_row0) {
  if(rows<1 || rows>p_->prefill_rows || slot<0 || slot>=p_->slots || pos<0 || pos>p_->cap-rows)
    throw std::invalid_argument("trained QSA layer3: invalid prefill slot/positions");
  auto &q=p_->fab.compute(card); auto &d=p_->cards.at(card);
  const auto &l=d.layer.at(bank_layer);
  for(int i=0;i<3;++i) {
    const auto w=p_->projection(d,l,q,bank_layer,i);
    k::pf_dense(q,rows,w,x,w.k,k::PfOut{d.x[i],nullptr,width});
  }
  History h=l.h;
  if(thw_row0>=0) {
    if(thw_row0+rows>d.thw_rows) throw std::logic_error("trained QSA: the prefill's M-RoPE rows were not staged");
    h.thw=d.thw+size_t(thw_row0)*3;
  }
  norm_index(q,h,d.x,p_->eps,rows,slot,pos,{},false);
  index_blocks(q,h,p_->eps,rows,slot,pos,{},nullptr,false);
  publish_tail(q,h,slot,pos,rows,h.pf_raw);
  if(bank_layer+1==int(d.layer.size())) {
    // after every bank read the old entry: the open block's first row, if this call wrote it
    const int end=pos+rows, first=end-end%4;
    q.single_task([=] {
      int32_t *o=h.open+size_t(slot)*4;
      if(first==end) { o[0]=-1; return; }
      if(first<pos) return;  // the block opened before this call; its entry stands
      const Thw p=row_thw(h,first-pos,slot,first);
      o[0]=first; o[1]=p.t; o[2]=p.h; o[3]=p.w;
    });
  }
  if(!fa_rows) return {};  // the default still builds only the decode history
  d.thw_now=h.thw;
  auto plan=prefill_plan(card,rows,slot,pos,bank_layer,fa_rows,p_->subtile);
  d.thw_now=nullptr;
  return plan;
}
void Device::set_rows_thw(int card,const int32_t *thw,int rows) {
  auto &d=p_->cards.at(card);
  if(!thw) { d.thw_rows=0; return; }
  if(rows<1 || rows>p_->prefill_rows) throw std::invalid_argument("trained QSA: invalid M-RoPE row count");
  p_->fab.compute(card).memcpy(d.thw,thw,size_t(rows)*3*sizeof(int32_t)).wait_and_throw();
  d.thw_rows=rows;
}
void Device::set_delta(int card,int slot,int delta) {
  if(slot<0 || slot>=p_->slots) throw std::invalid_argument("trained QSA: invalid delta slot");
  p_->fab.compute(card).fill(p_->cards.at(card).delta+slot,int32_t(delta),1);
}
void Device::set_mrope(int sec_h,int sec_w) {
  uint32_t hm=0,wm=0;
  for(int i=0;i<32;++i) {
    if(i%3==1 && i<3*sec_h) hm|=1u<<i;
    else if(i%3==2 && i<3*sec_w) wm|=1u<<i;
  }
  for(auto &c:p_->cards) for(auto &l:c.layer) { l.h.hmask=hm; l.h.wmask=wm; }
}
k::GqaTrainedPrefill Device::prefill_plan(int card,int rows,int slot,int pos,int bank,int fa_rows,bool skip_subtiles) {
  auto &q=p_->fab.compute(card); auto &d=p_->cards.at(card);
  const auto &l=d.layer.at(bank);
  auto &w=d.pf;
  if(!q.is_in_order() || !w.masks || (fa_rows!=16 && fa_rows!=32 && fa_rows!=40) || pos/4<=k::kTrainedQsaTopBlocks)
    throw std::invalid_argument("trained XMX prefill: expected sparse workspace, supported FA geometry and long prefix");
  if(skip_subtiles && (!w.subtiles || fa_rows!=40)) throw std::invalid_argument("prefill subtile path needs native 40-row groups");
  auto *subtiles=skip_subtiles?w.subtiles:nullptr;
  const int pitch=p_->cap/k::kQsaPage, live_pages=(pos+rows+k::kQsaPage-1)/k::kQsaPage;
  auto *masks=w.masks;
  auto *pages=w.pages;
  auto *counts=w.counts;
  q.fill(masks,uint16_t(0),size_t(rows)*pitch);
  const size_t origin=size_t(slot)*(p_->cap/4)*dim;
  // Keep the existing FP16 or FP32 resident summaries. Convert to the upstream
  // BF16 XMX operand layout only for this chunk; no changes to decode history.
  if(l.h.pool16) k::qsa_trained_xmx_pack_k(q,l.h.pool16+origin,(pos+rows)/4,w.xmx_keys);
  else k::qsa_trained_xmx_pack_k(q,l.h.pool+origin,(pos+rows)/4,w.xmx_keys);
  History h=l.h;
  h.query=w.query;
  h.thw=d.thw_now;
  for(int q0=0;q0<rows;q0+=pf_score_tile) {
    const int n=std::min(pf_score_tile,rows-q0), start=pos+q0, bound=(start+n)/4;
    std::array<float*,3> projected=d.x;
    for(auto &p:projected) p+=size_t(q0)*width;
    History ht=h;
    if(ht.thw) ht.thw+=size_t(q0)*3;
    norm_index<true>(q,ht,projected,p_->eps,n,slot,start,{},false);
    k::qsa_trained_xmx_pack_q(q,w.query,n,w.xmx_queries);
    k::qsa_trained_xmx_score(q,w.xmx_queries,w.xmx_keys,n,start,bound,w.sc.score,p_->cap/4,w.sc.nblk);
    k::qsa_select(q,n,w.sc.score,p_->cap/4,w.sc.nblk,bound,w.sc);
    const auto *ids=w.sc.sel,*ns=w.sc.sel_n;
    auto *entries=w.entries;
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id) {
      const int r=id[0], rp=start+r;
      auto *row=masks+size_t(q0+r)*pitch;
      auto *packed=entries+size_t(r)*k::kTrainedQsaPages;
      const int count=k::trained_qsa_pack_row(rp,ids+size_t(r)*k::kTrainedQsaTopBlocks,ns[r],packed);
      if (count<=0) {  // same safe fallback as trained decode; the FA causal mask still applies
        for(int pg=0;pg<=rp/k::kQsaPage;++pg) row[pg]=0xffffu;
      } else {
        for(int j=0;j<count;++j) row[k::trained_qsa_page(packed[j])]=k::trained_qsa_mask(packed[j]);
      }
    });
  }
  // Compact the union in increasing logical page order. FA retains the row's
  // mask, not the union mask; the latter decides what to stage exactly once.
  constexpr int items=128;
  const int blocks=(rows+fa_rows-1)/fa_rows;
  auto build_union=[&]<bool UseSubtiles>() {
  q.parallel_for(sycl::nd_range<1>(size_t(blocks)*items,items),[=](sycl::nd_item<1> it) {
    const int g=it.get_group(0),lane=it.get_local_id(0),r0=g*fa_rows;
    const auto group=it.get_group();
    const int rn=sycl::min(r0+fa_rows,rows), pn=(pos+rn+k::kQsaPage-1)/k::kQsaPage;
    int base=0;
    for(int p0=0;p0<pn;p0+=items) {
      const int pg=p0+lane;
      bool keep=false;
      uint32_t work=0;
      if(pg<pn && pg<live_pages) {
        if constexpr(UseSubtiles) {
          for(int r=r0;r<rn;r+=8) {
            uint16_t bits=0;
            for(int j=r;j<sycl::min(r+8,rn);++j) bits|=masks[size_t(j)*pitch+pg];
            const uint32_t nibble=uint32_t((bits&0x000fu)!=0)|(uint32_t((bits&0x00f0u)!=0)<<1)|
                                 (uint32_t((bits&0x0f00u)!=0)<<2)|(uint32_t((bits&0xf000u)!=0)<<3);
            work|=nibble|(nibble<<(4+(r-r0)/2));
          }
          keep=(work&15)!=0;
        } else for(int r=r0;r<rn && !keep;++r) keep=masks[size_t(r)*pitch+pg]!=0;
      }
      const int before=sycl::exclusive_scan_over_group(group,int(keep),sycl::plus<int>());
      if (keep) {
        const size_t i=size_t(g)*pitch+base+before;
        pages[i]=pg;
        if constexpr(UseSubtiles) subtiles[i]=work;
      }
      base+=sycl::group_broadcast(group,before+int(keep),items-1);
    }
    if (!lane) counts[g]=base;
  });
  };
  if(skip_subtiles) build_union.template operator()<true>();
  else build_union.template operator()<false>();
  return {pages,counts,masks,pitch,fa_rows,subtiles};
}

void Device::stage(int card,const k::gqa_bf16 *x,int rows,int hkv,k::GqaStep &step,int bank_layer,int model_layer,uint32_t track,bool sparse,uint32_t dense_rows) {
  if(rows<1 || rows>p_->step_rows || step.pos_bound<rows || step.pos_bound>p_->cap)
    throw std::invalid_argument("trained QSA layer3: invalid verification chain");
  auto &q=p_->fab.compute(card); auto &d=p_->cards.at(card);
  auto &l=d.layer.at(bank_layer);
  d.scored_rows=0;
  if(!track) return;
  const uint32_t all=rows==32?~uint32_t(0):((uint32_t(1)<<rows)-1);
  if(sparse && (track&all&~dense_rows)!=(all&~dense_rows)) throw std::logic_error("trained QSA: sparse rows must be tracked");
  for(int i=0;i<3;++i) k::dense_gemm(q,rows,x,p_->projection(d,l,q,bank_layer,i),d.x[i]);
  const bool tree_sparse=step.tree && sparse && p_->tree;
  if(tree_sparse && k::seg_size(rows,step.seg_rows)>k::kTreeRows)
    throw std::logic_error("trained sparse tree segments hold at most 16 rows");
  norm_index(q,l.h,d.x,p_->eps,rows,0,0,step,true,track,tree_sparse);
  const bool can_tile=step.stride>=k::kSegRec && k::seg_rows_ok(rows,step.seg_rows,step.stride) && k::seg_size(rows,step.seg_rows)<=16;
  const bool tiled=p_->tile && can_tile;
  if(step.tree && !tree_sparse) return;
  if(!step.tree) index_blocks(q,l.h,p_->eps,rows,0,0,step,d.sc.nblk,true,track);
  if(!sparse) return;  // retained history only; target runs the unchanged dense kernel
  const int bound=step.pos_bound/k::kQsaCompress;
  if (bound<=k::kTrainedQsaTopBlocks) return;
  if(tree_sparse) {
    if(!tiled) throw std::logic_error("trained sparse tree requires tiled layout");
    // The page planner keeps the entire current ancestor path. Rank only
    // committed summaries, never staged keys from any speculative branch.
    static const bool xmx_score=[]{const char *e=std::getenv("XEN_QSA_XMX_SCORE");return e && e[0]=='1';}();
    static const int score_check=[]{const char *e=std::getenv("XEN_QSA_SCORE_CHECK");return e?std::atoi(e):0;}();
    static int score_checked=0;
    if(p_->half && xmx_score) {
      if(score_check>0 && score_checked<score_check && card==0) {
        ++score_checked;
        const int pitch=l.h.cap/4;
        // reference into the score buffer, copy out, then the DPAS kernel over it
        index_scores_tile<true,true>(q,l.h,rows,bound,step,d.sc.score,d.sc.nblk);
        std::vector<uint32_t> ref(size_t(rows)*pitch),got(size_t(rows)*pitch);
        std::vector<int32_t> nb(rows);
        q.memcpy(ref.data(),d.sc.score,ref.size()*4).wait();
        k::qsa_trained_dec_xmx_score(q,l.h.query,l.h.pool16,step.pos,step.stride,step.seg_rows,rows,pitch,bound,d.sc.score,d.sc.nblk);
        q.memcpy(got.data(),d.sc.score,got.size()*4).wait();
        q.memcpy(nb.data(),d.sc.nblk,nb.size()*4).wait();
        double maxrel=0; long overlap=0,total=0;
        for(int r=0;r<rows;++r) {
          const int nn=nb[r]; if(nn<=0) continue;
          std::vector<std::pair<float,int>> a(nn),b(nn);
          for(int i=0;i<nn;++i) {
            const float x=sycl::bit_cast<float>(ref[size_t(r)*pitch+i]),y=sycl::bit_cast<float>(got[size_t(r)*pitch+i]);
            maxrel=std::max(maxrel,double(std::fabs(x-y))/std::max(1e-6,double(std::fabs(x))));
            a[i]={-x,i}; b[i]={-y,i};
          }
          const int K=std::min(nn,k::kTrainedQsaTopBlocks);
          std::nth_element(a.begin(),a.begin()+K-1,a.end()); std::nth_element(b.begin(),b.begin()+K-1,b.end());
          std::vector<int> sa,sb; for(int i=0;i<K;++i){sa.push_back(a[i].second);sb.push_back(b[i].second);}
          std::sort(sa.begin(),sa.end()); std::sort(sb.begin(),sb.end());
          std::vector<int> in; std::set_intersection(sa.begin(),sa.end(),sb.begin(),sb.end(),std::back_inserter(in));
          overlap+=long(in.size()); total+=K;
        }
        std::fprintf(stderr,"QSA_SCORE_CHECK n=%d rows=%d bound=%d maxrel=%.3g top512_overlap=%ld/%ld\n",score_checked,rows,bound,maxrel,overlap,total);
      } else {
        k::qsa_trained_dec_xmx_score(q,l.h.query,l.h.pool16,step.pos,step.stride,step.seg_rows,rows,l.h.cap/4,bound,d.sc.score,d.sc.nblk);
      }
    } else if(p_->half) index_scores_tile<true,true>(q,l.h,rows,bound,step,d.sc.score,d.sc.nblk);
    else index_scores_tile<false,true>(q,l.h,rows,bound,step,d.sc.score,d.sc.nblk);
  } else if(tiled || (p_->half && k::seg_size(rows,step.seg_rows)<=16 && (step.seg_rows>=0 || step.stride>=k::kSegRec) && [] {const char *e=std::getenv("XEN_QSA_XMX_SCORE");return e && e[0]=='1';}())) {
    static const bool xmx_chain=[]{const char *e=std::getenv("XEN_QSA_XMX_SCORE");return e && e[0]=='1';}();
    if(p_->half && xmx_chain && k::seg_size(rows,step.seg_rows)<=16) {
      static const int score_check=[]{const char *e=std::getenv("XEN_QSA_SCORE_CHECK");return e?std::atoi(e):0;}();
      static int score_checked_c=0;
      if(score_check>0 && score_checked_c<score_check && card==0) {
        ++score_checked_c;
        const int pitch=l.h.cap/4;
        // reference into the score buffer, copy out, then the DPAS kernel over it
        index_scores_tile<true>(q,l.h,rows,bound,step,d.sc.score);
        std::vector<uint32_t> ref(size_t(rows)*pitch),got(size_t(rows)*pitch);
        std::vector<int32_t> nb(rows);
        q.memcpy(ref.data(),d.sc.score,ref.size()*4).wait();
        k::qsa_trained_dec_xmx_score(q,l.h.query,l.h.pool16,step.pos,step.stride,step.seg_rows,rows,pitch,bound,d.sc.score,nullptr);
        index_scores_tile<true,false,true>(q,l.h,rows,bound,step,d.sc.score);
        q.memcpy(got.data(),d.sc.score,got.size()*4).wait();
        q.memcpy(nb.data(),d.sc.nblk,nb.size()*4).wait();
        double maxrel=0; long overlap=0,total=0;
        for(int r=0;r<rows;++r) {
          const int nn=nb[r]; if(nn<=0) continue;
          std::vector<std::pair<float,int>> a(nn),b(nn);
          for(int i=0;i<nn;++i) {
            const float x=sycl::bit_cast<float>(ref[size_t(r)*pitch+i]),y=sycl::bit_cast<float>(got[size_t(r)*pitch+i]);
            maxrel=std::max(maxrel,double(std::fabs(x-y))/std::max(1e-6,double(std::fabs(x))));
            a[i]={-x,i}; b[i]={-y,i};
          }
          const int K=std::min(nn,k::kTrainedQsaTopBlocks);
          std::nth_element(a.begin(),a.begin()+K-1,a.end()); std::nth_element(b.begin(),b.begin()+K-1,b.end());
          std::vector<int> sa,sb; for(int i=0;i<K;++i){sa.push_back(a[i].second);sb.push_back(b[i].second);}
          std::sort(sa.begin(),sa.end()); std::sort(sb.begin(),sb.end());
          std::vector<int> in; std::set_intersection(sa.begin(),sa.end(),sb.begin(),sb.end(),std::back_inserter(in));
          overlap+=long(in.size()); total+=K;
        }
        std::fprintf(stderr,"QSA_SCORE_CHECK chain n=%d rows=%d bound=%d maxrel=%.3g top512_overlap=%ld/%ld\n",score_checked_c,rows,bound,maxrel,overlap,total);
      } else {
      k::qsa_trained_dec_xmx_score(q,l.h.query,l.h.pool16,step.pos,step.stride,step.seg_rows,rows,l.h.cap/4,bound,d.sc.score,nullptr);
      index_scores_tile<true,false,true>(q,l.h,rows,bound,step,d.sc.score);
      }
    } else if(p_->half) index_scores_tile<true>(q,l.h,rows,bound,step,d.sc.score);
    else index_scores_tile(q,l.h,rows,bound,step,d.sc.score);
  } else {
    if(p_->half) index_scores<true>(q,l.h,rows,bound,step,d.sc.score);
    else index_scores(q,l.h,rows,bound,step,d.sc.score);
  }
  k::qsa_select(q,rows,d.sc.score,p_->cap/4,d.sc.nblk,bound,d.sc);
  static const bool gather=[]{const char *e=std::getenv("XEN_QSA_GATHER");return e && e[0]=='1';}();
  k::gqa_pack_trained_qsa(q,rows,hkv,step,d.sc.sel,d.sc.sel_n,d.entries,d.counts,d.status,dense_rows,gather);
  static const int pack_check=[]{const char *e=std::getenv("XEN_QSA_PACK_CHECK");return e?std::atoi(e):0;}();
  static int pack_checked=0;
  if(pack_check>0 && pack_checked<pack_check && card==0) {
    ++pack_checked;
    q.wait();
    const int K=k::kTrainedQsaTopBlocks,E=k::kTrainedQsaPages;
    std::vector<int32_t> ids(size_t(rows)*K),ns(rows),ent(size_t(rows)*E),st(rows),rc(size_t(step.stride)*64);
    q.memcpy(ids.data(),d.sc.sel,ids.size()*4); q.memcpy(ns.data(),d.sc.sel_n,ns.size()*4);
    q.memcpy(ent.data(),d.entries,ent.size()*4); q.memcpy(st.data(),d.status,st.size()*4);
    const int nseg=k::seg_count(rows,step.seg_rows);
    q.memcpy(rc.data(),step.pos,size_t(nseg)*step.stride*4).wait();
    int bad=0;
    for(int r=0;r<rows;++r) {
      const auto sr=k::seg_row(rc.data(),step.stride,step.seg_rows,r);
      int32_t ref[E]; int want=-1;
      if(sr.pos>=0 && sr.pos<k::kQsaMaxTokens) {
        if(r<32 && ((dense_rows>>r)&1u)) want=0;
        else if((sr.pos+1)/k::kQsaCompress<=K) want=0;
        else if(step.tree) want=k::trained_qsa_pack_tree_row(rc[sr.seg*step.stride+k::kSegPos],k::seg_vpos(rc.data(),step.stride,sr),ids.data()+size_t(r)*K,ns[r],ref);
        else want=k::trained_qsa_pack_row(sr.pos,ids.data()+size_t(r)*K,ns[r],ref);
      }
      if(gather && want>0) {  // the gather form of the same selection: ids, tiles, then the extra blocks' pages
        const int n=ns[r],ng=(n+15)/16;
        std::vector<int32_t> blocks;
        for(int i=0;i<want;++i) for(int b=0;b<16;++b) if((ref[i]>>b)&1) blocks.push_back((k::trained_qsa_page(ref[i])<<4)+b);
        std::vector<int32_t> g;
        for(int t=0;t<ng;++t) {const int nb=std::min(16,n-16*t); g.push_back(k::kTrainedQsaGather|(t<<16)|int32_t(nb==16?0xffffu:(1u<<nb)-1u));}
        for(size_t i=size_t(n);i<blocks.size();++i) {const int pi=blocks[i]>>4,bit=1<<(blocks[i]&15); if(g.size()==size_t(ng) || k::trained_qsa_page(g.back())!=pi) g.push_back(pi<<16); g.back()|=bit;}
        bool idok=true;
        for(int i=0;i<n;++i) idok&=ent[size_t(r)*E+k::kTrainedQsaPlanEntries+i]==ids[size_t(r)*K+i] && blocks[size_t(i)]==ids[size_t(r)*K+i];
        want=idok?int(g.size()):-2;
        for(size_t i=0;i<g.size();++i) ref[i]=g[i];
      }
      bool ok=st[r]==want;
      for(int i=0;ok && i<want;++i) ok=ent[size_t(r)*E+i]==ref[i];
      if(!ok) ++bad;
    }
    std::fprintf(stderr,"QSA_PACK_CHECK n=%d rows=%d tree=%d mismatched_rows=%d\n",pack_checked,rows,int(step.tree),bad);
  }
  step.trained_qsa={d.entries,d.counts};
  d.scored_rows=rows;
  if (tree_sparse && model_layer>=0 && !l.tree_logged) {
    std::fprintf(stderr,"trained QSA sparse tree submit: rank=%d layer=%d rows=%d bound=%d\n",card,model_layer,rows,step.pos_bound);
    l.tree_logged=true;
  }
  if (model_layer>=0 && (!l.sparse_logged || (bound>=50000 && !l.long_logged))) {
    std::fprintf(stderr,"trained QSA sparse submit: rank=%d layer=%d rows=%d max_eligible=%d\n",card,model_layer,rows,bound);
    l.sparse_logged=true;
    l.long_logged|=bound>=50000;
  }
}
size_t Device::copy_aligned_prefix(int dst,int src,int pos) {
  if(dst==src || dst<0 || src<0 || dst>=p_->slots || src>=p_->slots ||
     pos<=0 || pos>p_->cap || pos%k::kQsaPage)
    throw std::invalid_argument("trained QSA snapshot copy requires distinct slots and a page-aligned prefix");
  static_assert(k::kQsaPage%k::kQsaCompress==0);
  const size_t stride=size_t(p_->cap/k::kQsaCompress)*dim;
  const size_t words=size_t(pos/k::kQsaCompress)*dim;
  for(size_t c=0;c<p_->cards.size();++c) {
    auto &q=p_->fab.compute(c);
    for(const auto &layer:p_->cards[c].layer) {
      const auto &h=layer.h;
      // both stores when both exist (the FP16 audit keeps an FP32 mirror beside the FP16 summaries)
      if(h.pool16) q.memcpy(h.pool16+size_t(dst)*stride,h.pool16+size_t(src)*stride,words*sizeof(sycl::half));
      if(h.pool) q.memcpy(h.pool+size_t(dst)*stride,h.pool+size_t(src)*stride,words*sizeof(float));
    }
  }
  // No partial group is visible at the aligned boundary. The next append
  // creates the destination's own tail and queries; neither is copied.
  return words*p_->cards.front().layer.size()*(p_->half?sizeof(sycl::half):sizeof(float));
}
void Device::accept(int slot,int pos,int row0,int keep,bool tree_batch,uint64_t path) {
  if(slot<0 || slot>=p_->slots || keep<1 || pos<0 || pos>p_->cap-keep || row0<0 || row0>p_->step_rows-keep ||
     (tree_batch && keep>p_->prefill_rows) || (path && (!tree_batch || keep>k::kTreeRows || row0+k::path_row(path,keep-1)>=p_->step_rows)))
    throw std::invalid_argument("trained QSA layer3: invalid accepted prefix");
  for(int c=0;c<int(p_->cards.size());++c) {
    auto &q=p_->fab.compute(c);
    for (const auto &l:p_->cards[c].layer) {
      const auto &h=l.h;
      if(tree_batch) {
        // Pack the accepted path (identity for chain segments in this
        // batch); use the same closure/norm/rotary path as linear prefill.
        q.parallel_for(sycl::range<1>(size_t(keep)*dim),[=](sycl::id<1> id) {
          const size_t i=id[0];
          h.pf_raw[i]=h.staged[(size_t(row0)+k::path_row(path,static_cast<int>(i/dim)))*dim+i%dim];
        });
        index_blocks(q,h,p_->eps,keep,slot,pos,{},nullptr,false);
        publish_tail(q,h,slot,pos,keep,h.pf_raw);
        continue;
      }
      publish_tail(q,h,slot,pos,keep,h.staged+size_t(row0)*dim);
      const int blocks=(pos+keep)/4-pos/4;
      if(blocks && h.pool16) {
        const int i0=(pos/4)*4+3-pos;
        auto *out=h.pool16+(size_t(slot)*(p_->cap/4)+pos/4)*dim;
        const auto *src=h.closed+size_t(row0+i0)*dim;
        q.parallel_for(sycl::range<1>(size_t(blocks)*dim),[=](sycl::id<1> idx) {
          const size_t i=idx[0]; out[i]=sycl::half(src[(i/dim)*4*dim+i%dim]);
        });
      }
      if(h.pool) for(int i=0;i<keep;++i) if ((pos+i)%4==3)
        q.memcpy(h.pool+(size_t(slot)*(p_->cap/4)+(pos+i)/4)*dim,h.closed+size_t(row0+i)*dim,dim*sizeof(float));
    }
  }
}
Device::Selection Device::read_selection(int card,int row) {
  if (row<0 || row>=p_->cards.at(card).scored_rows) throw std::invalid_argument("trained QSA layer3: diagnostic row was not scored");
  auto &q=p_->fab.compute(card); const auto &d=p_->cards.at(card);
  Selection out{};
  int count=0;
  q.memcpy(&out.eligible,d.sc.nblk+row,4);
  q.memcpy(&out.pages,d.status+row,4);
  q.memcpy(&count,d.sc.sel_n+row,4).wait_and_throw();
  if(out.pages<0 || out.eligible<=k::kTrainedQsaTopBlocks || count!=k::kTrainedQsaTopBlocks)
    throw std::runtime_error("trained QSA layer3: invalid diagnostic selection: eligible="+std::to_string(out.eligible)+
                             " pages="+std::to_string(out.pages)+" selected="+std::to_string(count));
  out.ids.resize(count);
  q.memcpy(out.ids.data(),d.sc.sel+size_t(row)*k::kTrainedQsaTopBlocks,size_t(count)*4).wait_and_throw();
  return out;
}
std::vector<int32_t> Device::read_logical_pages(int card,int row,int pages) {
  if (row<0 || row>=p_->cards.at(card).scored_rows || pages<0 || pages>k::kTrainedQsaPages)
    throw std::invalid_argument("trained QSA layer3: invalid page diagnostic bounds");
  std::vector<int32_t> entries(pages);
  if (pages) p_->fab.compute(card).memcpy(entries.data(),p_->cards.at(card).entries+size_t(row)*k::kTrainedQsaPages,
                                         entries.size()*sizeof(int32_t)).wait_and_throw();
  for (auto &entry:entries) entry=k::trained_qsa_page(entry);
  return entries;
}
void Device::check_selector_reference(int card,int row,const Selection &picked) {
  if (row<0 || row>=p_->cards.at(card).scored_rows || picked.eligible<1 || picked.eligible>p_->cap/k::kQsaCompress)
    throw std::invalid_argument("trained QSA layer3: invalid score diagnostic bounds");
  const int pitch=p_->cap/k::kQsaCompress;
  std::vector<uint32_t> scores(picked.eligible);
  p_->fab.compute(card).memcpy(scores.data(),p_->cards.at(card).sc.score+size_t(row)*pitch,
                              scores.size()*sizeof(uint32_t)).wait_and_throw();
  const auto reference=k::qsa_ref_select(scores.data(),picked.eligible,k::kTrainedQsaTopBlocks);
  if (picked.ids!=reference) throw std::runtime_error("trained QSA layer3: selector differs from CPU score-bit oracle at eligible="+
                                                     std::to_string(picked.eligible));
}
}  // namespace xenithium::arch::qwen35::trained_qsa
