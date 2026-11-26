// Xenithium · the vision encoder's oneDNN ops (xenithium/kernels/encoder/vit.hpp): linear layers with bias +
// activation / residual post-ops and the fused bidirectional attention (oneDNN Graph SDPA). Primitives
// and compiled partitions are cached per queue and shape (an image's patch count is its own shape: a new one compiles
// once).
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "xenithium/kernels/encoder/vit.hpp"

namespace xenithium::kernels {

namespace {

using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;
using md = dnnl::memory::desc;
namespace g = dnnl::graph;
using lt = g::logical_tensor;
using ldt = lt::data_type;
using lay = lt::layout_type;
using prop = lt::property_type;

void *dev_malloc(size_t size, size_t align, const void *dev, const void *ctx) {
  return sycl::aligned_alloc_device(align < 64 ? 64 : align, size, *static_cast<const sycl::device *>(dev), *static_cast<const sycl::context *>(ctx));
}
void dev_free(void *buf, const void *, const void *ctx, void *ev) {
  if (ev) static_cast<sycl::event *>(ev)->wait();
  sycl::free(buf, *static_cast<const sycl::context *>(ctx));
}

struct Attn {
  int seq = 0, heads = 0, hd = 0;
  g::compiled_partition cp;
  lt q, k, v, scale, out;
  // the scale as a {1} f32 tensor on the device: oneDNN 3.9 takes host_scalar inputs only as s32
  std::shared_ptr<float> scale_dev;
};

struct Dev {
  sycl::queue q;
  dnnl::engine eng;    // primitives
  dnnl::engine geng;   // graph (with the allocator)
  dnnl::stream st, gst;
  std::map<std::tuple<int, int, int, int, int, int, int>, dnnl::matmul> mm;
  std::list<Attn> attn;  // most recent first
};
std::mutex g_mu;
std::vector<std::unique_ptr<Dev>> g_devs;

Dev &dev_of(sycl::queue &q) {
  for (auto &d : g_devs)
    if (d->q == q) return *d;
  auto d = std::make_unique<Dev>();
  d->q = q;
  d->eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  d->st = dnnl::sycl_interop::make_stream(d->eng, q);
  static g::allocator alloc = g::sycl_interop::make_allocator(dev_malloc, dev_free);
  d->geng = g::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc);
  d->gst = dnnl::sycl_interop::make_stream(d->geng, q);
  g_devs.push_back(std::move(d));
  return *g_devs.back();
}

dnnl::memory mem(Dev &d, const md &m, const void *p) {
  return dnnl::sycl_interop::make_memory(m, d.eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void *>(p));
}

Attn build_attn(Dev &d, int seq, int heads, int hd) {
  Attn a;
  a.seq = seq, a.heads = heads, a.hd = hd;
  const int64_t row = int64_t(3) * heads * hd;  // qkv row
  const lt::dims qsz = {1, heads, seq, hd}, qst = {seq * row, hd, row, 1};
  const lt::dims osz = {1, heads, seq, hd}, ost = {int64_t(seq) * heads * hd, hd, int64_t(heads) * hd, 1};
  const lt::dims ssz = {1, heads, seq, seq};
  size_t id = 0;
  g::graph gr(dnnl::engine::kind::gpu);
  a.q = lt(id++, ldt::bf16, qsz, qst);
  a.k = lt(id++, ldt::bf16, qsz, qst);
  lt sc(id++, ldt::f32, ssz, lay::strided);
  g::op mm1(id++, g::op::kind::MatMul, "qk");
  mm1.set_attr<bool>(g::op::attr::transpose_b, true);
  mm1.add_inputs({a.q, a.k});
  mm1.add_outputs({sc});
  a.scale = lt(id++, ldt::f32, lt::dims{1}, lay::strided);
  {
    sycl::queue q = d.q;
    a.scale_dev = std::shared_ptr<float>(sycl::malloc_device<float>(1, q), [q](float *p) mutable { q.wait(); sycl::free(p, q); });
    const float v = 1.f / std::sqrt(static_cast<float>(hd));
    q.memcpy(a.scale_dev.get(), &v, sizeof v).wait();
  }
  lt scaled(id++, ldt::f32, ssz, lay::strided);
  g::op mul(id++, g::op::kind::Multiply, "scale");
  mul.add_inputs({sc, a.scale});
  mul.add_outputs({scaled});
  lt probs(id++, ldt::bf16, ssz, lay::strided);
  g::op sm(id++, g::op::kind::SoftMax, "softmax");
  sm.set_attr<int64_t>(g::op::attr::axis, -1);
  sm.add_inputs({scaled});
  sm.add_outputs({probs});
  a.v = lt(id++, ldt::bf16, qsz, qst);
  a.out = lt(id++, ldt::bf16, osz, ost);
  g::op mm2(id++, g::op::kind::MatMul, "pv");
  mm2.add_inputs({probs, a.v});
  mm2.add_outputs({a.out});
  for (auto *o : {&mm1, &mul, &sm, &mm2}) gr.add_op(*o);
  gr.finalize();
  auto ps = gr.get_partitions();
  if (ps.size() != 1 || !ps[0].is_supported())
    throw std::runtime_error("vit_attention: oneDNN did not take the SDPA pattern as one partition (" + std::to_string(ps.size()) + ")");
  a.cp = ps[0].compile({a.q, a.k, a.scale, a.v}, {a.out}, d.geng);
  return a;
}

}  // namespace

sycl::event vit_linear(sycl::queue &q, int m, int n, int k, const vit_bf16 *a, int lda, const vit_bf16 *w, const vit_bf16 *bias, VitAct act,
                       const vit_bf16 *res, vit_bf16 *out, int ldo) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  const md amd({m, k}, dt::bf16, dnnl::memory::dims{lda, 1}), wmd({k, n}, dt::bf16, tag::ba), bmd({1, n}, dt::bf16, tag::ab),
      omd({m, n}, dt::bf16, dnnl::memory::dims{ldo, 1});
  const auto key = std::make_tuple(m, n, k, lda, ldo, static_cast<int>(act) | (bias ? 8 : 0), res ? 1 : 0);
  auto it = d.mm.find(key);
  if (it == d.mm.end()) {
    dnnl::primitive_attr attr;
    attr.set_deterministic(true);  // small-m calls (the merger's 1 / 4 rows) otherwise take split-k with atomic sums
    dnnl::post_ops po;
    if (act == VitAct::gelu_tanh) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
    if (act == VitAct::gelu_erf) po.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
    if (res) po.append_binary(dnnl::algorithm::binary_add, omd);
    attr.set_post_ops(po);
    const dnnl::matmul::primitive_desc pd = bias ? dnnl::matmul::primitive_desc(d.eng, amd, wmd, bmd, omd, attr) : dnnl::matmul::primitive_desc(d.eng, amd, wmd, omd, attr);
    it = d.mm.emplace(key, dnnl::matmul(pd)).first;
  }
  std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, mem(d, amd, a)}, {DNNL_ARG_WEIGHTS, mem(d, wmd, w)}, {DNNL_ARG_DST, mem(d, omd, out)}};
  if (bias) args[DNNL_ARG_BIAS] = mem(d, bmd, bias);
  if (res) args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(act == VitAct::none ? 0 : 1) | DNNL_ARG_SRC_1] = mem(d, omd, res);
  return dnnl::sycl_interop::execute(it->second, d.st, args);
}

sycl::event vit_attention(sycl::queue &q, int seq, int heads, int hd, const vit_bf16 *qkv, vit_bf16 *o) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  Attn *a = nullptr;
  for (auto it = d.attn.begin(); it != d.attn.end(); ++it)
    if (it->seq == seq && it->heads == heads && it->hd == hd) {
      if (it != d.attn.begin()) d.attn.splice(d.attn.begin(), d.attn, it);
      a = &d.attn.front();
      break;
    }
  if (!a) {
    d.attn.push_front(build_attn(d, seq, heads, hd));
    if (d.attn.size() > 16) d.attn.pop_back();
    a = &d.attn.front();
  }
  vit_bf16 *base = const_cast<vit_bf16 *>(qkv);
  std::vector<g::tensor> in = {g::tensor(a->q, d.geng, base), g::tensor(a->k, d.geng, base + static_cast<size_t>(heads) * hd),
                               g::tensor(a->scale, d.geng, a->scale_dev.get()), g::tensor(a->v, d.geng, base + static_cast<size_t>(2) * heads * hd)};
  std::vector<g::tensor> out = {g::tensor(a->out, d.geng, o)};
  return g::sycl_interop::execute(a->cp, d.gst, in, out);
}

}  // namespace xenithium::kernels
