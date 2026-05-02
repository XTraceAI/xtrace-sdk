// production_GPU_paillier_client/paillier_GPU_client.cu
//
// This file implements a CUDA-accelerated Paillier client whose
// public API mirrors xtrace_sdk.crypto.paillier_client.PaillierClient:
//
//   class PaillierGPUClient:
//       def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False)
//       def encrypt(self, embd: list[int]) -> list[int]
//       def encode_hamming_client(self, ct1: list[int], ct2: list[int]) -> list[int]
//       @staticmethod
//       def encode_hamming_server(ct1: list[int], ct2: list[int], pk: dict) -> list[int]
//       def decode_hamming_client(self, cipher: list[int]) -> int
//
// Keys are generated once in the constructor (unless skip_key_gen=True)
// using GMP big integers on the CPU, and stored inside the C++ class.
// GPU kernels use NVlabs CGBN for modular arithmetic on ciphertexts.

#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <gmp.h>
#include "cgbn/cgbn.h"

namespace py = pybind11;

// ---------------------- Compile-time configuration ----------------------
#ifndef KEY_BITS
#define KEY_BITS 1024
#endif

#ifndef CGBN_TPI
#define CGBN_TPI 32
#endif

// In the Python client, key_len is the prime bit-length.
// n is roughly 2*key_len bits, and we pad plaintext chunks to 2*key_len bits.
#define PAILLIER_N_SQUARED_BITS (KEY_BITS*4)
static constexpr int LIMBS = (PAILLIER_N_SQUARED_BITS + 31) / 32;
static constexpr int RANDOM_CANDIDATES_PER_INSTANCE = 16;

// ---------------------- Error handling helpers ----------------------
#define CUDA_CHECK(call) do {                                          \
  cudaError_t _e = (call);                                             \
  if (_e != cudaSuccess) {                                             \
    std::ostringstream _oss;                                           \
    _oss << "CUDA error " << cudaGetErrorName(_e)                      \
         << " (" << cudaGetErrorString(_e) << ") at " << __FILE__      \
         << ":" << __LINE__;                                           \
    throw std::runtime_error(_oss.str());                              \
  }                                                                    \
} while(0)

// ---------------------- Minimal GMP big-int wrapper ----------------------
namespace bigint {

class big_int_t {
public:
  mpz_t v;

  big_int_t() { mpz_init(v); }
  explicit big_int_t(unsigned long long x) { mpz_init(v); mpz_set_ui(v, static_cast<unsigned long>(x)); }
  explicit big_int_t(const std::string &s, int base = 10) {
    mpz_init(v);
    if (mpz_set_str(v, s.c_str(), base) != 0) {
      throw std::invalid_argument("big_int_t: invalid string");
    }
  }
  big_int_t(const big_int_t &o) { mpz_init_set(v, o.v); }
  big_int_t(big_int_t &&o) noexcept { mpz_init(v); mpz_swap(v, o.v); }
  ~big_int_t() { mpz_clear(v); }

  big_int_t &operator=(const big_int_t &o) {
    if (this != &o) mpz_set(v, o.v);
    return *this;
  }
  big_int_t &operator=(big_int_t &&o) noexcept {
    if (this != &o) mpz_swap(v, o.v);
    return *this;
  }

  std::string to_string(int base = 10) const {
    char *raw = mpz_get_str(nullptr, base, v);
    if (!raw) return {};
    std::string s(raw);
    void (*freefunc)(void*, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    if (freefunc) freefunc(raw, std::strlen(raw) + 1);
    return s;
  }
};

inline bigint::big_int_t powm(const bigint::big_int_t &base,
                              const bigint::big_int_t &exp,
                              const bigint::big_int_t &mod) {
  bigint::big_int_t r;
  mpz_powm(r.v, base.v, exp.v, mod.v);
  return r;
}

inline bigint::big_int_t gcd(const bigint::big_int_t &a, const bigint::big_int_t &b) {
  bigint::big_int_t r;
  mpz_gcd(r.v, a.v, b.v);
  return r;
}

inline void fill_os_random(void *data, std::size_t size) {
  std::ifstream rng("/dev/urandom", std::ios::in | std::ios::binary);
  if (!rng) {
    throw std::runtime_error("secure randomness: failed to open /dev/urandom");
  }
  const auto requested = static_cast<std::streamsize>(size);
  rng.read(static_cast<char *>(data), requested);
  if (rng.gcount() != requested) {
    throw std::runtime_error("secure randomness: failed to read enough bytes from /dev/urandom");
  }
}

inline bigint::big_int_t random_bits(unsigned bits) {
  if (bits == 0) throw std::invalid_argument("bits must be > 0");
  const std::size_t byte_count = (static_cast<std::size_t>(bits) + 7) / 8;
  std::vector<unsigned char> bytes(byte_count);
  fill_os_random(bytes.data(), bytes.size());

  const unsigned excess_bits = static_cast<unsigned>(byte_count * 8 - bits);
  if (excess_bits > 0) {
    bytes[0] &= static_cast<unsigned char>(0xFFu >> excess_bits);
  }

  bigint::big_int_t out;
  mpz_import(out.v, bytes.size(), 1, 1, 0, 0, bytes.data());
  return out;
}

inline bigint::big_int_t random_prime_bits(unsigned bits) {
  if (bits < 2) throw std::invalid_argument("bits must be >= 2");
  bigint::big_int_t x = random_bits(bits);
  mpz_setbit(x.v, bits - 1);
  if (mpz_even_p(x.v)) mpz_add_ui(x.v, x.v, 1);
  bigint::big_int_t p;
  mpz_nextprime(p.v, x.v);
  while (static_cast<unsigned>(mpz_sizeinbase(p.v, 2)) != bits) {
    x = random_bits(bits);
    mpz_setbit(x.v, bits - 1);
    if (mpz_even_p(x.v)) mpz_add_ui(x.v, x.v, 1);
    mpz_nextprime(p.v, x.v);
  }
  return p;
}

} // namespace bigint

// ---------------------- CPU Paillier key structures ----------------------
struct PaillierPublicKeyCPU {
  bigint::big_int_t g;
  bigint::big_int_t n;
  bigint::big_int_t n_squared;
};

struct PaillierSecretKeyCPU {
  bigint::big_int_t phi;
  bigint::big_int_t inv;
};

struct PaillierKeyPairCPU {
  PaillierPublicKeyCPU pk;
  PaillierSecretKeyCPU sk;
};

class PaillierCPU {
public:
  static PaillierKeyPairCPU key_gen(unsigned prime_bits) {
    using namespace bigint;

    big_int_t p = random_prime_bits(prime_bits);
    big_int_t q = random_prime_bits(prime_bits);

    big_int_t n;
    mpz_mul(n.v, p.v, q.v);

    big_int_t p_minus_1, q_minus_1;
    mpz_sub_ui(p_minus_1.v, p.v, 1);
    mpz_sub_ui(q_minus_1.v, q.v, 1);

    big_int_t phi;
    mpz_mul(phi.v, p_minus_1.v, q_minus_1.v);

    big_int_t g;
    mpz_add_ui(g.v, n.v, 1);

    big_int_t n2;
    mpz_mul(n2.v, n.v, n.v);

    big_int_t inv;
    if (!mpz_invert(inv.v, phi.v, n.v)) {
      throw std::runtime_error("Paillier key_gen: phi^{-1} mod n does not exist");
    }

    PaillierPublicKeyCPU pk{g, n, n2};
    PaillierSecretKeyCPU sk{phi, inv};
    return PaillierKeyPairCPU{pk, sk};
  }
};

// ---------------------- Hex <-> limb conversion ----------------------
static inline uint32_t hex_nibble(char c) {
  if (c >= '0' && c <= '9') return (uint32_t)(c - '0');
  if (c >= 'a' && c <= 'f') return (uint32_t)(10 + c - 'a');
  if (c >= 'A' && c <= 'F') return (uint32_t)(10 + c - 'A');
  throw std::runtime_error("Invalid hex digit");
}

static std::string normalize_hex(const std::string &s) {
  std::string h = s;
  if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
  size_t p = 0;
  while (p < h.size() && h[p] == '0') ++p;
  if (p == h.size()) return "0";
  return h.substr(p);
}

static std::vector<uint32_t> hex_to_le_words(const std::string &hex_in, int words) {
  std::string h = normalize_hex(hex_in);
  if (h == "0") return std::vector<uint32_t>(words, 0u);
  if (h.size() & 1) h = "0" + h;
  std::vector<uint8_t> bytes(h.size() / 2);
  for (size_t i = 0, j = 0; i < h.size(); i += 2, ++j) {
    uint8_t hi = (uint8_t)hex_nibble(h[i]);
    uint8_t lo = (uint8_t)hex_nibble(h[i + 1]);
    bytes[j] = (uint8_t)((hi << 4) | lo);
  }
  std::vector<uint32_t> words_le(words, 0u);
  const size_t num_bytes = bytes.size();
  for (size_t i = 0; i < num_bytes; ++i) {
    size_t limb_index = i / 4;
    size_t byte_index = i % 4;
    if (limb_index >= (size_t)words) break;
    words_le[limb_index] |= ((uint32_t)bytes[num_bytes - 1 - i]) << (8 * byte_index);
  }
  return words_le;
}

static std::string le_words_to_hex(const std::vector<uint32_t> &words) {
  std::vector<uint8_t> bytes;
  bytes.reserve(words.size() * 4);
  for (uint32_t w : words) {
    bytes.push_back((uint8_t)(w & 0xFF));
    bytes.push_back((uint8_t)((w >> 8) & 0xFF));
    bytes.push_back((uint8_t)((w >> 16) & 0xFF));
    bytes.push_back((uint8_t)((w >> 24) & 0xFF));
  }
  while (!bytes.empty() && bytes.back() == 0) bytes.pop_back();
  if (bytes.empty()) return "0";
  static const char *HEX = "0123456789abcdef";
  std::string hex;
  hex.resize(bytes.size() * 2);
  size_t idx = 0;
  for (int i = (int)bytes.size() - 1; i >= 0; --i) {
    uint8_t b = bytes[i];
    hex[idx++] = HEX[(b >> 4) & 0xF];
    hex[idx++] = HEX[b & 0xF];
  }
  size_t p = 0;
  while (p + 1 < hex.size() && hex[p] == '0') ++p;
  return hex.substr(p);
}

// hex -> binary string (MSB-first), no fixed width; "0" for zero
static inline std::string hex_to_bin_nopad(const std::string &hex) {
  auto hn = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return 0;
  };
  static const char *LUT[16] = {
      "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111",
      "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"};
  std::string out;
  out.reserve(hex.size() * 4);
  for (char c : hex) out.append(LUT[hn(c)]);
  size_t p = out.find_first_not_of('0');
  if (p == std::string::npos) return "0";
  return out.substr(p);
}

// ---------------------- CGBN types ----------------------
typedef cgbn_context_t<CGBN_TPI>               context_t;
typedef cgbn_env_t<context_t, PAILLIER_N_SQUARED_BITS> env_t;
typedef cgbn_error_report_t               error_report_t;
typedef cgbn_mem_t<env_t::BITS>  mem_t;



// ------------------ Arithmetic helpers ------------------

// Extract pointer to 32-bit limbs from mem_t (little-endian)
static __device__ __forceinline__ const uint32_t* limbs_of(const mem_t *m) {
  return reinterpret_cast<const uint32_t*>(m);
}

// Find MSB index (bit position) of exponent in mem_t (return -1 if zero)
static __device__ __forceinline__ int msb_index_mem(const mem_t *exp_mem) {
  const uint32_t *w = limbs_of(exp_mem);
  for (int i=LIMBS-1; i>=0; --i) {
    uint32_t v = w[i];
    if (v != 0) {
      // position of highest set bit within word
      int lz = __clz(v);             // 0..32
      int bit = 31 - lz;             // 0..31
      return i*32 + bit;
    }
  }
  return -1;
}

// Square-and-multiply: result = base^exp_mem (mod mod_n2)
// exp_mem is provided as mem_t (n's limbs)
static __device__ void pow_mod_mem(env_t &env,
                            typename env_t::cgbn_t &result,
                            const typename env_t::cgbn_t &base,
                            const mem_t *exp_mem,
                            const typename env_t::cgbn_t &mod_n2) {
  typename env_t::cgbn_t res, b;
  typename env_t::cgbn_wide_t wide;

  cgbn_set_ui32(env, res, 1);
  cgbn_set(env, b, base);

  const uint32_t *EW = reinterpret_cast<const uint32_t*>(exp_mem);

  // Iterate all LIMBS*32 bits, LSB to MSB
  for (int w = 0; w < LIMBS; ++w) {
    uint32_t x = EW[w];
    for (int k = 0; k < 32; ++k) {
      if (x & 1u) {
        cgbn_mul_wide(env, wide, res, b);
        cgbn_rem_wide(env, res, wide, mod_n2);
      }
      // b = b*b mod mod_n2
      cgbn_mul_wide(env, wide, b, b);
      cgbn_rem_wide(env, b, wide, mod_n2);
      x >>= 1;
    }
  }
  cgbn_set(env, result, res);
}

// Euclidean GCD (returns gcd(a,b) in 'a'); destructive to inputs
static __device__ void gcd_big(env_t &env,
                               typename env_t::cgbn_t &a,
                               typename env_t::cgbn_t &b) {
  typename env_t::cgbn_t t;
  while (cgbn_compare_ui32(env, b, 0) != 0) {
    cgbn_rem(env, t, a, b);
    cgbn_set(env, a, b);
    cgbn_set(env, b, t);
  }
}

// Sample r in [1, n-1] and (optionally) ensure gcd(r, n) = 1.
// Very unlikely to need retries.
static __device__ void sample_r(env_t &env,
                                typename env_t::cgbn_t &r,
                                const typename env_t::cgbn_t &n,
                                mem_t *random_candidates,
                                int instance_id,
                                bool ensure_coprime=true) {
  typename env_t::cgbn_t tmp, a, b;
  typename env_t::cgbn_t n_minus_1;
  cgbn_sub_ui32(env, n_minus_1, n, 1);

  for (int tries=0; tries<RANDOM_CANDIDATES_PER_INSTANCE; ++tries) {
    // r = (rmem % (n-1)) + 1  -> ensures 1..n-1
    cgbn_load(env, tmp, &random_candidates[instance_id * RANDOM_CANDIDATES_PER_INSTANCE + tries]);
    cgbn_rem(env, tmp, tmp, n_minus_1);
    cgbn_add_ui32(env, r, tmp, 1);

    if (!ensure_coprime) return;

    // Check gcd(r, n) == 1
    cgbn_set(env, a, n);
    cgbn_set(env, b, r);
    gcd_big(env, a, b);                 // gcd in 'a'
    if (cgbn_compare_ui32(env, a, 1) == 0)
      return; // good
  }

  // Fallback: r=1 (extremely unlikely path)
  cgbn_set_ui32(env, r, 1);
}



// Build a mask that selects bits at MSB-first positions 1,3,5,... within width=chunk_len
// mapped to little-endian limb storage used by cgbn_mem_t<BITS>.
static inline mem_t make_odd_mask_host(const int chunk_len) {
  mem_t mask{};
  uint32_t *w = reinterpret_cast<uint32_t*>(&mask);
  // zero init
  for (int i=0;i<LIMBS;i++) w[i]=0u;
  // For each odd MSB index tbit=1,3,..., set corresponding LSB index
  // lsb_index = chunk_len - 1 - tbit
  for (int tbit=1; tbit<chunk_len; tbit+=2) {
    const int lsb_index = chunk_len - 1 - tbit;
    const int idx = lsb_index >> 5;
    const int pos = lsb_index & 31;
    w[idx] |= (1u << pos);
  }
  return mask;
}




// ---------------------- CUDA kernel ----------------------
__global__ void modmul_pointwise_kernel(error_report_t *report,
                                        mem_t *A,
                                        mem_t *B,
                                        mem_t *MOD, // single modulus at index 0
                                        mem_t *R,
                                        int count) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= count) return;

  context_t context(cgbn_report_monitor, report, instance);
  env_t env(context);

  typename env_t::cgbn_t a, b, m, r;
  typename env_t::cgbn_wide_t wide;                 // <<< wide 2*BITS
  cgbn_load(env, a, &A[instance]);
  cgbn_load(env, b, &B[instance]);
  cgbn_load(env, m, &MOD[0]);

  // Use CGBN modular multiply: r = (a*b) mod m
  cgbn_mul_wide(env, wide, a, b);                   // wide = a*b
  cgbn_rem_wide(env, r, wide, m);                   // r = wide % m
  
  cgbn_store(env, &R[instance], r);
}


// Each instance handles one (vector, chunk)
__global__ void encrypt_kernel(error_report_t *report,
                               const uint8_t *all_bits, // [batch * embed_len], 0/1
                               int batch,
                               int embed_len,
                               int chunk_len,
                               int chunk_num,
                               mem_t *N,               // n
                               mem_t *N2,              // n^2
                               mem_t *random_candidates,
                               mem_t *out_ct) {          // [batch * chunk_num]
  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  int total    = batch * chunk_num;
  if (instance >= total) return;

  int vec_idx   = instance / chunk_num;
  int chunk_idx = instance % chunk_num;

  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  // Load n and n^2 into bigints
  typename env_t::cgbn_t n, n2;
  cgbn_load(env, n,  &N[0]);
  cgbn_load(env, n2, &N2[0]);

  // --- Build plaintext m from padded bits within this chunk ---
  // Padded scheme: for original bit x[j], padded has at positions: 2*j -> 0, 2*j+1 -> x[j].
  typename env_t::cgbn_t m;
  cgbn_set_ui32(env, m, 0);

  const int padded_total = 2 * embed_len;             // total padded bits for this vector
  const int chunk_base   = chunk_idx * chunk_len;     // start bit in padded stream
  const uint8_t *vec_bits = all_bits + vec_idx * embed_len;

  // Only consume the bits that actually exist in this chunk, do NOT extend to chunk_len
  int avail = 0;
  if (chunk_base < padded_total) {
    int max_take = padded_total - chunk_base;
    avail = (max_take > chunk_len) ? chunk_len : max_take;
  }

  // Build m by scanning MSB->LSB *for the avail bits only*
  for (int t=0; t<avail; ++t) {
    // m <<= 1
    cgbn_add(env, m, m, m);
    
    int p = chunk_base + t;  // global padded index (0..padded_total-1)
    uint32_t bit = 0;
    if ((p & 1) == 1) {      // odd -> carries original bit
      int j = p >> 1;        // source index in the embedding
      bit = (uint32_t)(vec_bits[j] & 1);
    }
    if (bit) {
      cgbn_add_ui32(env, m, m, 1);
    }
  }

  // Make sure 0 <= m < n  (same invariant as Python version)
  // If m >= n, reduce m modulo n (more permissive than Python assert)
  if (cgbn_compare(env, m, n) >= 0) {
    cgbn_rem(env, m, m, n);
  }

  // --- Compute a = 1 + m*n (mod n^2)  using (1+n)^m trick ---
  typename env_t::cgbn_t a, prod;
  typename env_t::cgbn_wide_t wide;
  cgbn_mul_wide(env, wide, m, n);
  cgbn_rem_wide(env, prod, wide, n2);
  cgbn_add_ui32(env, a, prod, 1);
  if (cgbn_compare(env, a, n2) >= 0) {
    cgbn_sub(env, a, a, n2);
  }

  // --- Sample r and compute b = r^n mod n^2 ---
  typename env_t::cgbn_t r, b;
  // DEBUG: force r=1 to remove the random factor during debugging
  // cgbn_set_ui32(env, r, 1);
  sample_r(env, r, n, random_candidates, instance, true);
  pow_mod_mem(env, b, r, N, n2); // exponent is n, supplied as mem_t N

  // --- c = (a * b) mod n^2 ---
  typename env_t::cgbn_t c;
  cgbn_mul_wide(env, wide, a, b);
  cgbn_rem_wide(env, c, wide, n2);

  // Store result
  cgbn_store(env, &out_ct[instance], c);
}

// ---------- GPU: decrypt-only kernel (per-chunk) ----------
__global__ void decrypt_only_kernel(error_report_t *report,
                                    mem_t *CT,   // [total] ciphertexts
                                    int total,         // batch * chunk_num
                                    mem_t *N,
                                    mem_t *N2,
                                    mem_t *PHI,
                                    mem_t *INV,
                                    mem_t *OUT_M) {    // [total] plaintext m
  const int thread   = blockIdx.x*blockDim.x + threadIdx.x;
  const int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  typedef cgbn_context_t<CGBN_TPI> context_t;
  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  typename env_t::cgbn_t n, n2, phi, inv, c, t, x, q, rrem, m;
  typename env_t::cgbn_wide_t wide;

  cgbn_load(env, c,   &CT[instance]);
  cgbn_load(env, n,   &N[0]);
  cgbn_load(env, n2,  &N2[0]);
  cgbn_load(env, phi, &PHI[0]);
  cgbn_load(env, inv, &INV[0]);

  // t = c^phi mod n^2  (LSB-first exp)
  pow_mod_mem(env, t, c, PHI, n2);

  // x = t - 1, then exact div by n: x = q*n + rrem
  cgbn_sub_ui32(env, x, t, 1);
  cgbn_div(env, q, x, n);
  cgbn_rem(env, rrem, x, n); // for debug you can check rrem==0

  // m = (q * inv) mod n
  cgbn_mul_wide(env, wide, q, inv);
  cgbn_rem_wide(env, m, wide, n);

  cgbn_store(env, &OUT_M[instance], m);
}



// Each instance handles one (vector, chunk) ciphertext -> produces count of ones at odd bit positions
__global__ void decode_hamming_kernel(error_report_t *report,
                                      mem_t *CT,          // [total] ciphertexts
                                      int total,          // batch * chunk_num
                                      int chunk_len,      // = 2*key_len
                                      mem_t *N, mem_t *N2,
                                      mem_t *PHI, mem_t *INV,
                                      uint32_t *out_odd,  // <-- NEW name
                                      uint32_t *out_even, // <-- NEW
                                      uint32_t *out_rem_nz, // <-- NEW
                                      mem_t *out_m_dbg,   // <-- OPTIONAL (can pass nullptr)
				      /* Additional parameters for GPU mask inputs for bitcounting*/
				      const mem_t *ODD_MASK,
                                      int words_used)
{

  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  // Load inputs
  typename env_t::cgbn_t n, n2, phi, inv, c;
  cgbn_load(env, n,   &N[0]);
  cgbn_load(env, n2,  &N2[0]);
  cgbn_load(env, phi, &PHI[0]);
  cgbn_load(env, inv, &INV[0]);
  cgbn_load(env, c,   &CT[instance]);

  // Decrypt: t = c^phi mod n^2
  typename env_t::cgbn_t t, x, q, rrem, m;
typename env_t::cgbn_wide_t wide;

pow_mod_mem(env, t, c, PHI, n2);       // t = c^phi mod n^2
// x = t - 1  (t is in [1, n^2-1], so this is >= 0)
cgbn_sub_ui32(env, x, t, 1);

// Divide exactly: x = q*n + rrem  (rrem must be 0)
cgbn_div(env, q, x, n);
cgbn_rem(env, rrem, x, n);
uint32_t rem_nz = (cgbn_compare_ui32(env, rrem, 0) != 0);

// m = (q * inv) mod n
cgbn_mul_wide(env, wide, q, inv);
cgbn_rem_wide(env, m, wide, n);

// Store m for host-side inspection if requested
if (out_m_dbg != nullptr) {
  cgbn_store(env, &out_m_dbg[instance], m);
}

// Count odd/even bit positions MSB-first over chunk_len
// ---- PURE GPU COUNTING via mask+popcount ----
  mem_t mmem;
  cgbn_store(env, &mmem, m);

  const uint32_t *mw = reinterpret_cast<const uint32_t*>(&mmem);
  const uint32_t *ow = reinterpret_cast<const uint32_t*>(&ODD_MASK[0]);

  uint32_t odd_cnt = 0;
  #pragma unroll
  for (int w=0; w<words_used; ++w) {
    odd_cnt += __popc(mw[w] & ow[w]);
  }

  out_odd[instance] = odd_cnt;

  if (out_even) {
    // optional parity sanity: even_count = popcount(m & ~ODD_MASK) but only within chunk_len region.
    // Build an implicit mask for "valid bits" (first words_used words all valid except possible high bits of last word).
    // Simpler: recompute even_count from MSB loop for debug only if needed; otherwise write 0.
    out_even[instance] = 0;
  }
  if (out_rem_nz) out_rem_nz[instance] = rem_nz;

}


// ---------------------- Host glue helpers ----------------------
static std::vector<std::string> gpu_pointwise_mulmod_hex(const std::vector<std::string> &ct1_hex,
                                                         const std::vector<std::string> &ct2_hex,
                                                         const std::string &n_squared_hex) {
  if (ct1_hex.size() != ct2_hex.size()) {
    throw std::runtime_error("encode_hamming_*: ct1 and ct2 must have the same length");
  }
  const int count = static_cast<int>(ct1_hex.size());
  if (count == 0) return {};

  std::vector<mem_t> h_A(count), h_B(count);
  std::vector<uint32_t> mod_words = hex_to_le_words(n_squared_hex, LIMBS);
  mem_t h_MOD_one{};
  std::memcpy(&h_MOD_one, mod_words.data(), LIMBS * sizeof(uint32_t));

  for (int i = 0; i < count; ++i) {
    auto a = hex_to_le_words(ct1_hex[i], LIMBS);
    auto b = hex_to_le_words(ct2_hex[i], LIMBS);
    std::memcpy(&h_A[i], a.data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_B[i], b.data(), LIMBS * sizeof(uint32_t));
  }

  error_report_t *d_report = nullptr;
  mem_t *d_A = nullptr, *d_B = nullptr, *d_MOD = nullptr, *d_R = nullptr;

  CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
  CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));

  CUDA_CHECK(cudaMalloc((void **)&d_A, count * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_B, count * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_R, count * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_MOD, sizeof(mem_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), count * sizeof(mem_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), count * sizeof(mem_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_MOD, &h_MOD_one, sizeof(mem_t), cudaMemcpyHostToDevice));

  const int INSTS_PER_BLK = 8;
  const int THREADS = INSTS_PER_BLK * CGBN_TPI;
  const int blocks = (count + INSTS_PER_BLK - 1) / INSTS_PER_BLK;

  modmul_pointwise_kernel<<<blocks, THREADS>>>(d_report, d_A, d_B, d_MOD, d_R, count);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<mem_t> h_R(count);
  CUDA_CHECK(cudaMemcpy(h_R.data(), d_R, count * sizeof(mem_t), cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_R);
  cudaFree(d_MOD);
  cudaFree(d_report);

  std::vector<std::string> out(count);
  for (int i = 0; i < count; ++i) {
    std::vector<uint32_t> words(LIMBS);
    std::memcpy(words.data(), &h_R[i], LIMBS * sizeof(uint32_t));
    out[i] = le_words_to_hex(words);
  }
  return out;
}

// ---------------------- PaillierGPUClient class ----------------------
class PaillierGPUClient {
public:
  static int max_key_len() { return KEY_BITS; }

  PaillierGPUClient(int embed_len = 512, int key_len = KEY_BITS, bool skip_key_gen = false)
      : embed_len_(embed_len),
        key_len_(key_len),
        chunk_len_(key_len * 2),
        chunk_num_(embed_len > key_len ? ((2 * embed_len) / (key_len * 2) +
                                          (((2 * embed_len) % (key_len * 2)) != 0))
                                       : 1),
        have_keys_(!skip_key_gen) {
    if (key_len_ > KEY_BITS) {
      throw std::runtime_error(
          "PaillierGPUClient: key_len=" + std::to_string(key_len_) +
          " exceeds the compile-time KEY_BITS=" + std::to_string(KEY_BITS) +
          " of this GPU extension. Rebuild with a larger KEY_BITS "
          "(see build_gpu_binaries.sh) or use a smaller key_len.");
    }
    if (!skip_key_gen) {
      keys_ = PaillierCPU::key_gen(static_cast<unsigned>(key_len_));
      pk_n_hex_ = keys_.pk.n.to_string(16);
      pk_n2_hex_ = keys_.pk.n_squared.to_string(16);
      sk_phi_hex_ = keys_.sk.phi.to_string(16);
      sk_inv_hex_ = keys_.sk.inv.to_string(16);
      // Light debug: print generated keys (hex)
      // std::cerr << "[PaillierGPUClient] Generated keys (hex):\n"
      //           << "  n        = " << pk_n_hex_ << "\n"
      //           << "  n_squared= " << pk_n2_hex_ << "\n"
      //           << "  phi      = " << sk_phi_hex_ << "\n"
      //           << "  inv      = " << sk_inv_hex_ << std::endl;
    }
  }

  int embed_len() const { return embed_len_; }
  int key_len() const { return key_len_; }
  int chunk_len() const { return chunk_len_; }
  int chunk_num() const { return chunk_num_; }

  // Expose public key components as hex for Python-side checks
  std::pair<std::string, std::string> get_pk_hex() const {
    if (!have_keys_) {
      throw std::runtime_error("get_pk_hex: keys not initialized");
    }
    return {pk_n_hex_, pk_n2_hex_};
  }

  // Expose full key components as hex: (n, n_squared, phi, inv)
  std::tuple<std::string, std::string, std::string, std::string> get_keys_hex() const {
    if (!have_keys_) {
      throw std::runtime_error("get_keys_hex: keys not initialized");
    }
    return {pk_n_hex_, pk_n2_hex_, sk_phi_hex_, sk_inv_hex_};
  }

  std::string stringify_pk() const {
    if (!have_keys_) {
      throw std::runtime_error("stringify_pk: keys not initialized");
    }
    py::object json = py::module_::import("json");
    bigint::big_int_t n(pk_n_hex_, 16);
    bigint::big_int_t n_squared(pk_n2_hex_, 16);
    bigint::big_int_t g;
    mpz_add_ui(g.v, n.v, 1);

    py::dict pk_dict;
    pk_dict["g"] = g.to_string(10);
    pk_dict["n"] = n.to_string(10);
    pk_dict["n_squared"] = n_squared.to_string(10);
    return py::cast<std::string>(json.attr("dumps")(pk_dict));
  }

  std::string stringify_sk() const {
    if (!have_keys_) {
      throw std::runtime_error("stringify_sk: keys not initialized");
    }
    py::object json = py::module_::import("json");
    py::dict sk_dict;
    sk_dict["phi"] = bigint::big_int_t(sk_phi_hex_, 16).to_string(10);
    sk_dict["inv"] = bigint::big_int_t(sk_inv_hex_, 16).to_string(10);
    return py::cast<std::string>(json.attr("dumps")(sk_dict));
  }

  std::string stringify_config() const {
    py::object json = py::module_::import("json");
    py::dict cfg;
    cfg["embed_len"] = embed_len_;
    cfg["key_len"] = key_len_;
    return py::cast<std::string>(json.attr("dumps")(cfg));
  }

  void load_stringified_keys(const std::string &pk_json, const std::string &sk_json) {
    py::object json = py::module_::import("json");
    py::dict pk_dict = json.attr("loads")(pk_json).cast<py::dict>();
    py::dict sk_dict = json.attr("loads")(sk_json).cast<py::dict>();

    auto decimal_field_as_hex = [](const py::dict &d, const char *name) {
      return bigint::big_int_t(py::cast<std::string>(py::str(d[name])), 10).to_string(16);
    };

    pk_n_hex_ = decimal_field_as_hex(pk_dict, "n");
    pk_n2_hex_ = decimal_field_as_hex(pk_dict, "n_squared");
    sk_phi_hex_ = decimal_field_as_hex(sk_dict, "phi");
    sk_inv_hex_ = decimal_field_as_hex(sk_dict, "inv");

    bigint::big_int_t n2(pk_n2_hex_, 16);
    const size_t n2_bits = mpz_sizeinbase(n2.v, 2);
    if (n2_bits > static_cast<size_t>(PAILLIER_N_SQUARED_BITS)) {
      throw std::runtime_error(
          "load_stringified_keys: n_squared has " + std::to_string(n2_bits) +
          " bits, exceeding the compile-time capacity of " +
          std::to_string(PAILLIER_N_SQUARED_BITS) +
          " bits (KEY_BITS=" + std::to_string(KEY_BITS) +
          "). Rebuild the GPU extension with a larger KEY_BITS.");
    }
    have_keys_ = true;
  }

  void load_config(const py::dict &config) {
    embed_len_ = py::cast<int>(config["embed_len"]);
    const int new_key_len = py::cast<int>(config["key_len"]);
    if (new_key_len > KEY_BITS) {
      throw std::runtime_error(
          "load_config: key_len=" + std::to_string(new_key_len) +
          " exceeds the compile-time KEY_BITS=" + std::to_string(KEY_BITS) +
          " of this GPU extension.");
    }
    key_len_ = new_key_len;
    chunk_len_ = key_len_ * 2;
    if (embed_len_ > key_len_) {
      chunk_num_ = 2 * embed_len_ / chunk_len_ + int((2 * embed_len_) % chunk_len_ != 0);
    } else {
      chunk_num_ = 1;
    }
  }

  // Batched encrypt: embeddings is [batch][embed_len] of 0/1, returns [batch][chunk_num] hex ciphers
  std::vector<std::vector<std::string>>
  encrypt(const std::vector<std::vector<int>> &embeddings) const {
    if (!have_keys_) {
      throw std::runtime_error("encrypt: keys not initialized");
    }
    const int batch = static_cast<int>(embeddings.size());
    const int embed_len = embed_len_;
    const int chunk_len = chunk_len_;
    const int chunk_num = chunk_num_;

    if (batch == 0) return {};

    // Validate shapes and binary values
    for (int i = 0; i < batch; ++i) {
      if ((int)embeddings[i].size() != embed_len) {
        throw std::runtime_error("encrypt: each embedding must have length == embed_len");
      }
      for (int v : embeddings[i]) {
        if (v != 0 && v != 1) {
          throw std::runtime_error("encrypt: embeddings must be binary 0/1");
        }
      }
    }

    // Flatten bits into [batch * embed_len]
    std::vector<uint8_t> h_bits;
    h_bits.reserve(static_cast<size_t>(batch) * static_cast<size_t>(embed_len));
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < embed_len; ++j) {
        h_bits.push_back(static_cast<uint8_t>(embeddings[i][j]));
      }
    }

    std::vector<uint32_t> n_words = hex_to_le_words(pk_n_hex_, LIMBS);
    std::vector<uint32_t> n2_words = hex_to_le_words(pk_n2_hex_, LIMBS);
    mem_t h_N{}, h_N2{};
    std::memcpy(&h_N, n_words.data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2, n2_words.data(), LIMBS * sizeof(uint32_t));

    error_report_t *d_report = nullptr;
    uint8_t *d_bits = nullptr;
    mem_t *d_N = nullptr, *d_N2 = nullptr, *d_random = nullptr, *d_out = nullptr;

    const int total = batch * chunk_num;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_bits,
                          static_cast<size_t>(batch) * static_cast<size_t>(embed_len) *
                              sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_bits,
                          h_bits.data(),
                          static_cast<size_t>(batch) * static_cast<size_t>(embed_len) *
                              sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_N, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_N2, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N, &h_N, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N2, &h_N2, sizeof(mem_t), cudaMemcpyHostToDevice));

    std::vector<mem_t> h_random(
        static_cast<size_t>(total) * static_cast<size_t>(RANDOM_CANDIDATES_PER_INSTANCE));
    bigint::fill_os_random(h_random.data(), h_random.size() * sizeof(mem_t));
    CUDA_CHECK(cudaMalloc((void **)&d_random, h_random.size() * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_random,
                          h_random.data(),
                          h_random.size() * sizeof(mem_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_out, total * sizeof(mem_t)));

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (total + INSTS_PER_BLK - 1) / INSTS_PER_BLK;

    encrypt_kernel<<<blocks, THREADS>>>(d_report, d_bits, batch, embed_len, chunk_len,
                                        chunk_num, d_N, d_N2, d_random, d_out);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_bits);
    cudaFree(d_N);
    cudaFree(d_N2);
    cudaFree(d_random);
    cudaFree(d_out);
    cudaFree(d_report);

    // Reshape back to [batch][chunk_num] hex strings
    std::vector<std::vector<std::string>> out(batch, std::vector<std::string>(chunk_num));
    for (int i = 0; i < batch; ++i) {
      for (int c = 0; c < chunk_num; ++c) {
        int idx = i * chunk_num + c;
        std::vector<uint32_t> words(LIMBS);
        std::memcpy(words.data(), &h_out[idx], LIMBS * sizeof(uint32_t));
        out[i][c] = le_words_to_hex(words);
      }
    }
    return out;
  }

  // Client-side homomorphic addition: ct1[i]*ct2[i] mod n^2
  std::vector<std::string> encode_hamming_client(const std::vector<std::string> &ct1,
                                                 const std::vector<std::string> &ct2) const {
    if (!have_keys_) {
      throw std::runtime_error("encode_hamming_client: keys not initialized");
    }
    return gpu_pointwise_mulmod_hex(ct1, ct2, pk_n2_hex_);
  }

  // Server-side version: takes pk dict with "n_squared"
  static std::vector<std::string> encode_hamming_server(const std::vector<std::string> &ct1,
                                                        const std::vector<std::string> &ct2,
                                                        const py::dict &pk) {
    auto n2_obj = pk["n_squared"];
    std::string n2_hex = py::cast<std::string>(py::str(n2_obj));
    return gpu_pointwise_mulmod_hex(ct1, ct2, n2_hex);
  }

  // GPU decrypt chunks to plaintext m (hex), then CPU bit-padding and odd-bit counting
  std::vector<long long>
  decode_hamming_client(const std::vector<std::vector<std::string>> &ciphers_hex_batch) const {
    if (!have_keys_) {
      throw std::runtime_error("decode_hamming_client: keys not initialized");
    }
    const int batch = static_cast<int>(ciphers_hex_batch.size());
    const int chunk_num = chunk_num_;
    const int chunk_len = chunk_len_;
    if (batch == 0) return {};

    for (int i = 0; i < batch; ++i) {
      if ((int)ciphers_hex_batch[i].size() != chunk_num) {
        throw std::runtime_error("decode_hamming_client: each inner list must have length == chunk_num");
      }
    }

    // 1) GPU decrypt per-chunk to get plaintext m as hex strings
    auto m_hex_batch = decrypt_chunks(ciphers_hex_batch);

    // 2) CPU: mirror Python's padding and odd-bit counting
    std::vector<long long> out(batch, 0);
    for (int i = 0; i < batch; ++i) {
      long long ham = 0;
      std::string bin_concat;
      bin_concat.reserve(static_cast<size_t>(chunk_len) * static_cast<size_t>(chunk_num));
      for (int c = 0; c < chunk_num; ++c) {
        const std::string &mh = m_hex_batch[i][c];
        std::string b = hex_to_bin_nopad(mh);  // MSB-first, minimal width ("0" for zero)
        if ((int)b.size() != chunk_len && !b.empty()) {
          if ((int)b.size() < chunk_len) {
            b.insert(0, chunk_len - (int)b.size(), '0');
          }
        } else if (b == "0") {
          b.assign(static_cast<size_t>(chunk_len), '0');
        }
        bin_concat += b;
      }
      for (size_t k = 1; k < bin_concat.size(); k += 2) {
        if (bin_concat[k] == '1') ++ham;
      }
      out[i] = ham;
    }
    return out;
  }

private:
  int embed_len_;
  int key_len_;
  int chunk_len_;
  int chunk_num_;
  bool have_keys_;

  PaillierKeyPairCPU keys_;
  std::string pk_n_hex_;
  std::string pk_n2_hex_;
  std::string sk_phi_hex_;
  std::string sk_inv_hex_;

  // Helper: GPU decrypt all chunks to plaintext m (hex), mirroring original GPU client
  std::vector<std::vector<std::string>>
  decrypt_chunks(const std::vector<std::vector<std::string>> &ciphers_hex_batch) const {
    const int batch = static_cast<int>(ciphers_hex_batch.size());
    const int chunk_num = this->chunk_num_;
    if (batch == 0) return {};

    for (int i = 0; i < batch; ++i) {
      if ((int)ciphers_hex_batch[i].size() != chunk_num) {
        throw std::runtime_error("decrypt_chunks: each inner list must have length == chunk_num");
      }
    }

    const int total = batch * chunk_num;

    // pack CT
    std::vector<mem_t> h_CT(total);
    for (int i = 0; i < batch; ++i) {
      for (int c = 0; c < chunk_num; ++c) {
        const int idx = i * chunk_num + c;
        auto w = hex_to_le_words(ciphers_hex_batch[i][c], LIMBS);
        std::memcpy(&h_CT[idx], w.data(), LIMBS * sizeof(uint32_t));
      }
    }

    // keys
    mem_t h_N{}, h_N2{}, h_PHI{}, h_INV{};
    std::memcpy(&h_N, hex_to_le_words(pk_n_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2, hex_to_le_words(pk_n2_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_PHI, hex_to_le_words(sk_phi_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_INV, hex_to_le_words(sk_inv_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));

    // device
    error_report_t *d_report = nullptr;
    mem_t *d_CT = nullptr, *d_N = nullptr, *d_N2 = nullptr, *d_PHI = nullptr, *d_INV = nullptr,
          *d_M = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_CT, total * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_CT, h_CT.data(), total * sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_N, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_N2, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_PHI, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_INV, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N, &h_N, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N2, &h_N2, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_PHI, &h_PHI, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_INV, &h_INV, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_M, total * sizeof(mem_t)));

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (total + INSTS_PER_BLK - 1) / INSTS_PER_BLK;
    decrypt_only_kernel<<<blocks, THREADS>>>(d_report, d_CT, total, d_N, d_N2, d_PHI, d_INV, d_M);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_M(total);
    CUDA_CHECK(cudaMemcpy(h_M.data(), d_M, total * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_CT);
    cudaFree(d_N);
    cudaFree(d_N2);
    cudaFree(d_PHI);
    cudaFree(d_INV);
    cudaFree(d_M);
    cudaFree(d_report);

    std::vector<std::vector<std::string>> out(batch, std::vector<std::string>(chunk_num));
    for (int i = 0; i < batch; ++i) {
      for (int c = 0; c < chunk_num; ++c) {
        const int idx = i * chunk_num + c;
        std::vector<uint32_t> w(LIMBS);
        std::memcpy(w.data(), &h_M[idx], LIMBS * sizeof(uint32_t));
        out[i][c] = le_words_to_hex(w);
      }
    }
    return out;
  }
};

// ---------------------- Pybind module ----------------------
PYBIND11_MODULE(paillier_GPU_client, m) {
  m.attr("KEY_BITS") = py::int_(KEY_BITS);
  py::class_<PaillierGPUClient>(m, "PaillierGPUClient")
      .def_static("max_key_len", &PaillierGPUClient::max_key_len)
      .def(py::init<int, int, bool>(),
           py::arg("embed_len") = 512,
           py::arg("key_len") = KEY_BITS,
           py::arg("skip_key_gen") = false)
      .def("encrypt", &PaillierGPUClient::encrypt,
           py::arg("embeddings"))
      .def("encode_hamming_client", &PaillierGPUClient::encode_hamming_client,
           py::arg("ct1"),
           py::arg("ct2"))
      .def_static("encode_hamming_server", &PaillierGPUClient::encode_hamming_server,
                  py::arg("ct1"),
                  py::arg("ct2"),
                  py::arg("pk"))
      .def("decode_hamming_client", &PaillierGPUClient::decode_hamming_client,
           py::arg("ciphers_hex_batch"))
      .def("get_pk_hex", &PaillierGPUClient::get_pk_hex)
      .def("get_keys_hex", &PaillierGPUClient::get_keys_hex)
      .def("stringify_pk", &PaillierGPUClient::stringify_pk)
      .def("stringify_sk", &PaillierGPUClient::stringify_sk)
      .def("stringify_config", &PaillierGPUClient::stringify_config)
      .def("load_stringified_keys", &PaillierGPUClient::load_stringified_keys)
      .def("load_config", &PaillierGPUClient::load_config)
      .def_property_readonly("embed_len", &PaillierGPUClient::embed_len)
      .def_property_readonly("key_len", &PaillierGPUClient::key_len)
      .def_property_readonly("chunk_len", &PaillierGPUClient::chunk_len)
      .def_property_readonly("chunk_num", &PaillierGPUClient::chunk_num);
}
