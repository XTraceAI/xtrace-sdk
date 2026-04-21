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
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iostream>
#include <ctime>

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

#ifndef ALPHA_LEN
#define ALPHA_LEN 280
#endif

#ifndef CGBN_TPI
#define CGBN_TPI 32
#endif

// In the Python client, key_len is the prime bit-length.
// n is roughly 2*key_len bits, and we pad plaintext chunks to 2*key_len bits.
#define PAILLIER_N_SQUARED_BITS (KEY_BITS*4)
static constexpr int LIMBS = (PAILLIER_N_SQUARED_BITS + 31) / 32;

// Lookup-table parameters (mirror Python paillier_lookup.py defaults)
#define PAILLIER_MSG_BITS 8
static constexpr int PAILLIER_MSG_TABLE_SIZE = (1 << PAILLIER_MSG_BITS);

// Precomputed noise table size and how many random entries to multiply
static constexpr int PAILLIER_NOISE_TABLE_SIZE = 1 << 8; // 2**8
static constexpr int PAILLIER_NOISE_MULTIPLES  = 14;

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

// RAII GMP RNG
struct gmp_rand_ctx {
  gmp_randstate_t st;
  gmp_rand_ctx() {
    gmp_randinit_default(st);
    // Simple seeding based on current time; no std::random_device / chrono,
    // to avoid extra dependencies in NVCC builds.
    unsigned long seed = static_cast<unsigned long>(std::time(nullptr));
    mpz_t mseed;
    mpz_init_set_ui(mseed, seed);
    gmp_randseed(st, mseed);
    mpz_clear(mseed);
  }
  ~gmp_rand_ctx() { gmp_randclear(st); }
  gmp_rand_ctx(const gmp_rand_ctx &) = delete;
  gmp_rand_ctx &operator=(const gmp_rand_ctx &) = delete;
};

inline bigint::big_int_t random_prime_bits(unsigned bits, gmp_rand_ctx &rng) {
  if (bits < 2) throw std::invalid_argument("bits must be >= 2");
  bigint::big_int_t x;
  mpz_urandomb(x.v, rng.st, bits);
  mpz_setbit(x.v, bits - 1);
  if (mpz_even_p(x.v)) mpz_add_ui(x.v, x.v, 1);
  bigint::big_int_t p;
  mpz_nextprime(p.v, x.v);
  while (static_cast<unsigned>(mpz_sizeinbase(p.v, 2)) != bits) {
    mpz_urandomb(x.v, rng.st, bits);
    mpz_setbit(x.v, bits - 1);
    if (mpz_even_p(x.v)) mpz_add_ui(x.v, x.v, 1);
    mpz_nextprime(p.v, x.v);
  }
  return p;
}

} // namespace bigint

// ---------------------- Lookup-style helpers (CPU, GMP) ----------------------
// Bit length of a big_int_t in base 2.
static inline unsigned bitlen_mpz(const bigint::big_int_t &x) {
  if (mpz_sgn(x.v) <= 0) return 0;
  return static_cast<unsigned>(mpz_sizeinbase(x.v, 2));
}

// L(x) = (x-1)/n, assuming x ≡ 1 (mod n)
static inline void L_div(bigint::big_int_t &out,
                         const bigint::big_int_t &x,
                         const bigint::big_int_t &n) {
  mpz_sub_ui(out.v, x.v, 1);
  mpz_tdiv_q(out.v, out.v, n.v);
}

// CRT: find r such that r ≡ a1 (mod m1), r ≡ a2 (mod m2).
static void crt_pair(const bigint::big_int_t &a1, const bigint::big_int_t &m1,
                     const bigint::big_int_t &a2, const bigint::big_int_t &m2,
                     bigint::big_int_t &r_out, bigint::big_int_t &M_out) {
  using namespace bigint;
  mpz_mul(M_out.v, m1.v, m2.v);

  big_int_t m1_mod_m2;
  mpz_mod(m1_mod_m2.v, m1.v, m2.v);

  big_int_t inv;
  if (!mpz_invert(inv.v, m1_mod_m2.v, m2.v)) {
    throw std::runtime_error("crt_pair: inverse does not exist");
  }

  big_int_t t, diff;
  mpz_sub(diff.v, a2.v, a1.v);
  mpz_mod(diff.v, diff.v, m2.v);
  mpz_mul(t.v, diff.v, inv.v);
  mpz_mod(t.v, t.v, m2.v);

  big_int_t tmp;
  mpz_mul(tmp.v, m1.v, t.v);
  mpz_add(r_out.v, a1.v, tmp.v);
  mpz_mod(r_out.v, r_out.v, M_out.v);
}

// Multiply by (1+p) to inject an order-p component in Z_{p^2}*.
static bigint::big_int_t lift_to_p2_with_p_component(const bigint::big_int_t &g_mod_p,
                                                     const bigint::big_int_t &p) {
  using namespace bigint;
  big_int_t p2, one_plus_p, r;
  mpz_mul(p2.v, p.v, p.v);
  mpz_add_ui(one_plus_p.v, p.v, 1);
  mpz_mul(r.v, g_mod_p.v, one_plus_p.v);
  mpz_mod(r.v, r.v, p2.v);
  return r;
}

// gen_dsa_params_custom: generate (p,q,g) with p_bits for p and q_bits for q, and g of order q mod p.
static void gen_dsa_params_custom(unsigned p_bits, unsigned q_bits,
                                  bigint::gmp_rand_ctx &rng,
                                  bigint::big_int_t &p_out,
                                  bigint::big_int_t &q_out,
                                  bigint::big_int_t &g_out) {
  using namespace bigint;
  if (q_bits >= p_bits) {
    throw std::invalid_argument("alpha_len too large: q_bits must be < p_bits");
  }

  // q: exact q_bits prime
  big_int_t q = random_prime_bits(q_bits, rng);

  // choose k so that p = k*q + 1 has exactly p_bits bits
  big_int_t two_pow_p_1, two_pow_p;
  mpz_set_ui(two_pow_p_1.v, 1);
  mpz_mul_2exp(two_pow_p_1.v, two_pow_p_1.v, p_bits - 1);

  mpz_set_ui(two_pow_p.v, 1);
  mpz_mul_2exp(two_pow_p.v, two_pow_p.v, p_bits);

  big_int_t k_lo, k_hi, tmp;
  mpz_tdiv_q(k_lo.v, two_pow_p_1.v, q.v);

  mpz_sub_ui(tmp.v, two_pow_p.v, 1);
  mpz_tdiv_q(k_hi.v, tmp.v, q.v);

  if (mpz_odd_p(k_lo.v)) {
    mpz_add_ui(k_lo.v, k_lo.v, 1);
  }

  big_int_t p, span, t, k;
  while (true) {
    // span = (k_hi - k_lo) / 2 + 1
    mpz_sub(span.v, k_hi.v, k_lo.v);
    mpz_tdiv_q_2exp(span.v, span.v, 1);
    mpz_add_ui(span.v, span.v, 1);

    mpz_urandomm(t.v, rng.st, span.v);        // t in [0, span-1]
    mpz_mul_2exp(k.v, t.v, 1);                // 2*t
    mpz_add(k.v, k.v, k_lo.v);                // k = k_lo + 2*t

    mpz_mul(p.v, k.v, q.v);
    mpz_add_ui(p.v, p.v, 1);

    if (bitlen_mpz(p) != p_bits) {
      continue;
    }
    if (mpz_probab_prime_p(p.v, 40) > 0) {
      break;
    }
  }

  // generator of order q
  big_int_t e;
  mpz_sub_ui(e.v, p.v, 1);
  mpz_tdiv_q(e.v, e.v, q.v);

  big_int_t h, g;
  big_int_t p_minus2;
  mpz_sub_ui(p_minus2.v, p.v, 2);

  while (true) {
    // pick h in [2, p-1]
    mpz_urandomm(h.v, rng.st, p_minus2.v);  // 0..p-3
    mpz_add_ui(h.v, h.v, 2);               // 2..p-1
    mpz_powm(g.v, h.v, e.v, p.v);
    if (mpz_cmp_ui(g.v, 1) != 0) {
      break;
    }
  }

  p_out = p;
  q_out = q;
  g_out = g;
}

// ---------------------- CPU Paillier key structures ----------------------
struct PaillierPublicKeyCPU {
  bigint::big_int_t g;
  bigint::big_int_t n;
  bigint::big_int_t n_squared;
};

struct PaillierSecretKeyCPU {
  bigint::big_int_t phi;
  bigint::big_int_t inv;
   // Lookup-optimized decryption exponent and inverse factor
  bigint::big_int_t a;
  bigint::big_int_t g_a_inv;
};

struct PaillierKeyPairCPU {
  PaillierPublicKeyCPU pk;
  PaillierSecretKeyCPU sk;
};

class PaillierCPU {
public:
  static PaillierKeyPairCPU key_gen(unsigned prime_bits, unsigned alpha_bits) {
    using namespace bigint;
    gmp_rand_ctx rng;
    while (true) {
      // --- pick q-bit split close to requested alpha_len ---
      unsigned q_bits_bound = alpha_bits;
      unsigned q_bits_1 = q_bits_bound / 2;
      unsigned q_bits_2 = q_bits_bound - q_bits_1;

      big_int_t p1, q1, g1;
      big_int_t p2, q2, g2;
      gen_dsa_params_custom(prime_bits, q_bits_1, rng, p1, q1, g1);
      gen_dsa_params_custom(prime_bits, q_bits_2, rng, p2, q2, g2);

      big_int_t p = p1;
      big_int_t q = p2;

      big_int_t n;
      mpz_mul(n.v, p.v, q.v);

      big_int_t n2;
      mpz_mul(n2.v, n.v, n.v);

      // Lift g1, g2 to p^2, q^2 so their orders gain factors p and q
      big_int_t g1_p2 = lift_to_p2_with_p_component(g1, p);
      big_int_t g2_q2 = lift_to_p2_with_p_component(g2, q);

      big_int_t p2_mod, q2_mod;
      mpz_mul(p2_mod.v, p.v, p.v);
      mpz_mul(q2_mod.v, q.v, q.v);

      big_int_t g, M_dummy;
      crt_pair(g1_p2, p2_mod, g2_q2, q2_mod, g, M_dummy);

      // alpha = lcm(q1, q2)
      big_int_t a;
      mpz_lcm(a.v, q1.v, q2.v);

      // (sanity) ensure ord(g) | n*a and n-part is present
      big_int_t n_mul_a;
      mpz_mul(n_mul_a.v, n.v, a.v);

      big_int_t check1, check2;
      mpz_powm(check1.v, g.v, n_mul_a.v, n2.v);
      mpz_powm(check2.v, g.v, n.v, n2.v);

      if (mpz_cmp_ui(check1.v, 1) != 0 || mpz_cmp_ui(check2.v, 1) == 0) {
        // extremely rare; try again
        continue;
      }

      big_int_t p_minus_1, q_minus_1;
      mpz_sub_ui(p_minus_1.v, p.v, 1);
      mpz_sub_ui(q_minus_1.v, q.v, 1);

      big_int_t phi;
      mpz_mul(phi.v, p_minus_1.v, q_minus_1.v);

      big_int_t g_n;
      mpz_powm(g_n.v, g.v, n.v, n2.v);

      big_int_t g_pow_a;
      mpz_powm(g_pow_a.v, g.v, a.v, n2.v);

      big_int_t L_ga;
      L_div(L_ga, g_pow_a, n);

      big_int_t g_a_inv;
      if (!mpz_invert(g_a_inv.v, L_ga.v, n.v)) {
        // extremely unlikely; regenerate keys
        continue;
      }

      big_int_t inv;
      if (!mpz_invert(inv.v, phi.v, n.v)) {
        // extremely unlikely; regenerate keys
        continue;
      }

      PaillierPublicKeyCPU pk{g, n, n2};
      PaillierSecretKeyCPU sk{phi, inv, a, g_a_inv};
      return PaillierKeyPairCPU{pk, sk};
    }
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

// Small set of global constants for decryption, shared across all threads
__device__ __constant__ mem_t CONST_N;
__device__ __constant__ mem_t CONST_N2;
__device__ __constant__ mem_t CONST_A;
__device__ __constant__ mem_t CONST_G_A_INV;

// Convert a GMP big integer to our fixed-size mem_t (little-endian limbs)
static inline mem_t bigint_to_mem(const bigint::big_int_t &x) {
  std::array<uint32_t, LIMBS> words{};
  size_t count = 0;
  mpz_export(words.data(), &count, -1, sizeof(uint32_t), 0, 0, x.v);
  if (count > LIMBS) {
    throw std::runtime_error("bigint_to_mem: value does not fit in mem_t");
  }
  mem_t out{};
  std::memcpy(&out, words.data(), LIMBS * sizeof(uint32_t));
  return out;
}

// Convert mem_t (little-endian limbs) to a GMP big integer
static inline bigint::big_int_t mem_to_bigint(const mem_t &mem) {
  bigint::big_int_t out;
  std::array<uint32_t, LIMBS> words{};
  std::memcpy(words.data(), &mem, LIMBS * sizeof(uint32_t));
  mpz_import(out.v, LIMBS, -1, sizeof(uint32_t), 0, 0, words.data());
  return out;
}

// ---------------------- Lookup-table precomputation (CPU) ----------------------
// Build g_table[i][j] = g^(j * 2^(MSG_BITS * i)) mod n^2, flattened as
// [message_chunks * PAILLIER_MSG_TABLE_SIZE] of mem_t in little-endian limbs.
static std::vector<mem_t>
precompute_g_table(const PaillierKeyPairCPU &keys, int key_len_bits) {
  using namespace bigint;

  const big_int_t &g = keys.pk.g;
  const big_int_t &n = keys.pk.n;
  const big_int_t &n2 = keys.pk.n_squared;

  const int msg_bits = PAILLIER_MSG_BITS;
  const int message_chunks = (key_len_bits * 2 + msg_bits - 1) / msg_bits;
  const int table_size = PAILLIER_MSG_TABLE_SIZE;

  std::vector<mem_t> table(static_cast<size_t>(message_chunks) *
                           static_cast<size_t>(table_size));

  // For each chunk i, base = g^(2^(msg_bits * i)) mod n^2, then table[i][j] = base^j.
  for (int i = 0; i < message_chunks; ++i) {
    big_int_t exp;
    mpz_set_ui(exp.v, 1);
    mpz_mul_2exp(exp.v, exp.v, msg_bits * i);

    big_int_t base;
    mpz_powm(base.v, g.v, exp.v, n2.v);

    big_int_t cur;
    mpz_set_ui(cur.v, 1);  // base^0

    const size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(table_size);
    for (int j = 0; j < table_size; ++j) {
      table[row_offset + static_cast<size_t>(j)] = bigint_to_mem(cur);
      if (j + 1 < table_size) {
        mpz_mul(cur.v, cur.v, base.v);
        mpz_mod(cur.v, cur.v, n2.v);
      }
    }
  }

  return table;
}

// Build noise_table[k] = g_n^r mod n^2 for random r in [1, n-1] with gcd(r,n)=1,
// as mem_t entries. This mirrors Paillier_Lookup.precompute_noise_table.
static std::vector<mem_t>
precompute_noise_table(const PaillierKeyPairCPU &keys) {
  using namespace bigint;

  const big_int_t &g = keys.pk.g;
  const big_int_t &n = keys.pk.n;
  const big_int_t &n2 = keys.pk.n_squared;

  gmp_rand_ctx rng;

  big_int_t g_n;
  mpz_powm(g_n.v, g.v, n.v, n2.v);

  std::vector<mem_t> table(PAILLIER_NOISE_TABLE_SIZE);

  for (int k = 0; k < PAILLIER_NOISE_TABLE_SIZE; ++k) {
    big_int_t r;
    // Sample r uniformly in [1, n-1] with gcd(r, n) = 1.
    while (true) {
      mpz_urandomm(r.v, rng.st, n.v);  // 0 <= r < n
      if (mpz_cmp_ui(r.v, 0) == 0) {
        continue;
      }
      big_int_t g_r;
      mpz_gcd(g_r.v, r.v, n.v);
      if (mpz_cmp_ui(g_r.v, 1) == 0) {
        break;
      }
    }

    big_int_t noise;
    mpz_powm(noise.v, g_n.v, r.v, n2.v);
    table[static_cast<size_t>(k)] = bigint_to_mem(noise);
  }

  return table;
}

// ---------------------- Host-side g_table cache ----------------------
struct GTableCacheEntry {
  std::shared_ptr<std::vector<mem_t>> table;
  std::list<std::string>::iterator lru_it;
};

static std::mutex g_table_cache_mutex;
static std::unordered_map<std::string, GTableCacheEntry> g_table_cache;
static std::list<std::string> g_table_lru;
static constexpr size_t G_TABLE_CACHE_MAX = 4;

static std::shared_ptr<std::vector<mem_t>>
get_cached_g_table(const std::string &cache_key, const PaillierKeyPairCPU &keys, int key_len_bits) {
  {
    std::lock_guard<std::mutex> lock(g_table_cache_mutex);
    auto it = g_table_cache.find(cache_key);
    if (it != g_table_cache.end()) {
      g_table_lru.erase(it->second.lru_it);
      g_table_lru.push_back(cache_key);
      it->second.lru_it = std::prev(g_table_lru.end());
      return it->second.table;
    }
  }

  auto computed = std::make_shared<std::vector<mem_t>>(precompute_g_table(keys, key_len_bits));

  std::lock_guard<std::mutex> lock(g_table_cache_mutex);
  auto it = g_table_cache.find(cache_key);
  if (it != g_table_cache.end()) {
    return it->second.table;
  }
  g_table_lru.push_back(cache_key);
  auto lru_it = std::prev(g_table_lru.end());
  g_table_cache.emplace(cache_key, GTableCacheEntry{computed, lru_it});
  if (g_table_cache.size() > G_TABLE_CACHE_MAX) {
    const std::string &old_key = g_table_lru.front();
    g_table_cache.erase(old_key);
    g_table_lru.pop_front();
  }
  return computed;
}



// ------------------ Arithmetic helpers ------------------

// --- RNG: xorshift64* ---
static __device__ __forceinline__ uint64_t xorshift64star(uint64_t &s) {
  s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
  return s * 0x2545F4914F6CDD1DULL;
}

static __device__ __forceinline__ uint32_t rand32(uint64_t &s) {
  return (uint32_t)(xorshift64star(s) >> 32);
}

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

// Square-and-multiply specialized for a small exponent with dynamic bit length.
// Iterates only over exp_bits bits of exp_mem instead of the full LIMBS*32.
static __device__ void pow_mod_small_exp(env_t &env,
                                         typename env_t::cgbn_t &result,
                                         const typename env_t::cgbn_t &base,
                                         const mem_t *exp_mem,
                                         const typename env_t::cgbn_t &mod_n2,
                                         int exp_bits) {
  typename env_t::cgbn_t res, b;
  typename env_t::cgbn_wide_t wide;

  cgbn_set_ui32(env, res, 1);
  cgbn_set(env, b, base);

  const uint32_t *EW = reinterpret_cast<const uint32_t*>(exp_mem);

  const int total_bits = exp_bits;
  const int full_words = total_bits / 32;
  const int rem_bits   = total_bits % 32;

  // Process full 32-bit words
  for (int w = 0; w < full_words; ++w) {
    uint32_t x = EW[w];
    for (int k = 0; k < 32; ++k) {
      if (x & 1u) {
        cgbn_mul_wide(env, wide, res, b);
        cgbn_rem_wide(env, res, wide, mod_n2);
      }
      cgbn_mul_wide(env, wide, b, b);
      cgbn_rem_wide(env, b, wide, mod_n2);
      x >>= 1;
    }
  }

  // Process remaining bits in the last word, if any
  if (rem_bits > 0) {
    uint32_t x = EW[full_words];
    for (int k = 0; k < rem_bits; ++k) {
      if (x & 1u) {
        cgbn_mul_wide(env, wide, res, b);
        cgbn_rem_wide(env, res, wide, mod_n2);
      }
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
                                uint64_t seed,
                                int instance_id,
                                bool ensure_coprime=true) {
  typename env_t::cgbn_t tmp, a, b;
  typename env_t::cgbn_t n_minus_1;
  cgbn_sub_ui32(env, n_minus_1, n, 1);

  // seed per instance
  uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(instance_id + 1)) ^ (uint64_t)clock64();

  for (int tries=0; tries<16; ++tries) {
    // Fill a mem_t with random words
    mem_t rmem;
    uint32_t *rw = reinterpret_cast<uint32_t*>(&rmem);
    for (int i=0; i<LIMBS; ++i) rw[i] = rand32(s);

    // r = (rmem % (n-1)) + 1  -> ensures 1..n-1
    cgbn_load(env, tmp, &rmem);
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
                               mem_t *G_TABLE,   // [message_chunks * table_size]
                               mem_t *NOISE_TABLE, // [noise_table_size]
                               int message_chunks,
                               int noise_table_size,
                               mem_t *out_ct,          // [batch * chunk_num]
                               uint64_t seed) {
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

  // --- Build per-chunk base-2^MSG_BITS digits from padded bits ---
  // Padded scheme: for original bit x[j], padded has at positions: 2*j -> 0, 2*j+1 -> x[j].
  const int padded_total = 2 * embed_len;             // total padded bits for this vector
  const int chunk_base   = chunk_idx * chunk_len;     // start bit in padded stream
  const uint8_t *vec_bits = all_bits + vec_idx * embed_len;

  // Only consume the bits that actually exist in this chunk, do NOT extend to chunk_len
  int avail = 0;
  if (chunk_base < padded_total) {
    int max_take = padded_total - chunk_base;
    avail = (max_take > chunk_len) ? chunk_len : max_take;
  }

  // Split the (at most avail)-bit chunk into MSG_BITS-wide digits, matching
  // Python's behavior: for the local substring s = padded_embd[chunk_base:chunk_base+avail],
  // interpret m_local = int(s, 2), then m_split[i] = (m_local >> (MSG_BITS*i)) & ((1<<MSG_BITS)-1).
  const int msg_bits = PAILLIER_MSG_BITS;
  const int table_size = PAILLIER_MSG_TABLE_SIZE;
  const int max_chunks = (KEY_BITS * 2 + msg_bits - 1) / msg_bits;

  uint8_t m_split_local[(KEY_BITS * 2 + PAILLIER_MSG_BITS - 1) / PAILLIER_MSG_BITS];
  for (int i = 0; i < max_chunks; ++i) {
    m_split_local[i] = 0;
  }

  const int local_len = avail;
  for (int t = 0; t < avail; ++t) {
    int p = chunk_base + t;  // global padded index (0..padded_total-1)
    uint32_t bit = 0;
    if ((p & 1) == 1) {      // odd -> carries original bit
      int j = p >> 1;        // source index in the embedding
      bit = (uint32_t)(vec_bits[j] & 1);
    }
    if (!bit) continue;

    // position within this chunk's substring, MSB-first
    int local_e = (local_len - 1) - t;  // exponent in 2^local_e
    int digit_idx = local_e / msg_bits;
    int offset = local_e % msg_bits;
    if (digit_idx >= 0 && digit_idx < message_chunks) {
      m_split_local[digit_idx] |= (uint8_t)(1u << offset);
    }
  }

  // --- Compute g^m via lookup table: g_m = prod_i g_table[i][m_split_local[i]] mod n^2 ---
  typename env_t::cgbn_t g_m;
  typename env_t::cgbn_wide_t wide;
  cgbn_set_ui32(env, g_m, 1);

  for (int i = 0; i < message_chunks; ++i) {
    uint8_t d = m_split_local[i];
    if (d == 0) continue;  // g_table[i][0] == 1
    int idx = i * table_size + (int)d;
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &G_TABLE[idx]);
    cgbn_mul_wide(env, wide, g_m, entry);
    cgbn_rem_wide(env, g_m, wide, n2);
  }

  // --- Sample noise using precomputed NOISE_TABLE entries ---
  typename env_t::cgbn_t noise;
  cgbn_set_ui32(env, noise, 1);

  // Simple per-instance RNG seed, reusing xorshift helper.
  uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(instance + 1)) ^ (uint64_t)clock64();
  for (int k = 0; k < PAILLIER_NOISE_MULTIPLES; ++k) {
    uint32_t r32 = rand32(s);
    int idx = (int)(r32 % (uint32_t)noise_table_size);
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &NOISE_TABLE[idx]);
    cgbn_mul_wide(env, wide, noise, entry);
    cgbn_rem_wide(env, noise, wide, n2);
  }

  // --- c = g_m * noise mod n^2 ---
  typename env_t::cgbn_t c;
  cgbn_mul_wide(env, wide, g_m, noise);
  cgbn_rem_wide(env, c, wide, n2);

  // Store result
  cgbn_store(env, &out_ct[instance], c);
}

// Each instance handles one plaintext integer (already split into MSG_BITS digits)
__global__ void encrypt_cts_kernel(error_report_t *report,
                                   const uint8_t *all_digits, // [batch * message_chunks]
                                   int batch,
                                   int message_chunks,
                                   mem_t *N,               // n
                                   mem_t *N2,              // n^2
                                   mem_t *G_TABLE,         // [message_chunks * table_size]
                                   mem_t *NOISE_TABLE,     // [noise_table_size]
                                   int noise_table_size,
                                   mem_t *out_ct,          // [batch]
                                   uint64_t seed) {
  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= batch) return;

  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  // Load n and n^2 into bigints
  typename env_t::cgbn_t n, n2;
  cgbn_load(env, n,  &N[0]);
  cgbn_load(env, n2, &N2[0]);

  const int table_size = PAILLIER_MSG_TABLE_SIZE;

  // --- Compute g^m via lookup table: g_m = prod_i g_table[i][digit_i] mod n^2 ---
  typename env_t::cgbn_t g_m;
  typename env_t::cgbn_wide_t wide;
  cgbn_set_ui32(env, g_m, 1);

  const uint8_t *digits = all_digits + instance * message_chunks;
  for (int i = 0; i < message_chunks; ++i) {
    uint8_t d = digits[i];
    if (d == 0) continue;  // g_table[i][0] == 1
    int idx = i * table_size + (int)d;
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &G_TABLE[idx]);
    cgbn_mul_wide(env, wide, g_m, entry);
    cgbn_rem_wide(env, g_m, wide, n2);
  }

  // --- Sample noise using precomputed NOISE_TABLE entries ---
  typename env_t::cgbn_t noise;
  cgbn_set_ui32(env, noise, 1);

  uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(instance + 1)) ^ (uint64_t)clock64();
  for (int k = 0; k < PAILLIER_NOISE_MULTIPLES; ++k) {
    uint32_t r32 = rand32(s);
    int idx = (int)(r32 % (uint32_t)noise_table_size);
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &NOISE_TABLE[idx]);
    cgbn_mul_wide(env, wide, noise, entry);
    cgbn_rem_wide(env, noise, wide, n2);
  }

  // --- c = g_m * noise mod n^2 ---
  typename env_t::cgbn_t c;
  cgbn_mul_wide(env, wide, g_m, noise);
  cgbn_rem_wide(env, c, wide, n2);

  cgbn_store(env, &out_ct[instance], c);
}

// ---------- GPU: decrypt-only kernel (per-chunk) ----------
__global__ void decrypt_only_kernel(error_report_t *report,
                                    mem_t *CT,   // [total] ciphertexts
                                    int total,         // batch * chunk_num
                                    mem_t *OUT_M,      // [total] plaintext m
                                    int exp_bits) {
  const int thread   = blockIdx.x*blockDim.x + threadIdx.x;
  const int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  typedef cgbn_context_t<CGBN_TPI> context_t;
  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  typename env_t::cgbn_t n, n2, a, g_a_inv, c, t, x, q, rrem, m;
  typename env_t::cgbn_wide_t wide;

  cgbn_load(env, c,        &CT[instance]);
  cgbn_load(env, n,        &CONST_N);
  cgbn_load(env, n2,       &CONST_N2);
  cgbn_load(env, a,        &CONST_A);
  cgbn_load(env, g_a_inv,  &CONST_G_A_INV);

  // t = c^a mod n^2  (LSB-first exp), using small exponent loop
  pow_mod_small_exp(env, t, c, &CONST_A, n2, exp_bits);

  // x = t - 1, then exact div by n: x = q*n + rrem
  cgbn_sub_ui32(env, x, t, 1);
  cgbn_div(env, q, x, n);
  cgbn_rem(env, rrem, x, n); // for debug you can check rrem==0
  // q = L(t) = (t - 1) / n

  // m = (q * g_a_inv) mod n
  cgbn_mul_wide(env, wide, q, g_a_inv);
  cgbn_rem_wide(env, m, wide, n);

  cgbn_store(env, &OUT_M[instance], m);
}

// Fused decrypt-then-reencrypt kernel (one ciphertext per instance)
__global__ void decrypt_then_reencrypt_kernel(error_report_t *report,
                                              mem_t *CT,     // [batch] ciphertexts under key_a
                                              int batch,
                                              mem_t *N_a,    // key_a n
                                              mem_t *N2_a,   // key_a n^2
                                              mem_t *A_a,    // key_a a
                                              mem_t *G_A_INV_a, // key_a g_a_inv
                                              int exp_bits_a,
                                              mem_t *N2_b,   // key_b n^2
                                              mem_t *G_TABLE_b,   // [message_chunks * table_size]
                                              mem_t *NOISE_TABLE_b, // [noise_table_size]
                                              int message_chunks,
                                              int noise_table_size,
                                              mem_t *OUT_CT,       // [batch] ciphertexts under key_b
                                              uint64_t seed) {
  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= batch) return;

  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  typename env_t::cgbn_t n_a, n2_a, a_a, g_a_inv_a;
  typename env_t::cgbn_t c_in, t, x, q, rrem, m;
  typename env_t::cgbn_wide_t wide;

  cgbn_load(env, c_in,       &CT[instance]);
  cgbn_load(env, n_a,        &N_a[0]);
  cgbn_load(env, n2_a,       &N2_a[0]);
  cgbn_load(env, a_a,        &A_a[0]);
  cgbn_load(env, g_a_inv_a,  &G_A_INV_a[0]);

  // Decrypt: t = c^a mod n^2
  pow_mod_small_exp(env, t, c_in, A_a, n2_a, exp_bits_a);

  // x = t - 1, then x = q*n + rrem
  cgbn_sub_ui32(env, x, t, 1);
  cgbn_div(env, q, x, n_a);
  cgbn_rem(env, rrem, x, n_a);

  // m = (q * g_a_inv) mod n
  cgbn_mul_wide(env, wide, q, g_a_inv_a);
  cgbn_rem_wide(env, m, wide, n_a);

  // Extract plaintext digits (base 2^MSG_BITS) from m
  const int table_size = PAILLIER_MSG_TABLE_SIZE;

  // Load n2_b for encryption
  typename env_t::cgbn_t n2_b;
  cgbn_load(env, n2_b, &N2_b[0]);

  // Compute g^m via lookup table
  typename env_t::cgbn_t g_m;
  cgbn_set_ui32(env, g_m, 1);
  for (int i = 0; i < message_chunks; ++i) {
    uint32_t d = env.extract_bits_ui32(m, i * PAILLIER_MSG_BITS, PAILLIER_MSG_BITS);
    if (d == 0) continue;
    int idx = i * table_size + (int)d;
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &G_TABLE_b[idx]);
    cgbn_mul_wide(env, wide, g_m, entry);
    cgbn_rem_wide(env, g_m, wide, n2_b);
  }

  // Sample noise using precomputed table
  typename env_t::cgbn_t noise;
  cgbn_set_ui32(env, noise, 1);
  uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(instance + 1)) ^ (uint64_t)clock64();
  for (int k = 0; k < PAILLIER_NOISE_MULTIPLES; ++k) {
    uint32_t r32 = rand32(s);
    int idx = (int)(r32 % (uint32_t)noise_table_size);
    typename env_t::cgbn_t entry;
    cgbn_load(env, entry, &NOISE_TABLE_b[idx]);
    cgbn_mul_wide(env, wide, noise, entry);
    cgbn_rem_wide(env, noise, wide, n2_b);
  }

  // c = g_m * noise mod n^2
  typename env_t::cgbn_t c_out;
  cgbn_mul_wide(env, wide, g_m, noise);
  cgbn_rem_wide(env, c_out, wide, n2_b);

  cgbn_store(env, &OUT_CT[instance], c_out);
}



// Fused decrypt + odd-bit popcount kernel
__global__ void decode_hamming_fused_kernel(error_report_t *report,
                                            mem_t *CT,          // [total] ciphertexts
                                            int total,          // batch * chunk_num
                                            int chunk_len,      // = 2*key_len
                                            const mem_t *ODD_MASK,
                                            int words_used,
                                            int exp_bits,
                                            uint32_t *out_odd) {
  const int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  const int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  typedef cgbn_context_t<CGBN_TPI> context_t;
  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  // Load keys from constant memory
  typename env_t::cgbn_t n, n2, a, g_a_inv, c, t, x, q, rrem, m;
  typename env_t::cgbn_wide_t wide;

  cgbn_load(env, c,       &CT[instance]);
  cgbn_load(env, n,       &CONST_N);
  cgbn_load(env, n2,      &CONST_N2);
  cgbn_load(env, a,       &CONST_A);
  cgbn_load(env, g_a_inv, &CONST_G_A_INV);

  // Decrypt: t = c^a mod n^2 using small exponent loop
  pow_mod_small_exp(env, t, c, &CONST_A, n2, exp_bits);

  // x = t - 1, then exact div by n: x = q*n + rrem
  cgbn_sub_ui32(env, x, t, 1);
  cgbn_div(env, q, x, n);
  cgbn_rem(env, rrem, x, n); // rrem should be 0; q = L(t)

  // m = (q * g_a_inv) mod n
  cgbn_mul_wide(env, wide, q, g_a_inv);
  cgbn_rem_wide(env, m, wide, n);

  // Count odd bit positions via mask + popcount
  mem_t mmem;
  cgbn_store(env, &mmem, m);

  const uint32_t *mw = reinterpret_cast<const uint32_t*>(&mmem);
  const uint32_t *ow = reinterpret_cast<const uint32_t*>(&ODD_MASK[0]);

  uint32_t odd_cnt = 0;
  #pragma unroll
  for (int w = 0; w < words_used; ++w) {
    odd_cnt += __popc(mw[w] & ow[w]);
  }

  out_odd[instance] = odd_cnt;
}


// Each instance handles one (vector, chunk) ciphertext -> produces count of ones at odd bit positions
__global__ void decode_hamming_kernel(error_report_t *report,
                                      mem_t *CT,          // [total] ciphertexts
                                      int total,          // batch * chunk_num
                                      int chunk_len,      // = 2*key_len
                                      mem_t *N, mem_t *N2,
                                      mem_t *A, mem_t *G_A_INV,
                                      uint32_t *out_odd,  // <-- NEW name
                                      uint32_t *out_even, // <-- NEW
                                      uint32_t *out_rem_nz, // <-- NEW
                                      mem_t *out_m_dbg,   // <-- OPTIONAL (can pass nullptr)
				      /* Additional parameters for GPU mask inputs for bitcounting*/
				      const mem_t *ODD_MASK,
                                      int words_used,
                                      int exp_bits)
{

  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  context_t context(cgbn_report_monitor, report, instance);
  env_t     env(context);

  // Load inputs
  typename env_t::cgbn_t n, n2, a, g_a_inv, c;
  cgbn_load(env, n,       &N[0]);
  cgbn_load(env, n2,      &N2[0]);
  cgbn_load(env, a,       &A[0]);
  cgbn_load(env, g_a_inv, &G_A_INV[0]);
  cgbn_load(env, c,       &CT[instance]);

  // Decrypt: t = c^a mod n^2
  typename env_t::cgbn_t t, x, q, rrem, m;
  typename env_t::cgbn_wide_t wide;

  pow_mod_small_exp(env, t, c, A, n2, exp_bits);  // t = c^a mod n^2
  // pow_mod_small_exp(env, t, c, A, n2, exp_bits);  // duplicate call kept commented for reversibility
  // x = t - 1  (t is in [1, n^2-1], so this is >= 0)
  cgbn_sub_ui32(env, x, t, 1);

// Divide exactly: x = q*n + rrem  (rrem must be 0)
cgbn_div(env, q, x, n);
cgbn_rem(env, rrem, x, n);
uint32_t rem_nz = (cgbn_compare_ui32(env, rrem, 0) != 0);

// q = L(t) = (t - 1) / n

// m = (q * g_a_inv) mod n
cgbn_mul_wide(env, wide, q, g_a_inv);
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

// Simple kernel: given plaintext m values and an odd-bit mask, count ones at odd positions.
__global__ void odd_popcount_kernel(const mem_t *M,
                                    int total,
                                    const mem_t *ODD_MASK,
                                    int words_used,
                                    uint32_t *out_odd) {
  int thread   = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = thread / CGBN_TPI;
  if (instance >= total) return;

  mem_t mmem = M[instance];
  const uint32_t *mw = reinterpret_cast<const uint32_t*>(&mmem);
  const uint32_t *ow = reinterpret_cast<const uint32_t*>(&ODD_MASK[0]);

  uint32_t odd_cnt = 0;
  #pragma unroll
  for (int w = 0; w < words_used; ++w) {
    odd_cnt += __popc(mw[w] & ow[w]);
  }
  out_odd[instance] = odd_cnt;
}

// ---------------------- Python <-> GMP helpers ----------------------
static bigint::big_int_t big_from_py(py::handle obj) {
  std::string s = py::cast<std::string>(py::str(obj));
  size_t l = s.find_first_not_of(" \t\n\r");
  size_t r = s.find_last_not_of(" \t\n\r");
  if (l == std::string::npos) {
    s = "0";
  } else {
    s = s.substr(l, r - l + 1);
  }
  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
    s = s.substr(2);
  } else if (s.size() > 3 && s[0] == '-' && s[1] == '0' &&
             (s[2] == 'x' || s[2] == 'X')) {
    base = 16;
    s = "-" + s.substr(3);
  }
  return bigint::big_int_t(s, base);
}

static py::int_ py_int_from_big(const bigint::big_int_t &x) {
  std::string dec = x.to_string(10);
  PyObject *pobj = PyLong_FromString(const_cast<char *>(dec.c_str()), nullptr, 10);
  if (!pobj) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::int_>(pobj);
}

static py::object py_mpz_from_big(const bigint::big_int_t &x) {
  static py::object mpz_ctor = py::module_::import("gmpy2").attr("mpz");
  return mpz_ctor(py::str(x.to_string(10)));
}

static py::dict extract_dict_field(const py::dict &d, const char *field) {
  if (d.contains(field)) {
    return py::cast<py::dict>(d[field]);
  }
  return d;
}

static PaillierPublicKeyCPU pk_from_py(const py::dict &d_in) {
  py::dict pk = extract_dict_field(d_in, "pk");
  if (!pk.contains("g") || !pk.contains("n") || !pk.contains("n_squared")) {
    throw std::runtime_error("key pk must contain g, n, n_squared");
  }
  PaillierPublicKeyCPU pk_out;
  pk_out.g = big_from_py(pk["g"]);
  pk_out.n = big_from_py(pk["n"]);
  pk_out.n_squared = big_from_py(pk["n_squared"]);
  return pk_out;
}

static PaillierSecretKeyCPU sk_from_py(const py::dict &d_in) {
  py::dict sk = extract_dict_field(d_in, "sk");
  if (!sk.contains("a") || !sk.contains("g_a_inv")) {
    throw std::runtime_error("key sk must contain a and g_a_inv");
  }
  PaillierSecretKeyCPU sk_out;
  sk_out.a = big_from_py(sk["a"]);
  sk_out.g_a_inv = big_from_py(sk["g_a_inv"]);
  if (sk.contains("phi")) sk_out.phi = big_from_py(sk["phi"]);
  if (sk.contains("inv")) sk_out.inv = big_from_py(sk["inv"]);
  return sk_out;
}

static int message_chunks_from_key(const py::dict &kb, const bigint::big_int_t &n) {
  if (kb.contains("message_chunks")) {
    return py::cast<int>(kb["message_chunks"]);
  }
  if (kb.contains("key_len")) {
    int key_len_bits = py::cast<int>(kb["key_len"]);
    return (key_len_bits * 2 + PAILLIER_MSG_BITS - 1) / PAILLIER_MSG_BITS;
  }
  unsigned n_bits = static_cast<unsigned>(mpz_sizeinbase(n.v, 2));
  return (static_cast<int>(n_bits) + PAILLIER_MSG_BITS - 1) / PAILLIER_MSG_BITS;
}

static std::vector<mem_t> g_table_from_py(const py::handle &obj, int message_chunks) {
  const int table_size = PAILLIER_MSG_TABLE_SIZE;
  std::vector<mem_t> table(static_cast<size_t>(message_chunks) * static_cast<size_t>(table_size));
  if (obj.is_none()) return {};

  auto load_row = [&](int i, const py::handle &row_obj) {
    py::sequence row = py::cast<py::sequence>(row_obj);
    if (static_cast<int>(py::len(row)) != table_size) {
      throw std::runtime_error("g_table row length mismatch");
    }
    size_t base = static_cast<size_t>(i) * static_cast<size_t>(table_size);
    for (int j = 0; j < table_size; ++j) {
      table[base + static_cast<size_t>(j)] = bigint_to_mem(big_from_py(row[j]));
    }
  };

  if (py::isinstance<py::dict>(obj)) {
    py::dict d = py::reinterpret_borrow<py::dict>(obj);
    for (int i = 0; i < message_chunks; ++i) {
      py::object row_obj;
      py::int_ key_i(i);
      if (d.contains(key_i)) {
        row_obj = d[key_i];
      } else {
        py::str key_s(std::to_string(i));
        if (!d.contains(key_s)) {
          throw std::runtime_error("g_table missing row");
        }
        row_obj = d[key_s];
      }
      load_row(i, row_obj);
    }
    return table;
  }

  if (py::isinstance<py::sequence>(obj)) {
    py::sequence outer = py::cast<py::sequence>(obj);
    if (static_cast<int>(py::len(outer)) != message_chunks) {
      throw std::runtime_error("g_table outer length mismatch");
    }
    for (int i = 0; i < message_chunks; ++i) {
      load_row(i, outer[i]);
    }
    return table;
  }

  throw std::runtime_error("g_table must be dict or sequence");
}

static std::vector<mem_t> noise_table_from_py(const py::handle &obj) {
  if (obj.is_none()) return {};
  py::sequence seq = py::cast<py::sequence>(obj);
  const int size = static_cast<int>(py::len(seq));
  if (size <= 0) return {};
  std::vector<mem_t> table(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    table[static_cast<size_t>(i)] = bigint_to_mem(big_from_py(seq[i]));
  }
  return table;
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

static std::vector<bigint::big_int_t>
gpu_pointwise_mulmod_bigint(const std::vector<bigint::big_int_t> &ct1,
                            const std::vector<bigint::big_int_t> &ct2,
                            const bigint::big_int_t &n_squared) {
  if (ct1.size() != ct2.size()) {
    throw std::runtime_error("encode_hamming_*: ct1 and ct2 must have the same length");
  }
  const int count = static_cast<int>(ct1.size());
  if (count == 0) return {};

  std::vector<mem_t> h_A(count), h_B(count);
  mem_t h_MOD_one = bigint_to_mem(n_squared);

  for (int i = 0; i < count; ++i) {
    h_A[i] = bigint_to_mem(ct1[i]);
    h_B[i] = bigint_to_mem(ct2[i]);
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

  std::vector<bigint::big_int_t> out;
  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.emplace_back(mem_to_bigint(h_R[i]));
  }
  return out;
}

// ---------------------- PaillierGPUClient class ----------------------
class PaillierGPULookupClient {
public:
  PaillierGPULookupClient(int embed_len = 512,
                          int key_len = KEY_BITS,
                          int alpha_len = ALPHA_LEN,
                          bool skip_key_gen = false)
      : embed_len_(embed_len),
        key_len_(key_len),
        chunk_len_(key_len * 2),
        chunk_num_(embed_len > key_len ? ((2 * embed_len) / (key_len * 2) +
                                          (((2 * embed_len) % (key_len * 2)) != 0))
                                       : 1),
        alpha_len_(alpha_len),
        a_bits_(alpha_len),
        have_keys_(!skip_key_gen),
        message_chunks_(0),
        tables_ready_(!skip_key_gen),
        device_tables_ready_(false),
        d_g_table_(nullptr),
        d_noise_table_(nullptr),
        d_g_table_entries_(0),
        d_noise_table_entries_(0) {
    if (!skip_key_gen) {
      keys_ = PaillierCPU::key_gen(static_cast<unsigned>(key_len_),
                                   static_cast<unsigned>(alpha_len_));
      pk_g_hex_ = keys_.pk.g.to_string(16);
      pk_n_hex_ = keys_.pk.n.to_string(16);
      pk_n2_hex_ = keys_.pk.n_squared.to_string(16);
      sk_phi_hex_ = keys_.sk.phi.to_string(16);
      sk_inv_hex_ = keys_.sk.inv.to_string(16);
      sk_a_hex_ = keys_.sk.a.to_string(16);
      sk_g_a_inv_hex_ = keys_.sk.g_a_inv.to_string(16);
      a_bits_ = static_cast<int>(mpz_sizeinbase(keys_.sk.a.v, 2));
      message_chunks_ = (key_len_ * 2 + PAILLIER_MSG_BITS - 1) / PAILLIER_MSG_BITS;
      g_table_ = std::make_shared<std::vector<mem_t>>(precompute_g_table(keys_, key_len_));
      noise_table_ = precompute_noise_table(keys_);
      // Light debug: print generated keys (hex)
      // std::cerr << "[PaillierGPUClient] Generated keys (hex):\n"
      //           << "  n        = " << pk_n_hex_ << "\n"
      //           << "  n_squared= " << pk_n2_hex_ << "\n"
      //           << "  phi      = " << sk_phi_hex_ << "\n"
      //           << "  inv      = " << sk_inv_hex_ << std::endl;
    }
  }

  ~PaillierGPULookupClient() {
    free_device_tables_();
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

  // JSON helpers for interoperability with the Python lookup client
  std::string stringify_pk() const {
    if (!have_keys_) {
      throw std::runtime_error("stringify_pk: keys not initialized");
    }
    using namespace bigint;
    py::object json = py::module_::import("json");

    // Compute g_n = g^n mod n^2
    big_int_t g_n;
    mpz_powm(g_n.v, keys_.pk.g.v, keys_.pk.n.v, keys_.pk.n_squared.v);

    py::dict pk_dict;
    auto to_py_int = [](const big_int_t &x) {
      std::string dec = x.to_string(10);
      py::str s_dec(dec);
      return py::int_(s_dec);
    };

    pk_dict["g"]         = to_py_int(keys_.pk.g);
    pk_dict["n"]         = to_py_int(keys_.pk.n);
    pk_dict["n_squared"] = to_py_int(keys_.pk.n_squared);
    pk_dict["g_n"]       = to_py_int(g_n);

    py::object s = json.attr("dumps")(pk_dict);
    return py::cast<std::string>(s);
  }

  std::string stringify_sk() const {
    if (!have_keys_) {
      throw std::runtime_error("stringify_sk: keys not initialized");
    }
    using namespace bigint;
    py::object json = py::module_::import("json");

    py::dict sk_dict;
    auto to_py_int = [](const big_int_t &x) {
      std::string dec = x.to_string(10);
      py::str s_dec(dec);
      return py::int_(s_dec);
    };

    sk_dict["phi"]     = to_py_int(keys_.sk.phi);
    sk_dict["a"]       = to_py_int(keys_.sk.a);
    sk_dict["g_a_inv"] = to_py_int(keys_.sk.g_a_inv);

    py::object s = json.attr("dumps")(sk_dict);
    return py::cast<std::string>(s);
  }

  std::string stringify_config() const {
    py::object json = py::module_::import("json");
    py::dict cfg;
    cfg["message_chunks"] = py::int_(message_chunks_);
    cfg["alpha_len"]      = py::int_(alpha_len_);
    cfg["embed_len"]      = py::int_(embed_len_);
    cfg["key_len"]        = py::int_(key_len_);
    py::object s = json.attr("dumps")(cfg);
    return py::cast<std::string>(s);
  }

  void load_stringified_keys(const std::string &pk_json,
                             const std::string &sk_json) {
    using namespace bigint;
    py::object json = py::module_::import("json");

    // Parse pk
    py::dict pk_dict = json.attr("loads")(pk_json).cast<py::dict>();

    auto get_big_int = [](const py::handle &v) -> big_int_t {
      py::str s_dec = py::str(v);
      std::string s = py::cast<std::string>(s_dec);
      return big_int_t(s, 10);
    };

    big_int_t g   = get_big_int(pk_dict["g"]);
    big_int_t n   = get_big_int(pk_dict["n"]);
    big_int_t n2  = get_big_int(pk_dict["n_squared"]);

    // Parse sk
    py::dict sk_dict = json.attr("loads")(sk_json).cast<py::dict>();
    big_int_t phi    = get_big_int(sk_dict["phi"]);
    big_int_t a      = get_big_int(sk_dict["a"]);
    big_int_t g_a_inv = get_big_int(sk_dict["g_a_inv"]);

    // Recompute inv = phi^{-1} mod n
    big_int_t inv;
    if (!mpz_invert(inv.v, phi.v, n.v)) {
      throw std::runtime_error("load_stringified_keys: phi^{-1} mod n does not exist");
    }

    // Install keys
    keys_.pk.g          = g;
    keys_.pk.n          = n;
    keys_.pk.n_squared  = n2;
    keys_.sk.phi        = phi;
    keys_.sk.inv        = inv;
    keys_.sk.a          = a;
    keys_.sk.g_a_inv    = g_a_inv;

    pk_g_hex_       = keys_.pk.g.to_string(16);
    pk_n_hex_       = keys_.pk.n.to_string(16);
    pk_n2_hex_      = keys_.pk.n_squared.to_string(16);
    sk_phi_hex_     = keys_.sk.phi.to_string(16);
    sk_inv_hex_     = keys_.sk.inv.to_string(16);
    sk_a_hex_       = keys_.sk.a.to_string(16);
    sk_g_a_inv_hex_ = keys_.sk.g_a_inv.to_string(16);
    a_bits_         = static_cast<int>(mpz_sizeinbase(keys_.sk.a.v, 2));
    alpha_len_      = a_bits_;

    g_table_.reset();
    noise_table_.clear();
    tables_ready_ = false;
    free_device_tables_();

    have_keys_ = true;
  }

  void load_config(const py::dict &config, const py::object &tables = py::none()) {
    if (!have_keys_) {
      throw std::runtime_error("load_config: keys must be loaded first");
    }

    embed_len_ = py::cast<int>(config["embed_len"]);
    key_len_   = py::cast<int>(config["key_len"]);
    alpha_len_ = py::cast<int>(config["alpha_len"]);
    chunk_len_ = key_len_ * 2;
    if (embed_len_ > key_len_) {
      chunk_num_ = 2 * embed_len_ / chunk_len_ +
                   int((2 * embed_len_) % chunk_len_ != 0);
    } else {
      chunk_num_ = 1;
    }

    message_chunks_ = py::cast<int>(config["message_chunks"]);

    free_device_tables_();
    a_bits_ = static_cast<int>(mpz_sizeinbase(keys_.sk.a.v, 2));

    if (!tables.is_none()) {
      if (!py::isinstance<py::dict>(tables)) {
        throw std::runtime_error("load_config: tables must be a dict");
      }
      py::dict tables_dict = py::reinterpret_borrow<py::dict>(tables);

      std::vector<mem_t> g_table;
      std::vector<mem_t> noise_table;
      if (tables_dict.contains("g_table")) {
        g_table = g_table_from_py(tables_dict["g_table"], message_chunks_);
      }
      if (tables_dict.contains("noise_table")) {
        noise_table = noise_table_from_py(tables_dict["noise_table"]);
      }

      const int expected_g_table_size = message_chunks_ * PAILLIER_MSG_TABLE_SIZE;
      if (!g_table.empty() && static_cast<int>(g_table.size()) != expected_g_table_size) {
        throw std::runtime_error("load_config: g_table size mismatch");
      }
      if (!noise_table.empty() && static_cast<int>(noise_table.size()) != PAILLIER_NOISE_TABLE_SIZE) {
        throw std::runtime_error("load_config: noise_table size mismatch");
      }

      if (!g_table.empty()) {
        g_table_ = std::make_shared<std::vector<mem_t>>(std::move(g_table));
      } else {
        g_table_.reset();
      }
      noise_table_ = std::move(noise_table);
      tables_ready_ = g_table_ != nullptr && !noise_table_.empty();
      return;
    }

    // Lazy init: defer expensive table computation to first use
    g_table_.reset();
    noise_table_.clear();
    tables_ready_ = false;
  }

  // Batched encrypt: embeddings is [batch][embed_len] of 0/1, returns [batch][chunk_num] ciphers
  py::list
  encrypt(const std::vector<std::vector<int>> &embeddings, std::uint64_t seed = 0) const {
    if (!have_keys_) {
      throw std::runtime_error("encrypt: keys not initialized");
    }
    ensure_tables_ready_();
    ensure_device_tables_();
    const int batch = static_cast<int>(embeddings.size());
    const int embed_len = embed_len_;
    const int chunk_len = chunk_len_;
    const int chunk_num = chunk_num_;

    if (batch == 0) return py::list();

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
    mem_t *d_N = nullptr, *d_N2 = nullptr, *d_out = nullptr;

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

    CUDA_CHECK(cudaMalloc((void **)&d_out, total * sizeof(mem_t)));

    const int msg_bits = PAILLIER_MSG_BITS;
    const int table_size = PAILLIER_MSG_TABLE_SIZE;
    const int message_chunks = message_chunks_;
    const int noise_table_size = PAILLIER_NOISE_TABLE_SIZE;

    if (!g_table_ || (int)g_table_->size() != message_chunks * table_size) {
      throw std::runtime_error("encrypt: g_table size mismatch");
    }
    if ((int)noise_table_.size() != noise_table_size) {
      throw std::runtime_error("encrypt: noise_table size mismatch");
    }

    // If seed == 0, derive a pseudo-random seed from time and simple bit-mixing
    uint64_t actual_seed = seed;
    if (actual_seed == 0) {
      actual_seed = static_cast<uint64_t>(std::time(nullptr));
      // simple scrambling to make it less predictable
      actual_seed ^= (actual_seed << 13);
      actual_seed ^= (actual_seed >> 7);
      actual_seed ^= 0x9E3779B97F4A7C15ULL;
    }

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (total + INSTS_PER_BLK - 1) / INSTS_PER_BLK;

    encrypt_kernel<<<blocks, THREADS>>>(d_report,
                                        d_bits,
                                        batch,
                                        embed_len,
                                        chunk_len,
                                        chunk_num,
                                        d_N,
                                        d_N2,
                                        d_g_table_,
                                        d_noise_table_,
                                        message_chunks,
                                        noise_table_size,
                                        d_out,
                                        actual_seed);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_bits);
    cudaFree(d_N);
    cudaFree(d_N2);
    cudaFree(d_out);
    cudaFree(d_report);

    // Reshape back to [batch][chunk_num] GMP ints
    py::list out(batch);
    for (int i = 0; i < batch; ++i) {
      py::list row(chunk_num);
      for (int c = 0; c < chunk_num; ++c) {
        int idx = i * chunk_num + c;
        row[c] = py_mpz_from_big(mem_to_bigint(h_out[idx]));
      }
      out[i] = row;
    }
    return out;
  }

  // Batched Paillier encryption of arbitrary plaintext integers m < n, using lookup tables.
  // Input: plaintexts as Python ints (converted via string), Output: ciphertexts as Python ints.
  py::list
  encrypt_cts(const std::vector<py::int_> &plaintexts, std::uint64_t seed = 0) const {
    if (!have_keys_) {
      throw std::runtime_error("encrypt_cts: keys not initialized");
    }
    ensure_tables_ready_();
    ensure_device_tables_();
    const int batch = static_cast<int>(plaintexts.size());
    if (batch == 0) return py::list();

    // Prepare n for range checks
    const bigint::big_int_t &n_big = keys_.pk.n;

    const int msg_bits = PAILLIER_MSG_BITS;
    const int message_chunks = message_chunks_;
    const uint64_t mask = (1u << msg_bits) - 1u;

    // Flatten digits: [batch * message_chunks]
    std::vector<uint8_t> h_digits(static_cast<size_t>(batch) *
                                  static_cast<size_t>(message_chunks));

    for (int i = 0; i < batch; ++i) {
      // Convert Python int to decimal string, then to GMP big_int_t
      py::object obj = plaintexts[i];
      py::str s_dec = py::str(obj);
      std::string s = py::cast<std::string>(s_dec);
      bigint::big_int_t m_big(s, 10);

      if (mpz_sgn(m_big.v) < 0 || mpz_cmp(m_big.v, n_big.v) >= 0) {
        throw std::runtime_error("encrypt_cts: plaintext out of range [0, n)");
      }

      for (int j = 0; j < message_chunks; ++j) {
        bigint::big_int_t shifted;
        mpz_tdiv_q_2exp(shifted.v, m_big.v, msg_bits * j);
        unsigned long d = mpz_get_ui(shifted.v) & mask;
        h_digits[static_cast<size_t>(i) * static_cast<size_t>(message_chunks) +
                 static_cast<size_t>(j)] = static_cast<uint8_t>(d);
      }
    }

    // Prepare n and n^2 for device
    std::vector<uint32_t> n_words = hex_to_le_words(pk_n_hex_, LIMBS);
    std::vector<uint32_t> n2_words = hex_to_le_words(pk_n2_hex_, LIMBS);
    mem_t h_N{}, h_N2{};
    std::memcpy(&h_N, n_words.data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2, n2_words.data(), LIMBS * sizeof(uint32_t));

    error_report_t *d_report = nullptr;
    uint8_t *d_digits = nullptr;
    mem_t *d_N = nullptr, *d_N2 = nullptr, *d_out = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_digits,
                          static_cast<size_t>(batch) *
                              static_cast<size_t>(message_chunks) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_digits,
                          h_digits.data(),
                          static_cast<size_t>(batch) *
                              static_cast<size_t>(message_chunks) * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_N, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_N2, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N, &h_N, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N2, &h_N2, sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_out, batch * sizeof(mem_t)));

    const int table_size = PAILLIER_MSG_TABLE_SIZE;
    const int noise_table_size = PAILLIER_NOISE_TABLE_SIZE;

    if (!g_table_ || (int)g_table_->size() != message_chunks * table_size) {
      throw std::runtime_error("encrypt_cts: g_table size mismatch");
    }
    if ((int)noise_table_.size() != noise_table_size) {
      throw std::runtime_error("encrypt_cts: noise_table size mismatch");
    }

    // Seed
    uint64_t actual_seed = seed;
    if (actual_seed == 0) {
      actual_seed = static_cast<uint64_t>(std::time(nullptr));
      actual_seed ^= (actual_seed << 13);
      actual_seed ^= (actual_seed >> 7);
      actual_seed ^= 0x9E3779B97F4A7C15ULL;
    }

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (batch + INSTS_PER_BLK - 1) / INSTS_PER_BLK;

    encrypt_cts_kernel<<<blocks, THREADS>>>(d_report,
                                            d_digits,
                                            batch,
                                            message_chunks,
                                            d_N,
                                            d_N2,
                                            d_g_table_,
                                            d_noise_table_,
                                            noise_table_size,
                                            d_out,
                                            actual_seed);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_out(batch);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, batch * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_digits);
    cudaFree(d_N);
    cudaFree(d_N2);
    cudaFree(d_out);
    cudaFree(d_report);

    // Convert to GMP ints
    py::list out(batch);
    for (int i = 0; i < batch; ++i) {
      out[i] = py_mpz_from_big(mem_to_bigint(h_out[i]));
    }
    return out;
  }

  // Decrypt ciphertexts under key_a, then re-encrypt under key_b (fused GPU kernel).
  py::list decrypt_then_reencrypt(const py::sequence &ciphers,
                                  const py::dict &key_a,
                                  const py::dict &key_b,
                                  std::uint64_t seed = 0) const {
    const int batch = static_cast<int>(py::len(ciphers));
    if (batch == 0) return py::list();

    PaillierPublicKeyCPU pk_a = pk_from_py(key_a);
    PaillierSecretKeyCPU sk_a = sk_from_py(key_a);
    PaillierPublicKeyCPU pk_b = pk_from_py(key_b);

    const int message_chunks = message_chunks_from_key(key_b, pk_b.n);
    const int table_size = PAILLIER_MSG_TABLE_SIZE;

    std::vector<mem_t> g_table;
    std::vector<mem_t> noise_table;

    if (key_b.contains("g_table")) {
      g_table = g_table_from_py(key_b["g_table"], message_chunks);
    }
    if (key_b.contains("noise_table")) {
      noise_table = noise_table_from_py(key_b["noise_table"]);
    }

    if (g_table.empty() || noise_table.empty()) {
      PaillierKeyPairCPU keys_b{};
      keys_b.pk = pk_b;
      if (g_table.empty()) {
        int key_len_bits = (message_chunks * PAILLIER_MSG_BITS) / 2;
        g_table = precompute_g_table(keys_b, key_len_bits);
      }
      if (noise_table.empty()) {
        noise_table = precompute_noise_table(keys_b);
      }
    }

    if ((int)g_table.size() != message_chunks * table_size) {
      throw std::runtime_error("decrypt_then_reencrypt: g_table size mismatch");
    }
    const int noise_table_size = static_cast<int>(noise_table.size());
    if (noise_table_size <= 0) {
      throw std::runtime_error("decrypt_then_reencrypt: noise_table is empty");
    }

    // Pack ciphertexts
    std::vector<mem_t> h_CT(batch);
    for (int i = 0; i < batch; ++i) {
      h_CT[i] = bigint_to_mem(big_from_py(ciphers[i]));
    }

    mem_t h_Na = bigint_to_mem(pk_a.n);
    mem_t h_N2a = bigint_to_mem(pk_a.n_squared);
    mem_t h_Aa = bigint_to_mem(sk_a.a);
    mem_t h_GaInv = bigint_to_mem(sk_a.g_a_inv);
    mem_t h_N2b = bigint_to_mem(pk_b.n_squared);

    error_report_t *d_report = nullptr;
    mem_t *d_CT = nullptr, *d_OUT = nullptr;
    mem_t *d_Na = nullptr, *d_N2a = nullptr, *d_Aa = nullptr, *d_GaInv = nullptr;
    mem_t *d_N2b = nullptr, *d_g_table = nullptr, *d_noise_table = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_CT, batch * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_CT, h_CT.data(), batch * sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_OUT, batch * sizeof(mem_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_Na, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_N2a, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_Aa, sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_GaInv, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_Na, &h_Na, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N2a, &h_N2a, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Aa, &h_Aa, sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_GaInv, &h_GaInv, sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_N2b, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N2b, &h_N2b, sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_g_table,
                          static_cast<size_t>(message_chunks) *
                              static_cast<size_t>(table_size) * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_g_table,
                          g_table.data(),
                          static_cast<size_t>(message_chunks) *
                              static_cast<size_t>(table_size) * sizeof(mem_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_noise_table,
                          static_cast<size_t>(noise_table_size) * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_noise_table,
                          noise_table.data(),
                          static_cast<size_t>(noise_table_size) * sizeof(mem_t),
                          cudaMemcpyHostToDevice));

    int exp_bits_a = static_cast<int>(mpz_sizeinbase(sk_a.a.v, 2));
    if (exp_bits_a <= 0) exp_bits_a = key_len_ * 2;

    uint64_t actual_seed = seed;
    if (actual_seed == 0) {
      actual_seed = static_cast<uint64_t>(std::time(nullptr));
      actual_seed ^= (actual_seed << 13);
      actual_seed ^= (actual_seed >> 7);
      actual_seed ^= 0x9E3779B97F4A7C15ULL;
    }

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (batch + INSTS_PER_BLK - 1) / INSTS_PER_BLK;

    decrypt_then_reencrypt_kernel<<<blocks, THREADS>>>(d_report,
                                                       d_CT,
                                                       batch,
                                                       d_Na,
                                                       d_N2a,
                                                       d_Aa,
                                                       d_GaInv,
                                                       exp_bits_a,
                                                       d_N2b,
                                                       d_g_table,
                                                       d_noise_table,
                                                       message_chunks,
                                                       noise_table_size,
                                                       d_OUT,
                                                       actual_seed);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_OUT(batch);
    CUDA_CHECK(cudaMemcpy(h_OUT.data(), d_OUT, batch * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_CT);
    cudaFree(d_OUT);
    cudaFree(d_Na);
    cudaFree(d_N2a);
    cudaFree(d_Aa);
    cudaFree(d_GaInv);
    cudaFree(d_N2b);
    cudaFree(d_g_table);
    cudaFree(d_noise_table);
    cudaFree(d_report);

    py::list out(batch);
    for (int i = 0; i < batch; ++i) {
      out[i] = py_mpz_from_big(mem_to_bigint(h_OUT[i]));
    }
    return out;
  }

  // Client-side homomorphic addition: ct1[i]*ct2[i] mod n^2
  py::list encode_hamming_client(const py::sequence &ct1,
                                 const py::sequence &ct2) const {
    if (!have_keys_) {
      throw std::runtime_error("encode_hamming_client: keys not initialized");
    }
    if (py::len(ct1) != py::len(ct2)) {
      throw std::runtime_error("encode_hamming_client: ct1 and ct2 must have the same length");
    }
    std::vector<bigint::big_int_t> a;
    std::vector<bigint::big_int_t> b;
    a.reserve(py::len(ct1));
    b.reserve(py::len(ct2));
    for (auto &&o : ct1) a.emplace_back(big_from_py(o));
    for (auto &&o : ct2) b.emplace_back(big_from_py(o));

    auto res = gpu_pointwise_mulmod_bigint(a, b, keys_.pk.n_squared);
    py::list out(res.size());
    for (size_t i = 0; i < res.size(); ++i) {
      out[i] = py_mpz_from_big(res[i]);
    }
    return out;
  }

  // Server-side version: takes pk dict with "n_squared"
  static py::list encode_hamming_server(const py::sequence &ct1,
                                        const py::sequence &ct2,
                                        const py::dict &pk) {
    if (py::len(ct1) != py::len(ct2)) {
      throw std::runtime_error("encode_hamming_server: ct1 and ct2 must have the same length");
    }
    if (!pk.contains("n_squared")) {
      throw std::runtime_error("encode_hamming_server: pk must contain n_squared");
    }
    std::vector<bigint::big_int_t> a;
    std::vector<bigint::big_int_t> b;
    a.reserve(py::len(ct1));
    b.reserve(py::len(ct2));
    for (auto &&o : ct1) a.emplace_back(big_from_py(o));
    for (auto &&o : ct2) b.emplace_back(big_from_py(o));

    bigint::big_int_t n2_big = big_from_py(pk["n_squared"]);
    auto res = gpu_pointwise_mulmod_bigint(a, b, n2_big);
    py::list out(res.size());
    for (size_t i = 0; i < res.size(); ++i) {
      out[i] = py_mpz_from_big(res[i]);
    }
    return out;
  }

  // GPU decrypt chunks to plaintext m, then CPU bit-padding and odd-bit counting
  std::vector<long long>
  decode_hamming_client(const py::sequence &ciphers_batch) const {
    if (!have_keys_) {
      throw std::runtime_error("decode_hamming_client: keys not initialized");
    }
    const int batch = static_cast<int>(py::len(ciphers_batch));
    const int chunk_num = chunk_num_;
    const int chunk_len = chunk_len_;
    if (batch == 0) return {};

    const int total = batch * chunk_num;

    // Pack ciphertexts into mem_t array
    std::vector<mem_t> h_CT(total);
    for (int i = 0; i < batch; ++i) {
      py::sequence row = py::cast<py::sequence>(ciphers_batch[i]);
      if (static_cast<int>(py::len(row)) != chunk_num) {
        throw std::runtime_error("decode_hamming_client: each inner list must have length == chunk_num");
      }
      for (int c = 0; c < chunk_num; ++c) {
        const int idx = i * chunk_num + c;
        h_CT[idx] = bigint_to_mem(big_from_py(row[c]));
      }
    }

    // Keys -> constant memory (for decrypt kernels)
    mem_t h_N{}, h_N2{}, h_A{}, h_G_A_INV{};
    std::memcpy(&h_N,       hex_to_le_words(pk_n_hex_, LIMBS).data(),       LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2,      hex_to_le_words(pk_n2_hex_, LIMBS).data(),      LIMBS * sizeof(uint32_t));
    std::memcpy(&h_A,       hex_to_le_words(sk_a_hex_, LIMBS).data(),       LIMBS * sizeof(uint32_t));
    std::memcpy(&h_G_A_INV, hex_to_le_words(sk_g_a_inv_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));

    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N,       &h_N,       sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N2,      &h_N2,      sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_A,       &h_A,       sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_G_A_INV, &h_G_A_INV, sizeof(mem_t)));

    // Build odd-position mask and compute words used
    mem_t h_MASK = make_odd_mask_host(chunk_len);
    const int words_used = (chunk_len + 31) / 32;

    // Device buffers
    error_report_t *d_report = nullptr;
    mem_t *d_CT = nullptr, *d_M = nullptr, *d_MASK = nullptr;
    uint32_t *d_out_odd = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_CT, total * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_CT, h_CT.data(), total * sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_M, total * sizeof(mem_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_MASK, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_MASK, &h_MASK, sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_out_odd, total * sizeof(uint32_t)));

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (total + INSTS_PER_BLK - 1) / INSTS_PER_BLK;
    int exp_bits = (a_bits_ > 0) ? a_bits_ : (key_len_ * 2);

    // First kernel: decrypt-only to plaintext m in d_M
    decrypt_only_kernel<<<blocks, THREADS>>>(d_report, d_CT, total, d_M, exp_bits);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Second kernel: GPU popcount on m with odd-bit mask
    odd_popcount_kernel<<<blocks, THREADS>>>(d_M,
                                             total,
                                             d_MASK,
                                             words_used,
                                             d_out_odd);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_out_odd(total);
    CUDA_CHECK(cudaMemcpy(h_out_odd.data(), d_out_odd, total * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_CT);
    cudaFree(d_M);
    cudaFree(d_MASK);
    cudaFree(d_out_odd);
    cudaFree(d_report);

    // Sum odd counts per embedding (across chunks)
    std::vector<long long> out(batch, 0);
    for (int i = 0; i < batch; ++i) {
      long long ham = 0;
      for (int c = 0; c < chunk_num; ++c) {
        const int idx = i * chunk_num + c;
        ham += static_cast<long long>(h_out_odd[idx]);
      }
      out[i] = ham;
    }
    return out;
  }

  // Batched Paillier decryption: ciphers is [batch] ciphertext ints, returns Python ints.
  std::vector<py::int_>
  decrypt_cts(const py::sequence &ciphers) const {
    if (!have_keys_) {
      throw std::runtime_error("decrypt_cts: keys not initialized");
    }
    const int batch = static_cast<int>(py::len(ciphers));
    if (batch == 0) return {};

    // Pack CT
    std::vector<mem_t> h_CT(batch);
    for (int i = 0; i < batch; ++i) {
      h_CT[i] = bigint_to_mem(big_from_py(ciphers[i]));
    }

    // Keys -> constant memory
    mem_t h_N{}, h_N2{}, h_A{}, h_G_A_INV{};
    std::memcpy(&h_N,       hex_to_le_words(pk_n_hex_, LIMBS).data(),       LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2,      hex_to_le_words(pk_n2_hex_, LIMBS).data(),      LIMBS * sizeof(uint32_t));
    std::memcpy(&h_A,       hex_to_le_words(sk_a_hex_, LIMBS).data(),       LIMBS * sizeof(uint32_t));
    std::memcpy(&h_G_A_INV, hex_to_le_words(sk_g_a_inv_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));

    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N,       &h_N,       sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N2,      &h_N2,      sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_A,       &h_A,       sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_G_A_INV, &h_G_A_INV, sizeof(mem_t)));

    // Device
    error_report_t *d_report = nullptr;
    mem_t *d_CT = nullptr, *d_M = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));

    CUDA_CHECK(cudaMalloc((void **)&d_CT, batch * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_CT, h_CT.data(), batch * sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_M, batch * sizeof(mem_t)));

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (batch + INSTS_PER_BLK - 1) / INSTS_PER_BLK;
    int exp_bits = (a_bits_ > 0) ? a_bits_ : (key_len_ * 2);
    decrypt_only_kernel<<<blocks, THREADS>>>(d_report, d_CT, batch, d_M, exp_bits);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_M(batch);
    CUDA_CHECK(cudaMemcpy(h_M.data(), d_M, batch * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_CT);
    cudaFree(d_M);
    cudaFree(d_report);

    // Convert plaintexts to Python ints via decimal strings (avoid base issues)
    std::vector<py::int_> out(batch);
    for (int i = 0; i < batch; ++i) {
      out[i] = py_int_from_big(mem_to_bigint(h_M[i]));
    }
    return out;
  }

private:
  int embed_len_;
  int key_len_;
  int chunk_len_;
  int chunk_num_;
  int alpha_len_;
  int a_bits_;
  bool have_keys_;

  PaillierKeyPairCPU keys_;
  std::string pk_g_hex_;
  std::string pk_n_hex_;
  std::string pk_n2_hex_;
  std::string sk_phi_hex_;
  std::string sk_inv_hex_;
  std::string sk_a_hex_;
  std::string sk_g_a_inv_hex_;

  mutable int message_chunks_;
  mutable std::shared_ptr<std::vector<mem_t>> g_table_; // [message_chunks_ * PAILLIER_MSG_TABLE_SIZE]
  mutable std::vector<mem_t> noise_table_;              // [PAILLIER_NOISE_TABLE_SIZE]
  mutable bool tables_ready_;
  mutable bool device_tables_ready_;
  mutable mem_t *d_g_table_;
  mutable mem_t *d_noise_table_;
  mutable int d_g_table_entries_;
  mutable int d_noise_table_entries_;

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

    // keys -> constant memory
    mem_t h_N{}, h_N2{}, h_A{}, h_G_A_INV{};
    std::memcpy(&h_N, hex_to_le_words(pk_n_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_N2, hex_to_le_words(pk_n2_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_A, hex_to_le_words(sk_a_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));
    std::memcpy(&h_G_A_INV, hex_to_le_words(sk_g_a_inv_hex_, LIMBS).data(), LIMBS * sizeof(uint32_t));

    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N, &h_N, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_N2, &h_N2, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_A, &h_A, sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CONST_G_A_INV, &h_G_A_INV, sizeof(mem_t)));

    // device
    error_report_t *d_report = nullptr;
    mem_t *d_CT = nullptr, *d_M = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_report, sizeof(error_report_t)));
    CUDA_CHECK(cudaMemset(d_report, 0, sizeof(error_report_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_CT, total * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_CT, h_CT.data(), total * sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_M, total * sizeof(mem_t)));

    const int INSTS_PER_BLK = 8;
    const int THREADS = INSTS_PER_BLK * CGBN_TPI;
    const int blocks = (total + INSTS_PER_BLK - 1) / INSTS_PER_BLK;
    int exp_bits = (a_bits_ > 0) ? a_bits_ : (key_len_ * 2);
    decrypt_only_kernel<<<blocks, THREADS>>>(d_report, d_CT, total, d_M, exp_bits);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<mem_t> h_M(total);
    CUDA_CHECK(cudaMemcpy(h_M.data(), d_M, total * sizeof(mem_t), cudaMemcpyDeviceToHost));

    cudaFree(d_CT);
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

  std::string g_table_cache_key_() const {
    return pk_g_hex_ + "|" + pk_n_hex_ + "|" + std::to_string(key_len_);
  }

  void ensure_tables_ready_() const {
    if (tables_ready_) return;
    if (!have_keys_) {
      throw std::runtime_error("lookup tables: keys not initialized");
    }
    if (message_chunks_ <= 0) {
      message_chunks_ = (key_len_ * 2 + PAILLIER_MSG_BITS - 1) / PAILLIER_MSG_BITS;
    }
    if (!g_table_) {
      const std::string cache_key = g_table_cache_key_();
      g_table_ = get_cached_g_table(cache_key, keys_, key_len_);
    }
    if (noise_table_.empty()) {
      noise_table_ = precompute_noise_table(keys_);
    }
    tables_ready_ = true;
  }

  void ensure_device_tables_() const {
    if (device_tables_ready_) return;
    ensure_tables_ready_();
    if (!g_table_ || g_table_->empty()) {
      throw std::runtime_error("lookup tables: g_table not initialized");
    }
    if (noise_table_.empty()) {
      throw std::runtime_error("lookup tables: noise_table not initialized");
    }
    d_g_table_entries_ = static_cast<int>(g_table_->size());
    d_noise_table_entries_ = static_cast<int>(noise_table_.size());

    CUDA_CHECK(cudaMalloc((void **)&d_g_table_,
                          static_cast<size_t>(d_g_table_entries_) * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_g_table_,
                          g_table_->data(),
                          static_cast<size_t>(d_g_table_entries_) * sizeof(mem_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_noise_table_,
                          static_cast<size_t>(d_noise_table_entries_) * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_noise_table_,
                          noise_table_.data(),
                          static_cast<size_t>(d_noise_table_entries_) * sizeof(mem_t),
                          cudaMemcpyHostToDevice));

    device_tables_ready_ = true;
  }

  void free_device_tables_() const {
    if (d_g_table_ != nullptr) {
      cudaFree(d_g_table_);
      d_g_table_ = nullptr;
    }
    if (d_noise_table_ != nullptr) {
      cudaFree(d_noise_table_);
      d_noise_table_ = nullptr;
    }
    device_tables_ready_ = false;
    d_g_table_entries_ = 0;
    d_noise_table_entries_ = 0;
  }
};

// Backwards-compatible alias
using PaillierGPUClient = PaillierGPULookupClient;

// ---------------------- Pybind module ----------------------
PYBIND11_MODULE(paillier_GPU_lookup_client, m) {
  py::class_<PaillierGPULookupClient> lookup_cls(m, "PaillierGPULookupClient");
  lookup_cls
      .def(py::init<int, int, int, bool>(),
           py::arg("embed_len") = 512,
           py::arg("key_len") = KEY_BITS,
           py::arg("alpha_len") = ALPHA_LEN,
           py::arg("skip_key_gen") = false)
      .def("encrypt", &PaillierGPULookupClient::encrypt,
           py::arg("embeddings"),
           py::arg("seed") = 0ULL)
      .def("encrypt_cts", &PaillierGPULookupClient::encrypt_cts,
           py::arg("plaintexts"),
           py::arg("seed") = 0ULL)
      .def("decrypt_then_reencrypt", &PaillierGPULookupClient::decrypt_then_reencrypt,
           py::arg("ciphers"),
           py::arg("key_a"),
           py::arg("key_b"),
           py::arg("seed") = 0ULL)
      .def("encode_hamming_client", &PaillierGPULookupClient::encode_hamming_client,
           py::arg("ct1"),
           py::arg("ct2"))
      .def_static("encode_hamming_server", &PaillierGPULookupClient::encode_hamming_server,
                  py::arg("ct1"),
                  py::arg("ct2"),
                  py::arg("pk"))
      .def("decode_hamming_client", &PaillierGPULookupClient::decode_hamming_client,
           py::arg("ciphers_batch"))
      .def("decrypt_cts", &PaillierGPULookupClient::decrypt_cts,
           py::arg("ciphers"))
      .def("get_pk_hex", &PaillierGPULookupClient::get_pk_hex)
      .def("get_keys_hex", &PaillierGPULookupClient::get_keys_hex)
      .def("stringify_pk", &PaillierGPULookupClient::stringify_pk)
      .def("stringify_sk", &PaillierGPULookupClient::stringify_sk)
      .def("stringify_config", &PaillierGPULookupClient::stringify_config)
      .def("load_stringified_keys", &PaillierGPULookupClient::load_stringified_keys)
      .def("load_config", &PaillierGPULookupClient::load_config,
           py::arg("config"),
           py::arg("tables") = py::none())
      .def_property_readonly("embed_len", &PaillierGPULookupClient::embed_len)
      .def_property_readonly("key_len", &PaillierGPULookupClient::key_len)
      .def_property_readonly("chunk_len", &PaillierGPULookupClient::chunk_len)
      .def_property_readonly("chunk_num", &PaillierGPULookupClient::chunk_num);
  // Alias for compatibility with existing Python code
  m.attr("PaillierGPUClient") = lookup_cls;
}
