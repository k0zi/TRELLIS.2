/*
 * CPU-only serialize implementations (no CUDA required).
 *
 * Provides z-order and Hilbert encoding/decoding in pure C++,
 * plus stub wrappers that alias the CUDA-named symbols to their
 * CPU counterparts so Python code can call either name.
 *
 * Used when building without CUDA/HIP (setup.py CPU_ONLY path).
 */

#include <torch/extension.h>
#include <cstdint>
#include <cstddef>


// ---------------------------------------------------------------------------
// Bit-manipulation helpers (identical to the __host__ __device__ helpers in
// z_order.cu / hilbert.cu, but without CUDA decorators)
// ---------------------------------------------------------------------------

static inline uint32_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint32_t extractBits(uint32_t v)
{
    v = v & 0x49249249;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0x030000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}


// ---------------------------------------------------------------------------
// Z-order (Morton) encoding – CPU
// ---------------------------------------------------------------------------

torch::Tensor z_order_encode_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z)
{
    const size_t N = x.size(0);
    auto codes = torch::empty_like(x, torch::dtype(torch::kInt32));

    const uint32_t* px = reinterpret_cast<const uint32_t*>(x.contiguous().data_ptr<int>());
    const uint32_t* py = reinterpret_cast<const uint32_t*>(y.contiguous().data_ptr<int>());
    const uint32_t* pz = reinterpret_cast<const uint32_t*>(z.contiguous().data_ptr<int>());
    uint32_t* out = reinterpret_cast<uint32_t*>(codes.data_ptr<int>());

    for (size_t i = 0; i < N; ++i) {
        out[i] = expandBits(px[i]) * 4 + expandBits(py[i]) * 2 + expandBits(pz[i]);
    }
    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
z_order_decode_cpu(const torch::Tensor& codes)
{
    const size_t N = codes.size(0);
    auto tx = torch::empty_like(codes, torch::dtype(torch::kInt32));
    auto ty = torch::empty_like(codes, torch::dtype(torch::kInt32));
    auto tz = torch::empty_like(codes, torch::dtype(torch::kInt32));

    const uint32_t* in  = reinterpret_cast<const uint32_t*>(codes.contiguous().data_ptr<int>());
    uint32_t* ox = reinterpret_cast<uint32_t*>(tx.data_ptr<int>());
    uint32_t* oy = reinterpret_cast<uint32_t*>(ty.data_ptr<int>());
    uint32_t* oz = reinterpret_cast<uint32_t*>(tz.data_ptr<int>());

    for (size_t i = 0; i < N; ++i) {
        ox[i] = extractBits(in[i] >> 2);
        oy[i] = extractBits(in[i] >> 1);
        oz[i] = extractBits(in[i]);
    }
    return std::make_tuple(tx, ty, tz);
}


// ---------------------------------------------------------------------------
// Hilbert encoding – CPU
// ---------------------------------------------------------------------------

torch::Tensor hilbert_encode_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z)
{
    const size_t N = x.size(0);
    auto codes = torch::empty_like(x);

    const uint32_t* px = reinterpret_cast<const uint32_t*>(x.contiguous().data_ptr<int>());
    const uint32_t* py = reinterpret_cast<const uint32_t*>(y.contiguous().data_ptr<int>());
    const uint32_t* pz = reinterpret_cast<const uint32_t*>(z.contiguous().data_ptr<int>());
    uint32_t* out = reinterpret_cast<uint32_t*>(codes.data_ptr<int>());

    for (size_t i = 0; i < N; ++i) {
        uint32_t point[3] = {px[i], py[i], pz[i]};
        uint32_t m = 1u << 9, q, p, t;

        q = m;
        while (q > 1) {
            p = q - 1;
            for (int j = 0; j < 3; ++j) {
                if (point[j] & q) { point[0] ^= p; }
                else { t = (point[0] ^ point[j]) & p; point[0] ^= t; point[j] ^= t; }
            }
            q >>= 1;
        }
        for (int j = 1; j < 3; ++j) point[j] ^= point[j-1];
        t = 0; q = m;
        while (q > 1) { if (point[2] & q) t ^= q - 1; q >>= 1; }
        for (int j = 0; j < 3; ++j) point[j] ^= t;

        out[i] = expandBits(point[0]) * 4 + expandBits(point[1]) * 2 + expandBits(point[2]);
    }
    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hilbert_decode_cpu(const torch::Tensor& codes)
{
    const size_t N = codes.size(0);
    auto tx = torch::empty_like(codes);
    auto ty = torch::empty_like(codes);
    auto tz = torch::empty_like(codes);

    const uint32_t* in  = reinterpret_cast<const uint32_t*>(codes.contiguous().data_ptr<int>());
    uint32_t* ox = reinterpret_cast<uint32_t*>(tx.data_ptr<int>());
    uint32_t* oy = reinterpret_cast<uint32_t*>(ty.data_ptr<int>());
    uint32_t* oz = reinterpret_cast<uint32_t*>(tz.data_ptr<int>());

    for (size_t i = 0; i < N; ++i) {
        uint32_t point[3];
        point[0] = extractBits(in[i] >> 2);
        point[1] = extractBits(in[i] >> 1);
        point[2] = extractBits(in[i]);

        uint32_t m = 2u << 9, q, p, t;
        t = point[2] >> 1;
        for (int j = 2; j > 0; --j) point[j] ^= point[j-1];
        point[0] ^= t;

        q = 2;
        while (q != m) {
            p = q - 1;
            for (int j = 2; j >= 0; --j) {
                if (point[j] & q) { point[0] ^= p; }
                else { t = (point[0] ^ point[j]) & p; point[0] ^= t; point[j] ^= t; }
            }
            q <<= 1;
        }
        ox[i] = point[0]; oy[i] = point[1]; oz[i] = point[2];
    }
    return std::make_tuple(tx, ty, tz);
}


// ---------------------------------------------------------------------------
// Stubs: CUDA-named symbols that forward to CPU implementations.
// These let Python code call z_order_encode_cuda() even without a GPU.
// ---------------------------------------------------------------------------

torch::Tensor z_order_encode_cuda(
    const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) {
    return z_order_encode_cpu(x, y, z);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
z_order_decode_cuda(const torch::Tensor& codes) {
    return z_order_decode_cpu(codes);
}
torch::Tensor hilbert_encode_cuda(
    const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) {
    return hilbert_encode_cpu(x, y, z);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hilbert_decode_cuda(const torch::Tensor& codes) {
    return hilbert_decode_cpu(codes);
}
