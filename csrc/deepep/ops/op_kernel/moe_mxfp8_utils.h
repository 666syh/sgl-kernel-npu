#ifndef MOE_MXFP8_UTILS_H
#define MOE_MXFP8_UTILS_H

#include "kernel_operator.h"
#include "quantize_functions.h"

namespace MoeMxfp8 {

using namespace AscendC;

constexpr uint32_t MX_BLOCK_SIZE = 32U;
constexpr uint32_t MX_DATA_ALIGN = 256U;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

__aicore__ inline uint32_t ScaleCount(uint32_t tokenLen)
{
    return AlignUp((tokenLen + MX_BLOCK_SIZE - 1U) / MX_BLOCK_SIZE, 2U);
}

template <typename InputType, typename PacketType>
__aicore__ inline void QuantizeE4M3(LocalTensor<PacketType> &packet, LocalTensor<InputType> &input,
                                    LocalTensor<float> &scratch, uint32_t tokenLen)
{
    const uint32_t scaleCount = ScaleCount(tokenLen);
    const uint32_t scaleScratchCount = AlignUp(scaleCount, 32U);
    __ubuf__ InputType *src = (__ubuf__ InputType *)input.GetPhyAddr();
    __ubuf__ uint16_t *maxExp = (__ubuf__ uint16_t *)scratch.GetPhyAddr();
    __ubuf__ uint16_t *halfScale = (__ubuf__ uint16_t *)scratch[scaleScratchCount].GetPhyAddr();
    LocalTensor<fp8_e4m3fn_t> fp8Packet = packet.template ReinterpretCast<fp8_e4m3fn_t>();
    __ubuf__ int8_t *out = (__ubuf__ int8_t *)fp8Packet.GetPhyAddr();
    __ubuf__ uint16_t *mxScale = (__ubuf__ uint16_t *)fp8Packet[AlignUp(tokenLen, MX_DATA_ALIGN)].GetPhyAddr();

    quant::ComputeMaxExp(src, maxExp, tokenLen);
    quant::ComputeScale<fp8_e4m3fn_t>(maxExp, mxScale, halfScale, scaleCount);
    quant::ComputeFp8Data<InputType, fp8_e4m3fn_t, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(src, halfScale, out,
                                                                                                tokenLen);
}

// Decode one packed [FP8 E4M3 data | E8M0 scales] token directly into FP32.
// The scale conversion is E8M0 -> 2 BF16 lanes -> 4 FP32 lanes, so callers
// must reserve scaleCount * 2 BF16 elements and scaleCount * 4 FP32 elements.
template <typename PacketType>
__aicore__ inline void DequantizeE4M3ToFloat(LocalTensor<PacketType> &packet, LocalTensor<float> &output,
                                             LocalTensor<bfloat16_t> &scaleBf16, LocalTensor<float> &scaleFloat,
                                             uint32_t tokenLen)
{
    const uint32_t scaleCount = ScaleCount(tokenLen);
    LocalTensor<fp8_e4m3fn_t> fp8Packet = packet.template ReinterpretCast<fp8_e4m3fn_t>();
    LocalTensor<fp8_e8m0_t> scales = fp8Packet[AlignUp(tokenLen, MX_DATA_ALIGN)].template ReinterpretCast<fp8_e8m0_t>();
    __ubuf__ fp8_e4m3fn_t *token = (__ubuf__ fp8_e4m3fn_t *)fp8Packet.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *scale = (__ubuf__ fp8_e8m0_t *)scales.GetPhyAddr();
    __ubuf__ bfloat16_t *scaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16.GetPhyAddr();
    __ubuf__ float *scaleFloatPtr = (__ubuf__ float *)scaleFloat.GetPhyAddr();
    __ubuf__ float *out = (__ubuf__ float *)output.GetPhyAddr();

    const uint32_t bf16Vl = quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
    const uint32_t fp32Vl = quant::GetVRegSizeDispatch() / sizeof(float);
    const uint16_t scaleRepeat = (scaleCount + bf16Vl - 1U) / bf16Vl;
    const uint16_t scaleFloatRepeat = (scaleCount * 2U + fp32Vl - 1U) / fp32Vl;
    const uint16_t tokenRepeat = (tokenLen + fp32Vl - 1U) / fp32Vl;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<fp8_e8m0_t> scaleReg;
        MicroAPI::RegTensor<fp8_e4m3fn_t> tokenReg;
        MicroAPI::RegTensor<float> tokenFloatReg;
        MicroAPI::RegTensor<bfloat16_t> scaleBf16Reg;
        MicroAPI::RegTensor<bfloat16_t> convertedScaleBf16Reg;
        MicroAPI::RegTensor<float> scaleFloatReg;
        MicroAPI::RegTensor<float> outReg;
        MicroAPI::MaskReg mask;
        static constexpr MicroAPI::CastTrait fp8ToBf16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait bf16ToFp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < scaleRepeat; ++i) {
            mask = MicroAPI::UpdateMask<bfloat16_t>(scaleCount - i * bf16Vl);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(scaleReg, scale + i * bf16Vl);
            MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, fp8ToBf16>(convertedScaleBf16Reg, scaleReg, mask);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleBf16Ptr + i * bf16Vl * 2U, convertedScaleBf16Reg, convertedScaleBf16Reg, mask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < scaleFloatRepeat; ++i) {
            mask = MicroAPI::UpdateMask<float>(scaleCount * 2U - i * fp32Vl);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleBf16Reg,
                                                                                scaleBf16Ptr + i * fp32Vl);
            MicroAPI::Cast<float, bfloat16_t, bf16ToFp32>(scaleFloatReg, scaleBf16Reg, mask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(scaleFloatPtr + i * fp32Vl * 2U,
                                                                           scaleFloatReg, scaleFloatReg, mask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < tokenRepeat; ++i) {
            mask = MicroAPI::UpdateMask<float>(tokenLen - i * fp32Vl);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(scaleFloatReg, scaleFloatPtr + i * 8U);
            MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenReg, token + i * fp32Vl);
            MicroAPI::Cast<float, fp8_e4m3fn_t, fp8ToBf16>(tokenFloatReg, tokenReg, mask);
            MicroAPI::Mul(outReg, scaleFloatReg, tokenFloatReg, mask);
            MicroAPI::DataCopy(out + i * fp32Vl, outReg, mask);
        }
    }
}

}  // namespace MoeMxfp8

#endif  // MOE_MXFP8_UTILS_H
