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
// E8M0 is an exponent-only format. Decode it by constructing the FP32
// exponent bits, matching the A5 combine implementation used by MC2.
template <typename PacketType>
__aicore__ inline void DequantizeE4M3AndAccumulate(LocalTensor<PacketType> &packet, LocalTensor<float> &sum,
                                                   LocalTensor<float> &scaleFloat, LocalTensor<float> &debugToken,
                                                   float expertScale, uint32_t tokenLen)
{
    const uint32_t scaleCount = ScaleCount(tokenLen);
    LocalTensor<fp8_e4m3fn_t> fp8Packet = packet.template ReinterpretCast<fp8_e4m3fn_t>();
    LocalTensor<fp8_e8m0_t> scales = fp8Packet[AlignUp(tokenLen, MX_DATA_ALIGN)].template ReinterpretCast<fp8_e8m0_t>();
    __ubuf__ fp8_e4m3fn_t *token = (__ubuf__ fp8_e4m3fn_t *)fp8Packet.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *scale = (__ubuf__ fp8_e8m0_t *)scales.GetPhyAddr();
    __ubuf__ float *scaleFloatPtr = (__ubuf__ float *)scaleFloat.GetPhyAddr();
    __ubuf__ float *debugTokenPtr = (__ubuf__ float *)debugToken.GetPhyAddr();
    __ubuf__ float *sumPtr = (__ubuf__ float *)sum.GetPhyAddr();

    const uint32_t fp32Vl = quant::GetVRegSizeDispatch() / sizeof(float);
    const uint16_t scaleRepeat = (scaleCount + fp32Vl - 1U) / fp32Vl;
    const uint16_t tokenRepeat = (tokenLen + fp32Vl * 2U - 1U) / (fp32Vl * 2U);
    uint32_t remainingScale = scaleCount;
    uint32_t remainingToken = tokenLen;
    uint32_t remainingTokenBytes = tokenLen * 4U;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<fp8_e8m0_t> scaleReg;
        MicroAPI::RegTensor<fp8_e4m3fn_t> tokenReg;
        MicroAPI::RegTensor<float> tokenFloatReg0;
        MicroAPI::RegTensor<float> tokenFloatReg1;
        MicroAPI::RegTensor<float> scaleFloatReg;
        MicroAPI::RegTensor<float> outReg0;
        MicroAPI::RegTensor<float> outReg1;
        MicroAPI::RegTensor<float> sumReg0;
        MicroAPI::RegTensor<float> sumReg1;
        MicroAPI::MaskReg scaleMask;
        MicroAPI::MaskReg tokenMask;
        static constexpr MicroAPI::CastTrait castTraitZero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTraitTwo = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < scaleRepeat; ++i) {
            scaleMask = MicroAPI::UpdateMask<float>(remainingScale);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(scaleReg, scale + i * fp32Vl);
            MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint32_t> &)scaleFloatReg,
                                 (MicroAPI::RegTensor<uint32_t> &)scaleReg, static_cast<int16_t>(23), scaleMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(scaleFloatPtr + i * fp32Vl * 2U,
                                                                           scaleFloatReg, scaleFloatReg, scaleMask);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < tokenRepeat; ++i) {
            tokenMask = MicroAPI::UpdateMask<fp8_e4m3fn_t>(remainingTokenBytes);
            MicroAPI::MaskReg outputMask = MicroAPI::UpdateMask<float>(remainingToken);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(scaleFloatReg, scaleFloatPtr + i * 8U);
            MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(tokenReg, token + i * fp32Vl * 2U);
            MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitZero>(tokenFloatReg0, tokenReg, tokenMask);
            MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitTwo>(tokenFloatReg1, tokenReg, tokenMask);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(sumReg0, sumReg1, sumPtr + i * fp32Vl * 2U);
            MicroAPI::Mul(outReg0, scaleFloatReg, tokenFloatReg0, outputMask);
            MicroAPI::Mul(outReg1, scaleFloatReg, tokenFloatReg1, outputMask);
            // Temporary debug staging: keep the pure dequantized token before
            // expert weighting so the dump can distinguish decode from reduce.
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(debugTokenPtr + i * fp32Vl * 2U, outReg0,
                                                                           outReg1, outputMask);
            MicroAPI::Muls(outReg0, outReg0, expertScale, outputMask);
            MicroAPI::Muls(outReg1, outReg1, expertScale, outputMask);
            MicroAPI::Add(sumReg0, sumReg0, outReg0, outputMask);
            MicroAPI::Add(sumReg1, sumReg1, outReg1, outputMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(sumPtr + i * fp32Vl * 2U, sumReg0, sumReg1,
                                                                           outputMask);
        }
    }
}

}  // namespace MoeMxfp8

#endif  // MOE_MXFP8_UTILS_H
