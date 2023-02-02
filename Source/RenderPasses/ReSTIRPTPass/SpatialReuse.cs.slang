/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Params;
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingData;
import Rendering.Materials.MaterialShading;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import NRDHelpers;

struct PathReusePass
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    // Resources
    RestirPathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture1D<float2> neighborOffsets;

    ByteAddressBuffer nRooksPattern;

    RWTexture2D<float4> outputColor;                      ///< Output resolved color.
    StructuredBuffer<PathReservoir> outputReservoirs;     // reservoir from previous pass
    RWStructuredBuffer<PathReservoir> temporalReservoirs; // resulting reservoir for next frame
    StructuredBuffer<PixelReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<PathReuseMISWeight> misWeightBuffer;

    RWTexture2D<float4> outputNRDDiffuseRadianceHitDist;    ///< Output resolved diffuse color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDSpecularRadianceHitDist;   ///< Output resolved specular color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDResidualRadianceHitDist;   ///< Output resolved residual color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    Texture2D<float4> primaryHitEmission;                     ///< Output per-pixel primary hit emission. Only valid if kOutputNRDData == true.

    int gSpatialRoundId;
    int  gNumSpatialRounds;
    uint gSpatialReusePattern;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;
    // for small window
    int gSmallWindowRadius;

    bool gIsLastRound;

    bool gFeatureBasedRejection;

    uint gSppId;

    Texture2D<float4> directLighting;  
    bool useDirectLighting;

    bool isValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    bool isValidGeometry(ShadingData centralSd, ShadingData neighborSd)
    {
        if (!gFeatureBasedRejection) return true;
        float3 camPos = gScene.camera.data.posW;
        float centralDist = distance(camPos, centralSd.posW);
        float neighborDist = distance(camPos, neighborSd.posW);
        return dot(centralSd.N, neighborSd.N) >= 0.5f && abs(centralDist - neighborDist) < 0.1f * centralDist;
    }

    ShadingData getPixelShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);

            TriangleHit th = PrimaryHit.getTriangleHit();
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    int getNeighborCount()
    {
        if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathReuse) return 16; //include self
        else
        {
            if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Default)
            {
                return gNeighborCount; // does not include self
            }
            else
            {
                int smallWindowDiameter = 2 * gSmallWindowRadius + 1;
                return smallWindowDiameter * smallWindowDiameter; // count self, but not use self as neighbor
            }
        }
    }

    int2 getPathReuseNextNeighborPixel(int4 NRookQuery, int2 pixel, int i)
    {
        // 
        if (PathReusePattern(gSpatialReusePattern) == PathReusePattern::Block)
        {
            int2 block = pixel / 4;
            return 4 * block + int2(i % 4, i / 4);
        }
        else
        {
            // fetch neighbors
            int groupId = NRookQuery.z;
            uint2 words = nRooksPattern.Load2(NRookQuery.w * 256 + groupId * 8);
            int rowId = (words[i / 8] >> (4 * (i % 8))) & 0xF;

            return int2(i, rowId) - NRookQuery.xy + pixel;
        }
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        int2 neighborPixel = int2(0,0);

        if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Default)
        {
            uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
            neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);
        }
        else if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::SmallWindow)
        {
            int smallWindowDiameter = 2 * gSmallWindowRadius + 1;
            neighborPixel = pixel + int2(-gSmallWindowRadius + (i % smallWindowDiameter),
                -gSmallWindowRadius + (i / smallWindowDiameter));
            if (all(neighborPixel == pixel)) neighborPixel = int2(-1);
        }
 
        return neighborPixel;
    }

    float pairwiseMisWeight(float w_i, float w_c, float M_i, float M_c)
    {
        float denom = w_i * M_i + w_c * M_c;
        return denom <= 0 ? 1 : ((w_i * M_i) / denom);
    }
    
    void ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples + 1 + gSpatialRoundId);

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = outputReservoirs[centralOffset];

        PackedHitInfo centralPrimaryHitPacked;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);

        if (!isValidPackedHitInfo(centralPrimaryHitPacked)) return;

        PathReservoir dstReservoir = centralReservoir;
        dstReservoir.init();

        float3 color = 0.f;
        ReconnectionData dummyRcData;
        dummyRcData.Init();

        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        /////////////////////
        /// PAIRWISE RMIS ///
        /////////////////////
        if (ReSTIRMISKind(kSpatialReSTIRMISKind) == ReSTIRMISKind::Pairwise)
        {
            int validNeighborCount = 0;
            float canonicalWeight = 1;

            for (int i = 0; i < neighborCount; ++i)
            {
                int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);

                if (!isValidScreenRegion(neighborPixel)) continue;

                PackedHitInfo neighborPrimaryHitPacked;
                ShadingData neighborPrimarySd = getPixelShadingData(neighborPixel, neighborPrimaryHitPacked);
                if (!isValidPackedHitInfo(neighborPrimaryHitPacked)) continue;
                if (!isValidGeometry(centralPrimarySd, neighborPrimarySd)) continue;

                PathReservoir neighborReservoir = outputReservoirs[params.getReservoirOffset(neighborPixel)];

                float dstJacobian;
                float prefix_approxPdf = 0.f;
                float prefixJacobian;

                ReconnectionData rcData;
                if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && centralReservoir.pathFlags.rcVertexLength() > 1)
                    rcData = reconnectionDataBuffer[centralOffset].data[2 * i];
                else
                    rcData = dummyRcData;

                float3 prefixIntegrand = computeShiftedIntegrand(params, prefixJacobian, neighborPrimaryHitPacked, neighborPrimarySd,
                                                                 centralPrimarySd, centralReservoir, rcData, true);
                prefix_approxPdf = PathReservoir::computeWeight(prefixIntegrand) * prefixJacobian;

                canonicalWeight += 1;
                if (prefix_approxPdf > 0.f)
                {
                    float w = pairwiseMisWeight(prefix_approxPdf, PathReservoir::computeWeight(centralReservoir.F),
                                                neighborReservoir.M * neighborCount, centralReservoir.M);
                    canonicalWeight -= w;
                }

                if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && neighborReservoir.pathFlags.rcVertexLength() > 1)
                    rcData = reconnectionDataBuffer[centralOffset].data[2 * i + 1];
                else
                    rcData = dummyRcData;

                float3 dstIntegrand = computeShiftedIntegrand(params, dstJacobian, centralPrimaryHitPacked, centralPrimarySd,
                                                              neighborPrimarySd, neighborReservoir, rcData, true);
                float dst_approxPdf = PathReservoir::computeWeight(dstIntegrand) * dstJacobian;

                float neighborWeight = 0.f;
                if (dst_approxPdf > 0.f)
                {
                    neighborWeight = pairwiseMisWeight(PathReservoir::computeWeight(neighborReservoir.F), dst_approxPdf,
                                                       neighborReservoir.M * neighborCount, centralReservoir.M);

                    if (isnan(neighborWeight) || isinf(neighborWeight)) neighborWeight = 0.f;
                }

                mergeReservoirWithResamplingMIS(params, dstIntegrand, dstJacobian, dstReservoir, neighborReservoir, neighborReservoir, sg, true, neighborWeight);

                validNeighborCount++;
            }

            mergeReservoirWithResamplingMIS(params, centralReservoir.F, 1.f, dstReservoir, centralReservoir, centralReservoir, sg, true, canonicalWeight);

            if (dstReservoir.weight > 0)
            {
                dstReservoir.finalizeGRIS();
                dstReservoir.weight /= (validNeighborCount + 1); // compensate for the fact that pairwise resampling MIS was not divided by (k+1)
            }
            color = dstReservoir.F * dstReservoir.weight;
        }

        if (dstReservoir.weight < 0.f) dstReservoir.weight = 0.f;
        if (isnan(dstReservoir.weight) || isinf(dstReservoir.weight)) dstReservoir.weight = 0.f;

        temporalReservoirs[centralOffset] = dstReservoir;

        if (any(isnan(color) || isinf(color) || color < 0.f)) color = 0.f;
        if (gIsLastRound)
        {
            outputColor[pixel] += float4(color / kSamplesPerPixel, 1.f);
        }

    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
};

cbuffer CB
{
    PathReusePass gPathReusePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathReusePass.execute(dispatchThreadId.xy);
}
