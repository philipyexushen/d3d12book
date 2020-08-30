//***************************************************************************************
// color.hlsl by Frank Luna (C) 2015 All Rights Reserved.
//
// Transforms and colors geometry.
//***************************************************************************************

cbuffer cbPerObject : register(b0)
{
	float4x4 gWorldViewProj; 
};

/*
* cbuffer cbPerGtimeObject : register(b1)
{
	float gTime;
};
*/

struct VPosData
{
	float3 PosL  : POSITION;
};

struct VExtraData
{
	float4 Color : COLOR;
};

struct VertexIn
{
	float3 PosL  : POSITION;
    float4 Color : COLOR;
};

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float4 Color : COLOR;
};

VertexOut VS(VPosData posData, VExtraData extraData)
{
	VertexOut vout;
	// posData.PosL.xy += 0.5f * sin(posData.PosL.x) * sin(3.0f * gTime);
	// posData.PosL.z *= 0.6f + 0.4f * sin(2.0f * gTime);
	
	// Transform to homogeneous clip space.
	vout.PosH = mul(float4(posData.PosL, 1.0f), gWorldViewProj);
	
	// Just pass vertex color into the pixel shader.
    vout.Color = extraData.Color;
    
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    return pin.Color;
}


