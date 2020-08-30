//***************************************************************************************
// BoxApp.cpp by Frank Luna (C) 2015 All Rights Reserved.
//
// Shows how to draw a box in Direct3D 12.
//
// Controls:
//   Hold the left mouse button down and move the mouse to rotate.
//   Hold the right mouse button down and move the mouse to zoom in and out.
//***************************************************************************************

#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

struct Vertex
{
	XMFLOAT3 Pos;
	XMFLOAT4 Color;
};

struct VPosData
{
    XMFLOAT3 Pos;
};

struct VExtraData
{
    XMFLOAT4 Color;
	XMFLOAT4 GameTime;
};

struct ObjectConstants
{
    XMFLOAT4X4 WorldViewProj = MathHelper::Identity4x4();
};

struct ObjectTimerConstants
{
	float gTime = 0.0f;
};

class BoxApp : public D3DApp
{
public:
	BoxApp(HINSTANCE hInstance);
    BoxApp(const BoxApp& rhs) = delete;
    BoxApp& operator=(const BoxApp& rhs) = delete;
	~BoxApp();

	virtual bool Initialize()override;

private:
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

    void BuildDescriptorHeaps();
	void BuildConstantBuffers();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildTriangleGeometry();
    void BuildBoxGeometry();
    void BuildPSO();

private:
    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;

	std::unique_ptr<UploadBuffer<ObjectConstants>> mObjectCB;
	// std::unique_ptr<UploadBuffer<ObjectTimerConstants>> mTimerObjectCB;

	std::unique_ptr<MeshGeometry> mTriGeo;
	std::unique_ptr<MeshGeometry> mBoxGeo;

    ComPtr<ID3DBlob> mvsByteCode = nullptr;
    ComPtr<ID3DBlob> mpsByteCode = nullptr;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

    ComPtr<ID3D12PipelineState> mPSO = nullptr;

    XMFLOAT4X4 mWorld = MathHelper::Identity4x4();
	XMFLOAT4X4 mWorld2 = MathHelper::Identity4x4();
    XMFLOAT4X4 mView = MathHelper::Identity4x4();
    XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    float mTheta = 1.5f*XM_PI;
    float mPhi = XM_PIDIV4;
    float mRadius = 5.0f;

    POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
				   PSTR cmdLine, int showCmd)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    try
    {
        BoxApp theApp(hInstance);
        if(!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch(DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

BoxApp::BoxApp(HINSTANCE hInstance)
: D3DApp(hInstance) 
{
	mWorld2._41 = -2.0f;
	mWorld2._43 = -1.0f;
}

BoxApp::~BoxApp()
{
}

bool BoxApp::Initialize()
{
    if(!D3DApp::Initialize())
		return false;
		
    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));
 
    BuildDescriptorHeaps();
	BuildConstantBuffers();
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildTriangleGeometry();
	BuildBoxGeometry();
    BuildPSO();

    // Execute the initialization commands.
    ThrowIfFailed(mCommandList->Close());
	ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait until initialization is complete.
    FlushCommandQueue();

	return true;
}

void BoxApp::OnResize()
{
	D3DApp::OnResize();

    // The window resized, so update the aspect ratio and recompute the projection matrix.
	// 透视矩阵
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f*MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
    XMStoreFloat4x4(&mProj, P);
}

void BoxApp::Update(const GameTimer& gt)
{
    // Convert Spherical to Cartesian coordinates.
    float x = mRadius*sinf(mPhi)*cosf(mTheta);
    float z = mRadius*sinf(mPhi)*sinf(mTheta);
    float y = mRadius*cosf(mPhi);

    // Build the view matrix.
    XMVECTOR pos = XMVectorSet(x, y, z, 1.0f);
    XMVECTOR target = XMVectorZero();
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	// 找到观察矩阵
    XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
    XMStoreFloat4x4(&mView, view);

    XMMATRIX world = XMLoadFloat4x4(&mWorld);
    XMMATRIX proj = XMLoadFloat4x4(&mProj); // 我们必须在齐次裁剪控件内
    XMMATRIX worldViewProj = world*view*proj;

	// Update the constant buffer with the latest worldViewProj matrix.
	ObjectConstants objConstants;
    XMStoreFloat4x4(&objConstants.WorldViewProj, XMMatrixTranspose(worldViewProj));
    mObjectCB->CopyData(0, objConstants);

	XMMATRIX world2 = XMLoadFloat4x4(&mWorld2);
	XMMATRIX worldViewProj2 = world2 * view * proj;

	ObjectConstants objConstants2;
	XMStoreFloat4x4(&objConstants2.WorldViewProj, XMMatrixTranspose(worldViewProj2));
	mObjectCB->CopyData(1, objConstants2);

// 	ObjectTimerConstants objTimerConstants;
// 	objTimerConstants.gTime = gt.TotalTime();
// 	mTimerObjectCB->CopyData(0, objTimerConstants);
}

void BoxApp::Draw(const GameTimer& gt)
{
    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
	ThrowIfFailed(mDirectCmdListAlloc->Reset());

	// A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mPSO.Get()));

    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear the back buffer and depth buffer.
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(), Colors::LightSteelBlue, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
	
    // Specify the buffers we are going to render to.
	mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

	ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

	// 先画三角形
    // draw的时候，要给VS传参数，参数就是IASetVertexBuffers决定
    D3D12_VERTEX_BUFFER_VIEW bufferViewList[2];
    bufferViewList[0] = mTriGeo->VertexBufferView();
    bufferViewList[1] = mTriGeo->VertexColorBufferView();

	mCommandList->IASetVertexBuffers(0, 2, bufferViewList);
	mCommandList->IASetIndexBuffer(&mTriGeo->IndexBufferView());
    mCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    mCommandList->SetGraphicsRootDescriptorTable(0, mCbvHeap->GetGPUDescriptorHandleForHeapStart());

	auto& triArgs = mTriGeo->DrawArgs["box"];
    mCommandList->DrawIndexedInstanced(
		triArgs.IndexCount,
		1, 
		triArgs.StartIndexLocation, 
		triArgs.BaseVertexLocation,
		0);

	// 再画正方体
	bufferViewList[0] = mBoxGeo->VertexBufferView();
	bufferViewList[1] = mBoxGeo->VertexColorBufferView();

	mCommandList->IASetVertexBuffers(0, 2, bufferViewList);
	mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());
	mCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	auto handleIncrementSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	CD3DX12_GPU_DESCRIPTOR_HANDLE hanlde(mCbvHeap->GetGPUDescriptorHandleForHeapStart(), handleIncrementSize);
	mCommandList->SetGraphicsRootDescriptorTable(0, hanlde);

	auto& cubeArgs = mBoxGeo->DrawArgs["box"];
	mCommandList->DrawIndexedInstanced(
		cubeArgs.IndexCount,
		1,
		cubeArgs.StartIndexLocation,
		cubeArgs.BaseVertexLocation,
		0);
	
    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    // Done recording commands.
	ThrowIfFailed(mCommandList->Close());
 
    // Add the command list to the queue for execution.
	ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);
	
	// swap the back and front buffers
	ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

	// Wait until frame commands are complete.  This waiting is inefficient and is
	// done for simplicity.  Later we will show how to organize our rendering code
	// so we do not have to wait per frame.
	FlushCommandQueue();
}

void BoxApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void BoxApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void BoxApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if((btnState & MK_LBUTTON) != 0)
    {
        // Make each pixel correspond to a quarter of a degree.
        float dx = XMConvertToRadians(0.25f*static_cast<float>(x - mLastMousePos.x));
        float dy = XMConvertToRadians(0.25f*static_cast<float>(y - mLastMousePos.y));

        // Update angles based on input to orbit camera around box.
        mTheta += dx;
        mPhi += dy;

        // Restrict the angle mPhi.
        mPhi = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    }
    else if((btnState & MK_RBUTTON) != 0)
    {
        // Make each pixel correspond to 0.005 unit in the scene.
        float dx = 0.005f*static_cast<float>(x - mLastMousePos.x);
        float dy = 0.005f*static_cast<float>(y - mLastMousePos.y);

        // Update the camera radius based on input.
        mRadius += dx - dy;

        // Restrict the radius.
        mRadius = MathHelper::Clamp(mRadius, 3.0f, 15.0f);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}

void BoxApp::BuildDescriptorHeaps()
{
    // 描述符堆
    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = 2;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&cbvHeapDesc,
        IID_PPV_ARGS(&mCbvHeap)));
}

void BoxApp::BuildConstantBuffers()
{
	mObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(md3dDevice.Get(), 2, true);
	
	{
		UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
		// Offset to the ith object constant buffer in the buffer.

		D3D12_GPU_VIRTUAL_ADDRESS cbAddress = mObjectCB->Resource()->GetGPUVirtualAddress();
		int boxCBufIndex = 0;
		cbAddress += boxCBufIndex * objCBByteSize;
		// DESC是Describes的意思
		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
		cbvDesc.BufferLocation = cbAddress;
		cbvDesc.SizeInBytes = objCBByteSize;

		md3dDevice->CreateConstantBufferView(
			&cbvDesc,
			mCbvHeap->GetCPUDescriptorHandleForHeapStart());
	}

	{
		UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
		// Offset to the ith object constant buffer in the buffer.

		// 内存在CPU和GPU的占用都是一个大小，所以可以直接用偏移的方式定位到第二个资源在GPU的位置
		D3D12_GPU_VIRTUAL_ADDRESS cbAddress = mObjectCB->Resource()->GetGPUVirtualAddress();
		int boxCBufIndex = 1;
		cbAddress += boxCBufIndex * objCBByteSize;

		auto handleIncrementSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		CD3DX12_CPU_DESCRIPTOR_HANDLE hGPUHeapStart(mCbvHeap->GetCPUDescriptorHandleForHeapStart(), 1 * handleIncrementSize);
		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
		cbvDesc.BufferLocation = cbAddress; // 绑定到第二个资源
		cbvDesc.SizeInBytes = objCBByteSize;

		// 绑定到hlsl的第二个描述附上
		md3dDevice->CreateConstantBufferView(
			&cbvDesc,
			hGPUHeapStart);
	}

// 	mTimerObjectCB = std::make_unique<UploadBuffer<ObjectTimerConstants>>(md3dDevice.Get(), 1, true);
// 	{
// 		UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectTimerConstants));
// 		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
// 		cbvDesc.BufferLocation = mTimerObjectCB->Resource()->GetGPUVirtualAddress();
// 		cbvDesc.SizeInBytes = objCBByteSize;
// 
// 		auto handleIncrementSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
// 		CD3DX12_CPU_DESCRIPTOR_HANDLE hGPUHeapStart(mCbvHeap->GetCPUDescriptorHandleForHeapStart(), 1 * handleIncrementSize);
// 
// 		md3dDevice->CreateConstantBufferView(
// 			&cbvDesc,
// 			hGPUHeapStart);
// 	}
}

void BoxApp::BuildRootSignature()
{
	// Shader programs typically require resources as input (constant buffers,
	// textures, samplers).  The root signature defines the resources the shader
	// programs expect.  If we think of the shader programs as a function, and
	// the input resources as function parameters, then the root signature can be
	// thought of as defining the function signature.  

	// Root parameter can be a table, root descriptor or root constants.
	CD3DX12_ROOT_PARAMETER slotRootParameter[1];

	// Create a single descriptor table of CBVs.
	CD3DX12_DESCRIPTOR_RANGE cbvTable;
	// cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 2, 0);
	cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
	slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable);

	// A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1, slotRootParameter, 0, nullptr, 
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	// create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

	if(errorBlob != nullptr)
	{
		::OutputDebugStringW((wchar_t*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&mRootSignature)));
}

void BoxApp::BuildShadersAndInputLayout()
{
    HRESULT hr = S_OK;
    
	mvsByteCode = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "VS", "vs_5_0");
	mpsByteCode = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "PS", "ps_5_0");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };
}

void BoxApp::BuildTriangleGeometry()
{
	std::array<VPosData, 5> vertices =
	{
		VPosData({ XMFLOAT3(+1.0f, +1.0f, +0.0f) }),
		VPosData({ XMFLOAT3(+1.0f, -1.0f, +0.0f) }),
		VPosData({ XMFLOAT3(-1.0f, -1.0f, +0.0f) }),
		VPosData({ XMFLOAT3(-1.0f, +1.0f, +0.0f) }),
		VPosData({ XMFLOAT3(+0.0f, -0.0f, -1.41421f) }),
	};

	std::array<VExtraData, 5> verticesColor =
	{
		VExtraData({ XMFLOAT4(Colors::Violet),  }),
		VExtraData({ XMFLOAT4(Colors::Violet) }),
		VExtraData({ XMFLOAT4(Colors::Violet) }),
		VExtraData({ XMFLOAT4(Colors::Violet) }),
		VExtraData({ XMFLOAT4(Colors::Yellow) }),
	};

	// 顶点索引
	std::array<std::uint16_t, 18> indices =
	{
		// 底部两个三角形
		0, 3, 2, 
		0, 2, 1,

		// 四个斜面
		4, 2, 3,
		4, 3, 0,
		4, 0, 1,
		4, 1, 2
	};

	const UINT vbByteSize = (UINT)vertices.size() * sizeof(VPosData);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);
	const UINT vbColorByteSize = (UINT)verticesColor.size() * sizeof(VExtraData);

	mTriGeo = std::make_unique<MeshGeometry>();
	mTriGeo->Name = "triGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &mTriGeo->VertexBufferCPU));
	CopyMemory(mTriGeo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &mTriGeo->IndexBufferCPU));
	CopyMemory(mTriGeo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	ThrowIfFailed(D3DCreateBlob(vbColorByteSize, &mTriGeo->VertexColorBufferCPU));
	CopyMemory(mTriGeo->VertexColorBufferCPU->GetBufferPointer(), verticesColor.data(), vbColorByteSize);

	mTriGeo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, mTriGeo->VertexBufferUploader);

	mTriGeo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, mTriGeo->IndexBufferUploader);

	mTriGeo->VertexColorBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), verticesColor.data(), vbColorByteSize, mTriGeo->VertexColorBufferUploader);

	mTriGeo->VertexByteStride = sizeof(VPosData);
	mTriGeo->VertexBufferByteSize = vbByteSize;
	mTriGeo->IndexFormat = DXGI_FORMAT_R16_UINT;
	mTriGeo->IndexBufferByteSize = ibByteSize;
	mTriGeo->VertexColorBufferStride = sizeof(VExtraData);
	mTriGeo->VertexColorBufferByteSize = vbColorByteSize;

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	mTriGeo->DrawArgs["box"] = submesh;
}

void BoxApp::BuildBoxGeometry()
{
    std::array<VPosData, 8> vertices =
    {
        VPosData({ XMFLOAT3(-1.0f, -1.0f, -1.0f) }),
        VPosData({ XMFLOAT3(-1.0f, +1.0f, -1.0f) }),
        VPosData({ XMFLOAT3(+1.0f, +1.0f, -1.0f) }),
        VPosData({ XMFLOAT3(+1.0f, -1.0f, -1.0f) }),
        VPosData({ XMFLOAT3(-1.0f, -1.0f, +1.0f) }),
        VPosData({ XMFLOAT3(-1.0f, +1.0f, +1.0f) }),
        VPosData({ XMFLOAT3(+1.0f, +1.0f, +1.0f) }),
        VPosData({ XMFLOAT3(+1.0f, -1.0f, +1.0f) })
    };

    std::array<VExtraData, 8> verticesColor =
    {
        VExtraData({ XMFLOAT4(Colors::White) }),
        VExtraData({ XMFLOAT4(Colors::Black) }),
        VExtraData({ XMFLOAT4(Colors::Red) }),
        VExtraData({ XMFLOAT4(Colors::Green) }),
        VExtraData({ XMFLOAT4(Colors::Blue) }),
        VExtraData({ XMFLOAT4(Colors::Yellow) }),
        VExtraData({ XMFLOAT4(Colors::Cyan) }),
        VExtraData({ XMFLOAT4(Colors::Magenta) })
    };

    // 顶点索引
	std::array<std::uint16_t, 36> indices =
	{
		// front face
		0, 1, 2,
		0, 2, 3,

		// back face
		4, 6, 5,
		4, 7, 6,

		// left face
		4, 5, 1,
		4, 1, 0,

		// right face
		3, 2, 6,
		3, 6, 7,

		// top face
		1, 5, 6,
		1, 6, 2,

		// bottom face
		4, 0, 3,
		4, 3, 7
	};

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(VPosData);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);
    const UINT vbColorByteSize = (UINT)verticesColor.size() * sizeof(VExtraData);

	mBoxGeo = std::make_unique<MeshGeometry>();
	mBoxGeo->Name = "boxGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &mBoxGeo->VertexBufferCPU));
	CopyMemory(mBoxGeo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &mBoxGeo->IndexBufferCPU));
	CopyMemory(mBoxGeo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	ThrowIfFailed(D3DCreateBlob(vbColorByteSize, &mBoxGeo->VertexColorBufferCPU));
	CopyMemory(mBoxGeo->VertexColorBufferCPU->GetBufferPointer(), verticesColor.data(), vbColorByteSize);

	mBoxGeo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, mBoxGeo->VertexBufferUploader);

	mBoxGeo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, mBoxGeo->IndexBufferUploader);

	mBoxGeo->VertexColorBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), verticesColor.data(), vbColorByteSize, mBoxGeo->VertexColorBufferUploader);

	mBoxGeo->VertexByteStride = sizeof(VPosData);
	mBoxGeo->VertexBufferByteSize = vbByteSize;
	mBoxGeo->IndexFormat = DXGI_FORMAT_R16_UINT;
	mBoxGeo->IndexBufferByteSize = ibByteSize;
	mBoxGeo->VertexColorBufferStride = sizeof(VExtraData);
	mBoxGeo->VertexColorBufferByteSize = vbColorByteSize;

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	mBoxGeo->DrawArgs["box"] = submesh;
}

void BoxApp::BuildPSO()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
    ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    psoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.VS = 
	{ 
		reinterpret_cast<BYTE*>(mvsByteCode->GetBufferPointer()), 
		mvsByteCode->GetBufferSize() 
	};
    psoDesc.PS = 
	{ 
		reinterpret_cast<BYTE*>(mpsByteCode->GetBufferPointer()), 
		mpsByteCode->GetBufferSize() 
	};
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = mBackBufferFormat;
    psoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    psoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    psoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPSO)));
}