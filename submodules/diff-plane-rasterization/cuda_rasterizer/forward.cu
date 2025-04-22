/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 高斯球中心点坐标变换到相机坐标系
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 限制高斯球中心点坐标范围
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	// xy除了个z
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 限制大小后又把z给乘回去了
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// Σ′=JWΣW^TJ^T
	// 注:上面的雅可比矩阵J蕴含相机内参的信息、视角变换矩阵 
	// W蕴含相机外参信息。 
	// 于是已知相机内外参和3D协方差矩阵Σ则可计算高斯球投影在成像平面上的2D协方差矩阵Σ′
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	// 这里是对称的，所以返回三个值就行了
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	// 缩放矩阵
	// mod是外面python传来的scale_modifier,用于调节高斯球大小
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	// 从四元数计算旋转矩阵R
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Σ=RSS^TR^T
	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// 对每个高斯球进行处理
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,						// 高斯球中心在世界空间下的坐标
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,						// view transform matrix，世界空间坐标转相机空间坐标
	const float* projmatrix,						// 这里的projmatrix其实是projection transform和view transform的合体，直接把世界空间坐标转NDC空间坐标
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,								// block块的范围
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();		// 线程在组内的标号，区间为[0,num_threads)
	if (idx >= P)									// 如果编号大于高斯球的数量，则返回，也就是说一个线程对应一个高斯球处理	
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;				// 高斯球半径
	tiles_touched[idx] = 0;		// 高斯球在图片上覆盖的tile数量

	// Perform near culling, quit if outside.
	float3 p_view;
	// 执行近剔除，如果高斯球离成像平面太近则退出	
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	// 高斯球中心在世界空间下的坐标
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	// 世界空间坐标直接转NDC坐标，并且自动变齐次坐标
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);

	// 齐次转回非齐次,此时处于NDC坐标
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;	//  预设的3D协方差
	}
	else
	{
		// 用高斯球的scales和rotations其3D协方差，放进cov3Ds里
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		// 从cov3Ds里读取高斯球的3D协方差矩阵
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 3D协方差矩阵计算投影到成像平面上的2D协方差
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);	// 协方差矩阵[abbc]的模(行列式)ac-b^2
	if (det == 0.0f)		// 协方差矩阵的模为0说明投影出来椭圆面积为0，退出
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };		// 协方差矩阵[abbc]的逆 = det*(c,-b,a)

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	
	// 计算两个特征值，作为椭圆的两个半径
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

	//乘3是为了使高斯分布的正负三个标准差的值都覆盖进来
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// point_image: 中心点坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;

	// 计算这个圆覆盖了哪些tile
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// if (opacities[idx] > 0.9 && point_image.x > 0 && point_image.x < 1264 && point_image.y > 0 && point_image.y < 832) {
	// 	glm::vec4 q = rotations[idx];
	// 	glm::vec3 cp = *cam_pos;
	// 	printf("q(wxyz) %lf %lf %lf %lf, scale %lf %lf %lf, mean3d %lf %lf %lf, c %lf %lf %lf\n viewmatrix %lf %lf %lf %lf, %lf %lf %lf %lf, %lf %lf %lf %lf, %lf %lf %lf %lf\n", 
	// 		q.x, q.y, q.z, q.w, scales[idx].x, scales[idx].y, scales[idx].z,
	// 		p_orig.x, p_orig.y, p_orig.z, cp.x, cp.y, cp.z, 
	// 		viewmatrix[0],viewmatrix[4],viewmatrix[8],viewmatrix[12],
	// 		viewmatrix[1],viewmatrix[5],viewmatrix[9],viewmatrix[13],
	// 		viewmatrix[2],viewmatrix[6],viewmatrix[10],viewmatrix[14],
	// 		viewmatrix[3],viewmatrix[7],viewmatrix[11],viewmatrix[15]);
	// }

	// Store some useful helper data for the next steps.
	// 该高斯椭球中心距成像平面距离(深度)
	depths[idx] = p_view.z;
	// 该高斯椭球投影在成像平面上的长轴半径
	radii[idx] = my_radius;
	// 该高斯椭球中心在成像平面上的像素坐标
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	// 高斯椭球的协方差逆矩阵和透明度放在一起了
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	
	// 总共覆盖了几个tile
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 对每个tile中的每个像素一个renderCUDA像素
template <uint32_t CHANNELS, uint32_t ALL_MAP>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const float cx, const float cy,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ cam_pos,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ all_map,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ out_observe,
	float* __restrict__ out_all_map,
	float* __restrict__ out_plane_depth,
	const bool render_geo)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	
	// 获取当前tile对应的像素range
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	
	// 该thread需要处理的像素的坐标
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	
	// 该thread对应的像素编号
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const float2 ray = { (pixf.x - cx) / focal_x, (pixf.y - cy) / focal_y };

	// Check if this thread is associated with a valid pixel or outside.
	// 是否超出图像区域范围
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 当前tile在point_list_keys中的范围，包括x下界和y上界
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	//当前tile在point_list_keys中的range大小除BLOCK_SIZE，也即把当前线程即要处理的高斯球每BLOCK_SIZE个分一个batch，之后for循环里是一个batch一个batch的处理
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 当前线程即要处理的高斯球数量
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// 存储block内各thread处理的高斯球的编号
	// 渲染前的初始化
	// 公用BLOCK_SIZE个thread分批读取数据
	// 存储block内各thread处理的高斯球的编号
	__shared__ int collected_id[BLOCK_SIZE];
	// 存储block内各thread处理的高斯球中心在2D平面的投影坐标
	__shared__ float2 collected_xy[BLOCK_SIZE];
	// 存储block内各thread处理的高斯球的2D协方差逆矩阵和不透明度
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	/**
	block(tile)里的每个thread(像素)都要对这个tile中的所有高斯球进行渲染，所以一个thread需要访问上面的collected_*数组里的每一项。 
	这里最速度的情况是一次把这个tile中的所有高斯球都读进来渲染，collected_*数组长度设置为高斯球的数量。 当然，这种方法占内存太高，
	本文采用的方法是利用thread进行并行读取，所以才把collected_*数组长度设置为BLOCK_SIZE，这样就是充分利用block中的BLOCK_SIZE个thread，读一批处理一批。
	*/
	float T = 1.0f;					// 透射率
	uint32_t contributor = 0;		// 总过经过了多少个高斯
	uint32_t last_contributor = 0;	// 存储最终经过的高斯求数量
	float C[CHANNELS] = { 0 };		// 高斯球颜色
	float All_map[ALL_MAP] = { 0 };

	// Iterate over batches until all done or range is complete
	// for循环处理该像素的每个高斯球，每次处理一个batch
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);		// 计算一个block里done为true的数量
		if (num_done == BLOCK_SIZE)						// 如果done有BLOCK_SIZE个true，则表明所有批次的高斯球全部处理完成，可以退出
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// thread_rank是当前线程在组内的标号，区间为[0, num_threads)；progress是当前批次应该从第几个高斯球开始读
		int progress = i * BLOCK_SIZE + block.thread_rank();
		
		// 当前线程有效，即处理的高斯球不越界
		if (range.x + progress < range.y)
		{	
			int coll_id = point_list[range.x + progress]; //point_list表示与已排序的point_list_keys对应的高斯球编号，coll_id为当前线程处理的高斯球编号
			collected_id[block.thread_rank()] = coll_id; //读取待处理的高斯球id
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; //读取待处理的高斯球中心在像素平面的坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; //读取待处理的高斯球2D协方差逆矩阵和不透明度
		}
		block.sync();		// 同步，确保读取全部完成

		// Iterate over current batch
		// 然后每个thread各自处理读进来的BLOCK_SIZE个高斯球：
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			// 当前这个像素与高斯的距离
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			// 高斯球2D协方差逆矩阵和不透明度
			float4 con_o = collected_conic_opacity[j]; 
			// 通过相对位置、高斯球2D协方差逆矩阵计算高斯分布的指数部分
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 通过高斯分布的指数部分和不透明度计算alpha值
			float alpha = min(0.99f, con_o.w * exp(power));
			// 由alpha值计算透射率
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			// 透射率太低表明前几个高斯球已经遮住了这个像素，则该像素渲染结束
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			if (render_geo) {
				for (int ch = 0; ch < ALL_MAP; ch++)
					All_map[ch] += all_map[collected_id[j] * ALL_MAP + ch] * alpha * T;
			}
			
			if (T > 0.5)
			{
				atomicAdd(&(out_observe[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;			 //记录最终的透射率
		n_contrib[pix_id] = last_contributor;		//记录经过几个高斯球才被遮挡
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];			// 记录颜色
		if (render_geo) {
			for (int ch = 0; ch < ALL_MAP; ch++)
				out_all_map[ch * H * W + pix_id] = All_map[ch];
			out_plane_depth[pix_id] = All_map[4] / -(All_map[0] * ray.x + All_map[1] * ray.y + All_map[2] + 1.0e-8);
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const float cx, const float cy,
	const float* viewmatrix,
	const float* cam_pos,
	const float2* means2D,
	const float* colors,
	const float* all_map,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* out_observe,
	float* out_all_map,
	float* out_plane_depth,
	const bool render_geo)
{
	// 对grid个tile各启动block(16x16)个线程,即每个像素一个线程运行renderCUDA
	renderCUDA<NUM_CHANNELS,NUM_ALL_MAP> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		cx, cy,
		viewmatrix,
		cam_pos,
		means2D,
		colors,
		all_map,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_observe,
		out_all_map,
		out_plane_depth,
		render_geo);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}