// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <cstdlib>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/patch_score.h>
#include <svo/matcher.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/feature_alignment.h>

namespace svo {

	namespace warp {

		//计算仿射变换矩阵
		//原理：以参考关键帧上的某一像点为中心（记为r1），再取两个点（记为r2,r3），构成一个直角三角形（程序在像点正右边和正下边取了两个点），
		//然后将其投影到当前帧中，得到三个点坐标，与参考帧中的一一对应，分别为c1,c2,c3，，由于r2,r3相对r1只有一维上的运动，运动的距离都为
		//固定的像素halfpatch_size（窗口大小的一半），可用c2,c3分别减去c1除以halfpatch_size即可得参考帧像素块变换到当前帧中缩放的尺度，如果
		//只有拉伸，那么c2,c3减去c1所得的结果必有一个为0，但是一般都会带有一些旋转的，故而不单有拉伸，还有旋转的补偿，最终得到的A_cur_ref矩阵
		//主对角线元素接近1，其他两个元素不等于0，但接近0，作为旋转带来的补偿
		//要强调的是：计算出来的A_cur_ref是从参考帧某一层图像（特征点所在层）到当前帧底层图像的仿射变换矩阵，而不是
		//参考帧底层图像与当前帧底层图像之间的仿射变换矩阵；对A_cur_ref取逆，就可以得到从当前帧底层图像变换至参考帧
		//特征点点所在层图像的仿射矩阵
		//取块时，不是恒在底层图像上取8*8(或10*10)方形区域，而是在对应层图像中取8*8(或10*10)的方形区域；
		//最后强调一下，此处的仿射矩阵与特征点对应的，不是所有特征点都具有相同的仿射矩阵，而是每个特征点都有自己的
		//一个局部仿射变换矩阵；
		void getWarpMatrixAffine(
			const vk::AbstractCamera& cam_ref,	//参考关键帧相机（含参考关键帧相机参数）
			const vk::AbstractCamera& cam_cur,	//当前帧相机（含当前帧相机参数）
			const Vector2d& px_ref,				//像点在参考关键帧（创建该像点）底层金字塔图像上的坐标
			const Vector3d& f_ref,				//参考帧观测该点的视线方向向量
			const double depth_ref,				//在参考帧中的平均逆深
			const SE3& T_cur_ref,				//参考帧到当前帧的SE3矩阵
			const int level_ref,				//特征点所在层级（参考关键帧中的层级）
			Matrix2d& A_cur_ref)				//最终的计算结果：仿射矩阵
		{
			// Compute affine warp matrix A_ref_cur
			const int halfpatch_size = 5;

			//实现方向乘以深度，得到该像点3D坐标，要注意的是，depth_ref是视线的长度，不是z坐标的值，
			//本文中，深度是指视线距离，不是垂直像平面的距离
			const Vector3d xyz_ref(f_ref*depth_ref);

			//在参考关键帧中另取两个点，每个点仅在一个方向上有位移；然后将像素坐标投影至世界坐标中；
			//注意乘以了level_ref，说明首先是在特征点所在层取一个halfpatch_size区，而后变换到底层，其长度有所增长
			Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size, 0)*(1 << level_ref)));	//仅在x方向向右偏移了5个像素
			Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0, halfpatch_size)*(1 << level_ref)));	//仅在y方向向下偏移了5个像素

			//下面语句应该这样解读，首先是xyz_du_ref / xyz_du_ref[2]，将参考关键帧像点对应的3D坐标归一化，得到归一化图像坐标；
			//然后乘以xyz_ref[2]得到两个点的3D坐标
			//解释一下：像点在上一帧的深度是已知的，但是另外取的两个点，其深度是未知的，为了能够将另外取的两个点变换至当前帧相机坐标系中，
			//首先需要恢复其在世界坐标系中的坐标，因此需要计算其深度，那怎么计算了？因为这两个点距离像点并不远，因此，认为其与像点的z坐标
			//相同（注意不是使其视线长度相同）
			xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
			xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];

			//将获取到的三个点变换至当前帧相机坐标系中
			const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));
			const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
			const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));

			//当前帧中的点作差，除以在关键帧中的位移量，即可得拉伸因子
			//注意，除的不是halfpatch_size*(1 << level_ref)，而是halfpatch_size，即最终计算的不是当前帧底层图像与
			//参考帧底层图像的仿射变换矩阵，而是当前帧底层图像到参考帧特征点所在层图像的仿射变换矩阵
			A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
			A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
		}

		//根据仿射矩阵行列式的值，来决定尺度，也就是该特征点，最有可能出现在当前帧的哪一层图像中
		int getBestSearchLevel(
			const Matrix2d& A_cur_ref,		//该特征点对应的参考帧特征点所在层图像到当前帧底层图像的仿射矩阵
			const int max_level)			//最大金字塔层数
		{
			// Compute patch level in other image
			int search_level = 0;
			double D = A_cur_ref.determinant();
			while (D > 3.0 && search_level < max_level)
			{
				search_level += 1;
				D *= 0.25;
			}
			return search_level;
		}

		//在当前帧中取一个方形区域，而后对其进行仿射变换，得到参考帧的区域，并通过插值获取这些区域的像素值
		void warpAffine(
			const Matrix2d& A_cur_ref,		//该特征点处，参考关键帧到当前帧的仿射变换矩阵
			const cv::Mat& img_ref,			//参考关键帧图像（特征点所在层图像）
			const Vector2d& px_ref,			//特征点在参考关键帧底层图像的x坐标
			const int level_ref,			//特征点在参考关键帧底层图像的y坐标
			const int search_level,			//特征点在当前帧中所在层数
			const int halfpatch_size,		//区域块大小的一半+1（也即5）
			uint8_t* patch)					//一维数组，大小为10*10，对应参考帧上的一个10*10区域块
		{
			//10*10区域
			const int patch_size = halfpatch_size * 2;

			//该特征点处，从当前帧到参考帧的仿射变换矩阵
			const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();

			//A_ref_cur除了主对角线上的元素，其他元素其实大多情况下接近0，如果取逆得到的矩阵的第一号元素无穷大，
			//说明相机没有发生平移运动，如果要搞清为什么，那就必须取看一下getWarpMatrixAffine()函数是怎么计算A_cur_ref的了。
			//不过，这种情况应该是很少发生的，但是，不明白的是，为什么这里直接返回了？如果直接返回了，那么patch_with_border_
			//都没赋值，后面从patch_with_border_提取的就是上一个种子周边的像素值了，不会出错吗（不过一般情形下，不会进入到if中）？？
			//另外，上面注释中只提到平移，并没提到旋转，旋转情况呢？所以这种处理是不是存在一定的侥幸啊。。。
			if (isnan(A_ref_cur(0, 0)))
			{
				printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
				return;
			}

			// Perform the warp on a larger patch.
			//获取patch_的首地址
			uint8_t* patch_ptr = patch;

			//特征点在参考帧对应层图像中的坐标
			const Vector2f px_ref_pyr = px_ref.cast<float>() / (1 << level_ref);
			
			//外层为行，内层为列，可知，最终得到的一维patch_with_border_是按行存储的
			for (int y = 0; y < patch_size; ++y)
			{
				for (int x = 0; x < patch_size; ++x, ++patch_ptr)
				{
					//在当前帧方形区域内的相对坐标
					Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);	

					//相对坐标变换到当前帧的底层图像上的坐标
					px_patch *= (1 << search_level);

					//A_ref_cur是直接从当前帧底层图像变换到参考关键帧特征点所在层图像的仿射变换矩阵（相见getWarpMatrixAffine()函数）
					//当前帧中方形区域的相对坐标乘以仿射矩阵得到参考帧中的相对区域中心的相对坐标，加上特征点在参考帧对应层图像的坐标，
					//得到参考帧区域的绝对坐标
					const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);

					//检验是否在图像区域内，如果不再直接取像素值为0；如果在，则利用双线性插值计算得到像素值
					if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
						*patch_ptr = 0;
					else
						*patch_ptr = (uint8_t)vk::interpolateMat_8u(img_ref, px[0], px[1]);
				}
			}
		}

	} // namespace warp

	//三角化得到深度，这里求解深度的方法很有意思，再强调一次，本程序中的深度是视线的长度，不是与成像平面垂直的z坐标，
	//另外要时刻记住一点的是，我们想要的是像点在参考帧中的深度，在当前帧中的深度我们并不关心，只是用其来更新参考帧中的深度。
	//三角化深度求法：假设在参考帧中像点的深度为dr，在当前帧中像点的深度为dc，在参考帧中像点的视线的方向向量为f_ref，在
	//当前帧中像点的视线方向向量为f_cur，参考帧帧到当前帧的旋转矩阵为R，平移为t，那么理想情况下有R*f_ref*dr-f_cur*dc=t，
	//程序中将减号写成了加号，这不影响结果，只是求得的dc是负值而已，使用伪逆即可求解出dr,dc，其中dr是我们想要的，
	//你会发现其实dr,dc的绝对值比较接近，那肯定吗，毕竟相机的运动是比较连续的
	bool depthFromTriangulation(
		const SE3& T_search_ref,		//参考帧到当前帧的SE3矩阵
		const Vector3d& f_ref,			//像点在参考帧中视线的方向向量
		const Vector3d& f_cur,			//像点在当前帧中匹配点的世界坐标
		double& depth)					//最终恢复出的深度
	{
		//T_search_ref是Sophus中的类，用rotation_matrix()转化为Eigen类，再乘以
		Matrix<double, 3, 2> A; A << T_search_ref.rotation_matrix() * f_ref, f_cur;
		const Matrix2d AtA = A.transpose()*A;
		if (AtA.determinant() < 0.000001)
			return false;

		//利用伪逆求解深度
		const Vector2d depth2 = -AtA.inverse()*A.transpose()*T_search_ref.translation();

		//第一个深度值才是参考关键帧中像点的深度
		depth = fabs(depth2[0]);
		return true;
	}

	//从带边界的图像块中提取出不带边界的图像块（所谓带边界的图像块就是10*10图像块，而我们用的是8*8的图像块）
	//在此之前已经通过warpAffine()计算得到ref_patch_border_ptr的值
	void Matcher::createPatchFromPatchWithBorder()
	{
		//patch_size_是静态常值变量，在match.h中赋值为8
		//获取patch_首地址，数组名就是指针
		uint8_t* ref_patch_ptr = patch_;

		//patch_with_border总共10行10列
		//直接从1开始，终止于8，意在提取第2行到第9行之间的块
		for (int y = 1; y < patch_size_ + 1; ++y, ref_patch_ptr += patch_size_)
		{
			//最后加一个1，以及下面循环终止于7，也是意在提取第2列到第9列中的块
			uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_ + 2) + 1;
			
			//x从0开始是因为上面已经加了一个1，ref_patch_border_ptr已经不是每行的首地址，而是每行的第二个元素的地址
			//终止于7，也就是从每行第二个地址往后再偏移7个地址，即提取到第9列
			for (int x = 0; x < patch_size_; ++x)
				ref_patch_ptr[x] = ref_patch_border_ptr[x];
		}
	}

	bool Matcher::findMatchDirect(
		const Point& pt,
		const Frame& cur_frame,
		Vector2d& px_cur)
	{
		if (!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_))
			return false;

		if (!ref_ftr_->frame->cam_->isInFrame(
			ref_ftr_->px.cast<int>() / (1 << ref_ftr_->level), halfpatch_size_ + 2, ref_ftr_->level))
			return false;

		// warp affine
		warp::getWarpMatrixAffine(
			*ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
			(ref_ftr_->frame->pos() - pt.pos_).norm(),
			cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level, A_cur_ref_);
		search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels() - 1);
		warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
			ref_ftr_->level, search_level_, halfpatch_size_ + 1, patch_with_border_);
		createPatchFromPatchWithBorder();

		// px_cur should be set
		Vector2d px_scaled(px_cur / (1 << search_level_));

		bool success = false;
		if (ref_ftr_->type == Feature::EDGELET)
		{
			Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
			dir_cur.normalize();
			success = feature_alignment::align1D(
				cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
				patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
		}
		else
		{
			success = feature_alignment::align2D(
				cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
				options_.align_max_iter, px_scaled);
		}
		px_cur = px_scaled * (1 << search_level_);
		return success;
	}


	//沿着极线搜索参考帧中特征点在当前帧中的匹配点；
	//根据最大最小深度，得到在当前帧中的极线段，沿着该极线段搜索匹配点；
	//最终如果找到匹配点，返回true，并得到depth
	bool Matcher::findEpipolarMatchDirect(
		const Frame& ref_frame,				//参考关键帧
		const Frame& cur_frame,				//当前帧
		const Feature& ref_ftr,				//参考帧中的特征点
		const double d_estimate,			//上一帧估计得到的深度均值（逆深的倒数）
		const double d_min,					//最小深度（最大逆深的倒数）
		const double d_max,					//最大深度（最小逆深的倒数）
		double& depth)						//计算得到的深度（如果匹配成功的话，会计算并返回深度值）
	{
		//从参考帧到当前帧的SE3矩阵
		SE3 T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
		int zmssd_best = PatchScore::threshold();		//调试时，打印出PatchScore::threshold()的值为128000
		Vector2d uv_best;		//最佳匹配点（归一化平面上的坐标，即第三维坐标恒为1，归一化图像坐标）

		// Compute start and end of epipolar line in old_kf for match search, on unit plane!（在单位平面上）
		//视线方向乘以最大最小深度得到像点在参考帧中相机坐标系中的最远最近3D坐标，然后再乘以参考帧到当前帧的SE3矩阵，
		//获得在当前帧相机坐标系的最近最远3D坐标，最后投影到当前帧图像上，得到在当前帧基线的起点端点像素坐标（project2d仅仅
		//是除以第三维坐标，相当于是得到归一化图像坐标）
		Vector2d A = vk::project2d(T_cur_ref * (ref_ftr.f*d_min));		//最近点
		Vector2d B = vk::project2d(T_cur_ref * (ref_ftr.f*d_max));		//最远点
		epi_dir_ = A - B;

		// Compute affine warp matrix：A_cur_ref
		//根据当前帧和参考帧以及当前考虑的特征点计算仿射变换矩阵（主要是缩放变换，也有旋转，
		//因为是当前帧与关键帧之间的关系，还是需要考虑仿射变换的）
		//获取的A_cur_ref是参考帧某一层图像（特征点所在层）到当前帧底层图像的仿射变换矩阵
		warp::getWarpMatrixAffine(
			*ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
			d_estimate, T_cur_ref, ref_ftr.level, A_cur_ref_);

		// feature pre-selection
		//直线特征？？有用到直线特征吗？
		reject_ = false;
		if (ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering)
		{
			const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
			const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
			if (cosangle < options_.epi_search_edgelet_max_angle) {
				reject_ = true;
				return false;
			}
		}

		//获取最佳搜索尺度，也就是特征点在当前帧应该最好处于哪一层图像中（在ORB-SLAM中也有类似的，估计特征点在当前帧所在层），
		//根据参考帧与当前帧之间的仿射变换的行列式来估计特征点在当前帧中的所在层
		search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels() - 1);

		// Find length of search range on epipolar line
		//A是当前帧相机坐标系下的归一化图像坐标，这里使用world2cam函数进一步将其变换到像素坐标（乘以内参矩阵）
		Vector2d px_A(cur_frame.cam_->world2cam(A));
		Vector2d px_B(cur_frame.cam_->world2cam(B));

		//根据在参考帧中的最大最小深度求得的在当前帧中的极线长度，像素单位
		epi_length_ = (px_A - px_B).norm() / (1 << search_level_);

		// Warp reference patch at ref_level
		//注意实际用的是8*8的区域块，但该函数计算得到的patch_with_border_是一个10*10的区域块（虽说是二维的，但是按行存储为
		//一维数组patch_with_border_），至于为什么要取10*10而后进行提取，个人感觉没有什么必要，可以直接计算一个8*8的区域；
		//另外，虽然上面getWarpMatrixAffine()计算的是从参考帧到当前帧的仿射矩阵，但实际上，还是以当前帧作为参考的，也就是用的时候，
		//是用A_cur_ref_的逆阵，当前帧中的区域是方形的，而参考关键帧的区域是不确定的，最终patch_with_border_存储了10*10个参考帧中
		//的像素值；
		//warpAffine()在当前帧中取了一个10*10的区域，如果按照从1到10的顺序排的话，那么特征点在第6位，也就是由于计算得到的是一个偶数
		//列区域，特征点左边（上边）有5个像素，右边（下边）有4个像素，最终提取8*8个，也就是最终提取的区域特征点左边（上边）有4个像素，
		//右边（下边）有3个像素；
		//另外，记住这里最终提取的区域在特征点左边（上边）有4个像素，在右边有3个像素，所以下面在当前帧中提取时，也要与之对应，
		//实际上，由于使用了UZH rpg实验组自己的库rpg_vikit，实际并没有提取完整的区域，只是将区域的首地址传入就可以了，这个首地址的计算
		//对应上就可以了。
		warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
			ref_ftr.level, search_level_, halfpatch_size_ + 1, patch_with_border_);

		//patch_（参考关键帧的图像块区域）在该函数中被赋值，patch_with_border_在上面的warpAffine()函数中被赋值，其尺寸为10*10，
		//但实际我们用的是8*8的尺寸，所以接下来从该10*10区域块中，抠出8*8的区域，即从partch_with_border_中获取patch_
		//该函数仅仅时完成提取操作，不做任何修改
		createPatchFromPatchWithBorder();

		//如果搜索极线过短，都小于2个像素了，那么就直接考虑极线的中点（不用一步一步搜索了）
		if (epi_length_ < 2.0)
		{
			px_cur_ = (px_A + px_B) / 2.0;

			//当前帧的坐标变换至对应层金字塔图像上的坐标
			Vector2d px_scaled(px_cur_ / (1 << search_level_));

			bool res;

			//精确到亚像素（利用KLT光流法跟踪，精确到亚像素精度）
			//注意px_scaled已经是变换到对应层金字塔图像上的坐标，调用align2D或align1D图像时，也是输入对应层金字塔图像
			if (options_.align_1d)
				res = feature_alignment::align1D(
					cur_frame.img_pyr_[search_level_], (px_A - px_B).cast<float>().normalized(),
					patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
			else
				res = feature_alignment::align2D(
					cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
					options_.align_max_iter, px_scaled);
			if (res)
			{
				//变换回当前帧底层图像上的坐标
				px_cur_ = px_scaled*(1 << search_level_);
				//三角化计算深度
				if (depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
					return true;
			}
			return false;
		}

		//epi_length/0.7得到最大迭代次数（比如极线总长70个像素，那么最大迭代次数为100次，每次步长为0.7个像素）
		size_t n_steps = epi_length_ / 0.7; // one step per pixel

		//极线在归一化平面上的长度除以迭代次数，得到每个步长，极线的偏移量（注意是归一化平面上的单位，不是像素单位）
		Vector2d step = epi_dir_ / n_steps;

		//max_epi_search_steps每个点在极线能上设定的最大迭代次数，在matcher.h中指定值为1000，也就说，如果搜索极线过长，导致迭代次数过多，
		//使得最大迭代次数大于指定的最大迭代次数，那么直接放弃该点，认为搜索极线过长（相当于最大最小逆深相差太大，不确定度太大）（这种策略也是
		//为了保证速度啊）
		if (n_steps > options_.max_epi_search_steps)
		{
			printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
				n_steps, epi_length_, d_min, d_max);
			return false;
		}

		// for matching, precompute sum and sum2 of warped reference patch
		int pixel_sum = 0;
		int pixel_sum_square = 0;

		//PatchScore是第三方库rgp_vikit中的模板类，可用于计算两个块之间的NCC值，输入参数patch_是参考关键帧的图像块
		PatchScore patch_score(patch_);

		// now we sample along the epipolar line
		//开始沿着极线搜索最佳匹配点
		//搜索起点为B点（在归一化平面上进行迭代搜索）
		Vector2d uv = B - step;

		//上一次搜索的坐标（初始化（0，0））
		Vector2i last_checked_pxi(0, 0);
		++n_steps;
		for (size_t i = 0; i < n_steps; ++i, uv += step)
		{
			//搜索点转换至像素坐标（uv是归一化平面上的坐标）
			Vector2d px(cur_frame.cam_->world2cam(uv));

			//将搜索点坐标变换至相应层图像上的坐标
			//注意这是Vector2i类型，也就是整型的，如果不是整型数据，那么始终回向下取整，正是因为这个原因，才需要加上0.5，
			//这样起到四舍五入的效果，使得取整为最接近的值
			Vector2i pxi(px[0] / (1 << search_level_) + 0.5,
				px[1] / (1 << search_level_) + 0.5); // +0.5 to round to closest int

			if (pxi == last_checked_pxi)
				continue;
			//可能会出现当前搜索点与当一次搜索点一样的情况吗？也许有吧，可能有些坐标经过四舍五入之后取整之后，还真就相等了
			last_checked_pxi = pxi;

			// check if the patch is full within the new frame
			//验证在当前帧中取8*8块，是否全在图像区域内
			if (!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_))
				continue;

			// TODO interpolation would probably be a good idea
			//获取当前帧8*8块图像的首地址：.data是Mat类型的数据头，也就是矩阵的首地址，
			//加上(pxi[1] - halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
			//为行数*列数，再加上(pxi[0] - halfpatch_size_)得到最终的块首地址，注意是减去halfpatch_size_=4，
			//正如上述，在特征点左边（上边）是有4个元素的，首地址的计算要与上面对齐
			uint8_t* cur_patch_ptr = cur_frame.img_pyr_[search_level_].data
				+ (pxi[1] - halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
				+ (pxi[0] - halfpatch_size_);
			//计算块之间的得分：计算块之间的SSD值
			int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);

			//统计最小的SSD值
			if (zmssd < zmssd_best) {
				zmssd_best = zmssd;
				uv_best = uv;
			}
		}

		//如果最佳匹配SSD值小于指定值，则接受该匹配（打印输出PatchScore::threshold()的值为128000）
		if (zmssd_best < PatchScore::threshold())
		{
			//subpix_refinement在matcher.h中已经设置为true
			//精确到亚像素精度
			if (options_.subpix_refinement)
			{
				px_cur_ = cur_frame.cam_->world2cam(uv_best);

				//将当前帧中底层图像的坐标变换到指定层（search_level_上）图像上
				Vector2d px_scaled(px_cur_ / (1 << search_level_));
				bool res;
				if (options_.align_1d)
					res = feature_alignment::align1D(
						cur_frame.img_pyr_[search_level_], (px_A - px_B).cast<float>().normalized(),
						patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
				else
					res = feature_alignment::align2D(
						cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
						options_.align_max_iter, px_scaled);
				if (res)
				{
					//变换回底层图像的坐标
					px_cur_ = px_scaled*(1 << search_level_);
					
					//cam2world函数会对最后的坐标归一化，也就是得到的是世界坐标的归一化坐标（方向向量）
					if (depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
						return true;
				}
				return false;
			}

			px_cur_ = cur_frame.cam_->world2cam(uv_best);

			//unprojected()函数回返回归一化像素坐标，因此还需要进行归一化得到当前帧中的像点视线的方向向量
			if (depthFromTriangulation(T_cur_ref, ref_ftr.f, vk::unproject2d(uv_best).normalized(), depth))
				return true;
		}
		return false;
	}

} // namespace svo
