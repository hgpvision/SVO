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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo {

	//结构体静态变量
	int Seed::batch_counter = 0;
	int Seed::seed_counter = 0;

	Seed::Seed(Feature* ftr, float depth_mean, float depth_min) :
		batch_id(batch_counter),		//创建该Seed的关键帧Id
		id(seed_counter++),				//仅用来可视化
		ftr(ftr),						//特征点（需要计算深度的特征点）
		a(10),							//Beta分布中的a参数
		b(10),							//Beta分布中的b参数
		mu(1.0 / depth_mean),			//正态分布的初始均值，设置为平均深度的倒数（逆深）
		z_range(1.0 / depth_min),		//最大逆深
		sigma2(z_range*z_range / 36)	//Patch covariance in reference image.
	{}

	//深度滤波器的初始化
	//输入：	feature				feature_detector
	//			seed_converged_cb	一个函数指针，在此之前，FrameHandlerMono::initialize()函数已经将seed_converged_cb绑定到
	//								地图中候选关键点的成员函数newCandidatePoint()
	DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
		feature_detector_(feature_detector),
		seed_converged_cb_(seed_converged_cb),
		seeds_updating_halt_(false),
		thread_(NULL),
		new_keyframe_set_(false),
		new_keyframe_min_depth_(0.0),
		new_keyframe_mean_depth_(0.0)
	{}

	DepthFilter::~DepthFilter()
	{
		stopThread();
		SVO_INFO_STREAM("DepthFilter destructed.");
	}

	void DepthFilter::startThread()
	{
		//启动深度滤波（种子更新线程）
		thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
	}

	void DepthFilter::stopThread()
	{
		SVO_INFO_STREAM("DepthFilter stop thread invoked.");
		if (thread_ != NULL)
		{
			SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
			seeds_updating_halt_ = true;
			thread_->interrupt();
			thread_->join();
			thread_ = NULL;
		}
	}

	//普通帧深度滤波函数，实际上在一般情况下，深度滤波线程是一直开着的，也即thread_一般都是不等于NULL的，
	//因此，一般都会执行if中的语句，不会调用updateSeeds()，实际上，在后台还运行着一个深度滤波线程（可能处于阻塞状态，但线程不为NULL），
	//一直在等待帧队列中有帧插入，一旦有帧插入，notify_one()就会唤醒深度滤波线程，在深度滤波线程中绑定执行了updateSeedsLoop()函数，
	//该函数同样会调用updateSeeds()函数进行种子深度滤波（可以看出，正常情况下还是交由专门的深度滤波线程来进行深度滤波，而不是在主线程
	//中完成）。
	void DepthFilter::addFrame(FramePtr frame)
	{
		if (thread_ != NULL)
		{
			{
				lock_t lock(frame_queue_mut_);
				//看来SVO为了速度，真的是放弃了太多，帧队列中不超过2帧，
				//超过两帧就会丢弃掉一帧不处理
				if (frame_queue_.size() > 2)
					frame_queue_.pop();
				frame_queue_.push(frame);
			}
			seeds_updating_halt_ = false;
			frame_queue_cond_.notify_one();
		}
		else
			updateSeeds(frame);
	}

	// Add new keyframe to the queue
	//对于关键帧，不需要添加进帧队列，只是将其标记为关键帧就可以了，然后后台深度滤波线程同样会被唤醒，去处理最新的关键帧；
	//和addFrame()一样，调用addKeyframe()函数的是主线程，因此一般也不会执行else中的语句，而是用专门的深度滤波线程来进行种子滤波
	void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
	{
		new_keyframe_min_depth_ = depth_min;
		new_keyframe_mean_depth_ = depth_mean;
		if (thread_ != NULL)
		{
			new_keyframe_ = frame;
			new_keyframe_set_ = true;
			seeds_updating_halt_ = true;
			frame_queue_cond_.notify_one();
		}
		else
			initializeSeeds(frame);
	}

	//关键帧扩增、初始化新种子
	void DepthFilter::initializeSeeds(FramePtr frame)
	{
		Features new_features;
		feature_detector_->setExistingFeatures(frame->fts_);
		feature_detector_->detect(frame.get(), frame->img_pyr_,
			Config::triangMinCornerScore(), new_features);

		// initialize a seed for every new feature
		seeds_updating_halt_ = true;
		lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
		++Seed::batch_counter;
		std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr) {
			seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
		});

		if (options_.verbose)
			SVO_INFO_STREAM("DepthFilter: Initialized " << new_features.size() << " new seeds");
		seeds_updating_halt_ = false;
	}

	void DepthFilter::removeKeyframe(FramePtr frame)
	{
		seeds_updating_halt_ = true;
		lock_t lock(seeds_mut_);
		list<Seed>::iterator it = seeds_.begin();
		size_t n_removed = 0;
		while (it != seeds_.end())
		{
			if (it->ftr->frame == frame.get())
			{
				it = seeds_.erase(it);
				++n_removed;
			}
			else
				++it;
		}
		seeds_updating_halt_ = false;
	}

	void DepthFilter::reset()
	{
		seeds_updating_halt_ = true;
		{
			lock_t lock(seeds_mut_);
			seeds_.clear();
		}
		lock_t lock();
		while (!frame_queue_.empty())
			frame_queue_.pop();
		seeds_updating_halt_ = false;

		if (options_.verbose)
			SVO_INFO_STREAM("DepthFilter: RESET.");
	}

	//A thread that is continuously updating the seeds.
	//深度滤波线程种子深度滤波
	void DepthFilter::updateSeedsLoop()
	{
		while (!boost::this_thread::interruption_requested())
		{
			FramePtr frame;
			{
				lock_t lock(frame_queue_mut_);
				while (frame_queue_.empty() && new_keyframe_set_ == false)
					frame_queue_cond_.wait(lock);
				//如果有新关键帧插入，则清空帧队列
				if (new_keyframe_set_)
				{
					//回置new_keyframe_set_为false
					new_keyframe_set_ = false;
					//当然不能停止更新种子了，设置该标记为false
					seeds_updating_halt_ = false;
					clearFrameQueue();
					frame = new_keyframe_;
				}
				else
				{
					frame = frame_queue_.front();
					frame_queue_.pop();
				}
			}
			//种子更新
			updateSeeds(frame);
			if (frame->isKeyframe())
				initializeSeeds(frame);
		}
	}

	//对所有种子进行更新（包括逆深以及不确定度）
	void DepthFilter::updateSeeds(FramePtr frame)
	{
		// update only a limited number of seeds, because we don't have time to do it
		// for all the seeds in every frame!
		//为保证速度，只能更新一定数量的种子
		size_t n_updates = 0, n_failed_matches = 0, n_seeds = seeds_.size();
		lock_t lock(seeds_mut_);
		list<Seed>::iterator it = seeds_.begin();		//获取种子列表中的第一颗种子

		//*************************计算逆深方差（不确定度）*************************//
		//参考：REODE: Probabilistic, Monocular Dense Reconstruction in Real Time
		//*************************计算逆深方差（不确定度）*************************//
		const double focal_length = frame->cam_->errorMultiplier2();		//获取相机的焦距（这里视所选择相机的类型而定，如果是针孔相机返回fx
		double px_noise = 1.0;												//像点位置噪声固定为1个像素
		double px_error_angle = atan(px_noise / (2.0*focal_length))*2.0;	//law of chord (sehnensatz)，式（16），计算beta+

		while (it != seeds_.end())
		{
			// set this value true when seeds updating should be interrupted
			//如果seeds_updateing_halt_被设置为true，那么种子滤波将暂停（当新关键帧插入的时候，将会暂停种子滤波，而后重新开启）
			if (seeds_updating_halt_)
				return;

			// check if seed is not already too old
			//有些种子，无论进行多少次更新（过了多少帧），也没能收敛，那就将其从种子集中删除，不在对其进行更新（再进行下去也无益）
			if ((Seed::batch_counter - it->batch_id) > options_.max_n_kfs) {
				it = seeds_.erase(it);
				continue;
			}

			// check if point is visible in the current image
			//ftr是该种子对应的特征，frame是创建该特征点的关键帧，T_f_w_是从世界到相机帧坐标系的SE3矩阵，frame是当前帧
			//（可能是普通帧也可能是关键帧），最终计算得到从当前帧到参考帧的SE3矩阵
			SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();

			//mu是种子的平均逆深（注意是平均逆深，正态分布的均值，也是种子上一次滤波得到的后验参数），f是特征点视线的方向向量，
			//也就是该特征点视线的方向向量，可由归一化图像坐标归一化得到，1.0 / it->mu * it->ftr->f得到特征点的3D坐标
			//（使用平均深度恢复，视线方向向量乘以视线总长），乘以参考帧到当前帧的SE3矩阵，得到该点在当前帧中的3D坐标，
			//注意种子里面存储的特征点的坐标是在创建该种子的关键帧中的坐标
			const Vector3d xyz_f(T_ref_cur.inverse()*(1.0 / it->mu * it->ftr->f));

			//验证该种子是否还在当前帧的前面，如果不在，显然是看不到的，跳过
			if (xyz_f.z() < 0.0) {
				++it; // behind the camera
				continue;
			}
			//检验该种子投影到当前帧中是否落入当前帧的图像区域内：f2c首先将3D坐标归一化，然后乘以内参矩阵得到像素坐标
			if (!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
				++it; // point does not project in image
				continue;
			}

			// we are using inverse depth coordinates
			//sigma2是Patch covariance in reference image.
			//it->mu是种子上一次更新后的后验参数：平均逆深，现在要求当前更新的新参数；
			//it->sigma2是种子上一次更新后的后验参数：平均逆深的方差（不确定度），现在要求当前更新的新参数；
			//根据上一次估计的后验平均逆深以及逆深的不确定度（对当前帧来说，相当于是先验信息），求得在当前帧中的最大最小逆深，
			//当前帧最终得到的估计值，应该是在这个范围内的（不然就说明上一帧的估计值是比较离谱的）
			float z_inv_min = it->mu + sqrt(it->sigma2);					//种子在当前帧中的最大逆深，对应最小深度（此处命名略有点乱）
			float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);	//种子在当前帧中的最小逆深，对应最大深度

			//z是视线的长度（注意不是逆深），也就是本程序中所定义的深度，乘以视线的方向向量，即得点的3D坐标；
			//注意z是新的深度测量值，上面的it->mu是上一次深度滤波得到的深度后验平均深度；
			//新的测量值与上一次的后验平均深度（在当前帧中相当于是先验信息了），将用来估计当前帧的新的后验参数。
			double z;
			//新的深度测量值z的计算通过findEpipolarMatchDirect()来完成
			if (!matcher_.findEpipolarMatchDirect(
				*it->ftr->frame, *frame, *it->ftr, 1.0 / it->mu, 1.0 / z_inv_min, 1.0 / z_inv_max, z))
			{
				it->b++; // increase outlier probability when no match was found，外点标记直接加1
				++it;
				++n_failed_matches;
				continue;
			}

			// compute tau
			//计算深度方差的平方根（标准差），注意是深度的，不是逆深
			//参考：REODE: Probabilistic, Monocular Dense Reconstruction in Real Time
			double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);				//计算深度的标准差

			//首先由深度的标准差计算得到最大逆深和最小逆深，二者相减得到最大最小逆深之差，取中值得逆深的标准差（深度标准差的倒数）
			//tau_inverse这个标准差，是测量值的标准差（逆深）
			double tau_inverse = 0.5 * (1.0 / max(0.0000001, z - tau) - 1.0 / (z + tau));

			// update the estimate
			//单个种子更新
			//参考：Video-based, Real-Time Multi View Stereo及其补充材料
			updateSeed(1. / z, tau_inverse*tau_inverse, &*it);

			++n_updates;										//这个参数并没有用到

			//如果当前处理帧是关键帧，那么该点对应所在当前关键帧的网格内不需要再生成新点，标记已经生成了点（如果网格内没有点，
			//则会对其进行提点，以扩增种子）
			if (frame->isKeyframe())
			{
				// The feature detector should not initialize new seeds close to this location
				feature_detector_->setGridOccpuancy(matcher_.px_cur_);
			}

			// if the seed has converged, we initialize a new candidate point and remove the seed
			//z_range是最大深度（注意不是最大逆深，而是最大深度），it->sigma2是种子逆深的方差，seed_convergence_sigma2_thresh赋值为200；
			//这里有点乱，逆深和深度交叉，很容易混淆哪个是逆深，哪个是深度；
			//最关键的是，这里的收敛判定准则有点不科学啊：逆深的标准差小于最大深度除以一个常值，什么情况？至少两边都是逆深吧，怎么一个
			//是逆深，一个是深度，虽然除以了指定的值，还是觉得非常的不合理。
			//更合理的方式也许应该用启发式判断规则，可以试着改改
			if (sqrt(it->sigma2) < it->z_range / options_.seed_convergence_sigma2_thresh)
			{
				assert(it->ftr->point == NULL); // TODO this should not happen anymore

				//像素点对应的新的世界坐标：it->mu已经是新的逆深均值了
				Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0 / it->mu)));

				//收敛的种子成为地图点
				Point* point = new Point(xyz_world, it->ftr);
				it->ftr->point = point;
				/* FIXME it is not threadsafe to add a feature to the frame here.
				if(frame->isKeyframe())
				{
				  Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
				  ftr->point = point;
				  point->addFrameRef(ftr);
				  frame->addFeature(ftr);
				  it->ftr->frame->addFeature(it->ftr);
				}
				else
				*/

				//在FrameHandlerMono::initialize()函数已经将seed_converged_cb绑定到地图中候选关键点的成员函数newCandidatePoint()，
				//该函数就是将地图点point插入到候选地图点candidates_列表中
				{
					//第一个参数为地图点，第二个参数为该地图点的方差（逆深不确定度）
					//这里需要点Boost的语法知识
					seed_converged_cb_(point, it->sigma2); // put in candidate list
				}
				//对于收敛的点，将其从种子列表中删除，不再迭代
				it = seeds_.erase(it);
			}
			//如果逆深太大，认为其发散，同样删除点
			else if (isnan(z_inv_min))
			{
				SVO_WARN_STREAM("z_min is NaN");
				it = seeds_.erase(it);
			}
			//除却上面两种情况，说明该点还需要继续进行迭代，还未收敛
			else
				++it;
		}
	}

	void DepthFilter::clearFrameQueue()
	{
		while (!frame_queue_.empty())
			frame_queue_.pop();
	}

	void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
	{
		lock_t lock(seeds_mut_);
		for (std::list<Seed>::iterator it = seeds_.begin(); it != seeds_.end(); ++it)
		{
			if (it->ftr->frame == frame.get())
				seeds.push_back(*it);
		}
	}

	//更新某个种子：更新种子后验分布的四个参数a,b,mu,sigma2
	//参考：Video-based, Real-Time Multi View Stereo及其补充材料
	//输入：	x	视线总长度的倒数，即逆深
	//			tau2	逆深方差
	//			seed	种子指针
	void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
	{
		float norm_scale = sqrt(seed->sigma2 + tau2);
		if (std::isnan(norm_scale))
			return;

		//利用boost库，构建一个均值为seed->mu，标准差为norm_scale的正态分布
		boost::math::normal_distribution<float> nd(seed->mu, norm_scale);

		float s2 = 1. / (1. / seed->sigma2 + 1. / tau2);		//s平方，补充材料（19）式
		float m = s2*(seed->mu / seed->sigma2 + x / tau2);		//m值，补充材料（20）式

		float C1 = seed->a / (seed->a + seed->b) * boost::math::pdf(nd, x);	//补充材料（21）式
		float C2 = seed->b / (seed->a + seed->b) * 1. / seed->z_range;		//补充材料（22）式

		float normalization_constant = C1 + C2;								//这是补充材料所没有的，对C1和C2进行归一化处理
		C1 /= normalization_constant;
		C2 /= normalization_constant;

		float f = C1*(seed->a + 1.) / (seed->a + seed->b + 1.) + C2*seed->a / (seed->a + seed->b + 1.);		//补充材料式（25）
		float e = C1*(seed->a + 1.)*(seed->a + 2.) / ((seed->a + seed->b + 1.)*(seed->a + seed->b + 2.))	//补充材料式（26）
			+ C2*seed->a*(seed->a + 1.0f) / ((seed->a + seed->b + 1.0f)*(seed->a + seed->b + 2.0f));

		// update parameters
		float mu_new = C1*m + C2*seed->mu;														//补充材料式（23），新的均值
		seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;	//新的方差，补充材料式（24）
		seed->mu = mu_new;
		seed->a = (e - f) / (f - e / f);		//根据补充材料式（25），（26）计算新的a，b参数
		seed->b = seed->a*(1.0f - f) / f;
	}

	//计算深度方差的平方根（标准差），注意是深度的，不是逆深
	//参考：REODE: Probabilistic, Monocular Dense Reconstruction in Real Time
	//输入：	z	视线的总长度，乘以视线的方向向量，可得3D坐标（最新深度测量值）
	//			px_error_angle	当前帧中的误差视线角，已在函数外求得
	//返回：	tau		方差的平方根（标准差）
	double DepthFilter::computeTau(
		const SE3& T_ref_cur,
		const Vector3d& f,
		const double z,
		const double px_error_angle)
	{
		Vector3d t(T_ref_cur.translation());
		Vector3d a = f*z - t;		//f*z获取真实3D坐标
		double t_norm = t.norm();
		double a_norm = a.norm();
		double alpha = acos(f.dot(t) / t_norm); // dot product
		double beta = acos(a.dot(-t) / (t_norm*a_norm));	// dot product
		double beta_plus = beta + px_error_angle;			//式（16）
		double gamma_plus = PI - alpha - beta_plus; // triangle angles sum to PI
		double z_plus = t_norm*sin(beta_plus) / sin(gamma_plus); // law of sines
		return (z_plus - z); // tau
	}

} // namespace svo
