#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <deque>

#include "sophus/so3.hpp"
#include "imu_factor.h"
#include "pre_integration.h"
#include "tool_color_printf.hpp"
#include "poseParameterization.h"
#include "utility.h"
#include "imu_factor_pnp.h"

//extrinsic param
#define RIC_y ((double)0.0)
#define RIC_p ((double)0.0)
#define RIC_r ((double)180.0)
#define MIN_LOOP_NUM 22
#define LOOP_FREQ 3
#define WINDOW_SIZE 10
#define PNP_SIZE 6
#define SIZE_POSE 7
#define SIZE_SPEEDBIAS 9
#define SIZE_SPEED 3
#define SIZE_BIAS  6

using namespace std;

/*
 *   参考clic写的测试自动求导和解析求导的demo文件
 *   需要注意的是，最好参数都是vector的形式会比较好
 *   确保参差计算方式一模一样才可以
 */
namespace Eigen
{
     template <typename T>
     using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

     template <typename T>
     using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

     template <typename K, typename V>
     using aligned_map = std::map<K, V, std::less<K>,
                                  Eigen::aligned_allocator<std::pair<K const, V>>>;

     template <typename K, typename V>
     using aligned_unordered_map =
         std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                            Eigen::aligned_allocator<std::pair<K const, V>>>;
}
class FactorTest
{
public:
     FactorTest()
     {
          parameterization = new RotationParameterization();
          autodiff_parameterization = new ceres::EigenQuaternionParameterization();

          solver_options_.max_num_iterations = 1;
          solver_options_.num_threads = 1;
          solver_options_.minimizer_progress_to_stdout = false;
          solver_options_.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
          GenerateData();
          // double laser_point_cov = 1;
          // CT_ICP::LidarPlaneNormFactor::t_il = Eigen::Vector3d::Zero();
          // CT_ICP::LidarPlaneNormFactor::q_il = Eigen::Quaterniond::Identity();
          // CT_ICP::LidarPlaneNormFactor::sqrt_info = sqrt(1 / laser_point_cov);
     }
     ~FactorTest() {}

     // void TestPointPlaneFactorAutoDiff(Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobians)
     // {
     //      Eigen::Vector3d norm_vector(0.3, 1.5, -2);
     //      norm_vector.normalize();
     //      Eigen::Vector3d vector_neig(1, 3, 5);
     //      double norm_offset = -norm_vector.dot(vector_neig);
     //      Eigen::Vector3d point_end(10, 12, 14);
     //      double weight = 1;

     //      Eigen::Quaterniond end_quat = Eigen::Quaterniond(0.2, 0.6, 1.3, -0.9);
     //      end_quat.normalize();
     //      std::cout << "normal: " << norm_vector.transpose() << std::endl;
     //      Eigen::Vector3d end_t(11, 13, 15);

     //      auto *cost_function =
     //          CT_ICP::PointToPlaneFunctor::Create(vector_neig, point_end, norm_vector, weight);

     //      ceres::Problem problem(problem_options_);
     //      std::vector<double *> vec;
     //      vec.emplace_back(end_t.data());
     //      vec.emplace_back(end_t.data());
     //      problem.AddParameterBlock(&end_quat.x(), 4, autodiff_parameterization);
     //      problem.AddParameterBlock(&end_t.x(), 3);

     //      problem.AddResidualBlock(cost_function, nullptr, &end_t.x(), &end_quat.x());

     //      GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
     // }

     // void TestPointPlaneFactorAnalytic(Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobians)
     // {
     //      Eigen::Vector3d norm_vector(0.3, 1.5, -2);
     //      norm_vector.normalize();
     //      Eigen::Vector3d vector_neig(1, 3, 5);
     //      double norm_offset = -norm_vector.dot(vector_neig);
     //      Eigen::Vector3d point_end(10, 12, 14);
     //      double weight = 1;

     //      Eigen::Quaterniond end_quat = Eigen::Quaterniond(0.2, 0.6, 1.3, -0.9);
     //      end_quat.normalize();
     //      Eigen::Vector3d end_t(11, 13, 15);

     //      CT_ICP::LidarPlaneNormFactor *cost_function =
     //          new CT_ICP::LidarPlaneNormFactor(point_end, norm_vector, norm_offset, weight);

     //      ceres::Problem problem(problem_options_);
     //      std::vector<double *> vec;
     //      vec.emplace_back(end_t.data());
     //      vec.emplace_back(end_t.data());
     //      problem.AddParameterBlock(&end_quat.x(), 4, parameterization);
     //      problem.AddParameterBlock(&end_t.x(), 3);

     //      problem.AddResidualBlock(cost_function, nullptr, &end_t.x(), &end_quat.x());

     //      GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
     // }

     void TestIMUFactorAnalytic(Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobians)
     {
          Eigen::Vector3d para_last_position(0, 0, 0);
          Eigen::Quaterniond para_last_quat(1, 0, 0, 0);
          Eigen::Vector3d para_last_v(0, 0, 1);
          Eigen::Vector3d para_last_ba(0.01, 0.01, 0.01);
          Eigen::Vector3d para_last_bg(0.01, 0.01, 0.01);

          Eigen::Vector3d para_pre_position(0.5, 0.5, 0.5);
          Eigen::Quaterniond para_pre_quat(0.707, 0.707, 0, 0);
          Eigen::Vector3d para_pre_v(0, 0, 2);
          Eigen::Vector3d para_pre_ba(0.02, 0.02, 0.02);
          Eigen::Vector3d para_pre_bg(0.02, 0.02, 0.02);
          
          Eigen::Vector3d acc0(0, 0, 0.1);
          Eigen::Vector3d gyr0(0, 0.1, 0.1);
          // double dt = 0.01;
          // processIMU(dt, acc0, gyr0);
          IMUFactor* cost_function = new IMUFactor(pre_integration);
          //这个注释了依然不能跑，说明不是这个原因
          ceres::Problem problem(problem_options_);
          std::vector<double *> vec;
          vec.emplace_back(para_last_position.data());
          vec.emplace_back(para_last_v.data());
          vec.emplace_back(para_last_ba.data());
          vec.emplace_back(para_last_bg.data());

          vec.emplace_back(para_pre_position.data());
          vec.emplace_back(para_pre_v.data());
          vec.emplace_back(para_pre_ba.data());
          vec.emplace_back(para_pre_bg.data());

          //加入了
          problem.AddParameterBlock(&para_last_position.x(), 3);
          problem.AddParameterBlock(&para_last_quat.x(), 4, parameterization);
          problem.AddParameterBlock(&para_last_v.x(), 3);
          problem.AddParameterBlock(&para_last_ba.x(), 3);
          problem.AddParameterBlock(&para_last_bg.x(), 3);

          problem.AddParameterBlock(&para_pre_position.x(), 3);
          problem.AddParameterBlock(&para_pre_quat.x(), 4, parameterization);
          problem.AddParameterBlock(&para_pre_v.x(), 3);
          problem.AddParameterBlock(&para_pre_ba.x(), 3);
          problem.AddParameterBlock(&para_pre_bg.x(), 3);

          problem.AddResidualBlock(cost_function, nullptr, &para_last_position.x(), &para_last_quat.x()
                                                       , &para_last_v.x(), &para_last_ba.x(), &para_last_bg.x()
                                                       , &para_pre_position.x(), &para_pre_quat.x()
                                                       , &para_pre_v.x(), &para_pre_ba.x(), &para_pre_bg.x());
          GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);

     }

     void TestIMUFactorAnalytic2(Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobians)
     {
          vec2double();
          IMUFactorPnP* imu_factor = new IMUFactorPnP(pre_integration);

          //这个注释了依然不能跑，说明不是这个原因
          ceres::Problem problem(problem_options_);
          std::vector<double *> vec;

          vec.emplace_back(para_Pose_i);
          vec.emplace_back(para_Speed_i);
          vec.emplace_back(para_Bias_i);

          vec.emplace_back(para_Pose_j);
          vec.emplace_back(para_Speed_j);
          vec.emplace_back(para_Bias_j);
          
          //加入了
          problem.AddParameterBlock(para_Pose_i, 7);
          problem.AddParameterBlock(para_Speed_i, 3);
          problem.AddParameterBlock(para_Bias_i, 6);

          problem.AddParameterBlock(para_Pose_j, 7);
          problem.AddParameterBlock(para_Speed_j, 3);
          problem.AddParameterBlock(para_Bias_j, 6);
   
          problem.AddResidualBlock(imu_factor, NULL, para_Pose_i,para_Speed_i, para_Bias_i, para_Pose_j, para_Speed_j, para_Bias_j);
          GetJacobian(vec, problem, imu_factor->num_residuals(), jacobians);
     }


     void CheckJacobian(
         std::string factor_descri,
         Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobs_automatic,
         Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobs_analytic,
         const std::vector<double *> &parameters = {},
         const std::vector<std::string> &param_descri = {})
     {
          bool check_pass = true;
          size_t cnt = 0;

          std::map<double *, int> parameters_map;
          if (!parameters.empty() && !param_descri.empty())
          {
               for (int i = 0; i < (int)parameters.size(); ++i)
                    parameters_map[parameters.at(i)] = i;
          }

          for (auto const &v : jacobs_analytic)
          {
               auto iter = jacobs_automatic.find(v.first);
               if (jacobs_automatic.end() != iter)
               {
                    Eigen::MatrixXd diff = iter->second - v.second;
                    if (diff.cwiseAbs().maxCoeff() > 1e-6)
                    {
                         //   按内存地址大小的距离，不是误差项中参数添加顺序的索引
                         int idx;
                         if (parameters.empty())
                         {
                              idx = std::distance(jacobs_automatic.begin(), iter);
                         }
                         else
                         {
                              idx = parameters_map.at(iter->first);
                         }

                         std::cout << std::setiosflags(ios::fixed) << std::setw(15)
                                   << std::setprecision(15);

                         if (parameters.empty())
                              cout << " ======== index " << idx << " ========\n";
                         else
                              cout << " ======== index " << idx << " " << param_descri.at(idx)
                                   << " ========\n";
                         cout << "auto diff\n"
                              << iter->second << "\nanalytic:\n"
                              << v.second << endl;
                         check_pass = false;

                         std::cout << std::setiosflags(ios::fixed) << std::setw(3)
                                   << std::setprecision(3);
                    }
                    else
                    {
                         cnt++;
                    }
               }
          }

          cout << factor_descri << " check [" << cnt << "/" << jacobs_analytic.size()
               << "] jacobians ok.\n\n";
          if (!check_pass)
          {
               cout << ANSI_COLOR_RED << factor_descri << " has some problems.\n"
                    << ANSI_COLOR_RESET;
          }
     }

     void GetJacobian(std::vector<double *> param_vec, ceres::Problem &problem,
                      int residual_num,
                      Eigen::aligned_map<double *, Eigen::MatrixXd> &jacobians)
     {
          double cost = 0.0;
          ceres::CRSMatrix J;
          std::vector<double> residuals;
          problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals,
                           nullptr, &J);//错误

          Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
          dense_jacobian.setZero();
          for (int r = 0; r < J.num_rows; ++r)
          {
               for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
               {
                    const int c = J.cols[idx];
                    dense_jacobian(r, c) = J.values[idx];
               }
          }

          int cnt = 0;
          std::string right_descri = ")= ";
          if (residual_num > 1)
               right_descri += "\n";
          for (size_t i = 0; i < param_vec.size(); i++)
          {
               int local_size = problem.ParameterBlockLocalSize(param_vec.at(i));
               // if (i == 1)
               //      local_size = 4;
               Eigen::MatrixXd jacob = Eigen::MatrixXd::Zero(residual_num, local_size);
               jacob = dense_jacobian.block(0, cnt, residual_num, local_size);
               cnt += local_size;

               jacobians[param_vec.at(i)] = jacob;
               // jacobians[param_vec.at(i)] = jacob;
               cout << "J(" << std::setw(2) << i << right_descri << jacob << std::endl;
          }

          std::cout << "cost = " << cost << "; redisual: ";
          for (auto &r : residuals)
               std::cout << r << ", ";
          std::cout << "\n";

          // std::cout << "J = (" << J.num_rows << ", " << J.num_cols
          //           << ") with non - zero value; \n ";
          // for (int i = 0; i < J.num_rows; i++) {
          //   for (int j = J.rows[i]; j < J.rows[i + 1]; j++) {
          //     std::cout << "J(" << std::setw(2) << i << "," << std::setw(2)
          //               << J.cols[j] << ") = " << std::setw(10)
          //               << std::setiosflags(ios::fixed) << std::setprecision(3)
          //               << J.values[j] << "; ";
          //   }
          //   cout << endl;
          // }
     }

     void GetJacobian(std::vector<double *> param_vec,
                      const ceres::CostFunction *cost_function)
     {
          int num_residuals = cost_function->num_residuals();
          Eigen::MatrixXd residuals;
          residuals.setZero(num_residuals, 1);

          std::vector<double *> J_vec;
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              Jacob[cost_function->parameter_block_sizes().size()];
          size_t cnt = 0;
          for (auto const v : cost_function->parameter_block_sizes())
          {
               Jacob[cnt].setZero(num_residuals, v);
               J_vec.emplace_back(Jacob[cnt++].data());
          }

          cost_function->Evaluate(param_vec.data(), residuals.data(), J_vec.data());
          cout << "residuals = " << residuals.transpose() << endl;

          for (size_t i = 0; i < J_vec.size(); ++i)
          {
               if (num_residuals == 1)
                    cout << "J[" << i << "] = " << Jacob[i] << endl;
               else
                    cout << "J[" << i << "] = \n"
                         << Jacob[i] << endl;
          }
     }


     // void processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
     // {

     //      Eigen::Vector3d acc_0 = linear_acceleration;
     //      Eigen::Vector3d gyr_0 = angular_velocity;
     //      Eigen::Vector3d m_acceleration_bias(0.01, 0.01, 0.02);
     //      Eigen::Vector3d m_angular_bias(0.01, 0.01, 0.02);
     //      pre_integration = new IntegrationBase{acc_0, gyr_0, m_acceleration_bias,m_angular_bias};
     //      // if(!pre_integration)
     //      // {
     //      //      pre_integration = new IntegrationBase{acc_0, gyr_0, m_acceleration_bias,m_angular_bias};
     //      // }

     //      // pre_integration->push_back(dt, linear_acceleration,angular_velocity);
     // }

private:
    void GenerateData() 
    {
     Eigen::Vector3d para_last_position{0, 0, 0};
     Eigen::Quaterniond para_last_quat{1, 0, 0, 0};
     Eigen::Vector3d para_last_v{0, 0, 1};
     Eigen::Vector3d para_last_ba{0.01, 0.01, 0.01};
     Eigen::Vector3d para_last_bg{0.01, 0.01, 0.01};
     Eigen::Vector3d para_pre_position{0.5, 0.5, 0.5};
     Eigen::Quaterniond para_pre_quat{0.707, 0.707, 0, 0};
     Eigen::Vector3d para_pre_v{0, 0, 2};
     Eigen::Vector3d para_pre_ba{0.02, 0.02, 0.02};
     Eigen::Vector3d para_pre_bg{0.02, 0.02, 0.02};

     Eigen::Vector3d sqrt_info_vec = Eigen::Vector3d(0.33, 0.27, 0.11);

     pre_integration_ = new IntegrationBase(
          Eigen::Vector3d(0.11, 0.12, -9.83), Eigen::Vector3d(0.1, 0.2, -0.1),
          Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.01, 0.20, -0.01));
     pre_integration_->push_back(0.025, Eigen::Vector3d(0.1, 0.2, -9.83),
                                   Eigen::Vector3d(0.1, 0.2, 0.3));
     pre_integration_->push_back(0.46, Eigen::Vector3d(0.6, 0.4, -9.83),
                                   Eigen::Vector3d(0.3, 0.1, 0.17));
     t_pre_inte[0] = 0.21;
     t_pre_inte[1] = 0.695;

  }
private:
     //ceres set
     ceres::LocalParameterization *parameterization;
     ceres::LocalParameterization *autodiff_parameterization;
     ceres::Problem::Options problem_options_;
     ceres::Solver::Options solver_options_;
     
     //imu integration factor
     IntegrationBase* pre_integration_;
     double t_pre_inte[2];
     Eigen::Vector3d gravity_;
     Eigen::Vector2d g_refine_;

     double para_Pose_i[7];
     double para_Speed_i[3];
     double para_Bias_i[6];

     double para_Pose_j[7];
     double para_Speed_j[3];
     double para_Bias_j[6];

     Eigen::Vector3d para_last_position;
     Eigen::Quaterniond para_last_quat;
     Eigen::Vector3d para_last_v;
     Eigen::Vector3d para_last_ba;
     Eigen::Vector3d para_last_bg;

     Eigen::Vector3d para_pre_position;
     Eigen::Quaterniond para_pre_quat;
     Eigen::Vector3d para_pre_v;
     Eigen::Vector3d para_pre_ba;
     Eigen::Vector3d para_pre_bg;
};

int main()
{
     FactorTest factor_test;
     std::cout << std::setiosflags(ios::fixed) << std::setprecision(3);
     std::cout << "\n ===== TEST ===== \n\n";

     std::string descri =
         "test point plane factor";
     Eigen::aligned_map<double *, Eigen::MatrixXd> jacobs_automatic;
     Eigen::aligned_map<double *, Eigen::MatrixXd> jacobs_analytic;

     // factor_test.TestPointPlaneFactorAutoDiff(jacobs_automatic);
     // factor_test.TestPointPlaneFactorAnalytic(jacobs_analytic);
     // factor_test.TestIMUFactorAnalytic(jacobs_analytic);
     factor_test.TestIMUFactorAnalytic2(jacobs_analytic);
     // factor_test.CheckJacobian(descri, jacobs_automatic, TestIMUFactorAnalytic);

     return 0;
}