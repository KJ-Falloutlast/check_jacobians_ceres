/*
 * Continuous-Time Fixed-Lag Smoothing for LiDAR-Inertial-Camera SLAM
 * Copyright (C) 2022 Jiajun Lv
 * Copyright (C) 2022 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include "sophus/so3.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <deque>
#include "imu_factor.h"
#include "pre_integration.h"
#include "tool_color_printf.hpp"
#include "poseParameterization.h"
#include "utility.h"
#include "imu_factor_pnp.h"

#define RESET "\033[0m"
#define BLACK "\033[30m"   /* Black */
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

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

class FactorTest {
 public:
  FactorTest() {
    parameterization = new RotationParameterization();
    autodiff_parameterization = new ceres::EigenQuaternionParameterization();

    problem_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options_.local_parameterization_ownership =
        ceres::DO_NOT_TAKE_OWNERSHIP;

    solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options_.minimizer_progress_to_stdout = true;
    solver_options_.update_state_every_iteration = true;
    solver_options_.max_num_iterations = 1;//最大迭代次数为1
    solver_options_.num_threads = 1;

    GenerateData();
  }

//   void TestPreIntegrationFactorAutoDiff(
//       Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
//       bool use_analytic_factor = false) {
//     int64_t time0_ns = t_pre_inte[0] * S_TO_NS;
//     int64_t time1_ns = t_pre_inte[1] * S_TO_NS;
//     SplineMeta<SplineOrder> spline_meta;
//     SplineMeta<BiasSplineOrder> spline_meta_bias;
//     trajectory_->CaculateSplineMeta({{time0_ns, time1_ns}}, spline_meta);

//     auto* cost_function = auto_diff::PreIntegrationFactor::Create(
//         pre_integration_, time0_ns, time1_ns, spline_meta);

//     for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
//       cost_function->AddParameterBlock(4);  /// add SO3 knots
//     }
//     for (size_t i = 0; i < spline_meta.NumParameters(); i++) {
//       cost_function->AddParameterBlock(3);  /// add R3 knots
//     }
//     cost_function->AddParameterBlock(3);
//     cost_function->AddParameterBlock(3);
//     cost_function->AddParameterBlock(3);
//     cost_function->AddParameterBlock(3);

//     cost_function->SetNumResiduals(15);

//     ceres::Problem problem(problem_options_);
//     std::vector<double*> vec;
//     AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
//     AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
//     vec.emplace_back(imu_bias_.gyro_bias.data());
//     vec.emplace_back(imu_bias2_.gyro_bias.data());
//     vec.emplace_back(imu_bias_.accel_bias.data());
//     vec.emplace_back(imu_bias2_.accel_bias.data());

//     problem.AddResidualBlock(cost_function, NULL, vec);

//     GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);
//   }

//  void TestPreIntegrationFactorAnalytic(
//       Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
//       std::vector<double*>& vec, std::vector<std::string>& param_descri,
//       bool use_analytic_factor = true) {
//     int64_t time0_ns = t_pre_inte[0] * S_TO_NS;
//     int64_t time1_ns = t_pre_inte[1] * S_TO_NS;
//     SplineMeta<SplineOrder> spline_meta;
//     SplineMeta<BiasSplineOrder> spline_meta_bias;
//     trajectory_->CaculateSplineMeta({{time0_ns, time1_ns}}, spline_meta);

//     ceres::CostFunction* cost_function =
//         new analytic_derivative::PreIntegrationFactor(
//             pre_integration_, time0_ns, time1_ns, spline_meta);

//     ceres::Problem problem(problem_options_);

//     AddControlPoints(spline_meta, vec, false, use_analytic_factor, problem);
//     AddControlPoints(spline_meta, vec, true, use_analytic_factor, problem);
//     vec.emplace_back(imu_bias_.gyro_bias.data());
//     vec.emplace_back(imu_bias2_.gyro_bias.data());
//     vec.emplace_back(imu_bias_.accel_bias.data());
//     vec.emplace_back(imu_bias2_.accel_bias.data());

//     problem.AddResidualBlock(cost_function, NULL, vec);

//     GetJacobian(vec, problem, cost_function->num_residuals(), jacobians);

//     for (int i = 0; i < (int)spline_meta.NumParameters(); i++)
//       param_descri.push_back("rotation control point");
//     for (int i = 0; i < (int)spline_meta.NumParameters(); i++)
//       param_descri.push_back("position control point");

//     param_descri.push_back("gyro bias control point");
//     param_descri.push_back("gyro bias control point");
//     param_descri.push_back("accel bias control point");
//     param_descri.push_back("accel bias control point");
//   }


  void TestPreIntegrationFactorPNPAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      std::vector<double*>& vec, std::vector<std::string>& param_descri,
      bool use_analytic_factor = true) 
  {
      //转换数据
      vec2doublePNP();
      
      //传入优化量
      ceres::Problem problem(problem_options_);
      IMUFactorPnP* imu_factor = new IMUFactorPnP(pre_integration_);
      vec.emplace_back(para_Pose_i);
      vec.emplace_back(para_Speed_i);
      vec.emplace_back(para_Bias_i);

      vec.emplace_back(para_Pose_j);
      vec.emplace_back(para_Speed_j);
      vec.emplace_back(para_Bias_j);
      
      //用double arr[3]类型的可以
      problem.AddParameterBlock(para_Pose_i, 7);
      problem.AddParameterBlock(para_Speed_i, 3);
      problem.AddParameterBlock(para_Bias_i, 6);

      problem.AddParameterBlock(para_Pose_j, 7);
      problem.AddParameterBlock(para_Speed_j, 3);
      problem.AddParameterBlock(para_Bias_j, 6);

      problem.AddResidualBlock(imu_factor, NULL, para_Pose_i,para_Speed_i, para_Bias_i, para_Pose_j, para_Speed_j, para_Bias_j);
      // problem.AddResidualBlock(imu_factor, NULL, vec);
      param_descri.push_back("para_Pose_i");
      param_descri.push_back("para_Speed_i");
      param_descri.push_back("para_Bias_i");
      param_descri.push_back("para_Pose_j");
      param_descri.push_back("para_Speed_j");
      param_descri.push_back("para_Bias_j");

      //得到雅可比
      GetJacobian(vec, problem, imu_factor->num_residuals(), jacobians);
  }


  void TestPreIntegrationFactorUWBAnalytic(
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians,
      std::vector<double*>& vec, std::vector<std::string>& param_descri,
      bool use_analytic_factor = true) 
  {
      //传入优化量
      ceres::Problem problem(problem_options_);
      IMUFactor* imu_factor = new IMUFactor(pre_integration_);

      //方案1：利用Eigen::Vector加入
      // double last_quat_coeffs[4], pre_quat_coeffs[4];  
      // last_quat_coeffs[0] = para_last_quat.x();//order can‘t be messed！double arr[] to eigen:(0, 1, 2, 3)->(x, y, z, w)
      // last_quat_coeffs[1] = para_last_quat.y();//eigen to double arr[]: (w, x, y, z)->(3, 0, 1, 2) 
      // last_quat_coeffs[2] = para_last_quat.z();  
      // last_quat_coeffs[3] = para_last_quat.w();  

      // pre_quat_coeffs[0] = para_pre_quat.x();  
      // pre_quat_coeffs[1] = para_pre_quat.y();  
      // pre_quat_coeffs[2] = para_pre_quat.z();  
      // pre_quat_coeffs[3] = para_pre_quat.w();  

      // vec.emplace_back(para_last_position.data());
      // vec.emplace_back(last_quat_coeffs);  
      // vec.emplace_back(para_last_v.data());
      // vec.emplace_back(para_last_ba.data());
      // vec.emplace_back(para_last_bg.data());

      // vec.emplace_back(para_pre_position.data());
      // vec.emplace_back(pre_quat_coeffs);  
      // vec.emplace_back(para_pre_v.data());
      // vec.emplace_back(para_pre_ba.data());
      // vec.emplace_back(para_pre_bg.data());


      // problem.AddParameterBlock(&para_last_position.x(), 3);
      // problem.AddParameterBlock(&para_last_quat.x(), 4, autodiff_parameterization);//旋转一定要加parameterization
      // problem.AddParameterBlock(&para_last_quat.x(), 4);
      // problem.AddParameterBlock(&para_last_v.x(), 3);
      // problem.AddParameterBlock(&para_last_ba.x(), 3);
      // problem.AddParameterBlock(&para_last_bg.x(), 3);

      // problem.AddParameterBlock(&para_pre_position.x(), 3);
      // problem.AddParameterBlock(&para_pre_quat.x(), 4, autodiff_parameterization);
      // problem.AddParameterBlock(&para_pre_quat.x(), 4);
      // problem.AddParameterBlock(&para_pre_v.x(), 3);
      // problem.AddParameterBlock(&para_pre_ba.x(), 3);
      // problem.AddParameterBlock(&para_pre_bg.x(), 3);

      //方案2：利用double arr[3]加入
      vec2doubleUWB();
      //加入vec
      vec.emplace_back(para_position_i);
      vec.emplace_back(para_quat_i);
      vec.emplace_back(para_speed_i);
      vec.emplace_back(para_ba_i);
      vec.emplace_back(para_bg_i);

      vec.emplace_back(para_position_j);
      vec.emplace_back(para_quat_j);
      vec.emplace_back(para_speed_j);
      vec.emplace_back(para_ba_j);
      vec.emplace_back(para_bg_j);
      //加入parameter，为residual做准备
      problem.AddParameterBlock(para_position_i, 3);
      problem.AddParameterBlock(para_quat_i, 4);
      problem.AddParameterBlock(para_speed_i, 3);
      problem.AddParameterBlock(para_ba_i, 3);
      problem.AddParameterBlock(para_bg_i, 3);

      problem.AddParameterBlock(para_position_j, 3);
      problem.AddParameterBlock(para_quat_j, 4);
      problem.AddParameterBlock(para_speed_j, 3);
      problem.AddParameterBlock(para_ba_j, 3);
      problem.AddParameterBlock(para_bg_j, 3);

      //##########测试#############
      //测试的目的必须保证vec和&para_last_position.x()的雅可比一样
      //double arr[3]测试
      problem.AddResidualBlock(imu_factor, NULL, para_position_i, para_quat_i, para_speed_i
                                               , para_ba_i, para_bg_i
                                               , para_position_j, para_quat_j, para_speed_j
                                               , para_ba_j, para_bg_j);
      //Eigen测试
      // problem.AddResidualBlock(imu_factor, NULL, &para_last_position.x(), &para_last_quat.x()
      //                                             , &para_last_v.x(), &para_last_ba.x(), &para_last_bg.x()
      //                                             , &para_pre_position.x(), &para_pre_quat.x()
      //                                             , &para_pre_v.x(), &para_pre_ba.x(), &para_pre_bg.x());
      //通用测试
      // problem.AddResidualBlock(imu_factor, NULL, vec);
      //##########测试#############

      //得到雅可比
      GetJacobian(vec, problem, imu_factor->num_residuals(), jacobians);
  }

  void CheckJacobian(
      std::string factor_descri,
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobs_automatic,
      Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobs_analytic,
      const std::vector<double*>& parameters = {},
      const std::vector<std::string>& param_descri = {}) {
    bool check_pass = true;
    size_t cnt = 0;

    std::map<double*, int> parameters_map;
    if (!parameters.empty() && !param_descri.empty()) {
      for (int i = 0; i < (int)parameters.size(); ++i)
        parameters_map[parameters.at(i)] = i;
    }

    //只有当解析求导和自动求导两个雅可比都获得时，才能够使用这里的对比功能，否则cnt不会++
    //所以才会出现[0/6]的状况
    for (auto const& v : jacobs_analytic) 
    {
      auto iter = jacobs_automatic.find(v.first);//找到J_analy[i]对应的J_auto[i]
      if (jacobs_automatic.end() != iter)
      {
        Eigen::MatrixXd diff = iter->second - v.second;
        if (diff.cwiseAbs().maxCoeff() > 1e-6)//如果找到，则作差，距离过大,则J_analy[i]不能作为pass的雅可比
        {
          // 按内存地址大小的距离，不是误差项中参数添加顺序的索引
          /*
          如果parameters为空，说明没有额外的参数描述信息，直接根据迭代器位置计算索引。
          否则，从parameters_map中获取索引，并可能从param_descri中获取对应的参数描述信息。 
          然后，按固定格式打印差异信息，包括索引和可能的参数描述，以及两个矩阵的具体值。
          */
          int idx;
          if (parameters.empty()) {
            idx = std::distance(jacobs_automatic.begin(), iter);
          } else {
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

    //6个雅可比
    cout << factor_descri << " check [" << cnt << "/" << jacobs_analytic.size()
         << "] jacobians ok.\n\n";
    if (!check_pass) {
      cout << RED << factor_descri << " has some problems.\n" << RESET;
    }
  }

 private:
  void GenerateData() {
    Eigen::Vector3d sqrt_info_vec = Eigen::Vector3d(0.33, 0.27, 0.11);

    pre_integration_ = new IntegrationBase(
        Eigen::Vector3d(0.11, 0.12, -9.83), Eigen::Vector3d(0.1, 0.2, -0.1),
        Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.01, 0.20, -0.01));
    pre_integration_->push_back(0.025, Eigen::Vector3d(0.1, 0.2, -9.83),
                                Eigen::Vector3d(0.1, 0.2, 0.3));
    pre_integration_->push_back(0.46, Eigen::Vector3d(0.6, 0.4, -9.83),
                                Eigen::Vector3d(0.3, 0.1, 0.17));

    //初始化位姿
    para_last_position = Eigen::Vector3d(0, 0, 0);
    para_last_quat = Eigen::Quaterniond(1, 0, 0, 0);
    para_last_v = Eigen::Vector3d(0, 0, 1);
    para_last_ba = Eigen::Vector3d(0.01, 0.01, 0.01);
    para_last_bg = Eigen::Vector3d(0.01, 0.01, 0.01);

    para_pre_position = Eigen::Vector3d(0.5, 0.5, 0.5);
    para_pre_quat = Eigen::Quaterniond(0.707, 0.707, 0, 0);
    para_pre_quat.normalized();
    para_pre_v = Eigen::Vector3d(0, 0, 2);
    para_pre_ba = Eigen::Vector3d(0.02, 0.02, 0.02);
    para_pre_bg = Eigen::Vector3d(0.02, 0.02, 0.02);


  }


  void GetJacobian(std::vector<double*> param_vec, ceres::Problem& problem,
                   int residual_num,
                   Eigen::aligned_map<double*, Eigen::MatrixXd>& jacobians) {
    double cost = 0.0;
    ceres::CRSMatrix J;
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals,
                     nullptr, &J);

    Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
    dense_jacobian.setZero();
    for (int r = 0; r < J.num_rows; ++r) {
      for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx) {
        const int c = J.cols[idx];
        dense_jacobian(r, c) = J.values[idx];
      }
    }

    int cnt = 0;
    std::string right_descri = ")= ";
    if (residual_num > 1) right_descri += "\n";
    for (size_t i = 0; i < param_vec.size(); i++) {
      int local_size = problem.ParameterBlockLocalSize(param_vec.at(i));
      Eigen::MatrixXd jacob = Eigen::MatrixXd::Zero(residual_num, local_size);
      jacob = dense_jacobian.block(0, cnt, residual_num, local_size);
      cnt += local_size;

      jacobians[param_vec.at(i)] = jacob;
      cout << "J(" << std::setw(2) << i << right_descri << jacob <<
      std::endl; //打印雅可比
    }

    std::cout << "cost = " << cost << "; redisual: ";
    for (auto& r : residuals) std::cout << r << ", ";
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

  void GetJacobian(std::vector<double*> param_vec,
                   const ceres::CostFunction* cost_function) {
    int num_residuals = cost_function->num_residuals();
    Eigen::MatrixXd residuals;
    residuals.setZero(num_residuals, 1);

    std::vector<double*> J_vec;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Jacob[cost_function->parameter_block_sizes().size()];
    size_t cnt = 0;
    for (auto const v : cost_function->parameter_block_sizes()) {
      Jacob[cnt].setZero(num_residuals, v);
      J_vec.emplace_back(Jacob[cnt++].data());
    }

    cost_function->Evaluate(param_vec.data(), residuals.data(), J_vec.data());
    cout << "residuals = " << residuals.transpose() << endl;

    for (size_t i = 0; i < J_vec.size(); ++i) {
      if (num_residuals == 1)
        cout << "J[" << i << "] = " << Jacob[i] << endl;
      else
        cout << "J[" << i << "] = \n" << Jacob[i] << endl;
    }
  }

  void vec2doublePNP()
  {
      para_Pose_i[0] = para_last_position.x();
      para_Pose_i[1] = para_last_position.y();
      para_Pose_i[2] = para_last_position.z();
      para_Pose_i[3] = para_last_quat.x();
      para_Pose_i[4] = para_last_quat.y();
      para_Pose_i[5] = para_last_quat.z();
      para_Pose_i[6] = para_last_quat.w();

      para_Speed_i[0] = para_last_v.x();
      para_Speed_i[1] = para_last_v.y();
      para_Speed_i[2] = para_last_v.z();

      para_Bias_i[0] = para_last_ba.x();
      para_Bias_i[1] = para_last_ba.y();
      para_Bias_i[2] = para_last_ba.z();
      para_Bias_i[3] = para_last_bg.x();
      para_Bias_i[4] = para_last_bg.y();
      para_Bias_i[5] = para_last_bg.z();

      para_Pose_j[0] = para_pre_position.x();
      para_Pose_j[1] = para_pre_position.y();
      para_Pose_j[2] = para_pre_position.z();
      para_Pose_j[3] = para_pre_quat.x();
      para_Pose_j[4] = para_pre_quat.y();
      para_Pose_j[5] = para_pre_quat.z();
      para_Pose_j[6] = para_pre_quat.w();

      para_Speed_j[0] = para_pre_v.x();
      para_Speed_j[1] = para_pre_v.y();
      para_Speed_j[2] = para_pre_v.z();

      para_Bias_j[0] = para_pre_ba.x();
      para_Bias_j[1] = para_pre_ba.y();
      para_Bias_j[2] = para_pre_ba.z();
      para_Bias_j[3] = para_pre_bg.x();
      para_Bias_j[4] = para_pre_bg.y();
      para_Bias_j[5] = para_pre_bg.z();
  }

  void vec2doubleUWB()
  {

    para_position_i[0] = para_last_position.x();
    para_position_i[1] = para_last_position.y();
    para_position_i[2] = para_last_position.z();
    
    para_quat_i[0] = para_last_quat.x();
    para_quat_i[1] = para_last_quat.y();
    para_quat_i[2] = para_last_quat.z();
    para_quat_i[3] = para_last_quat.w();

    para_speed_i[0] = para_last_v.x();
    para_speed_i[1] = para_last_v.y();
    para_speed_i[2] = para_last_v.z();

    para_ba_i[0] = para_last_ba.x();
    para_ba_i[1] = para_last_ba.y();
    para_ba_i[2] = para_last_ba.z();

    para_bg_i[0] = para_last_bg.x();
    para_bg_i[1] = para_last_bg.y();
    para_bg_i[2] = para_last_bg.z();

    para_position_j[0] = para_pre_position.x();
    para_position_j[1] = para_pre_position.y();
    para_position_j[2] = para_pre_position.z();

    para_quat_j[0] = para_pre_quat.x();
    para_quat_j[1] = para_pre_quat.y();
    para_quat_j[2] = para_pre_quat.z();
    para_quat_j[3] = para_pre_quat.w();

    para_speed_j[0] = para_pre_v.x();
    para_speed_j[1] = para_pre_v.y();
    para_speed_j[2] = para_pre_v.z();

    para_ba_j[0] = para_pre_ba.x();
    para_ba_j[1] = para_pre_ba.y();
    para_ba_j[2] = para_pre_ba.z();
    
    para_bg_j[0] = para_pre_bg.x();
    para_bg_j[1] = para_pre_bg.y();
    para_bg_j[2] = para_pre_bg.z();
  }

 public:

  // PreIntegration factor
  double t_pre_inte[2];
  IntegrationBase* pre_integration_;

  //测试imu_pnp_factor
  double para_Pose_i[7];
  double para_Speed_i[3];
  double para_Bias_i[6];

  double para_Pose_j[7];
  double para_Speed_j[3];
  double para_Bias_j[6];
  
  //测试imu_uwb_factor
  double para_position_i[3];
  double para_quat_i[4];
  double para_speed_i[3];
  double para_ba_i[3];
  double para_bg_i[3];

  double para_position_j[3];
  double para_quat_j[4];
  double para_speed_j[3];
  double para_ba_j[3];
  double para_bg_j[3];

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

  ceres::LocalParameterization *parameterization;
  ceres::LocalParameterization *autodiff_parameterization;
  ceres::Problem::Options problem_options_;
  ceres::Solver::Options solver_options_;
};

int main(int argc, char** argv) {
  FactorTest factor_test;

  std::cout << std::setiosflags(ios::fixed) << std::setprecision(3);
  cout << "\n ===== TEST ===== \n\n";


  // double ti[2] = {0.21, 0.21};
  // double tj[2] = {0.35, 0.695};
  // for (int i = 0; i < 2; ++i) {
  //   factor_test.t_pre_inte[0] = ti[i];
  //   factor_test.t_pre_inte[1] = tj[i];

  //   Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
  //   Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;
  //   std::vector<double*> param_vec;
  //   std::vector<std::string> param_descri;

  //   std::string descri =
  //       "PreIntegration factor [case " + std::to_string(i) + "]";
  //   factor_test.TestPreIntegrationFactorAutoDiff(jacobs_automatic);
  //   factor_test.TestPreIntegrationFactorAnalytic(jacobs_analytic, param_vec,
  //                                                param_descri);
  //   factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic,
  //                             param_vec, param_descri);
  // }


  for (int i = 0; i < 2; ++i) {

    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;
    std::vector<double*> param_vec;
    std::vector<std::string> param_descri;

    std::string descri =
        "PreIntegration factor [case " + std::to_string(i) + "]";
    // factor_test.TestPreIntegrationFactorAutoDiff(jacobs_automatic);
    factor_test.TestPreIntegrationFactorPNPAnalytic(jacobs_analytic, param_vec,
                                                 param_descri);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic,
                              param_vec, param_descri);
  }

  for (int i = 0; i < 2; ++i) {

    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_automatic;
    Eigen::aligned_map<double*, Eigen::MatrixXd> jacobs_analytic;
    std::vector<double*> param_vec;
    std::vector<std::string> param_descri;

    std::string descri =
        "PreIntegrationUwb factor [case " + std::to_string(i) + "]";
    // factor_test.TestPreIntegrationFactorAutoDiff(jacobs_automatic);
    factor_test.TestPreIntegrationFactorUWBAnalytic(jacobs_analytic, param_vec,
                                                 param_descri);
    factor_test.CheckJacobian(descri, jacobs_automatic, jacobs_analytic,
                              param_vec, param_descri);
  }

  return 0;
}
