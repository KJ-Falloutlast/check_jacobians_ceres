#ifndef _IMU_FACTOR_H_
#define _IMU_FACTOR_H_

#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "utility.h"
#include "pre_integration.h"

#include <ceres/ceres.h>

//after fixed


// class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>  //原始方案：res = 15, p,q = 7, v, ba, bg = 9
// class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 1, 1, 1, 1, 1, 1, 1, 9>  //分开方案1：res = 15, p,q = 7, v, ba, bg = 9
class IMUFactor : public ceres::SizedCostFunction<15, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3> //分开方案2
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
        ACC_N = 0.1;
        ACC_W = 0.01;
        GYR_N = 0.01;
        GYR_W = 2.0e-5;

        G = Eigen::Vector3d(0.0,0.0,9.805);
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //原始方案：
        //         Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        //         Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        //         Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        //         Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        //         Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        //         Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        //         Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        //         Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        //         Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        //         Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);


        //方案2
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Bai(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Bgi(parameters[4][0], parameters[4][1], parameters[4][2]);

        Eigen::Vector3d Pj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Eigen::Quaterniond Qj(parameters[6][3], parameters[6][0], parameters[6][1], parameters[6][2]);//w, x, y, z

        Eigen::Vector3d Vj(parameters[7][0], parameters[7][1], parameters[7][2]);
        Eigen::Vector3d Baj(parameters[8][0], parameters[8][1], parameters[8][2]);
        Eigen::Vector3d Bgj(parameters[9][0], parameters[9][1], parameters[9][2]);
        

        //方案3：单独列出来position优化
        // Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        // Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        // Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        // Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        // Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);


        // Eigen::Vector3d Pj(parameters[2][0], parameters[3][0], parameters[4][0]);
        // Eigen::Quaterniond Qj(parameters[8][0], parameters[5][0], parameters[6][0], parameters[7][0]);

        // Eigen::Vector3d Vj(parameters[9][0], parameters[9][1], parameters[9][2]);
        // Eigen::Vector3d Baj(parameters[9][3], parameters[9][4], parameters[9][5]);
        // Eigen::Vector3d Bgj(parameters[9][6], parameters[9][7], parameters[9][8]);
//Eigen::Matrix<double, 15, 15> Fd;
//Eigen::Matrix<double, 15, 12> Gd;

//Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
//Eigen::Quaterniond pQj = Qi * delta_q;
//Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
//Eigen::Vector3d pBaj = Bai;
//Eigen::Vector3d pBgj = Bgi;

//Vi + Qi * delta_v - g * sum_dt = Vj;
//Qi * delta_q = Qj;

//delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
//delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
//delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();
        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            if (jacobians[0])//pi
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_position_i(jacobians[0]);
                jacobian_position_i.setZero();

                jacobian_position_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();

                jacobian_position_i = sqrt_info * jacobian_position_i;
                if (jacobian_position_i.maxCoeff() > 1e8 || jacobian_position_i.minCoeff() < -1e8)
                {
                    std::cout << sqrt_info << std::endl;
                    //assert(false);
                }
            }

            if (jacobians[1])//Ri
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_rotation_i(jacobians[1]);
                jacobian_rotation_i.setZero();

                //(0, 0)
                jacobian_rotation_i.block<3, 3>(O_P, O_R - O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_rotation_i.block<3, 3>(O_R, O_R - O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                //(3, 0)
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_rotation_i.block<3, 3>(O_R, O_R - O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif
                //(6, 0)
                jacobian_rotation_i.block<3, 3>(O_V, O_R - O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                jacobian_rotation_i = sqrt_info * jacobian_rotation_i;

                if (jacobian_rotation_i.maxCoeff() > 1e8 || jacobian_rotation_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }

            if (jacobians[2])//vi：务必要注意<3, 3>(0, 0), (3, 0), (6, 0)分别代表p, R, v
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_i(jacobians[2]);
                jacobian_speed_i.setZero();
                //(0, 0)
                jacobian_speed_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                //(6, 0)
                jacobian_speed_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();

                jacobian_speed_i = sqrt_info * jacobian_speed_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }

            if (jacobians[3])//bai
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[3]);
                jacobian_speedbias_i.setZero();
                //(0, 0)
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_BA) = -dp_dba;
                //(6, 0)
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_BA) = -dv_dba;//!fix
                //(3, 0)
                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_BA) = -Eigen::Matrix3d::Identity();//!fix

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }

            if (jacobians[4])//bgi
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_omegabias_i(jacobians[4]);
                jacobian_omegabias_i.setZero();
                //(0, 0)
                jacobian_omegabias_i.block<3, 3>(O_P, O_BG - O_BG) = -dp_dbg;
                //(6, 0)
                jacobian_omegabias_i.block<3, 3>(O_V, O_BG - O_BG) = -dv_dbg;//!fix
#if 0
                jacobian_omegabias_i.block<3, 3>(O_R, O_BG - O_BG) = -dq_dbg;
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //(3, 0)
                jacobian_omegabias_i.block<3, 3>(O_R, O_BG - O_BG) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;//!fix
                // origin
                // jacobian_omegabias_i.block<3, 3>(O_R, O_BG - O_BG) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif
                //(12,0)
                jacobian_omegabias_i.block<3, 3>(O_BG, O_BG - O_BG) = -Eigen::Matrix3d::Identity();//!fix

                jacobian_omegabias_i = sqrt_info * jacobian_omegabias_i;
                
                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }

            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_position_j(jacobians[5]);
                jacobian_position_j.setZero();
                //(0, 0)
                jacobian_position_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
                jacobian_position_j = sqrt_info * jacobian_position_j;
            }

            if (jacobians[6])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_rotation_j(jacobians[6]);
                jacobian_rotation_j.setZero();
#if 0
            jacobian_rotation_j.block<3, 3>(O_R, O_R - O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //(3, 0)
                jacobian_rotation_j.block<3, 3>(O_R, O_R - O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif
                jacobian_rotation_j = sqrt_info * jacobian_rotation_j;

                //ROS_ASSERT(fabs(jacobian_rotation_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_rotation_j.minCoeff()) < 1e8);
            }

            if (jacobians[7])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_j(jacobians[7]);
                jacobian_speed_j.setZero();
                //(6, 0)
                jacobian_speed_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speed_j = sqrt_info * jacobian_speed_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }

            if (jacobians[8])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[8]);
                jacobian_speedbias_j.setZero();
                //(9, 0)
                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_BA) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }

            if (jacobians[9])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_omegabias_j(jacobians[9]);
                jacobian_omegabias_j.setZero();
                //(12,0)
                jacobian_omegabias_j.block<3, 3>(O_BG, O_BG - O_BG) = Eigen::Matrix3d::Identity();
                jacobian_omegabias_j = sqrt_info * jacobian_omegabias_j;

                //ROS_ASSERT(fabs(jacobian_omegabias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_omegabias_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    double ACC_N;
    double ACC_W;
    double GYR_N;
    double GYR_W;

    Eigen::Vector3d G;

    IntegrationBase* pre_integration;


};

// class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
// {
//   public:
//     IMUFactor() = delete;
//     IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
//     {
//         ACC_N = 0.1;
//         ACC_W = 0.01;
//         GYR_N = 0.01;
//         GYR_W = 2.0e-5;

//         G = Eigen::Vector3d(0.0,0.0,9.805);
//     }
//     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
//     {

//         Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
//         Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

//         Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
//         Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
//         Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

//         Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
//         Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

//         Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
//         Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
//         Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

// //Eigen::Matrix<double, 15, 15> Fd;
// //Eigen::Matrix<double, 15, 12> Gd;

// //Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
// //Eigen::Quaterniond pQj = Qi * delta_q;
// //Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
// //Eigen::Vector3d pBaj = Bai;
// //Eigen::Vector3d pBgj = Bgi;

// //Vi + Qi * delta_v - g * sum_dt = Vj;
// //Qi * delta_q = Qj;

// //delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
// //delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
// //delta_q = Qi.inverse() * Qj;

// #if 0
//         if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
//             (Bgi - pre_integration->linearized_bg).norm() > 0.01)
//         {
//             pre_integration->repropagate(Bai, Bgi);
//         }
// #endif

//         Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
//         residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
//                                             Pj, Qj, Vj, Baj, Bgj);

//         Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
//         //sqrt_info.setIdentity();
//         residual = sqrt_info * residual;

//         if (jacobians)
//         {
//             double sum_dt = pre_integration->sum_dt;
//             Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
//             Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

//             Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

//             Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
//             Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

//             if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
//             {
//                 ROS_WARN("numerical unstable in preintegration");
//                 //std::cout << pre_integration->jacobian << std::endl;
// ///                ROS_BREAK();
//             }

//             if (jacobians[0])
//             {
//                 Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
//                 jacobian_pose_i.setZero();

//                 jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
//                 jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

// #if 0
//             jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
// #else
//                 Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
//                 jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
// #endif

//                 jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

//                 jacobian_pose_i = sqrt_info * jacobian_pose_i;

//                 if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
//                 {
//                     ROS_WARN("numerical unstable in preintegration");
//                     //std::cout << sqrt_info << std::endl;
//                     //ROS_BREAK();
//                 }
//             }
//             if (jacobians[1])
//             {
//                 Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
//                 jacobian_speedbias_i.setZero();
//                 jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
//                 jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
//                 jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

// #if 0
//             jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
// #else
//                 //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
//                 //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
//                 jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
// #endif

//                 jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
//                 jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
//                 jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

//                 jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

//                 jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

//                 jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

//                 //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
//                 //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
//             }
//             if (jacobians[2])
//             {
//                 Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
//                 jacobian_pose_j.setZero();

//                 jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

// #if 0
//             jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
// #else
//                 Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
//                 jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
// #endif

//                 jacobian_pose_j = sqrt_info * jacobian_pose_j;

//                 //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
//                 //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
//             }
//             if (jacobians[3])
//             {
//                 Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
//                 jacobian_speedbias_j.setZero();

//                 jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

//                 jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

//                 jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

//                 jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

//                 //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
//                 //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
//             }
//         }

//         return true;
//     }

//     //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

//     //void checkCorrection();
//     //void checkTransition();
//     //void checkJacobian(double **parameters);
//     double ACC_N;
//     double ACC_W;
//     double GYR_N;
//     double GYR_W;

//     Eigen::Vector3d G;

//     IntegrationBase* pre_integration;


// };

#endif