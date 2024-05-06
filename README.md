# check_jacobians_ceres

This is repo how to check jacobians with ceres

# 自动求导和解析求导传入参数的区别

* 问题定义：`residuals[0]`$\in {R}^{r\times1}$, 变量$x_i\in R^{m\times1}$
* 优化变量:$X=[x_0,\dots,x_n]^T$，雅可比:$J[i]\in R^{r \times m}$, 总雅可比$J=[J_0,\dots,J_n]^T \in R^{n . r \times m}$

## 自动求导法

> 自动求导法的参数是分隔开来

* 参数1：`m1[0]`
* 参数2：`m2[0]`
* 参数3：`m3[0]`
* 残差：`residual[0] =f(m1[0], m2[0],m3[0])`

```cpp
struct ExponentialResidual {
  ExponentialResidual(double x, double y) : x_(x), y_(y) {} //已知量x_, y_
  template <typename T>
  bool operator()(const T* const m1, const T* const m2, const T* const m3,  T* residual) const {
    residual[0] = y_ - exp(m1[0] * x_ + m2[0] + m3[0]);//损失函数 = 残差
    return true;
  }
 private:
  const double x_;
  const double y_;
};

int main(int argc, char** argv) 
{
  google::InitGoogleLogging(argv[0]);
  const double initial_m = 0.0;
  const double initial_c = 0.0;
  double m = initial_m;
  double c = initial_c;
  ceres::Problem problem;

  //正确
  for (int i = 0; i < kNumObservations; ++i) 
  {
    ExponentialResidual* residual = new ExponentialResidual(data[2 * i], data[2 * i + 1]);//自动求导costfunction
    problem.AddResidualBlock(
        cost_function,
        nullptr, //corefunction
        &m, //优化变量1
        &c);//优化变量2
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Initial m: " << initial_m << " c: " << initial_c << "\n";
  std::cout << "Final   m: " << m << " c: " << c << "\n";
  return 0;
}
```



## 解析求导法

* 参数1，雅可比1:`parameters[0][0],parameters[0][1],...,parameters[0][m]`; `jacobians[0]`
* 参数2，雅可比2:`parameters[1][0],parameters[0][1],...,parameters[0][m]`; `jacobians[1]`
* 参数N，雅可比N:`parameters[n][0],parameters[0][1],...,parameters[0][m]`; `jacobians[n]`

```cpp
//解析求导-方式2：ceres::CostFunction
class ExponentialResidual : public ceres::CostFunction
{   
public:
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}

  bool Evaluate(const double* const* parameters,
                double* residuals,
                double** jacobians) const override 
  {
    residuals[0] = y_ - exp(parameters[0][0] * x_ + parameters[1][0]);//损失函数 = 残差
    if (jacobians == nullptr) 
    {
      return true;
    }

    jacobians[0][0] = - x_ * exp(parameters[0][0] * x_ + parameters[1][0]);
    jacobians[1][0] = - exp(parameters[0][0] * x_ + parameters[1][0]);
  }

  static ceres::CostFunction* Create(double x, double y) {
    return new ExponentialResidual(x, y);
  }

 private:
  const double x_;
  const double y_;
};

int main(int argc, char** argv) 
{
  google::InitGoogleLogging(argv[0]);
  const double initial_m = 0.0;
  const double initial_c = 0.0;
  double m = initial_m;
  double c = initial_c;
  ceres::Problem problem;

  //正确
  for (int i = 0; i < kNumObservations; ++i) 
  {
    // ExponentialResidual* residual = new ExponentialResidual(data[2 * i], data[2 * i + 1]);//自动求导costfunction
    // ceres::CostFunction* cost_function = new ExponentialResidual(data[2 * i], data[2 * i + 1]);//解析求导costfunction-方法1
    ceres::CostFunction* cost_function = ExponentialResidual::Create(data[2 * i], data[2 * i + 1]);//解析求导costfunction-方法2
    problem.AddResidualBlock(
        cost_function,
        nullptr, //corefunction
        &m, //优化变量1
        &c);//优化变量2
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Initial m: " << initial_m << " c: " << initial_c << "\n";
  std::cout << "Final   m: " << m << " c: " << c << "\n";
  return 0;
}
```
