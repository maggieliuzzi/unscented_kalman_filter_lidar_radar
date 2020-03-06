#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


// Unscented Kalman Filter Initialiser
UKF::UKF() {

  // SENSORS
  use_lidar_ = true;  // take measurement into account, not only during init
  use_radar_ = true;

  time_us_ = 0;  // start time
  state_initialised_ = false;  // set to true with first ProcessMeasurement() call

  // STATE
  state_dim_ = 5;  // state dimension
  state_ = VectorXd(state_dim_);  // initial state
  STATE_COV_ = MatrixXd(state_dim_, state_dim_);  // initial state covariance

  // PROCESS NOISE
  dev_lin_acc = 2.31;  // std dev longitudinal/linear acceleration, m/s^2  // TOTUNE  // researchgate.net/publication/223922575_Design_speeds_and_acceleration_characteristics_of_bicycle_traffic_for_use_in_planning_design_and_appraisal
  dev_ang_acc = 1.0;  // std dev yaw/angular acceleration, rad/s^2  // TOTUNE
  
  // MEASUREMENT NOISE
  // Provided by sensor manufacturers
  // LiDAR measurement noise standard deviation
  std_lidpx_ = 0.15;  // px, m
  std_lidpy_ = 0.15;  // py, m
  // RADAR measurement noise standard deviation
  std_radr_ = 0.3;  // radius, m
  std_radphi_ = 0.03;  // angle, rad
  std_radrd_ = 0.3;  // radius change, m/s
  
  // GAUSSIANISATION
  // augmented state dimension
  state_dim_aug_ = 7;
  lambda_ = 0;
  weights_ = VectorXd(2 * state_dim_aug_ + 1);  // weights vector
  SIG_PT_pred_ = MatrixXd(state_dim_, 2 * state_dim_aug_ + 1);  // sigma points matrix
  
  // NIS
  NIS_lidar_ = 0;
  NIS_radar_ = 0;
}


UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // initialising of state and state covariance if not already initialised
  if (!state_initialised_) {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhodot = meas_package.raw_measurements_(2);
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rhodot * cos(phi);
      double vy = rhodot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      state_ << x, y, v, rho, rhodot;
      
      STATE_COV_ << std_radr_*std_radr_, 0, 0, 0, 0,
                    0, std_radr_*std_radr_, 0, 0, 0,
                    0, 0, std_radrd_*std_radrd_, 0, 0,
                    0, 0, 0, std_radphi_, 0,
                    0, 0, 0, 0, std_radphi_;
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::LIDAR) {

      state_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;  // TOTUNE: last values
      
      STATE_COV_ << std_lidpx_*std_lidpx_, 0, 0, 0, 0,
                    0, std_lidpy_*std_lidpy_, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 1;  // TOTUNE
    }
    
    state_initialised_ = true;
    time_us_ = meas_package.timestamp_;
    return;  // no prediction nor updating during initialisation
  }
  
  // if state already initialised
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;  // seconds
  time_us_ = meas_package.timestamp_;  // store current time for future reference

  // PREDICTION based on delta t
  Prediction(delta_t);

  // MEASUREMENT UPDATE depending on incoming sensor package
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
}


void UKF::Prediction(double delta_t) {
  /* Estimating object location after delta_t seconds
      Modifying state vector, predicting sigma points, the state, and the state covariance matrix */

  // calculating sigma points and setting them as matrix columns
  lambda_ = 3 - state_dim_;  // spreading parameter
  MatrixXd SIG_PT_ = MatrixXd(state_dim_, 2 * state_dim_ + 1);  // sigma point matrix
  MatrixXd SQR_C = STATE_COV_.llt().matrixL();  // square root of P
  SIG_PT_.col(0) = state_;
  for (int i = 0; i < state_dim_; i++) {
    SIG_PT_.col(i+1) = state_ + std::sqrt(lambda_+state_dim_) * SQR_C.col(i);
    SIG_PT_.col(i+1+state_dim_) = state_ - std::sqrt(lambda_+state_dim_) * SQR_C.col(i);
  }
  
  // setting augmented mean state
  VectorXd state_aug_ = VectorXd(state_dim_aug_);  // augmented mean vector
  state_aug_.head(5) = state_;
  state_aug_(5) = 0;
  state_aug_(6) = 0;
  
  // setting augmented covariance matrix
  MatrixXd STATE_COV_aug_ = MatrixXd(state_dim_aug_, state_dim_aug_);  // augmented state covariance
  MatrixXd Q = MatrixXd(2, 2);
  Q << dev_lin_acc*dev_lin_acc, 0,
       0, dev_ang_acc*dev_ang_acc;
  STATE_COV_aug_.fill(0.0);
  STATE_COV_aug_.topLeftCorner(5, 5) = STATE_COV_;
  STATE_COV_aug_.bottomRightCorner(2, 2) = Q;
  
  // setting augmented sigma point matrix
  MatrixXd SIG_PT_aug_ = MatrixXd(state_dim_aug_, 2 * state_dim_aug_ + 1);  // augmented sigma point matrix
  lambda_ = 3 - state_dim_aug_;  // spreading parameter for augmentation
  MatrixXd SQR_C_aug = STATE_COV_aug_.llt().matrixL();  // augmented square root matrix
  SIG_PT_aug_.col(0) = state_aug_;
  for (int i = 0; i < state_dim_aug_; i++) {
    SIG_PT_aug_.col(i+1) = state_aug_ + std::sqrt(lambda_+state_dim_aug_) * SQR_C_aug.col(i);
    SIG_PT_aug_.col(i+1+state_dim_aug_) = state_aug_ - std::sqrt(lambda_+state_dim_aug_) * SQR_C_aug.col(i);
  }
  
  // SIGMA POINT PREDICTION
  VectorXd vec1 = VectorXd(5);
  VectorXd vec2 = VectorXd(5);
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd calc_col = SIG_PT_aug_.col(i);
    double px = calc_col(0);
    double py = calc_col(1);
    double v = calc_col(2);
    double yaw = calc_col(3);
    double yawd = calc_col(4);
    double v_aug = calc_col(5);
    double v_yawdd = calc_col(6);
    VectorXd orig = calc_col.head(5);  // original
    
    // setting vec1 and vec2
    if (fabs(yawd) > .001) {  // yawd not zero
      vec1 << (v/yawd)*(sin(yaw+yawd*delta_t) - sin(yaw)),
              (v/yawd)*(-cos(yaw+yawd*delta_t) + cos(yaw)),
              0,
              yawd * delta_t,
              0;
    } else {  // yawd zero
      vec1 << v*cos(yaw)*delta_t,
              v*sin(yaw)*delta_t,
              0,
              yawd*delta_t,
              0;
    }
    vec2 << .5 * delta_t * delta_t*cos(yaw) * v_aug,
            .5 * delta_t * delta_t*sin(yaw) * v_aug,
            delta_t * v_aug,
            .5 * delta_t * delta_t * v_yawdd,
            delta_t * v_yawdd;

    SIG_PT_pred_.col(i) << orig + vec1 + vec2;  // populate columns with predicted sigma points
  }
  
  // STATE AND STATE COVARIANCE PREDICTION AND UPDATING
  VectorXd state_pred = VectorXd(state_dim_);
  state_pred.fill(0.0);
  MatrixXd STATE_COV_pred = MatrixXd(state_dim_, state_dim_);
  STATE_COV_pred.fill(0.0);
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    // setting weights
    if (i == 0) {
      weights_(i) = lambda_ / (lambda_ + state_dim_aug_);
    } else {
      weights_(i) = .5 / (lambda_ + state_dim_aug_);
    }

    // predicting state mean
    state_pred += weights_(i) * SIG_PT_pred_.col(i);
  }
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd x_diff = SIG_PT_pred_.col(i) - state_pred;
    // normalising angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;

    // predicting state covariance
    STATE_COV_pred += weights_(i) * x_diff * x_diff.transpose();
  }
  
  // updating state and state covariance
  state_ = state_pred;
  STATE_COV_ = STATE_COV_pred;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /* Updates the belief about the object's position using a LiDAR measurement. 
      Modifies the state vector and covariance
      Calculate the lidar NIS */

  int meas_dim = 2;  // measurement dimensions - LiDAR: px, py

  VectorXd meas_pred = VectorXd(meas_dim);  // mean predicted measurement
  meas_pred.fill(0.0);
  MatrixXd MEAS_COV = MatrixXd(meas_dim, meas_dim);  // measurement covariance matrix
  MEAS_COV.fill(0.0);
  MatrixXd SIG_PT_meas = MatrixXd(meas_dim, 2 * state_dim_aug_ + 1);  // matrix for sigma points in measurement space
  SIG_PT_meas.fill(0.0);
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    // transforming sigma points into measurement space
    VectorXd state_vec = SIG_PT_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    SIG_PT_meas.col(i) << px,
                          py;
    
    // calculating mean predicted measurement
    meas_pred += weights_(i) * SIG_PT_meas.col(i);
  }
  
  // calculating measurement covariance matrix
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd z_diff = SIG_PT_meas.col(i) - meas_pred;
    MEAS_COV += weights_(i) * z_diff * z_diff.transpose();
  }
  MatrixXd R = MatrixXd(2, 2);
  R << std_lidpx_*std_lidpx_, 0,
       0, std_lidpy_*std_lidpy_;
  MEAS_COV += R;  // adding R to MEAS_COV
  
  // processing incoming lidar measurement
  VectorXd z = VectorXd(meas_dim);
  double meas_px = meas_package.raw_measurements_(0);
  double meas_py = meas_package.raw_measurements_(1);
  z << meas_px,
       meas_py;
  
  MatrixXd Tc = MatrixXd(state_dim_, meas_dim);  // cross correlation matrix
  Tc.fill(0.0);
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd x_diff = SIG_PT_pred_.col(i) - state_;
    // normalising angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;
    
    VectorXd z_diff = SIG_PT_meas.col(i) - meas_pred;

    // calculating cross correlation matrix
    Tc += weights_(i) * x_diff * z_diff.transpose();  
  }
  
  VectorXd z_diff = z - meas_pred;  // residual
  
  NIS_lidar_ = z_diff.transpose() * MEAS_COV.inverse() * z_diff;  // LiDAR NIS
  
  MatrixXd K = Tc * MEAS_COV.inverse();  // Kalman Gain K
  
  // updating state mean and covariance
  state_ += K * z_diff;
  STATE_COV_ -= K * MEAS_COV * K.transpose();
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /* Uses radar data to update the belief about the object's position. 
      Updates the state vector and covariance
      Calculates radar NIS */
  
  int meas_dim = 3;  // radar: r, phi, and r_dot
  
  MatrixXd SIG_PT_meas = MatrixXd(meas_dim, 2 * state_dim_aug_ + 1);  // sigma points in measurement space
  SIG_PT_meas.fill(0.0);

  VectorXd meas_pred = VectorXd(meas_dim);  // mean predicted measurement
  meas_pred.fill(0.0);

  MatrixXd MEAS_COV = MatrixXd(meas_dim, meas_dim);  // measurement covariance matrix
  MEAS_COV.fill(0.0);
  
  double rho = 0;
  double phi = 0;
  double rho_d = 0;
  
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    // transforming sigma points into measurement space
    VectorXd state_vec = SIG_PT_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    double v = state_vec(2);
    double yaw = state_vec(3);
    double yaw_d = state_vec(4);
    
    rho = sqrt(px * px + py * py);
    phi = atan2(py, px);
    rho_d = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;
    
    SIG_PT_meas.col(i) << rho,
                          phi,
                          rho_d;
    
    // calculating mean predicted measurement
    meas_pred += weights_(i) * SIG_PT_meas.col(i);
  }
  
  // calculating measurement covariance matrix
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd z_diff = SIG_PT_meas.col(i) - meas_pred;
    while (z_diff(1) > M_PI) 
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < - M_PI) 
      z_diff(1) += 2. * M_PI;
    
    MEAS_COV += weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  MEAS_COV += R;  // adding R
  
  // processing incoming radar measurement
  VectorXd z = VectorXd(meas_dim);
  double meas_rho = meas_package.raw_measurements_(0);
  double meas_phi = meas_package.raw_measurements_(1);
  double meas_rhod = meas_package.raw_measurements_(2);
  z << meas_rho,
       meas_phi,
       meas_rhod;
  
  MatrixXd Tc = MatrixXd(state_dim_, meas_dim);  // cross correlation matrix
  Tc.fill(0.0);
  
  // calculating cross correlation matrix
  for (int i = 0; i < 2 * state_dim_aug_ + 1; i++) {
    VectorXd x_diff = SIG_PT_pred_.col(i) - state_;
    // normalising angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;
    
    VectorXd z_diff = SIG_PT_meas.col(i) - meas_pred;
    //normalising angles
    while (z_diff(1) > M_PI) 
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) 
      z_diff(1) += 2. * M_PI;
  
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  VectorXd z_diff = z - meas_pred;  // residual
  
  // normalising angles
  while (z_diff(1) > M_PI) 
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) 
    z_diff(1) += 2. * M_PI;
  
  NIS_radar_ = z_diff.transpose() * MEAS_COV.inverse() * z_diff;  // radar NIS
  
  MatrixXd K = Tc * MEAS_COV.inverse();  // Kalman Gain K
  
  // updating state mean and covariance matrix
  state_ += K * z_diff;
  STATE_COV_ -= K * MEAS_COV * K.transpose();
}
