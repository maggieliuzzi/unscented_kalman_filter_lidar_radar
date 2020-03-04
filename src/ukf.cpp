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
  // state dimension
  n_x_ = 5;
  // initial state
  x_ = VectorXd(n_x_);
  // initial state covariance
  P_ = MatrixXd(n_x_, n_x_);

  // PROCESS NOISE
  // Process noise std deviation longitudinal/linear acceleration, m/s^2
  std_a_ = 2.31;  // TOTUNE  // researchgate.net/publication/223922575_Design_speeds_and_acceleration_characteristics_of_bicycle_traffic_for_use_in_planning_design_and_appraisal
  // Process noise std deviation yaw/angular acceleration, rad/s^2
  std_yawdd_ = 1.0;  // TOTUNE
  
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
  n_aug_ = 7;
  lambda_ = 0;
  weights_ = VectorXd(2*n_aug_+1);  // weights vector
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);  // sigma points matrix
  
  // NIS
  NIS_lidar_ = 0;
  NIS_radar_ = 0;
}


UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // INITIALISATION of state (x_) and state covariance (P_)
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

      x_ << x, y, v, rho, rhodot;
      
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::LIDAR) {

      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0.0;  // TOTUNE: last values
      
      // TOTUNE
      P_ << std_lidpx_*std_lidpx_, 0, 0, 0, 0,
            0, std_lidpy_*std_lidpy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    
    state_initialised_ = true;
    time_us_ = meas_package.timestamp_;
    return;  // no prediction nor updating during initialisation
  }
  
  // if state already initialised

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
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
  // Estimating object location. 
  // Modifies the state vector, predicts sigma points, the state, and the state covariance matrix.

  lambda_ = 3 - n_x_;  // spreading parameter
  MatrixXd Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);  // sigma point matrix
  
  MatrixXd SQR_C = P_.llt().matrixL();  // square root of P
  
  // calculate sigma points and set them as matrix columns
  Xsig_.col(0) = x_;
  for (int i = 0; i < n_x_; i++) {
    Xsig_.col(i+1) = x_ + std::sqrt(lambda_+n_x_) * SQR_C.col(i);
    Xsig_.col(i+1+n_x_) = x_ - std::sqrt(lambda_+n_x_) * SQR_C.col(i);
  }
  
  lambda_ = 3 - n_aug_;  // spreading parameter for augmentation
  
  VectorXd x_aug_ = VectorXd(n_aug_);  // augmented mean vector
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);  // augmented state covariance
  
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);  // another sigma point matrix
  
  // augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  
  // augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_.bottomRightCorner(2, 2) = Q;
   
  MatrixXd SQR_C_aug = P_aug_.llt().matrixL();  // augmented square root matrix
  
  // augmented sigma points
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i+1) = x_aug_ + std::sqrt(lambda_+n_aug_) * SQR_C_aug.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - std::sqrt(lambda_+n_aug_) * SQR_C_aug.col(i);
  }
  
  // SIGMA POINT PREDICTION
  // x + vectors for each part
  VectorXd vec1 = VectorXd(5);  // TODO: neatify
  VectorXd vec2 = VectorXd(5);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd calc_col = Xsig_aug_.col(i);
    double px = calc_col(0);
    double py = calc_col(1);
    double v = calc_col(2);
    double yaw = calc_col(3);
    double yawd = calc_col(4);
    double v_aug = calc_col(5);
    double v_yawdd = calc_col(6);
    
    // original
    VectorXd orig = calc_col.head(5);
    
    if (fabs(yawd) > .001) { // yawd not zero
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
    
    // unchanged
    vec2 << .5 * delta_t * delta_t*cos(yaw) * v_aug,
            .5 * delta_t * delta_t*sin(yaw) * v_aug,
            delta_t * v_aug,
            .5 * delta_t * delta_t * v_yawdd,
            delta_t * v_yawdd;

    Xsig_pred_.col(i) << orig + vec1 + vec2;  // populate columns with predicted sigma points
  }
  

  VectorXd x_pred = VectorXd(n_x_);  // predicted state
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);  // predicted covariance matrix
  x_pred.fill(0.0);
  P_pred.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    // set weights
    if (i == 0) {
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    } else {
      weights_(i) = .5 / (lambda_ + n_aug_);
    }
    
    // predict state mean
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    // predict state covariance matrix
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    
    // normalise angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;
    
    P_pred += weights_(i) * x_diff * x_diff.transpose();
  }
  
  x_ = x_pred;
  P_ = P_pred;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Updates the belief about the object's position using a LiDAR measurement. 
  // Modifies the state vector and covariance
  // Calculate the lidar NIS

  int n_z = 2;  // measurement dimension - LiDAR: px, py   ----------------------------------------------------------
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //transform sigma points into measurement space
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    
    Zsig.col(i) << px,
                   py;
    
    //calculate mean predicted measurement
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  // Add R to S
  MatrixXd R = MatrixXd(2,2);
  R << std_lidpx_*std_lidpx_, 0,
       0, std_lidpy_*std_lidpy_;
  S += R;
  
  //create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  
  double meas_px = meas_package.raw_measurements_(0);
  double meas_py = meas_package.raw_measurements_(1);
  
  z << meas_px,
       meas_py;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //normalize angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;
    
    
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc += weights_(i) * x_diff * z_diff.transpose();

  }
  
  // residual
  VectorXd z_diff = z - z_pred;
  
  //calculate NIS
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
    //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);
  double rho = 0;
  double phi = 0;
  double rho_d = 0;
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //transform sigma points into measurement space
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    double v = state_vec(2);
    double yaw = state_vec(3);
    double yaw_d = state_vec(4);
    
    rho = sqrt(px*px+py*py);
    phi = atan2(py,px);
    rho_d = (px*cos(yaw)*v+py*sin(yaw)*v) / rho;
    
    Zsig.col(i) << rho,
                   phi,
                   rho_d;
    
    //calculate mean predicted measurement
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) 
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < - M_PI) 
      z_diff(1) += 2. * M_PI;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  // Add R to S
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S += R;
  
  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  
  double meas_rho = meas_package.raw_measurements_(0);
  double meas_phi = meas_package.raw_measurements_(1);
  double meas_rhod = meas_package.raw_measurements_(2);
  
  z << meas_rho,
       meas_phi,
       meas_rhod;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    while (x_diff(3) > M_PI) 
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) 
      x_diff(3) += 2. * M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //normalize angles
    while (z_diff(1) > M_PI) 
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) 
      z_diff(1) += 2. * M_PI;
  
    Tc += weights_(i) * x_diff * z_diff.transpose();
    
  }
  
  // residual
  VectorXd z_diff = z - z_pred;
  
  //normalize angles
  while (z_diff(1) > M_PI) 
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) 
    z_diff(1) += 2. * M_PI;
  
  
  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();
}
