#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


/* Initializes Unscented Kalman filter */
UKF::UKF() {

  /* Sensors used */  // if false, measurements will be ignored (except during init)
  use_laser_ = true;  // TOTRY: compare results with only one sensor
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.231;  // TOTRY: different  // Orig: 30  // Based on researchgate.net/publication/223922575_Design_speeds_and_acceleration_characteristics_of_bicycle_traffic_for_use_in_planning_design_and_appraisal
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;  // TOTRY: different  // Orig: 30
  
  /* Measurement noise values provided by the sensor manufacturer. Do not modify. */
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;  // TOTRY: 0.0175
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;  // TOTRY: 0.1


  // TODO: Complete the initialization. See ukf.h for other member properties. // Hint: one or more values initialized above might be wildly off...
  is_initialized_ = false;


  // State dimension
  int n_x_ = 5;

  // initial state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd(n_x_);  // Orig: 5
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);  // Orig: 5, 5
  // Augmented state dimension
  int n_aug_ = 7;

  // Sigma point spreading parameter for augmentation
  double lambda_ = 3 - n_aug_;

  // Time when the state is true, in us
  long long time_us_;  //// = ?



  VectorXd x_aug_ = VectorXd(n_aug_);  // augmented mean vector
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);  // augmented state covariance

  MatrixXd Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);  // sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);  // predicted sigma points as columns

  VectorXd weights_ = VectorXd(2*n_aug_+1);  // vector for weights
  VectorXd x_pred_ = VectorXd(n_x_);  // vector for predicted state
  MatrixXd P_pred_ = MatrixXd(n_x_, n_x_);  // covariance matrix for prediction

  int n_z_ = 3;
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);  // sigma points in measurement space
  VectorXd z_pred_ = VectorXd(n_z_);  // mean predicted measurement
  MatrixXd S_ = MatrixXd(n_z_, n_z_);  // measurement covariance matrix S
  MatrixXd R_ = MatrixXd(n_z_, n_z_); // measurement noise covariance matrix

}


UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // TODO: Complete this function! Make sure you switch between lidar and radar measurements.

  // if all positions of Vector have been initialised
  // is_initialized_ = true;

  if (is_initialized_ == false)  // if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == meas_package.SensorType::LASER)  // LiDAR
    {
      // state
      x_[0] = meas_package.raw_measurements_[0];  // px
      x_[1] = meas_package.raw_measurements_[1];  // py
      x_[2] = 0;  // TOTRY: diff value
      x_[3] = 0;  // TOTRY: diff value
      x_[4] = 0;  // TOTRY: diff value

      // covariance
      P_ << std_laspx_*std_laspx_,                     0, 0, 0, 0,
                                0, std_laspy_*std_laspy_, 0, 0, 0,
                                0,                     0, 1, 0, 0,
                                0,                     0, 0, 1, 0,
                                0,                     0, 0, 0, 1;
    }
    else if (meas_package.sensor_type_ == meas_package.SensorType::RADAR)
    {
      // state
      double rho = meas_package.raw_measurements_[0];  // range
      double phi = meas_package.raw_measurements_[1];  // bearing
      double rho_dot = meas_package.raw_measurements_[2];  // velocity

      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      x_ << x, y, v, rho, rho_dot;  // TOTRY: x, y, v, vx, vy and x, y, v, 0, 0, tune v

      // covariance
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,  // TOTRY: 0, 0, 1, 0, 0
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
    }
    
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  }
  else // is_initialized_ == true  // TODO: consider adding return to if and omit else
  {
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;  // diff in seconds between this and previous measurement timestamps
    time_us_ = meas_package.timestamp_;

    // Prediction
    Prediction(delta_t);

    // Measurement update
    if (meas_package.sensor_type_ == meas_package.SensorType::LASER && use_laser_)
    {
      UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == meas_package.SensorType::RADAR && use_radar_)
    {
      UpdateRadar(meas_package);
    }
  }
}


void UKF::Prediction(double delta_t) {  // eg. delta_t = 0.1; time diff in sec
  // TODO: Complete this function! Estimate the object's location. 
  // Modify the state vector (x_).
  // Predict sigma points, the state, and the state covariance matrix.


  // augment mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  // augment covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_aug_-2, n_aug_-2) = std_a_*std_a_;
  P_aug_(n_aug_-1, n_aug_-1) = std_yawdd_*std_yawdd_;


  // TOTRY: check order - if this before the augmentation, P_ and Xsig_, Xsig_aug, n_x_. to try, copy from kalman_filter repo
  MatrixXd sq_root = P_aug_.llt().matrixL();  // square root of P
  Xsig_aug_.col(0) = x_aug_;  // set first column of sigma point matrix
  for (int i = 0; i < n_aug_; ++i) {  // calculate and set remaining sigma points as columns of Xsig_
    Xsig_aug_.col(i+1)        = x_aug_ + sqrt(lambda_+n_aug_) * sq_root.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * sq_root.col(i);
  }


  // predict sigma points
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    // predicted states
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (-cos(yaw) + cos(yaw+yawd*delta_t) );  // TOTRY: cos -cos
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // modifying state vector; writing predicted sigma point into right column
    x_(0,i) = px_p;
    x_(1,i) = py_p;
    x_(2,i) = v_p;
    x_(3,i) = yaw_p;
    x_(4,i) = yawd_p;
  }

  // set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i = 1; i < 2*n_aug_+1; ++i) {
    weights_(i) = 0.5/(n_aug_+lambda_);
  }

  // predicted state mean
  x_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_pred_ = x_pred_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
    P_pred_ = P_pred_ + weights_(i) * x_diff * x_diff.transpose();
  }

  // modifying state vector
  x_ = x_pred_;
  P_ = P_pred_;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // TODO: Complete this function! Use lidar data to update the belief about the object's position. 
  // Modify the state vector (x_), and covariance (P_).
  // You can also calculate the lidar NIS, if desired.


}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // TODO: Complete this function! Use radar data to update the belief about the object's position. 
  // Modify the state vector (x_), and covariance (P_).
  // You can also calculate the radar NIS, if desired.


  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights_(i) = weight;
  }

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig_(1,i) = atan2(p_y,p_x);                                // phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  // innovation covariance matrix S
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R_ <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0,std_radrd_ * std_radrd_;
  S_ = S_ + R_;

}
