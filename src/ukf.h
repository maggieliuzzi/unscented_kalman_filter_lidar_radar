#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or LiDAR
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a LiDAR measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);


  bool use_lidar_;

  bool use_radar_;
  

  // time when the state is true, in us
  long long time_us_;

  bool state_initialised_;


  // state dimension
  int n_x_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate], SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;


  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;


  // LiDAR measurement noise standard deviation, m (px)
  double std_lidpx_;

  // LiDAR measurement noise standard deviation, m (py)
  double std_lidpy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;


  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  
  // current NISs
  double NIS_lidar_;

  double NIS_radar_;
  
};

#endif  // UKF_H
