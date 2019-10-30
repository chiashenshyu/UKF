#include "ukf.hpp"

/**
 * private
 */
void UKF::generateSigmaPoints(const Eigen::MatrixXd& u){
    Eigen::MatrixXd L = P.llt().matrixL(); 
    for(int i = 0; i < nStateDim; i++){
        sigmaPointsP.col(i) = xm + L.col(i);
        sigmaPointsP.col(i+nStateDim) = xm - L.col(i);
    }
    stateFunctionSigmaPoints(u); 
}

void UKF::stateFunctionSigmaPoints(const Eigen::MatrixXd& u){
    for(int i = 0; i < sigmaPoints.cols(); i++){
        Eigen::MatrixXd sx = sigmaPointsP.col(i);
        stateFunction(sx, u);
        sigmaPointsP.col(i) = sx; 
    }
}

void UKF::stateFunction(Eigen::MatrixXd& sx, const Eigen::MatrixXd& u){
    Eigen::MatrixXd B(3,2);
    B << cos(sx(2)),0, sin(sx(2)),0, 0,1;
    sx = sx + B * u * dt; 
}

void UKF::measurementFunctionSigmaPoints(){
    for(int i = 0; i < sigmaPoints.cols(); i++){
        Eigen::MatrixXd sz = sigmaPointsP.col(i); 
        measurementFunction(sz); 
        sigmaPointsZ.col(i) = sz;         
    }
}

void UKF::measurementFunction(Eigen::MatrixXd& sz){
    Eigen::MatrixXd A;
    A.setIdentity(sz.rows(), sz.rows());
    sz = A * sz; 
}   

/** 
 * public
 */

UKF::UKF(){
    dt = 0.1; 
    nStateDim = 3; 
    sigmaPointsP.resize(nStateDim, nStateDim*2);
    sigmaPointsP.resize(nStateDim, nStateDim*2);
    xp.resize(nStateDim, 1); 
    xm.resize(nStateDim, 1); 
    P.setIdentity(nStateDim, nStateDim);
    Q.setIdentity(nStateDim, nStateDim);  
    R.setIdentity(nStateDim, nStateDim);
}

UKF::~UKF(){

}

void UKF::priorUpdate(const Eigen::MatrixXd& u){
    generateSigmaPoints(u); 
    xp = sigmaPointsP.rowwise().mean();
    int n = sigmaPointsP.cols();
    P = Eigen::MatrixXd::Zero(n,n); 
    for(int i = 0; i < n; i++){
        Eigen::MatrixXd xd = sigmaPointsP.col(i) - xp;
        P = P + ((xd * xd.transpose()) / 2 / n); 
    }
    P = P + Q; 
}

void UKF::posterioriUpdate(const Eigen::MatrixXd& z){
    measurementFunctionSigmaPoints(); 
    zBar = sigmaPointsZ.rowwise().mean(); 
    int n = sigmaPointsP.cols(); 
    Pzz = Eigen::MatrixXd::Zero(n,n); 
    Pxz = Pzz; 
    for(int i = 0; i < n; i++){
        Eigen::MatrixXd zd = sigmaPointsZ.col(i) - zBar; 
        Eigen::MatrixXd xd = sigmaPointsP.col(i) - xp;
        Pzz = Pzz + ((zd * zd.transpose()) / 2 / n);
        Pxz = Pxz + ((xd * zd.transpose()) / 2 / n);
    }
    Pzz = Pzz + R;  
    Eigen::MatrixXd K = Pxz * Pzz.inverse(); 

    xm = xp + K * (z - zBar);
    P = P - K * Pzz * K.transpose();
}

int main(){
    Eigen::MatrixXd A(2,3);
    // A << 4,-1,2, -1,6,0;
    // cout << "The matrix A is" << endl << A << endl;
    // A = A * 2; 
    // cout << "The matrix A is" << endl << A << endl;
    // cout << A.rowwise().mean() << endl;
    A = Eigen::MatrixXd::Zero(2,2);
    A.setIdentity(3,3); 
    cout << A << endl;
    return 0; 
}