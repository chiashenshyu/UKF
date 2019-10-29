#include "ukf.hpp"

/**
 * private
 */
void UKF::generateSigmaPoints(const Eigen::MatrixXd& u){
    Eigen::MatrixXd L = P.llt().matrixL(); 
    for(int i = 0; i < nStateDim; i++){
        sigmaPoints.col(i) = xm + L.col(i);
        sigmaPoints.col(i+nStateDim) = xm - L.col(i);
    }
    stateFunctionSigmaPoints(u); 
}

void UKF::stateFunctionSigmaPoints(const Eigen::MatrixXd& u){
    for(int i = 0; i < sigmaPoints.cols(); i++){
        Eigen::MatrixXd x = sigmaPoints.col(i);
        stateFunction(x, u);
        sigmaPoints.col(i) = x; 
    }
}

void UKF::stateFunction(Eigen::MatrixXd& x, const Eigen::MatrixXd& u){
    Eigen::MatrixXd B(3,2);
    B << cos(x(2)),0, sin(x(2)),0, 0,1;
    x = x + B * u * dt; 
}

/** 
 * public
 */

UKF::UKF(){
    dt = 0.1; 
    nStateDim = 3; 
    sigmaPoints.resize(nStateDim, nStateDim*2);
    // P.setIdentity(nStateDim); 
    // R.setIdentity(nStateDim);
}

UKF::~UKF(){

}

void UKF::priorUpdate(){

}

void UKF::posterioriUpdate(){

}

int main(){
    Eigen::MatrixXd A(2,3);
    A << 4,-1,2, -1,6,0;
    cout << "The matrix A is" << endl << A << endl;
    A = A * 2; 
    cout << "The matrix A is" << endl << A << endl;
    cout << A.cols() << endl;
    return 0; 
}