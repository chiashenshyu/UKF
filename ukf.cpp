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
    for(int i = 0; i < sigmaPointsP.cols(); i++){
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
    for(int i = 0; i < sigmaPointsP.cols(); i++){
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
    sigmaPointsZ.resize(nStateDim, nStateDim*2);
    xp.resize(nStateDim, 1); 
    xm.resize(nStateDim, 1); 
    P.setIdentity(nStateDim, nStateDim);
    Q.setIdentity(nStateDim, nStateDim);  
    R.setIdentity(nStateDim, nStateDim);
}

UKF::~UKF(){

}

void UKF::setInitialCondition(const Eigen::MatrixXd& x){
    xm = x; 
}

void UKF::priorUpdate(const Eigen::MatrixXd& u){
    generateSigmaPoints(u); 
    xp = sigmaPointsP.rowwise().mean();
    int n = sigmaPointsP.cols();
    P = Eigen::MatrixXd::Zero(nStateDim,nStateDim); 
    for(int i = 0; i < n; i++){
        Eigen::MatrixXd xd = sigmaPointsP.col(i) - xp;
        P = P + ((xd * xd.transpose()) / static_cast<double>(2 * n)); 
    }
    P = P + Q; 
}

void UKF::posterioriUpdate(const Eigen::MatrixXd& z){
    measurementFunctionSigmaPoints(); 
    Eigen::MatrixXd zBar = sigmaPointsZ.rowwise().mean(); 
    int n = sigmaPointsP.cols(); 
    Pzz = Eigen::MatrixXd::Zero(nStateDim,nStateDim); 
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

void UKF::getEstimation(Eigen::MatrixXd& est){
    est = xm; 
}

int main(){
    bool init = false; 
    UKF ukf; 
    Eigen::MatrixXd a(3,1), u(2,1), est(3,1); 
    a << 0,0,0; 
    u << 3, M_PI/180*5;
    ukf.setInitialCondition(a);

    std::vector<double> x, y, estX, estY;

    for(int i = 1; i < 100; i++){
        if(init) ukf.stateFunction(a, u); 
        ukf.priorUpdate(u); 
        ukf.posterioriUpdate(a); 
        ukf.getEstimation(est);
        init = true;

        x.push_back(a(0,0)); 
        y.push_back(a(1,0));
        estX.push_back(est(0,0));
        estY.push_back(est(1,0)); 

        cout << "difference in x: " << abs(a(0,0)-est(0,0));
        cout << ", in y: " << abs(a(1,0)-est(1,0)) << endl;
        plt::clf();
        plt::xlim(0,30); 
        plt::ylim(0,18);
        plt::plot(x,y, "r*");
        plt::plot(estX, estY, "bo");
        plt::pause(0.001); 
        
    }
    plt::show();
    return 0; 
}