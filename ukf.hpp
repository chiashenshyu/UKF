#pragma once 
#include "common.hpp"

/**
 * systems where the noise is purely additive - working on 
 * systens where the noise enters in a nonlinear fashion - future work
 * 
 * step:
 * 0. setInitialCondition
 * 1. priorUpdate
 * 2. posterioriUpdate
 * 3. getEstimation
 */
class UKF{
private:
    int nStateDim; // state dimension    
    double dt; // delta time 
    /**
     * xm : state after measurement update
     * xp : state after prior update
     * z  : input measurement 
     * sigmaPointsP: sigma points after prior update
     * sigmaPointsZ: sigma points after measurement update
     * P, Pxz, Pzz : covariance matrix
     * Q, R : noise matrices, can be time variant 
     */
    Eigen::MatrixXd xm, xp, z;       
    Eigen::MatrixXd sigmaPointsP, sigmaPointsZ; 
    Eigen::MatrixXd P, Q, R, Pzz, Pxz;
    
    void generateSigmaPoints(const Eigen::MatrixXd& u); 
    void stateFunctionSigmaPoints(const Eigen::MatrixXd& u); 
    
    void measurementFunctionSigmaPoints(); 
    void measurementFunction(Eigen::MatrixXd& sz);  

public:
    UKF(); 
    ~UKF(); 

    void setInitialCondition(const Eigen::MatrixXd& x);
    void priorUpdate(const Eigen::MatrixXd& u);
    void posterioriUpdate(const Eigen::MatrixXd& z); 
    void getEstimation(Eigen::MatrixXd& est);
    void stateFunction(Eigen::MatrixXd& x, const Eigen::MatrixXd& u);

};