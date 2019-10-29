#pragma once 
#include "common.hpp"

/**
 * systems where the noise is purely additive - working on 
 * systens where the noise enters in a nonlinear fashion - future work
 */
class UKF{
private:
    int nStateDim;    
    double dt; 
    Eigen::MatrixXd xm, xp;       
    Eigen::MatrixXd sigmaPoints; 
    Eigen::MatrixXd P, Q;
    
    void generateSigmaPoints(const Eigen::MatrixXd& u); 
    void stateFunctionSigmaPoints(const Eigen::MatrixXd& u); 
    void stateFunction(Eigen::MatrixXd& x, const Eigen::MatrixXd& u); 
public:
    UKF(); 
    ~UKF(); 

    void priorUpdate();
    void posterioriUpdate(); 

};