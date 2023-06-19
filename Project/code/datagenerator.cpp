#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

// Define limits and G
const double G = 9.81; // acceleration due to gravity in m/s^2
const double min_L = 0.5; //minimum 50cm
const double max_L = 1; //maximum 100cm
const double min_M = 0.25; //minimum 0.5kg
const double max_M = 0.5; //maximum 1kg
const double min_t = -M_PI/2; //minimum -pi/2 radians
const double max_t = M_PI/2; //maximum pi/2 radians
const double min_o = -0.05; //minimum angular velocity
const double max_o = 0.05; //maximum angular velocity

// Globals
double L1;
double L2;
double M1;
double M2;
double o1;
double o2;
double t1;
double t2;

// Define function to calculate the derivatives of the system of equations
void derivatives(double t, double y[], double dydx[]) {
    double theta1 = y[0];
    double theta2 = y[1];
    double omega1 = y[2];
    double omega2 = y[3];
    
    // Calculate some intermediate values
    double c1 = cos(theta1);
    double c2 = cos(theta2);
    double s1 = sin(theta1);
    double s2 = sin(theta2);
    double s12 = sin(theta1 - theta2);
    double c12 = cos(theta1 - theta2);
    
    // Calculate the derivatives  
    dydx[0] = omega1;
    dydx[1] = omega2;
    dydx[2] = (-G*(2*M1+M2)*s1-M2*G*sin(theta1-2*theta2)-2*s12*M2*(omega2*omega2*L2+c12))/(L1*(2*M1+M2-M2*cos(2*theta1-2*theta2)));
    dydx[3] = (2*s12*(omega1*omega1*L1*(M1+M2)+G*(M1+M2)*c1+omega2*omega2*L2*M2*c12))/(L2*(2*M1+M2-M2*cos(2*theta1-2*theta2)));
}

// Define the Runge-Kutta 4 method
void runge_kutta(double t0, double y[], double h, int n) {
    double k1[n], k2[n], k3[n], k4[n];
    double y_temp[n];
    
    derivatives(t0, y, k1);
    for (int i=0; i<n; i++) {
        y_temp[i] = y[i] + k1[i] * h/2.0;
    }
    derivatives(t0 + h/2.0, y_temp, k2);
    for (int i=0; i<n; i++) {
        y_temp[i] = y[i] + k2[i] * h/2.0;
    }
    derivatives(t0 + h/2.0, y_temp, k3);
    for (int i=0; i<n; i++) {
        y_temp[i] = y[i] + k3[i] * h;
    }
    derivatives(t0 + h, y_temp, k4);
    
    for (int i=0; i<n; i++) {
        y[i] = y[i] + (h/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
}

bool isValidRun(double y[], int n) { // Cull NaN or inf values
    for (int i = 0; i < n; i++) {
        if (std::isnan(y[i]) || std::isinf(y[i])) {
            return false;
        }
    }
    return true;
}

int main() {
    // Set initial conditions
    srand(time(NULL));
    double h = 0.005;
    int n = 4; // number of variables in the system
    std::ofstream outfile;
    outfile.open("../data/double_pendulum.txt");
    std::ofstream initialsFile;
    initialsFile.open("../data/initials.txt");

    for (int i = 0; i < 10000; i++) {
        double t = 0.0;
        L1 = min_L + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_L - min_L))); // length of pendulum 1 in m
        L2 = min_L + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_L - min_L))); // length of pendulum 2 in m
        M1 = min_M + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_M - min_M))); // mass of pendulum 1 in kg
        M2 = min_M + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_M - min_M))); // mass of pendulum 2 in kg
        o1 = min_o + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_o - min_o))); // angular velocity of pendulum 1 in m/s
        o2 = min_o + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_o - min_o))); // angular velocity of pendulum 2 in m/s
        t1 = min_t + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_t - min_t))); // angle of pendulum 1 in radians relative to the vertical
        t2 = min_t + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_t - min_t))); // angle of pendulum 2 in radians relative to the vertical
        double y[n] = {t1, t2, o1, o2}; // initial values of theta1, theta2, omega1, and omega2
        

        // Numerically solve the system and output the results
        std::vector<double> timeSteps;
        std::vector<double> theta1Vals;
        std::vector<double> theta2Vals;

        for (int i = 0; i < 1000; i++) {
            runge_kutta(t, y, h, n);
            if (!isValidRun(y, n)) {
                timeSteps.clear();
                theta1Vals.clear();
                theta2Vals.clear();
                break;
            }
            
            timeSteps.push_back(t);
            theta1Vals.push_back(y[0]);
            theta2Vals.push_back(y[1]);

            t += h;
        }

        if (!timeSteps.empty()) {
            for (int j = 0; j < timeSteps.size(); j++) {
                outfile << timeSteps[j] << " " << theta1Vals[j] << " " << theta2Vals[j] << std::endl;
            }
            initialsFile << L1 << " " << L2 << " " << M1 << " " << M2 << " " << o1 << " " << o2
                     << " " << t1 << " " << t2 << std::endl;
        }
    }
    
    outfile.close();
    initialsFile.close();

    return 0;
}
