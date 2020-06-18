data {
    int<lower=0> N_m;            // number of measurements
    int<lower=0> N_s;            // number of samples per measurement
    vector[N_m] y;              // measurement results before change
}

parameters {
    real p1;
    real p2;
}

model {

}