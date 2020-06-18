data {
    int<lower=0> N_1;         // number of measurements before
    int<lower=0> N_2;         // number of measurements after
    int<lower=0> N_s;         // number of samples per measurement
    int y1[N_1];              // measurement results before change
    int y2[N_2];              // measurement results after the change
}

parameters {
    real p1;
    real p2;
}

model {
   for (t1 in 1:N_1) {
       y1[t1] ~ binomial(N_s, p1);
   }
   for (t2 in 1:N_2) {
       y2[t2] ~ binomial(N_s, p2);
   }
}