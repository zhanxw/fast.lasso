// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

#if 0
// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
Eigen::MatrixXd rcppeigen_hello_world() {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(3, 3);

  return m1 + 3 * (m1 + m2);
}


// another simple example: outer product of a vector,
// returning a matrix
//
// [[Rcpp::export]]
Eigen::MatrixXd rcppeigen_outerproduct(const Eigen::VectorXd & x) {
  Eigen::MatrixXd m = x * x.transpose();
  return m;
}

// and the inner product returns a scalar
//
// [[Rcpp::export]]
double rcppeigen_innerproduct(const Eigen::VectorXd & x) {
  double v = x.transpose() * x;
  return v;
}

// and we can use Rcpp::List to return both at the same time
//
// [[Rcpp::export]]
Rcpp::List rcppeigen_bothproducts(const Eigen::VectorXd & x) {
  Eigen::MatrixXd op = x * x.transpose();
  double          ip = x.transpose() * x;
  return Rcpp::List::create(Rcpp::Named("outer")=op,
                            Rcpp::Named("inner")=ip);
}
# endif

void print(const char* str,
           const Eigen::MatrixXd& m) {
  Rprintf("%s\n", str);
  for (int i = 0; i < m.rows(); i ++) {
    for (int j = 0; j < m.cols(); j++ ) {
      Rprintf(" %g", m(i,j));
    }
    Rprintf("\n");
  }
}

void print(const char* str,
           const Eigen::VectorXd& m) {
  Rprintf("%s\n", str);
  for (int i = 0; i < m.size(); i ++) {
      Rprintf(" %g", m(i));
  }
    Rprintf("\n");
}

// rewrite lasso.sum.ess
// returning a matrix
//
// [[Rcpp::export]]
Eigen::MatrixXd fast_lasso_sum_ess(const Eigen::VectorXd& bVec,
                                   const Eigen::VectorXd& sVec,
                                   const Eigen::MatrixXd& r2Mat,
                                   int n,
                                   const Eigen::VectorXi& group,
                                   const Eigen::VectorXd& lambda,
                                   const Eigen::VectorXd& alpha) {


  // z.vec <- b.vec/s.vec;
  Eigen::VectorXd zVec = bVec.array() / sVec.array();
  // u.vec <- b.vec/s.vec^2;
  Eigen::VectorXd uVec = bVec.array() / sVec.array().square();
  // v.vec <- 1/s.vec;
  Eigen::VectorXd vVec = sVec.array().inverse();

  // cov.mat <- cor2cov(r2.mat,v.vec);
  Eigen::MatrixXd covMat = r2Mat;
  covMat.array().colwise() *= vVec.array();
  covMat.array().rowwise() *= vVec.transpose().array();
  // Eigen::MatrixXd covMat = vVec.asDiagonal() * r2Mat * vVec.asDiagonal();

  // print("vVec = ", vVec);
  // print("r2Mat = ", r2Mat);  
  // print("covMat = ", covMat);
  // // #beta.vec <- ginv(cov.mat)%*%u.vec;
  // beta.vec <- b.vec;
  Eigen::VectorXd betaVec = bVec;
  // print("betaVec = ", betaVec);
  // beta0.vec <- rep(0,length(z.vec));
  const int vecLen = bVec.size();
  Eigen::VectorXd beta0Vec = Eigen::VectorXd::Zero(vecLen);

  double x_j_times_r, x_j_times_x;
  double old_beta_jj;
  double n_times_alpha_group_jj;

  // print("betaVec = ", betaVec);
  
  int maxIter  = 100;
  // while(sum(abs(beta.vec-beta0.vec))>1e-5) {
  while ( (beta0Vec - betaVec).array().square().sum() > 1e-5 && maxIter > 0) {
    // print("betaVec = ", betaVec);
    maxIter --;
    // beta0.vec <- beta.vec;
    beta0Vec = betaVec;
    // for(jj in 1:length(z.vec)) {
    for (int jj = 0; jj < vecLen; ++jj) {
      // ##x_j * r_j = x_j (y - x_j' * beta_j' ) ;
      //x.j.times.r <- u.vec[jj]-sum(cov.mat[jj,-jj]*beta.vec[-jj]);
      old_beta_jj = betaVec(jj);
      betaVec(jj) = 0;
      x_j_times_r = uVec(jj) - (covMat.row(jj) * betaVec).sum();
      // Rprintf("x_j_times_r = %g\n", x_j_times_r);
      
      betaVec(jj) = old_beta_jj;

      // x.j.times.x <- cov.mat[jj,jj];
      x_j_times_x = covMat(jj, jj);
      // Rprintf("x_j_times_x = %g\n", x_j_times_x);
      // if(x.j.times.r > n*alpha[group[jj]] )
      //   beta.vec[jj] <- (x.j.times.r-n*(alpha[group[jj]]))/(x.j.times.x + 2*n*lambda[group[jj]]);
      // if(x.j.times.r < -n*alpha[group[jj]] )
      //   beta.vec[jj] <- (x.j.times.r+n*(alpha[group[jj]]))/(x.j.times.x + 2*n*lambda[group[jj]]);
      // if(x.j.times.r < n*alpha[group[jj]] & x.j.times.r > -n*alpha[group[jj]])
      //   beta.vec[jj] <- 0;
      n_times_alpha_group_jj = alpha( group(jj) - 1) * n;
      // Rprintf("n_times_alpha_group_jj = %g\n", n_times_alpha_group_jj);
      
      if (x_j_times_r > n_times_alpha_group_jj) {
        betaVec(jj) = (x_j_times_r - n_times_alpha_group_jj) / (x_j_times_x + 2.0 * n * lambda( group(jj) - 1) );
      } else if (x_j_times_r <  - n_times_alpha_group_jj) {
        betaVec(jj) = (x_j_times_r + n_times_alpha_group_jj) / (x_j_times_x + 2.0 * n * lambda( group(jj) - 1) );
      } else if (x_j_times_r < n_times_alpha_group_jj && x_j_times_r > -n_times_alpha_group_jj){
        betaVec(jj) = 0;
      } else {
        REprintf("something wrong!\n");
      }
    }
  }
  if (maxIter == 0) {
    REprintf("max iteration reached!!\n");
  }
  // return(beta.vec);
  return betaVec;
}

// sourceCpp('rcppeigen_hello_world.cpp')
