#' convert correlation to covariance matrix
#' @param r2.mat correlation matrix
#' @param v.vec variance vector
#' @return covariance matrix
#'
#' @export
cor2cov <- function(r2.mat, v.vec) {
  diag(v.vec) %*% r2.mat %*% diag(v.vec)
}

#' Lasso calculation for summary statistics
#'
#' @param b.vec genetic effect estimates
#' @param s.vec standard error of genetic effect estimates;
#' @param r2.mat residual errors;
#' @param n sample size;
#' @param group 1-based group indicator
#' @param lambda l1 penalty parameter
#' @param alpha l2 penalty parameter
#' @return fitted regression coefficient vector
#'
#' @export
lasso.sum.ess <- function(b.vec,s.vec,r2.mat,n,group,lambda,alpha) {
  z.vec <- b.vec/s.vec;
  u.vec <- b.vec/s.vec^2;
  v.vec <- 1/s.vec;
  cov.mat <- cor2cov(r2.mat,v.vec);
  cat('cov.mat = ', cov.mat, '\n')
                                        #beta.vec <- ginv(cov.mat)%*%u.vec;
  beta.vec <- b.vec;
  beta0.vec <- rep(0,length(z.vec));
  while(sum(abs(beta.vec-beta0.vec))>1e-5) {
    cat("beta.vec = ", beta.vec, "\n")
    beta0.vec <- beta.vec;
    for(jj in 1:length(z.vec)) {
      ##x_j * r_j = x_j (y - x_j' * beta_j' ) ;
      x.j.times.r <- u.vec[jj]-sum(cov.mat[jj,-jj]*beta.vec[-jj]);
      x.j.times.x <- cov.mat[jj,jj];
      if(x.j.times.r > n*alpha[group[jj]] )
        beta.vec[jj] <- (x.j.times.r-n*(alpha[group[jj]]))/(x.j.times.x + 2*n*lambda[group[jj]]);
      if(x.j.times.r < -n*alpha[group[jj]] )
        beta.vec[jj] <- (x.j.times.r+n*(alpha[group[jj]]))/(x.j.times.x + 2*n*lambda[group[jj]]);
      if(x.j.times.r < n*alpha[group[jj]] & x.j.times.r > -n*alpha[group[jj]])
        beta.vec[jj] <- 0;
    }
  }
  return(beta.vec);
}

if (FALSE) {
    set.seed(42)
b.vec <- rnorm(3)
s.vec <- rchisq(3, df = 1)
r2.mat <- matrix(rnorm(3^2), 3, 3)
group = c(1, 2, 3)
alpha = rep(0.1, 3)
lambda = rep(0.1, 3)
lasso.sum.ess(b.vec, s.vec, r2.mat, n = 3, group, lambda, alpha)

library(Rcpp)
library(RcppEigen)
sourceCpp('../src/rcppeigen_hello_world.cpp')
fast_lasso_sum_ess(b.vec, s.vec, r2.mat, n = 3, group, lambda, alpha)
}

#' Lasso calculation for summary statistics
#'
#' @param b.vec genetic effect estimates
#' @param s.vec standard error of genetic effect estimates;
#' @param r2.mat residual errors;
#' @param n sample size;
#' @param group 1-based group indicator
#' @param lambda l1 penalty parameter
#' @param alpha l2 penalty parameter
#' @param init.vec inital guess
#' @param max.iter maximum iteration
#' @return a list: `beta` is the fitted regression coefficient vector, and `iteration` is the actual iteration.
#'
#' @export
fast.lasso.sum.ess <- function(b.vec,s.vec,r2.mat,n,group,lambda,alpha, init.vec = NULL, max.iter = 100) {
  print(init.vec)
  if (is.null(init.vec)) {
    init.vec <- b.vec
  }
  fast_lasso_sum_ess(b.vec, s.vec, r2.mat, n, group, lambda, alpha, init.vec, max.iter)
}
