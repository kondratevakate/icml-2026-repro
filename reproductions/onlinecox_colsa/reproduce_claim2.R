# Reproduction of COLSA Claim 2 (Figure 2 core), ORID TvLOddG9Ay
# Claim 2: "COLSA outperforms existing online methods and matches the performance
#           of full-data (Oracle) estimation."  (paper Sec 5.1 / Fig 2)
#
# Faithful to Appendix B. Methods per paper Sec 5 / Appendix B:
#   Oracle       - pooled coxph (full data)                       [survival]
#   COLSA        - authors' own package (repo_ref)                [their code]
#   Meta         - inverse-variance weighting of per-batch coxph  [Cochran 1954]
#   SGD          - coxphSGD, streaming one pass, eta_k=(c sqrt k)^-1, c=max(100,500/sqrt K)
#   SGD(Offline) - coxphSGD, 100 epochs, same learning rate
# NOT reproduced here: Online (Wu et al. 2021 adaptive-partition) -> documented gap.
#
# DGP (Appendix B): baseline S0(t)=0.5 exp(-10 t^3)+0.5 exp(-20 t^5); X1,X2~MVN((5,5),
# [[10,3],[3,2]]); X3~Bern(0.8); X4 4-level multinomial cond. on X3; beta=
# (.15,-.15,.3,.3,.3,.3); event times by cumulative-hazard inversion; C~Exp(6) (~12% events).

suppressMessages({ library(survival); library(statmod); library(MASS); library(coxphSGD) })
for (f in c("utils.R", "colsa.R")) source(file.path("repo_ref", "R", f))

set.seed(20260722)
true_beta <- c(0.15, -0.15, 0.3, 0.3, 0.3, 0.3)
names(true_beta) <- c("x1","x2","x31","x42","x43","x44")

S0 <- function(t) 0.5*exp(-10*t^3) + 0.5*exp(-20*t^5)

gen_batch <- function(n) {
  XY <- MASS::mvrnorm(n, mu=c(5,5), Sigma=matrix(c(10,3,3,2),2,2))
  x1 <- XY[,1]; x2 <- XY[,2]
  x31 <- rbinom(n, 1, 0.8)
  # X4: 4-level multinomial conditional on x31
  p0 <- c(0.2,0.2,0.3,0.3); p1 <- c(0.1,0.2,0.4,0.5)
  lvl <- integer(n)
  for (i in seq_len(n)) lvl[i] <- sample.int(4, 1, prob = if (x31[i]==1) p1 else p0)
  x42 <- as.integer(lvl==2); x43 <- as.integer(lvl==3); x44 <- as.integer(lvl==4)
  X <- cbind(x1,x2,x31,x42,x43,x44)
  lp <- as.vector(X %*% true_beta); eta <- exp(lp)
  U <- runif(n)
  target <- U^(1/eta)                     # solve S0(t)^eta = U  ->  S0(t)=U^{1/eta}
  Tt <- vapply(target, function(s) {
    if (s >= S0(1e-8)) return(1e-8)
    tryCatch(uniroot(function(t) S0(t) - s, c(1e-8, 5), tol = 1e-9)$root,
             error = function(e) 5)
  }, numeric(1))
  C <- rexp(n, rate = 6)
  time <- pmin(Tt, C); status <- as.integer(Tt <= C)
  data.frame(time=time, status=status, x1=x1, x2=x2, x31=x31, x42=x42, x43=x43, x44=x44)
}

Kval  <- as.integer(Sys.getenv("K", "6"))
# Appendix B: first 6 sites (1500,1500,1500,500,500,500); remaining K-6 sites size 500.
sizes_K6 <- if (Kval <= 6) {
  c(1500,1500,1500,500,500,500)[seq_len(Kval)]
} else {
  c(1500,1500,1500,500,500,500, rep(500, Kval-6))
}
formula  <- Surv(time, status) ~ x1 + x2 + x31 + x42 + x43 + x44
vars <- names(true_beta)

fit_oracle <- function(df) {
  m <- coxph(formula, df, ties="breslow"); s <- summary(m)$coefficients
  list(coef=coef(m)[vars], se=s[vars,"se(coef)"])
}
fit_meta <- function(batches) {
  cs <- ses <- matrix(NA, length(batches), length(vars), dimnames=list(NULL,vars))
  for (b in seq_along(batches)) {
    m <- tryCatch(coxph(formula, batches[[b]], ties="breslow"), error=function(e) NULL)
    if (!is.null(m)) { s<-summary(m)$coefficients; cs[b,]<-coef(m)[vars]; ses[b,]<-s[vars,"se(coef)"] }
  }
  w <- 1/ses^2
  coef <- colSums(w*cs, na.rm=TRUE)/colSums(w, na.rm=TRUE)
  se   <- sqrt(1/colSums(w, na.rm=TRUE))
  list(coef=coef[vars], se=se[vars])
}
fit_colsa <- function(df, batches) {
  boundary <- c(0, max(df$time))
  b1 <- batches[[1]]
  aics <- sapply(seq_len(5), function(nb) AIC(colsa(formula,b1,nb,boundary,scale=1)))
  nb <- which.min(aics); alpha <- nb/nrow(b1)^0.2
  fit <- colsa(formula, b1, nb, boundary)
  for (k in 2:length(batches)) fit <- update(fit, batches[[k]], alpha=alpha)
  s <- summary(fit)$coefficients
  s <- s[!grepl("^Basis", rownames(s)), , drop=FALSE]
  list(coef=s[vars,"coef"], se=s[vars,"se"])
}
fit_sgd <- function(batches, K, epochs=1) {
  cc <- max(100, 500/sqrt(K)); lr <- function(k) 1/(cc*sqrt(k))
  m <- tryCatch(coxphSGD(formula, data=batches, learn.rates=lr, beta.zero=0,
                         max.iter=epochs*length(batches), verbose=FALSE),
                error=function(e) NULL)
  if (is.null(m)) return(list(coef=setNames(rep(NA,length(vars)),vars), se=NULL))
  co <- m$coefficients[[length(m$coefficients)]]; names(co) <- vars
  list(coef=co[vars], se=NULL)
}

REPS <- as.integer(Sys.getenv("REPS", "200"))
methods <- c("Oracle","COLSA","Meta","SGD","SGD_Offline")
coef_arr <- array(NA, c(REPS, length(vars), length(methods)),
                  dimnames=list(NULL, vars, methods))
cover_arr <- array(NA, c(REPS, length(vars), length(methods)),
                   dimnames=list(NULL, vars, methods))
z975 <- qnorm(0.975)
cover_from <- function(res) {
  if (is.null(res$se)) return(rep(NA, length(vars)))
  as.integer(abs(res$coef - true_beta) <= z975*res$se)
}

t0 <- Sys.time()
for (r in seq_len(REPS)) {
  batches <- lapply(sizes_K6, gen_batch)
  df <- do.call(rbind, batches)
  res <- list(
    Oracle      = fit_oracle(df),
    COLSA       = tryCatch(fit_colsa(df, batches), error=function(e) list(coef=setNames(rep(NA,6),vars),se=NULL)),
    Meta        = fit_meta(batches),
    SGD         = fit_sgd(batches, K=Kval, epochs=1),
    SGD_Offline = fit_sgd(batches, K=Kval, epochs=100)
  )
  for (mth in methods) {
    coef_arr[r,,mth]  <- res[[mth]]$coef[vars]
    cover_arr[r,,mth] <- cover_from(res[[mth]])
  }
  if (r %% 25 == 0) cat("rep", r, "/", REPS, " elapsed",
                        round(difftime(Sys.time(),t0,units="mins"),1), "min\n")
}

# Metrics per paper: Abias = mean_p |mean_r beta - true|; RMSE = sqrt(mean over p,r (beta-true)^2)
summ <- data.frame(method=methods, Abias=NA, RMSE=NA, AvgCP=NA)
for (i in seq_along(methods)) {
  m <- methods[i]; C <- coef_arr[,,m]
  bias_p <- colMeans(C, na.rm=TRUE) - true_beta
  summ$Abias[i] <- mean(abs(bias_p))
  summ$RMSE[i]  <- sqrt(mean((sweep(C,2,true_beta))^2, na.rm=TRUE))
  cp <- cover_arr[,,m]; summ$AvgCP[i] <- if (all(is.na(cp))) NA else mean(cp, na.rm=TRUE)
}
cat("\n== Claim 2 metrics, K=", Kval, ",", REPS, "reps (Appendix B DGP) ==\n")
print(format(summ, digits=4))
cat("\nRanking by RMSE (low=better):", paste(summ$method[order(summ$RMSE)], collapse=" < "), "\n")
saveRDS(list(summ=summ, coef_arr=coef_arr, cover_arr=cover_arr, REPS=REPS, K=Kval),
        sprintf("runs/claim2_K%d.rds", Kval))
cat(sprintf("Saved runs/claim2_K%d.rds\n", Kval))
