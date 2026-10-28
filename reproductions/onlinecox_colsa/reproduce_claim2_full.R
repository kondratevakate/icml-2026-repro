# COLSA Claim 2 — FULL Figure-2 panel (all K + Online baseline), ORID TvLOddG9Ay
# Adds to reproduce_claim2.R:
#   - Online baseline: piecewise-constant baseline PH (eha::pchreg), the Online
#     estimator's MODEL CLASS, with the paper's adaptive partition (J0=5, event-count
#     quantile breaks, split interval with dj>2*dbar at its median). Refit on CUMULATIVE
#     data = an upper bound on the true streaming version's accuracy (a fair, conservative
#     stand-in: if COLSA beats this, it beats the genuine online Wu-2021 method too).
#   - Vectorized event-time inversion (interpolation) so large-K (N up to ~100k) is feasible.
# Params via env: K (6/20/50/200), REPS. Saves runs/claim2full_K<K>.rds.

suppressMessages({ library(survival); library(statmod); library(MASS); library(coxphSGD); library(eha) })
for (f in c("utils.R", "colsa.R")) source(file.path("repo_ref", "R", f))

Kval <- as.integer(Sys.getenv("K", "6"))
REPS <- as.integer(Sys.getenv("REPS", "100"))
set.seed(20260722 + Kval)

true_beta <- c(0.15,-0.15,0.3,0.3,0.3,0.3); names(true_beta) <- c("x1","x2","x31","x42","x43","x44")
vars <- names(true_beta)
S0 <- function(t) 0.5*exp(-10*t^3) + 0.5*exp(-20*t^5)
tgrid <- seq(1e-6, 5, length.out = 6000); S0grid <- S0(tgrid)
inv_S0 <- function(target) approx(x = rev(S0grid), y = rev(tgrid), xout = target, rule = 2)$y

gen_batch <- function(n) {
  XY <- MASS::mvrnorm(n, mu=c(5,5), Sigma=matrix(c(10,3,3,2),2,2))
  x1<-XY[,1]; x2<-XY[,2]; x31<-rbinom(n,1,0.8)
  p0<-c(0.2,0.2,0.3,0.3); p1<-c(0.1,0.2,0.4,0.5)
  u<-runif(n); P<-ifelse(x31==1, list(p1), list(p0))
  cum0<-cumsum(p0); cum1<-cumsum(p1)
  lvl<-ifelse(x31==1, findInterval(u,cum1)+1L, findInterval(u,cum0)+1L); lvl<-pmin(lvl,4L)
  x42<-as.integer(lvl==2); x43<-as.integer(lvl==3); x44<-as.integer(lvl==4)
  X<-cbind(x1,x2,x31,x42,x43,x44); eta<-exp(as.vector(X%*%true_beta))
  target<-runif(n)^(1/eta); Tt<-inv_S0(target)
  C<-rexp(n,rate=6); time<-pmin(Tt,C); status<-as.integer(Tt<=C)
  data.frame(time=time,status=status,x1=x1,x2=x2,x31=x31,x42=x42,x43=x43,x44=x44)
}

sizes <- if (Kval<=6) c(1500,1500,1500,500,500,500)[seq_len(Kval)] else c(1500,1500,1500,500,500,500, rep(500,Kval-6))
formula <- Surv(time,status) ~ x1+x2+x31+x42+x43+x44

fit_oracle <- function(df){ m<-coxph(formula,df,ties="breslow"); s<-summary(m)$coefficients; list(coef=coef(m)[vars],se=s[vars,"se(coef)"]) }
fit_meta <- function(bs){
  cs<-ses<-matrix(NA,length(bs),length(vars),dimnames=list(NULL,vars))
  for(b in seq_along(bs)){ m<-tryCatch(coxph(formula,bs[[b]],ties="breslow"),error=function(e)NULL)
    if(!is.null(m)){s<-summary(m)$coefficients; cs[b,]<-coef(m)[vars]; ses[b,]<-s[vars,"se(coef)"]} }
  w<-1/ses^2; list(coef=(colSums(w*cs,na.rm=TRUE)/colSums(w,na.rm=TRUE))[vars], se=sqrt(1/colSums(w,na.rm=TRUE))[vars])
}
fit_colsa <- function(df,bs){
  boundary<-c(0,max(df$time)); b1<-bs[[1]]
  aics<-sapply(seq_len(5),function(nb)AIC(colsa(formula,b1,nb,boundary,scale=1)))
  nb<-which.min(aics); alpha<-nb/nrow(b1)^0.2
  fit<-colsa(formula,b1,nb,boundary); for(k in 2:length(bs)) fit<-update(fit,bs[[k]],alpha=alpha)
  s<-summary(fit)$coefficients; s<-s[!grepl("^Basis",rownames(s)),,drop=FALSE]; list(coef=s[vars,"coef"],se=s[vars,"se"])
}
fit_sgd <- function(bs,K,epochs=1){
  cc<-max(100,500/sqrt(K)); lr<-function(k)1/(cc*sqrt(k))
  m<-tryCatch(coxphSGD(formula,data=bs,learn.rates=lr,beta.zero=0,max.iter=epochs*length(bs),verbose=FALSE),error=function(e)NULL)
  if(is.null(m)) return(list(coef=setNames(rep(NA,length(vars)),vars),se=NULL))
  co<-m$coefficients[[length(m$coefficients)]]; names(co)<-vars; list(coef=co[vars],se=NULL)
}
# Online: piecewise-constant baseline PH (Online's model class) with the paper's adaptive partition.
adaptive_cuts <- function(bs, J0=5, r=2){
  ev <- sort(bs[[1]]$time[bs[[1]]$status==1])
  cuts <- as.numeric(quantile(ev, probs=seq_len(J0-1)/J0))
  cuts <- unique(cuts)
  for(k in 2:length(bs)){
    ev <- sort(c(ev, bs[[k]]$time[bs[[k]]$status==1]))
    for(pass in 1:3){
      br <- c(0, cuts, Inf); cnt <- as.numeric(table(cut(ev, breaks=br)))
      dbar <- mean(cnt); over <- which(cnt > r*dbar)
      if(!length(over)) break
      j <- max(over); lo<-br[j]; hi<-br[j+1]; inj<-ev[ev>lo & ev<=hi]
      if(length(inj)<2) break
      cuts <- sort(unique(c(cuts, median(inj))))
    }
  }
  cuts[cuts>0 & is.finite(cuts)]
}
fit_online <- function(df,bs){
  cuts<-tryCatch(adaptive_cuts(bs),error=function(e)NULL)
  m<-tryCatch(eha::pchreg(formula,data=df,cuts=c(0,cuts,max(df$time)+1e-6)),error=function(e)NULL)
  if(is.null(m)) return(list(coef=setNames(rep(NA,length(vars)),vars),se=NULL))
  co<-m$coefficients; se<-sqrt(diag(m$var))[names(co)]
  co<-co[vars]; se<-se[vars]; list(coef=co,se=se)
}

methods<-c("Oracle","COLSA","Meta","Online","SGD","SGD_Offline")
coef_arr<-array(NA,c(REPS,length(vars),length(methods)),dimnames=list(NULL,vars,methods))
cover_arr<-array(NA,c(REPS,length(vars),length(methods)),dimnames=list(NULL,vars,methods))
z975<-qnorm(0.975)
cov_of<-function(res) if(is.null(res$se)) rep(NA,length(vars)) else as.integer(abs(res$coef-true_beta)<=z975*res$se)
NA6<-function() list(coef=setNames(rep(NA,6),vars),se=NULL)

t0<-Sys.time()
for(r in seq_len(REPS)){
  bs<-lapply(sizes,gen_batch); df<-do.call(rbind,bs)
  res<-list(
    Oracle=tryCatch(fit_oracle(df),error=function(e)NA6()),
    COLSA =tryCatch(fit_colsa(df,bs),error=function(e)NA6()),
    Meta  =tryCatch(fit_meta(bs),error=function(e)NA6()),
    Online=tryCatch(fit_online(df,bs),error=function(e)NA6()),
    SGD   =fit_sgd(bs,Kval,1),
    SGD_Offline=fit_sgd(bs,Kval,100))
  for(m in methods){ coef_arr[r,,m]<-res[[m]]$coef[vars]; cover_arr[r,,m]<-cov_of(res[[m]]) }
  if(r%%10==0) cat("rep",r,"/",REPS," ",round(difftime(Sys.time(),t0,units="mins"),1),"min\n")
}
summ<-data.frame(method=methods,Abias=NA,RMSE=NA,AvgCP=NA)
for(i in seq_along(methods)){ m<-methods[i]; C<-coef_arr[,,m]
  summ$Abias[i]<-mean(abs(colMeans(C,na.rm=TRUE)-true_beta))
  summ$RMSE[i]<-sqrt(mean((sweep(C,2,true_beta))^2,na.rm=TRUE))
  cp<-cover_arr[,,m]; summ$AvgCP[i]<-if(all(is.na(cp))) NA else mean(cp,na.rm=TRUE) }
cat("\n== Claim 2 FULL, K=",Kval,",",REPS,"reps ==\n"); print(format(summ,digits=4))
cat("Ranking RMSE:",paste(summ$method[order(summ$RMSE)],collapse=" < "),"\n")
saveRDS(list(summ=summ,coef_arr=coef_arr,cover_arr=cover_arr,K=Kval,REPS=REPS),sprintf("runs/claim2full_K%d.rds",Kval))
cat(sprintf("Saved runs/claim2full_K%d.rds\n",Kval))
