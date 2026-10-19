# Reproduction of COLSA (Online Cox via Renewable Sieve Estimation), ORID TvLOddG9Ay
# Claim 1: "COLSA achieves ... same statistical efficiency as pooled-data partial
#           likelihood estimator with constant memory requirement."
#
# Strategy (no invention): load the authors' own R package from repo_ref, run their
# exact README pipeline on their bundled sim.rda, then compare against:
#   (a) the README's printed reference coefficients (their headline numbers), and
#   (b) survival::coxph fit on the POOLED 6000-row data == the pooled-data partial
#       likelihood estimator that claim 1 names as the efficiency benchmark.
#
# Memory control: COLSA only ever holds one batch + running sufficient statistics;
# we log object size after each sequential update to check "constant memory".

suppressMessages({
  library(survival)
  library(statmod)   # gauss.quad.prob, used un-namespaced inside utils.R
})

repo <- "repo_ref"
# Load the package sources directly (avoids a full R CMD INSTALL).
for (f in c("utils.R", "colsa.R")) source(file.path(repo, "R", f))
load(file.path(repo, "data", "sim.rda"))  # provides `sim`

cat("== Data ==\n")
cat("rows:", nrow(sim), " groups:", paste(table(sim$group), collapse="/"),
    " events:", sum(sim$status), "\n\n")

formula  <- Surv(time, status) ~ x1 + x2 + x31 + x42 + x43 + x44
boundary <- c(0, max(sim$time))
true_beta <- c(x1=0.15, x2=-0.15, x31=0.3, x42=0.3, x43=0.3, x44=0.3)

## ---- (1) Authors' COLSA sequential pipeline (verbatim from README) ----
df_sub <- sim[sim$group == 1, , drop = FALSE]
aics <- sapply(seq_len(5), function(nb)
  AIC(colsa(formula, df_sub, nb, boundary, scale = 1)))
n_basis_best <- which.min(aics)
alpha_best   <- n_basis_best / nrow(df_sub)^0.2
cat("AIC over 1..5 basis:", round(aics, 2), " -> best n_basis =", n_basis_best, "\n")

fit <- colsa(formula, df_sub, n_basis_best, boundary)
mem <- numeric(6); mem[1] <- as.numeric(object.size(fit))
for (batch in 2:6) {
  df_sub <- sim[sim$group == batch, , drop = FALSE]
  fit <- update(fit, df_sub, alpha = alpha_best)
  mem[batch] <- as.numeric(object.size(fit))
}
s <- summary(fit)
colsa_coef <- s$coefficients[!grepl("^Basis", rownames(s$coefficients)), , drop = FALSE]

cat("\n== COLSA sequential estimate (final, after 6 batches) ==\n")
print(round(colsa_coef, 5))
cat("\nfit object size after each batch (bytes):", mem, "\n")
cat("constant-memory check: max/min object.size ratio =",
    round(max(mem)/min(mem), 3), "\n")

## ---- (2) Pooled-data partial likelihood estimator (claim-1 benchmark) ----
pooled <- coxph(formula, data = sim, ties = "breslow")
ps <- summary(pooled)$coefficients  # coef, exp(coef), se(coef), z, Pr(>|z|)
cat("\n== Pooled coxph (all 6000 rows, Breslow) ==\n")
print(round(ps[, c("coef","se(coef)","z")], 5))

## ---- (3) Comparison table: COLSA vs pooled vs true ----
vars <- names(true_beta)
cmp <- data.frame(
  var        = vars,
  true       = true_beta[vars],
  colsa_coef = colsa_coef[vars, "coef"],
  pooled_coef= ps[vars, "coef"],
  colsa_se   = colsa_coef[vars, "se"],
  pooled_se  = ps[vars, "se(coef)"]
)
cmp$coef_absdiff <- abs(cmp$colsa_coef - cmp$pooled_coef)
cmp$se_ratio     <- cmp$colsa_se / cmp$pooled_se
row.names(cmp) <- NULL
cat("\n== Claim 1 comparison (COLSA vs pooled partial likelihood vs truth) ==\n")
print(format(cmp, digits = 4))

cat("\n== Summary metrics ==\n")
cat("max |COLSA - pooled| coef diff :", round(max(cmp$coef_absdiff), 4), "\n")
cat("SE ratio (COLSA/pooled) range  :", round(min(cmp$se_ratio),3), "-",
    round(max(cmp$se_ratio),3), "\n")
cat("RMSE COLSA vs true             :", round(sqrt(mean((cmp$colsa_coef-cmp$true)^2)),4), "\n")
cat("RMSE pooled vs true            :", round(sqrt(mean((cmp$pooled_coef-cmp$true)^2)),4), "\n")

saveRDS(list(cmp=cmp, mem=mem, aics=aics, n_basis_best=n_basis_best,
             colsa_coef=colsa_coef, pooled=ps),
        "runs/claim1_result.rds")
cat("\nSaved runs/claim1_result.rds\n")
