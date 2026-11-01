# MARS fitted on the full analytic cohort (imputed, standardised) for interpretation:
# selected hinge terms, variable importance (GCV) and partial dependence for every predictor.
# Usage: Rscript 06_mars_full.R <out_dir>   (expects full_imputed_scaled.csv from 04_interpretation.py)
suppressPackageStartupMessages({library(caret); library(earth)})
out <- commandArgs(trailingOnly = TRUE)[1]
d <- read.csv(file.path(out, "full_imputed_scaled.csv"))
set.seed(20261101)
grid <- expand.grid(degree = 1:2, nprune = c(2, 4, 6, 8, 10, 12, 15, 20))
# suppressWarnings: caret warns when a resample yields a constant prediction (R^2 undefined), which is expected here
fit <- suppressWarnings(train(abi ~ ., data = d, method = "earth", tuneGrid = grid, nk = 20, metric = "RMSE",
             trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3)))
fm <- fit$finalModel
sink(file.path(out, "mars_full_summary.txt")); print(fit$bestTune); print(summary(fm)); print(evimp(fm)); sink()
X <- d[, setdiff(names(d), "abi")]
pdp <- do.call(rbind, lapply(names(X), function(v) {
  g <- seq(quantile(X[[v]], 0.02), quantile(X[[v]], 0.98), length.out = 40)
  data.frame(feature = v, x = g, yhat = sapply(g, function(z) { Z <- X; Z[[v]] <- z; mean(predict(fm, Z)) }))
}))
write.csv(pdp, file.path(out, "mars_pdp.csv"), row.names = FALSE)
write.csv(data.frame(degree = fit$bestTune$degree, nprune = fit$bestTune$nprune,
                     n_terms = length(fm$selected.terms)), file.path(out, "mars_full_best.csv"), row.names = FALSE)
cat("done\n")
