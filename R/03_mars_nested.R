# MARS arm of the repeated nested cross-validation.
# Reads the outer-fold training/test files written by 02_nested_cv.py (already imputed and
# standardised within each outer training fold), tunes MARS by inner 5-fold CV and evaluates
# once on the untouched held-out fold.
# Usage: Rscript 03_mars_nested.R <run_dir>
suppressPackageStartupMessages({library(caret); library(earth)})
args <- commandArgs(trailingOnly = TRUE); run_dir <- args[1]
files <- sort(list.files(file.path(run_dir, "folds"), pattern = "_train.csv$", full.names = TRUE))
grid <- expand.grid(degree = 1:2, nprune = c(2, 4, 6, 8, 10, 12, 15, 20))   # 16 combinations; nk = 20 terms max
res <- list(); oof <- list(); best <- list()
for (f in files) {
  k <- as.integer(sub(".*fold(\\d+)_train.csv", "\\1", f))
  tr <- read.csv(f); te <- read.csv(sub("_train", "_test", f))
  set.seed(20261101 + k)
  ctrl <- trainControl(method = "cv", number = 5)
  # suppressWarnings: caret warns when a resample yields a constant prediction (R^2 undefined), which is expected here
  fit <- suppressWarnings(train(abi ~ ., data = tr, method = "earth", tuneGrid = grid, trControl = ctrl,
               metric = "RMSE", nk = 20))
  p <- as.numeric(predict(fit, newdata = te))
  y <- te$abi
  res[[length(res) + 1]] <- data.frame(fold = k, model = "MARS", rmse = sqrt(mean((y - p)^2)),
                                       mae = mean(abs(y - p)), r2 = 1 - sum((y - p)^2) / sum((y - mean(y))^2))
  oof[[length(oof) + 1]] <- data.frame(fold = k, idx = seq_along(y), MARS = p)
  best[[length(best) + 1]] <- data.frame(fold = k, degree = fit$bestTune$degree, nprune = fit$bestTune$nprune)
}
write.csv(do.call(rbind, res), file.path(run_dir, "fold_metrics_mars.csv"), row.names = FALSE)
write.csv(do.call(rbind, oof), file.path(run_dir, "oof_predictions_mars.csv"), row.names = FALSE)
write.csv(do.call(rbind, best), file.path(run_dir, "mars_best_params.csv"), row.names = FALSE)
cat("MARS done:", length(files), "folds\n")
