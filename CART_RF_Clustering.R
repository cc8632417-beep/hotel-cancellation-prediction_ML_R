# ==============================================================================
# Hotel Booking Cancellation Prediction - Member 5
# Techniques: CART (Category A), Random Forest (Category A), Clustering (Category B)
# Data      : Hotel_Reservations_Cleaned.csv (36,275 bookings, 17 predictors + booking_status)
#
# HOW TO RUN: put this script and Hotel_Reservations_Cleaned.csv in the same
#             folder, set that folder as the working directory, then source().
#             All tables and charts are written to the sub-folder "member5_results".
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. Set-up
# ------------------------------------------------------------------------------
pkgs <- c("rpart", "rpart.plot", "randomForest", "cluster")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}
library(rpart)
library(rpart.plot)
library(randomForest)
library(cluster)

data_file <- "Hotel_Reservations_Cleaned.csv"
out_dir   <- "member5_results"
seed      <- 2026

if (!file.exists(data_file)) {
  stop("Hotel_Reservations_Cleaned.csv not found. Set the working directory to the folder that holds the data file.")
}
dir.create(out_dir, showWarnings = FALSE)
if (file.exists(file.path(out_dir, "DONE.txt"))) file.remove(file.path(out_dir, "DONE.txt"))

while (sink.number() > 0) sink()
sink(file.path(out_dir, "console_log.txt"), split = TRUE)
t_start <- Sys.time()
cat("Run started:", format(t_start), "\n")
cat(R.version.string, "\n\n")

# Colours used in every chart
col_keep   <- "#2E75B6"   # blue   = not cancelled / CART
col_cancel <- "#E07B39"   # orange = cancelled / Random Forest
col_dark   <- "#1F3A5F"
col_grey   <- "#8C8C8C"

# ------------------------------------------------------------------------------
# 1. Data preparation
# ------------------------------------------------------------------------------
hotel <- read.csv(data_file, stringsAsFactors = FALSE)
cat("Rows:", nrow(hotel), " Columns:", ncol(hotel), "\n")
cat("Missing values:", sum(is.na(hotel)), "\n")

# Target: "Canceled" is the event of interest (positive class)
hotel$booking_status <- factor(hotel$booking_status, levels = c("Not_Canceled", "Canceled"))

# Categorical predictors
hotel$type_of_meal_plan          <- factor(hotel$type_of_meal_plan)
hotel$room_type_reserved         <- factor(hotel$room_type_reserved)
hotel$market_segment_type        <- factor(hotel$market_segment_type)
hotel$required_car_parking_space <- factor(hotel$required_car_parking_space, levels = c(0, 1), labels = c("No", "Yes"))
hotel$repeated_guest             <- factor(hotel$repeated_guest, levels = c(0, 1), labels = c("No", "Yes"))
hotel$arrival_month              <- factor(hotel$arrival_month, levels = 1:12)

cat("\nStructure after type conversion:\n")
str(hotel)
cat("\nOverall cancellation rate:", round(mean(hotel$booking_status == "Canceled"), 4), "\n")

# ------------------------------------------------------------------------------
# 2. Train / test split: 70% / 30%, stratified on booking_status
# ------------------------------------------------------------------------------
set.seed(seed)
idx_c <- which(hotel$booking_status == "Canceled")
idx_n <- which(hotel$booking_status == "Not_Canceled")
train_idx <- c(sample(idx_c, round(0.7 * length(idx_c))),
               sample(idx_n, round(0.7 * length(idx_n))))
train <- hotel[train_idx, ]
test  <- hotel[-train_idx, ]
cat("\nTrain rows:", nrow(train), " cancellation rate:", round(mean(train$booking_status == "Canceled"), 4), "\n")
cat("Test rows :", nrow(test),  " cancellation rate:", round(mean(test$booking_status == "Canceled"), 4), "\n")

test_pos <- test$booking_status == "Canceled"

# ------------------------------------------------------------------------------
# Helper functions: evaluation on the test set
# ------------------------------------------------------------------------------
auc_fn <- function(prob, pos) {
  r  <- rank(prob)                      # average ranks handle ties
  n1 <- sum(pos)
  n0 <- sum(!pos)
  (sum(r[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

eval_model <- function(model_name, prob, pos, threshold = 0.5) {
  pred <- prob > threshold
  TP <- sum(pred & pos)
  FP <- sum(pred & !pos)
  TN <- sum(!pred & !pos)
  FN <- sum(!pred & pos)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  data.frame(model = model_name, threshold = threshold,
             accuracy = (TP + TN) / length(pos),
             precision = precision, recall = recall,
             f1 = 2 * precision * recall / (precision + recall),
             specificity = TN / (TN + FP),
             auc = auc_fn(prob, pos),
             TP = TP, FP = FP, TN = TN, FN = FN,
             flagged_share = mean(pred),
             stringsAsFactors = FALSE)
}

roc_points <- function(prob, pos) {
  o    <- order(prob, decreasing = TRUE)
  tpr  <- cumsum(pos[o]) / sum(pos)
  fpr  <- cumsum(!pos[o]) / sum(!pos)
  keep <- !duplicated(prob[o], fromLast = TRUE)   # one point per distinct score
  data.frame(fpr = c(0, fpr[keep]), tpr = c(0, tpr[keep]))
}

# ------------------------------------------------------------------------------
# 3. CART
#    Step 1: grow the tree to the maximum (cp = 0, minsplit = 2)
#    Step 2: prune with the 1-standard-error rule on 10-fold cross-validation
#    Step 3: a small "management" tree (max 7 splits) to communicate the rules
# ------------------------------------------------------------------------------
cat("\n==================== CART ====================\n")
set.seed(seed)
cart_max <- rpart(booking_status ~ ., data = train, method = "class",
                  control = rpart.control(minsplit = 2, cp = 0))

cpt <- as.data.frame(cart_max$cptable)
names(cpt) <- c("CP", "nsplit", "rel_error", "xerror", "xstd")
write.csv(cpt, file.path(out_dir, "cart_cptable.csv"), row.names = FALSE)

cp_between <- function(i) {            # geometric mean of CP[i] and CP[i-1]
  if (i > 1) sqrt(cpt$CP[i] * cpt$CP[i - 1]) else 1
}

i_min  <- which.min(cpt$xerror)
cv_cap <- cpt$xerror[i_min] + cpt$xstd[i_min]
i_1se  <- min(which(cpt$xerror <= cv_cap))          # simplest tree within 1 SE of the minimum
cart_opt <- prune(cart_max, cp = cp_between(i_1se))

i_simple    <- max(which(cpt$nsplit <= 7))          # largest tree with at most 7 splits
cart_simple <- prune(cart_max, cp = cp_between(i_simple))

n_leaves <- function(m) sum(m$frame$var == "<leaf>")
root_err <- cart_max$frame$dev[1] / cart_max$frame$n[1]   # root node error (majority-class error)

cart_summary <- data.frame(
  tree          = c("Maximal tree", "Minimum CV error tree", "Optimal tree (1-SE rule)", "Management tree"),
  cp_row        = c(nrow(cpt), i_min, i_1se, i_simple),
  cp_used       = c(0, cp_between(i_min), cp_between(i_1se), cp_between(i_simple)),
  splits        = c(cpt$nsplit[nrow(cpt)], cpt$nsplit[i_min], cpt$nsplit[i_1se], cpt$nsplit[i_simple]),
  leaves        = c(n_leaves(cart_max), cpt$nsplit[i_min] + 1, n_leaves(cart_opt), n_leaves(cart_simple)),
  train_error   = c(cpt$rel_error[nrow(cpt)], cpt$rel_error[i_min], cpt$rel_error[i_1se], cpt$rel_error[i_simple]) * root_err,
  cv_error      = c(cpt$xerror[nrow(cpt)], cpt$xerror[i_min], cpt$xerror[i_1se], cpt$xerror[i_simple]) * root_err,
  stringsAsFactors = FALSE)
cat("\nRoot node error (always predict Not_Canceled):", round(root_err, 4), "\n")
cat("1-SE cap on relative CV error:", round(cv_cap, 5), "\n\n")
print(cart_summary)
write.csv(cart_summary, file.path(out_dir, "cart_summary.csv"), row.names = FALSE)

# Variable importance of the optimal CART (scaled to sum to 100)
cart_imp <- cart_opt$variable.importance
cart_imp <- data.frame(variable = names(cart_imp),
                       importance_pct = round(100 * as.numeric(cart_imp) / sum(cart_imp), 2))
cat("\nCART variable importance (optimal tree, % of total):\n")
print(cart_imp)
write.csv(cart_imp, file.path(out_dir, "cart_var_importance.csv"), row.names = FALSE)

# Rules of the management tree, validated on the test set
frame_s    <- cart_simple$frame
node_ids   <- as.integer(rownames(frame_s))
leaf_ids   <- node_ids[frame_s$var == "<leaf>"]
paths      <- path.rpart(cart_simple, nodes = leaf_ids, print.it = FALSE)
rule_text  <- sapply(paths, function(p) paste(p[-1], collapse = " & "))
leaf_train <- node_ids[cart_simple$where]
node_tree  <- cart_simple                 # copy whose "prediction" is the node number,
node_tree$frame$yval <- node_ids          # so predict() returns the leaf each test booking falls in
leaf_test  <- as.integer(predict(node_tree, newdata = test, type = "vector"))

leaf_tab <- data.frame(
  node              = leaf_ids,
  rule              = as.character(rule_text),
  n_train           = as.integer(sapply(leaf_ids, function(k) sum(leaf_train == k))),
  cancel_rate_train = sapply(leaf_ids, function(k) mean(train$booking_status[leaf_train == k] == "Canceled")),
  n_test            = as.integer(sapply(leaf_ids, function(k) sum(leaf_test == k))),
  cancel_rate_test  = sapply(leaf_ids, function(k) mean(test_pos[leaf_test == k])),
  stringsAsFactors = FALSE)
leaf_tab$share_test <- leaf_tab$n_test / nrow(test)
leaf_tab <- leaf_tab[order(-leaf_tab$cancel_rate_train), ]
cat("\nManagement tree: decision rules (sorted by cancellation rate):\n")
print(leaf_tab, row.names = FALSE, digits = 3)
write.csv(leaf_tab, file.path(out_dir, "cart_management_tree_rules.csv"), row.names = FALSE)

cat("\nManagement tree printout:\n")
print(cart_simple)

# Test-set predictions
p_cart_opt    <- predict(cart_opt,    newdata = test, type = "prob")[, "Canceled"]
p_cart_simple <- predict(cart_simple, newdata = test, type = "prob")[, "Canceled"]

# Figure 1: management tree
png(file.path(out_dir, "fig1_cart_management_tree.png"), width = 2600, height = 1400, res = 200)
rpart.plot(cart_simple, type = 4, extra = 104, under = TRUE, fallen.leaves = TRUE,
           box.palette = "BuOr", roundint = FALSE, tweak = 1.15, branch.lty = 1,
           shadow.col = 0, clip.right.labs = FALSE)
dev.off()

# Figure 2: cross-validation error versus tree size
png(file.path(out_dir, "fig2_cart_cv_error.png"), width = 1800, height = 1100, res = 200)
par(mar = c(4.5, 4.8, 1.5, 1), las = 1, bty = "l")
ok <- cpt$nsplit > 0
plot(cpt$nsplit[ok], cpt$xerror[ok] * root_err, log = "x", type = "l", lwd = 2, col = col_keep,
     xlab = "Number of splits in the tree (log scale)", ylab = "Error rate",
     ylim = c(0, max(cpt$xerror[ok] * root_err) * 1.05))
lines(cpt$nsplit[ok], cpt$rel_error[ok] * root_err, lwd = 2, lty = 2, col = col_grey)
abline(v = cpt$nsplit[i_1se], lty = 3, col = col_cancel, lwd = 2)
points(cpt$nsplit[i_1se], cpt$xerror[i_1se] * root_err, pch = 19, col = col_cancel, cex = 1.3)
legend("topright", bty = "n", lwd = c(2, 2, 2), lty = c(1, 2, 3),
       col = c(col_keep, col_grey, col_cancel),
       legend = c("10-fold cross-validation error", "Training error",
                  paste0("Chosen tree (1-SE rule): ", cpt$nsplit[i_1se], " splits")))
dev.off()

# ------------------------------------------------------------------------------
# 4. Random Forest (500 trees, default mtry = floor(sqrt(17)) = 4)
# ------------------------------------------------------------------------------
cat("\n==================== RANDOM FOREST ====================\n")
set.seed(seed)
rf <- randomForest(booking_status ~ ., data = train, ntree = 500, importance = TRUE)
print(rf)

rf_oob <- data.frame(ntree = 1:rf$ntree, rf$err.rate)
write.csv(rf_oob, file.path(out_dir, "rf_oob_error_by_ntree.csv"), row.names = FALSE)
cat("\nOOB error at 500 trees:", round(rf$err.rate[rf$ntree, "OOB"], 4), "\n")
cat("OOB error at 100 / 200 / 300 trees:",
    round(rf$err.rate[c(100, 200, 300), "OOB"], 4), "\n")

rf_imp <- as.data.frame(importance(rf))
rf_imp$variable <- rownames(rf_imp)
rf_imp <- rf_imp[order(-rf_imp$MeanDecreaseAccuracy),
                 c("variable", "MeanDecreaseAccuracy", "MeanDecreaseGini", "Not_Canceled", "Canceled")]
rf_imp$gini_pct <- round(100 * rf_imp$MeanDecreaseGini / sum(rf_imp$MeanDecreaseGini), 2)
cat("\nRandom Forest variable importance:\n")
print(rf_imp, row.names = FALSE, digits = 4)
write.csv(rf_imp, file.path(out_dir, "rf_var_importance.csv"), row.names = FALSE)

p_rf <- predict(rf, newdata = test, type = "prob")[, "Canceled"]

# Figure 3: Random Forest variable importance
nice_names <- c(
  lead_time = "Lead time (days booked ahead)",
  avg_price_per_room = "Average room price",
  no_of_special_requests = "Number of special requests",
  market_segment_type = "Booking channel",
  arrival_month = "Arrival month",
  arrival_date = "Arrival day of month",
  arrival_year = "Arrival year",
  no_of_week_nights = "Week nights",
  no_of_weekend_nights = "Weekend nights",
  no_of_adults = "Number of adults",
  no_of_children = "Number of children",
  type_of_meal_plan = "Meal plan",
  room_type_reserved = "Room type",
  required_car_parking_space = "Car parking requested",
  repeated_guest = "Repeat guest",
  no_of_previous_cancellations = "Previous cancellations",
  no_of_previous_bookings_not_canceled = "Previous bookings kept")

png(file.path(out_dir, "fig3_rf_importance.png"), width = 1900, height = 1300, res = 200)
par(mar = c(4.5, 13.5, 1, 2), las = 1, bty = "n")
imp_plot <- rf_imp[order(rf_imp$MeanDecreaseAccuracy), ]
bar_col  <- ifelse(rank(-imp_plot$MeanDecreaseAccuracy) <= 5, col_cancel, col_keep)
bp <- barplot(imp_plot$MeanDecreaseAccuracy, horiz = TRUE, col = bar_col, border = NA,
              names.arg = nice_names[imp_plot$variable], cex.names = 0.85,
              xlab = "Importance (mean decrease in accuracy when the variable is shuffled)",
              xlim = c(0, max(imp_plot$MeanDecreaseAccuracy) * 1.12))
text(imp_plot$MeanDecreaseAccuracy, bp, labels = round(imp_plot$MeanDecreaseAccuracy),
     pos = 4, cex = 0.8, col = col_dark, xpd = TRUE)
dev.off()

# Figure 4: OOB error versus number of trees
png(file.path(out_dir, "fig4_rf_oob_error.png"), width = 1800, height = 1100, res = 200)
par(mar = c(4.5, 4.8, 1.5, 1), las = 1, bty = "l")
matplot(1:rf$ntree, rf$err.rate, type = "l", lty = 1, lwd = 2,
        col = c(col_dark, col_keep, col_cancel),
        xlab = "Number of trees", ylab = "Out-of-bag error rate",
        ylim = c(0, max(rf$err.rate[20:rf$ntree, ]) * 1.4))
legend("topright", bty = "n", lwd = 2, col = c(col_dark, col_keep, col_cancel),
       legend = c("Overall (OOB)", "Not cancelled bookings", "Cancelled bookings"))
dev.off()

# ------------------------------------------------------------------------------
# 5. Test-set comparison: CART versus Random Forest
# ------------------------------------------------------------------------------
cat("\n==================== TEST-SET PERFORMANCE ====================\n")
metrics <- rbind(
  eval_model("CART - optimal tree (1-SE)", p_cart_opt,    test_pos, 0.5),
  eval_model("CART - management tree",     p_cart_simple, test_pos, 0.5),
  eval_model("Random Forest",              p_rf,          test_pos, 0.5),
  eval_model("Random Forest",              p_rf,          test_pos, 0.4),
  eval_model("Random Forest",              p_rf,          test_pos, 0.3),
  eval_model("CART - optimal tree (1-SE)", p_cart_opt,    test_pos, 0.4),
  eval_model("CART - optimal tree (1-SE)", p_cart_opt,    test_pos, 0.3))
metrics$baseline_accuracy <- mean(!test_pos)
print(metrics, row.names = FALSE, digits = 4)
write.csv(metrics, file.path(out_dir, "metrics_test_set.csv"), row.names = FALSE)

cat("\nConfusion matrix - CART optimal tree (threshold 0.5):\n")
print(table(Actual = test$booking_status,
            Predicted = factor(ifelse(p_cart_opt > 0.5, "Canceled", "Not_Canceled"), levels = levels(test$booking_status))))
cat("\nConfusion matrix - Random Forest (threshold 0.5):\n")
print(table(Actual = test$booking_status,
            Predicted = factor(ifelse(p_rf > 0.5, "Canceled", "Not_Canceled"), levels = levels(test$booking_status))))

# Risk bands from the Random Forest score (how well does the score rank bookings?)
band <- cut(p_rf, breaks = c(-Inf, 0.2, 0.5, 0.8, Inf),
            labels = c("Low (0-20%)", "Moderate (20-50%)", "High (50-80%)", "Very high (80-100%)"))
risk_bands <- data.frame(
  band               = levels(band),
  bookings           = as.integer(table(band)),
  share_of_bookings  = as.numeric(table(band)) / length(band),
  actual_cancel_rate = as.numeric(tapply(test_pos, band, mean)),
  share_of_all_cancellations = as.numeric(tapply(test_pos, band, sum)) / sum(test_pos))
cat("\nRandom Forest risk bands on the test set:\n")
print(risk_bands, row.names = FALSE, digits = 3)
write.csv(risk_bands, file.path(out_dir, "rf_risk_bands_test_set.csv"), row.names = FALSE)

# Figure 5: ROC curves
roc_cart <- roc_points(p_cart_opt, test_pos)
roc_rf   <- roc_points(p_rf, test_pos)
png(file.path(out_dir, "fig5_roc_curves.png"), width = 1500, height = 1400, res = 200)
par(mar = c(4.5, 4.8, 1.5, 1), las = 1, bty = "l")
plot(roc_rf$fpr, roc_rf$tpr, type = "l", lwd = 3, col = col_cancel, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "False alarm rate (kept bookings wrongly flagged)",
     ylab = "Detection rate (cancellations caught)")
lines(roc_cart$fpr, roc_cart$tpr, lwd = 3, col = col_keep)
abline(0, 1, lty = 2, col = col_grey)
legend("bottomright", bty = "n", lwd = c(3, 3, 1), lty = c(1, 1, 2),
       col = c(col_cancel, col_keep, col_grey),
       legend = c(paste0("Random Forest (AUC = ", sprintf("%.3f", auc_fn(p_rf, test_pos)), ")"),
                  paste0("CART (AUC = ", sprintf("%.3f", auc_fn(p_cart_opt, test_pos)), ")"),
                  "Random guess (AUC = 0.500)"))
dev.off()

# ------------------------------------------------------------------------------
# 6. Clustering (k-means): what kinds of bookings does the hotel receive?
#    booking_status is NOT used to form the clusters; it is only used afterwards
#    to compare the cancellation rate of each segment.
# ------------------------------------------------------------------------------
cat("\n==================== CLUSTERING ====================\n")
clus_data <- data.frame(
  lead_time              = hotel$lead_time,
  avg_price_per_room     = hotel$avg_price_per_room,
  no_of_special_requests = hotel$no_of_special_requests,
  total_nights           = hotel$no_of_weekend_nights + hotel$no_of_week_nights,
  total_guests           = hotel$no_of_adults + hotel$no_of_children)
clus_scaled <- scale(clus_data)

# Choice of k: elbow (within-cluster sum of squares) and average silhouette width
set.seed(seed)
k_grid <- 1:8
wss <- sapply(k_grid, function(k) kmeans(clus_scaled, centers = k, nstart = 20, iter.max = 100)$tot.withinss)

set.seed(seed)
sil_idx  <- sample(nrow(clus_scaled), 4000)
sil_dist <- dist(clus_scaled[sil_idx, ])
sil <- c(NA, sapply(2:8, function(k) {
  set.seed(seed)
  km_k <- kmeans(clus_scaled, centers = k, nstart = 20, iter.max = 100)
  mean(silhouette(km_k$cluster[sil_idx], sil_dist)[, "sil_width"])
}))
k_choice <- data.frame(k = k_grid, total_within_ss = wss,
                       pct_variance_explained = 100 * (1 - wss / wss[1]),
                       avg_silhouette = sil)
cat("\nChoice of k:\n")
print(k_choice, row.names = FALSE, digits = 4)
write.csv(k_choice, file.path(out_dir, "cluster_k_choice.csv"), row.names = FALSE)

# Final solution: k = 4
k_final <- 4
set.seed(seed)
km <- kmeans(clus_scaled, centers = k_final, nstart = 50, iter.max = 100)

# Relabel segments A-D from highest to lowest cancellation rate
cancel_by_cluster <- tapply(hotel$booking_status == "Canceled", km$cluster, mean)
seg_order <- order(-cancel_by_cluster)
hotel$segment <- factor(LETTERS[match(km$cluster, seg_order)], levels = LETTERS[1:k_final])

profile <- data.frame(
  segment            = levels(hotel$segment),
  bookings           = as.integer(table(hotel$segment)),
  share_of_bookings  = as.numeric(table(hotel$segment)) / nrow(hotel),
  cancel_rate        = as.numeric(tapply(hotel$booking_status == "Canceled", hotel$segment, mean)),
  share_of_all_cancellations = as.numeric(tapply(hotel$booking_status == "Canceled", hotel$segment, sum)) /
                               sum(hotel$booking_status == "Canceled"),
  avg_lead_time      = as.numeric(tapply(hotel$lead_time, hotel$segment, mean)),
  avg_price          = as.numeric(tapply(hotel$avg_price_per_room, hotel$segment, mean)),
  avg_special_requests = as.numeric(tapply(hotel$no_of_special_requests, hotel$segment, mean)),
  avg_total_nights   = as.numeric(tapply(clus_data$total_nights, hotel$segment, mean)),
  avg_total_guests   = as.numeric(tapply(clus_data$total_guests, hotel$segment, mean)),
  pct_online         = as.numeric(tapply(hotel$market_segment_type == "Online", hotel$segment, mean)),
  pct_repeat_guest   = as.numeric(tapply(hotel$repeated_guest == "Yes", hotel$segment, mean)))
cat("\nSegment profile (k = 4, segments ordered by cancellation rate):\n")
print(profile, row.names = FALSE, digits = 3)
write.csv(profile, file.path(out_dir, "cluster_profile.csv"), row.names = FALSE)

cat("\nBetween-cluster SS / total SS:", round(km$betweenss / km$totss, 4), "\n")
chi <- chisq.test(table(hotel$segment, hotel$booking_status))
cat("Chi-square test, segment vs booking_status: X2 =", round(chi$statistic, 1),
    " df =", chi$parameter, " p-value =", format.pval(chi$p.value), "\n")

# Figure 6: elbow and silhouette
png(file.path(out_dir, "fig6_cluster_k_choice.png"), width = 2200, height = 1000, res = 200)
par(mfrow = c(1, 2), mar = c(4.5, 4.8, 2.5, 1), las = 1, bty = "l")
plot(k_grid, wss / 1000, type = "b", pch = 19, lwd = 2, col = col_keep,
     xlab = "Number of clusters (k)", ylab = "Within-cluster sum of squares ('000)",
     main = "Elbow method")
points(k_final, wss[k_final] / 1000, pch = 19, cex = 1.8, col = col_cancel)
plot(2:8, sil[2:8], type = "b", pch = 19, lwd = 2, col = col_keep,
     xlab = "Number of clusters (k)", ylab = "Average silhouette width",
     main = "Silhouette method", ylim = c(0, max(sil, na.rm = TRUE) * 1.3))
points(k_final, sil[k_final], pch = 19, cex = 1.8, col = col_cancel)
dev.off()

# Figure 7: cancellation rate by segment
png(file.path(out_dir, "fig7_cluster_cancel_rate.png"), width = 1800, height = 1100, res = 200)
par(mar = c(4.5, 4.8, 1.5, 1), las = 1, bty = "n")
overall_rate <- mean(hotel$booking_status == "Canceled")
bp <- barplot(100 * profile$cancel_rate, names.arg = paste("Segment", profile$segment),
              col = ifelse(profile$cancel_rate > overall_rate, col_cancel, col_keep), border = NA,
              ylim = c(0, 100 * max(profile$cancel_rate) * 1.2), ylab = "Cancellation rate (%)")
abline(h = 100 * overall_rate, lty = 2, col = col_grey, lwd = 2)
text(bp, 100 * profile$cancel_rate, labels = paste0(round(100 * profile$cancel_rate), "%"),
     pos = 3, cex = 1.1, col = col_dark)
text(max(bp), 100 * overall_rate, labels = paste0("Hotel average ", round(100 * overall_rate), "%"),
     pos = 3, cex = 0.85, col = col_grey)
dev.off()

# Link to the supervised models: Random Forest score by segment (test set only)
seg_test <- hotel$segment[-train_idx]
seg_rf <- data.frame(
  segment            = levels(seg_test),
  test_bookings      = as.integer(table(seg_test)),
  actual_cancel_rate = as.numeric(tapply(test_pos, seg_test, mean)),
  avg_rf_score       = as.numeric(tapply(p_rf, seg_test, mean)),
  rf_accuracy        = as.numeric(tapply((p_rf > 0.5) == test_pos, seg_test, mean)))
cat("\nRandom Forest by segment (test set):\n")
print(seg_rf, row.names = FALSE, digits = 3)
write.csv(seg_rf, file.path(out_dir, "cluster_vs_rf_test_set.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------
# 7. Finish
# ------------------------------------------------------------------------------
cat("\n==================== SESSION INFO ====================\n")
print(sessionInfo())
t_end <- Sys.time()
cat("\nRun finished:", format(t_end), " (", round(as.numeric(difftime(t_end, t_start, units = "mins")), 1), "minutes )\n")
sink()
writeLines(format(t_end), file.path(out_dir, "DONE.txt"))
cat("All results saved in folder:", normalizePath(out_dir), "\n")
