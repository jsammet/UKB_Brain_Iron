library(dplyr)
library(psych)
library(nFactors)
library(base)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#range analysis of labels
full_file <- read.csv("../swi_brain_vol_info.csv")
View(full_file)
max(full_file$mean_corp_hb)
min(full_file$mean_corp_hb)
( ceiling(max(full_file$mean_corp_hb)) - floor(min(full_file$mean_corp_hb)) )/100

results <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/results/test_oneD_3class_3class_30_0.0001_5e-07_mean_corp_hb.csv")
View(results)
sd(results$True_Label)
min(results$True_Label)
max(results$True_Label)
median(results$True_Label)

sd(results$Prediction)
min(results$Prediction)
max(results$Prediction)
relation <- lm(results$True_Label~results$Prediction)
plot(results$Prediction,results$True_Label, xlim=range(20:40), ylim=range(20:40))
abline(relation)
View(relation)

df <- data.frame(results$True_Label)
df %>%
  mutate(tertiles = ntile(results$True_Label, 3)) %>%
  mutate(tertiles = if_else(tertiles == 1, 'Low', if_else(tertiles == 2, 'Medium', 'High'))) %>%
  arrange(results$True_Label)

input_df <- read.csv("../swi_brain_vol_info.csv")
View(input_df)
tert_df <- input_df %>%
  mutate(tertiles = ntile(input_df$mean_corp_hb, 3)) %>%
  mutate(tertiles = if_else(tertiles == 1, 'Low', if_else(tertiles == 2, 'Medium', 'High'))) %>%
  arrange(input_df$mean_corp_hb)
tert_df <- subset(tert_df, select = c(X.1,ID, mean_corp_hb, tertiles))
View(tert_df)

# Check accuracy
min_ <- min(results$True_Label)
max_ <- max(results$True_Label)
Sensitivity <- 0
Specificity <- 0
for (i in min_:max_) {
  i
  TP_ <- nrow(results[ which(results$Prediction == i & results$True_Label == i) ,])
  FP_ <- nrow(results[ which(results$Prediction == i & results$True_Label != i) ,])
  FN_ <- nrow(results[ which(results$Prediction != i & results$True_Label == i) ,])
  TN_ <- nrow(results[ which(results$Prediction != i & results$True_Label != i) ,])
  Sensitivity <- Sensitivity + TP_ / (TP_ + FN_)
  Specificity <- Specificity + TN_ / (TN_ + FP_)
}
Sensitivity <- Sensitivity / (max_+1)
Specificity <- Specificity / (max_+1)
Sensitivity
Specificity

# Plot loss
loss <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/results/train_valid_oneD_30_3class__0.0001mean_corp_hb.csv")
View(loss)

plot(results$True_Label,results$Orig..true.val)
min(results[ which(results$True_Label == 0) ,]$Orig..true.val)
max(results[ which(results$True_Label == 1) ,]$Orig..true.val)
min(results[ which(results$True_Label == 1) ,]$Orig..true.val)
max(results[ which(results$True_Label == 2) ,]$Orig..true.val)
results[ which(full_file$Orig..true.val == 31.52) ,]
View(full_file)
# plot iron measurements
hist(full_file$mean_corp_hb, breaks=50)
hist(full_file$mean_corp_hb_concent, breaks=50)
hist(full_file$mean_corp_vol, breaks=50)
hist(full_file$Hct_percent, breaks=50)
hist(full_file$hb_concent, breaks=50)
hist(full_file$erythrocyte_cnt, breaks=50)
hist(full_file$erythrocyte_dist_wdt, breaks=50)
