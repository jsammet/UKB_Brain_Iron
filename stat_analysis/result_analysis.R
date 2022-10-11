library(dplyr)
library(psych)
library(nFactors)
library(base)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#range analysis of labels
full_file <- read.csv("final_brain_vol_info.csv")
View(full_file)
max(full_file$mean_corp_hb)
min(full_file$mean_corp_hb)
( ceiling(max(full_file$mean_corp_hb)) - floor(min(full_file$mean_corp_hb)) )/100

results <- read.csv("../results/test_numeric_100_0.0001_5e-07_mean_corp_hb.csv")
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

# check for accuracy case 0
TP_0 <- nrow(results[ which(results$Prediction == "0" & results$True_Label == "0") ,])
FP_0 <- nrow(results[ which(results$Prediction == "0" & results$True_Label != "0") ,])
FN_0 <- nrow(results[ which(results$Prediction != "0" & results$True_Label == "0") ,])
TN_0 <- nrow(results[ which(results$Prediction != "0" & results$True_Label != "0") ,])
# Sensitivity
TP_0 / (TP_0 + FN_0)
# Specifictiy
TN_0 / (TN_0 + FP_0)

# check for accuracy case 1
TP_1 <- nrow(results[ which(results$Prediction == "1" & results$True_Label == "1") ,])
FP_1 <- nrow(results[ which(results$Prediction == "1" & results$True_Label != "1") ,])
FN_1 <- nrow(results[ which(results$Prediction != "1" & results$True_Label == "1") ,])
TN_1 <- nrow(results[ which(results$Prediction != "1" & results$True_Label != "1") ,])
# Sensitivity
TP_1 / (TP_1 + FN_1)
# Specifictiy
TN_1 / (TN_1 + FP_1)

# check for accuracy case 2
TP_2 <- nrow(results[ which(results$Prediction == "2" & results$True_Label == "2") ,])
FP_2 <- nrow(results[ which(results$Prediction == "2" & results$True_Label != "2") ,])
FN_2 <- nrow(results[ which(results$Prediction != "2" & results$True_Label == "2") ,])
TN_2 <- nrow(results[ which(results$Prediction != "2" & results$True_Label != "2") ,])
# Sensitivity
TP_2 / (TP_2 + FN_2)
# Specifictiy
TN_2 / (TN_2 + FP_2)
