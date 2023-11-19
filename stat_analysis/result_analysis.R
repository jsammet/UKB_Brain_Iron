library(dplyr)
library(psych)
library(nFactors)
library(base)
library('caret')
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#range analysis of labels
full_file <- read.csv("../swi_brain_vol_info.csv")
View(full_file)
max(full_file$mean_corp_hb)
min(full_file$mean_corp_hb)
( ceiling(max(full_file$mean_corp_hb)) - floor(min(full_file$mean_corp_hb)) )/100

results <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/results/test__hb_concent_batch_model_False_augment_60_eps_3_class_0.0001_lr_NT_IntGrad_seed_1337.csv")
results$True_Label[results$True_Label == 19] <- 9
results$Prediction[results$Prediction == 19] <- 9
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

### Test linear model
GLM_iron <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_beta_lin_model.csv")
names(GLM_iron)[1] <- "betas"
shuffle_iron <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_beta_shuffle.csv")
names(shuffle_iron)[1] <- "betas"
l <- GLM_iron - shuffle_iron
diff_iron <- data.frame(l)
# t test
subset_df <- diff_iron
t.test(l)
# Paired t test
t.test(GLM_iron$betas[2000000:2050000], shuffle_iron$betas[2000000:2050000], paired=TRUE)
dim(GLM_iron)

# Check accuracy
min_ <- min(results$True_Label)
max_ <- max(results$True_Label)
Sensitivity <- 0
Specificity <- 0
Accuracy <- 0
for (i in min_:max_) {
  i
  TP_ <- nrow(results[ which(results$Prediction == i & results$True_Label == i) ,])
  FP_ <- nrow(results[ which(results$Prediction == i & results$True_Label != i) ,])
  FN_ <- nrow(results[ which(results$Prediction != i & results$True_Label == i) ,])
  TN_ <- nrow(results[ which(results$Prediction != i & results$True_Label != i) ,])
  Sensitivity <- Sensitivity + TP_ / (TP_ + FN_)
  Specificity <- Specificity + TN_ / (TN_ + FP_)
  Accuracy <- Accuracy + TP_
}
Sensitivity <- Sensitivity / (max_+1)
Specificity <- Specificity / (max_+1)
Accuracy <- Accuracy / length(results$True_Label)
Sensitivity
Specificity
Accuracy
# Accuracy per class
class_sensitivity <- array(max_)
class_specificity <- array(max_)
label_list <- min_:max_
for (i in min_:max_) {
  i
  TP_ <- nrow(results[ which(results$Prediction == i & results$True_Label == i) ,])
  FP_ <- nrow(results[ which(results$Prediction == i & results$True_Label != i) ,])
  FN_ <- nrow(results[ which(results$Prediction != i & results$True_Label == i) ,])
  TN_ <- nrow(results[ which(results$Prediction != i & results$True_Label != i) ,])
  class_sensitivity[i+1] <- TP_ / (TP_ + FN_)
  class_specificity[i+1] <- TN_ / (TN_ + FP_)
}
barplot(height=class_sensitivity, names=label_list, ylim=c(0.0,1), xlab="Classes", ylab="Sensititvity",main="Sensitivity per class",
        cex.lab=1.5, cex.names=1.5, cex.axis=1.5, cex.main=1.5)
barplot(height=class_specificity, names=label_list, ylim=c(0.0,1), xlab="Classes", ylab="Specificity",main="Specificity per class",
        cex.lab=1.5, cex.names=1.5, cex.axis=1.5, cex.main=1.5)
# confusion matrix
table(results$Prediction, results$True_Label)
confusionMatrix(data = as.factor(results$Prediction), reference = as.factor(results$True_Label))
mosaicplot(table(results$Prediction, results$True_Label),xlab="Prediction",ylab="True Label", cex.axis=1.02, cex.lab=1.3,
           main="Confusion Matrix prediction of Hb concentration in 3 classes ",shade = TRUE)

# Plot loss
loss <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/results/train_valid__hb_concent_no_batch_model_False_augment_100_eps_3_class_0.0001_lr_NT_IntGrad.csv")
View(loss)
png("plot.png", res = 1200, width = 9, height = 6, units = "in")
par(mar = c(4.5, 5, 2, 1))  # Adjust margin values as needed
plot(loss$ID, loss$train_loss, type = "l", lty = 1, col = "red", xlab = "epochs", ylab = "Cross_entropy Loss",
     main = "3 class: Loss for training & validation for hb concentration",
     ylim = c(0.55, 1.3), cex.lab = 1.3, cex.axis = 1.3, cex.main = 1.5)
lines(loss$ID, loss$valid, type = "l", lty = 1, col = "green")
legend(x = "topright", inset = 0.1, cex = 1.5,
       legend = c("train loss", "valid loss"),
       lty = c(1, 1),
       col = c(2, 3),
       lwd = 2)
dev.off()


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

# Permutation test beta
beta_ <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_beta_lin_model.csv")
shuffle_ <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_beta_shuffle.csv")
View(beta_)
max(beta_)

# Volcano plot betas and p values
pval_ <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_pval_lin_model.csv")
betas_ <- read.csv("/home/jsammet/mnt_ox/UKB_Brain_Iron/iron_beta_lin_model.csv")
data <- data.frame(pval_, betas_)
colnames(data) <- c('pval_','betas_')
x_replace <- data$pval_
x_replace[is.nan(x_replace)] <- 1
data$pval_ <- x_replace
p1 <- ggplot(data, aes(pval_, betas_)) + # -log10 conversion  
  geom_point(size = 2/5) +
  xlab(expression("p values")) + 
  ylab(expression("betas"))
p1
