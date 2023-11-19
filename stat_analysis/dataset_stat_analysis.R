library(dplyr)
library(psych)
library(nFactors)
library(base)
library(corrplot)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

info_df <- read.csv("swi_brain_vol_info.csv")

# Age
mean(info_df$age)
median(info_df$age)
sd(info_df$age)
table(info_df$age)

# Haemoglobine
quantile(info_df$hb_concent, prob=c(1/3,2/3))
sum(info_df$hb_concent<13.63, na.rm=TRUE)
sum(info_df$hb_concent>=13.63 & info_df$hb_concent<14.7, na.rm=TRUE)
sum(info_df$hb_concent>=14.7, na.rm=TRUE)
