library(dplyr)
library(psych)
library(nFactors)
library(base)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

full_file <- read.csv("../../UKB_brain_mri_raw/ukb51917_T2_MRIvar_RBC-IDP_all.tab", sep="\t")
View(full_file)

# get rid of unwanted columns for better overview
reduce_file <- full_file %>% select('f.eid','f.31.0.0', "f.21003.2.0",
                                    "f.30050.2.0",	##	Mean corpuscular haemoglobin
                                    "f.30060.2.0",	##	Mean corpuscular haemoglobin concentration
                                    "f.30040.2.0",	##	Mean corpuscular volume
                                    "f.30030.2.0",	##	Haematocrit percentage
                                    "f.30020.2.0",	##	Haemoglobin concentration
                                    "f.30010.2.0",	##	Red blood cell (erythrocyte) count
                                    "f.30070.2.0",	##	Red blood cell (erythrocyte) distribution widt
                                    matches("^f.258.*.2.0$"),	##	Volume of grey matter areas
                                    matches("^f.259.*.2.0$"),	##	Volume of grey matter areas
                                    matches("^f.257.*.2.0$"), ##	Volume of grey matter areas
                                    matches("^f.2503.*.2.0$"), ##	Median T2star areas
                                    matches("^f.2502.*.2.0$"), ##	Median T2star areas
                                    matches("^f.42018.*.2.0$"),
                                    "f.25756.2.0",	##	Scanner lateral (X) brain position	T1 structural brain MRI
                                    "f.25758.2.0",	##	Scanner longitudinal (Z) brain position	T1 structural brain MRI
                                    "f.25759.2.0",	##	Scanner table position	T1 structural brain MRI
                                    "f.25757.2.0")	##	Scanner transverse (Y) brain position	## Date of all cause dementia report
View(reduce_file)

file_noNA <- na.omit(reduce_file)
View(file_noNA)

# Renaming for understandability
file_fin <- rename(file_noNA,'ID' = 'f.eid', 'sex'='f.31.0.0','age'="f.21003.2.0","mean_corp_hb"="f.30050.2.0", "mean_corp_hb_concent"= "f.30060.2.0",
                   "mean_corp_vol" = "f.30040.2.0", "Hct_percent"="f.30030.2.0",	"hb_concent"="f.30020.2.0", "erythrocyte_cnt"="f.30010.2.0",
                   "erythrocyte_dist_wdt"="f.30070.2.0",
                   "T2star_acc_L" = "f.25038.2.0", "T2star_acc_R" = "f.25039.2.0", "T2star_amyg_L" = "f.25036.2.0",
                   "T2star_amyg_R" = "f.25037.2.0", "T2star_caud_L" = "f.25028.2.0", "T2star_caud_R" = "f.25029.2.0",	##	Median T2star in caudate (right)
                   "T2star_hipp_L" = "f.25034.2.0",	"T2star_hipp_R" = "f.25035.2.0","T2star_palli_L" = "f.25032.2.0",	##	Median T2star in pallidum (left)
                   "T2star_palli_R" = "f.25033.2.0", "T2star_put_L" = "f.25030.2.0",	##	Median T2star in putamen (left)
                   "T2star_put_R" = "f.25031.2.0", "T2star_tha_L" = "f.25026.2.0", "T2star_tha_R" = "f.25027.2.0",
                   "scan_lat"="f.25756.2.0","scan_long"="f.25758.2.0","scan_tab"="f.25759.2.0","scan_trans"="f.25757.2.0")
View(file_fin)

file_cor <- file_fin %>% select('sex','age',"mean_corp_hb", "mean_corp_hb_concent","mean_corp_vol", "Hct_percent","hb_concent", "erythrocyte_cnt",
                                "erythrocyte_dist_wdt","T2star_acc_L", "T2star_acc_R", "T2star_amyg_L","T2star_amyg_R", "T2star_caud_L", "T2star_caud_R",
                                "T2star_hipp_L",	"T2star_hipp_R","T2star_palli_L","T2star_palli_R", "T2star_put_L","T2star_put_R", "T2star_tha_L", "T2star_tha_R")
View(file_cor)
#Associations Heatmap: self correlation
COR_comp <- cor(file_cor,file_cor, use="pairwise", method="spearman")
CorrPlot <- cor.plot(COR_comp,numbers=TRUE,colors=TRUE,n=100,main="Blood-T2* correlation",zlim=c(-1,1), show.legend=TRUE, diag=FALSE, labels=NULL,n.legend=10,keep.par=TRUE,select=NULL,pval=c(0.001, 0.01, 0.05),cuts=c(.001,.01),cex.axis=0.5,stars=TRUE)

s#DO a regression
x <- file_cor$Hct_percent
y <- file_cor$T2star_amyg_L
x_plus <- x#+file_cor$age+file_cor$sex
relation <- lm(y~x_plus)
# Plot the chart.
plot(x_plus,y,col = "blue",main = "T2star_amyg_L & Hct_percent",
     abline(relation),xlab = "Hct_percent",ylab = "T2star_amyg_L")

# DO regression over all areas
for (names_ in colnames(file_cor)) {
  print(names_)
  if (startsWith(names_,"T2star")) {
    x <- file_cor$hb_concent
    y <- file_cor[ , names_]
    relation <- lm(y~x)
    # Plot the chart.
    plot(y,x,col = "blue",main = sprintf("%s & hb_concent",names_),
         abline(lm(x~y)),xlab = "hb_concent",ylab = names_)
  }}

# DO regression over all blood measures
for (names_ in colnames(file_cor)) {
  print(names_)
  if (!startsWith(names_,"T2star")) {
    y <- file_cor$T2star_hipp_L
    x <- file_cor[ , names_]
    relation <- lm(y~x+file_cor$age+file_cor$sex)
    # Plot the chart.
    plot(y,x,col = "blue",main = sprintf("%s & T2star_hipp_L",names_),
         abline(lm(x~y)),xlab = names_,ylab = "T2star_hipp_L")
  }}

#Is data suited for PCA?
cortest.bartlett(file_cor, n=nrow(file_cor_red)) #should be significant -> error message?
KMO_UKB<-KMO(file_cor_red); dim(file_cor_red); KMO_UKB #should be >0.5; consider removal of variables <0.5

file_cor_red <- subset(file_cor, select = - c(mean_corp_vol, mean_corp_hb_concent,mean_corp_hb,erythrocyte_dist_wdt))

#Check Number of Components (Scree plot and parallel analysis)
fitPCA_UKB <- princomp(file_cor_red)
par(mfrow=c(1,1))
plot(fitPCA_UKB, type="lines") # scree plot with Eigenvalues
eigenvals <- eigen(cor(file_cor_red)) # compute eigenvalues
par <- parallel(subject=nrow(file_cor_red), var=ncol(file_cor_red), rep=100, cent=0.05)
Plot <- nScree(x= eigenvals$values, aparallel=par$eigen$qevpea)
plotnScree(Plot)

#.....parameters for factors & rotation method....#
nfac <- 5
rotation <-"varimax"
# ... choose data:
pcadata <- file_cor #data all measured on same scale
#pcadata <- as.data.frame(lapply(zproso, scale)) #if scales are measured in same units, otherwise optional
rownames(pcadata) <- rownames(file_cor)
#Run PCA
PCA <- psych::principal(pcadata, nfactors = nfac, rotate = rotation)
print.psych(PCA, cut=0.3, sort = TRUE)
capture.output(print.psych(PCA, cut=0.3, sort = TRUE), file="PCA_reduced.txt")
#Extract scores: this only workes if you have used data as input, not correlation matrix
PCAscores <- data.frame(PCA$scores)
PCAscores$twuid <- rownames(PCAscores)
save(PCAscores, file="DatafilePCA_noNA.RData")

# Save altered and optimized data file
write.csv(file_fin,'final_brain_vol_info.csv')

# Create script to only have table entries where a images exists
img_list <- list.files(path="../SWI_images")
swi_imgs <- data.frame(matrix(unlist(img_list), nrow=length(img_list), byrow=TRUE))
colnames(swi_imgs) <- c('SWI')
swi_df <- data.frame(do.call('rbind', strsplit(as.character(swi_imgs$SWI),'_')))
swi_df <- subset(swi_df, select = c(X1))
colnames(swi_df) <- c('ID')
swi_df$ID <- as.integer(swi_df$ID)

full_file <- read.csv("final_brain_vol_info.csv")
swi_joint <- merge(x = swi_df, y = full_file, by = "ID")
write.csv(swi_joint,'swi_brain_vol_info.csv')

