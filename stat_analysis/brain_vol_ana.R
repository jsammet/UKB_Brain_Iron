library(dplyr)
library(psych)
library(nFactors)
library(base)
library(corrplot)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

full_file <- read.csv("../../UKB_brain_mri_raw/ukb51917_T2_MRIvar_RBC-IDP_all.tab", sep="\t")
View(full_file)

dementia_file <- full_file %>% dplyr:: select(starts_with('f.eid') | 
                                                starts_with("f.42018") | ## Date of all cause dementia report
                                                starts_with("f.42019") | ## Source of all cause dementia report
                                                starts_with("f.42020") | ## Date of alzheimer's disease report
                                                starts_with("f.42021")) ## Source of alzheimer's disease report
View(dementia_file)
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

file_noNA <- reduce_file[complete.cases(reduce_file), ] 
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

# Save altered and optimized data file
write.csv(file_fin,'final_brain_vol_info.csv')

############################################START STAT ANALYSIS#############################################################################

# Plot distribution of blood measures
hist(file_fin$erythrocyte_dist_wdt, breaks=60,cex.axis=1.5,cex.lab=1.5, cex.main=1.5, xlab = "Erythrocyte distribution width", main="Histogram of Erythrocyte distribution width")

file_cor <- file_fin %>% select('sex','age',"mean_corp_hb", "mean_corp_hb_concent","mean_corp_vol", "Hct_percent","hb_concent", "erythrocyte_cnt",
                                "erythrocyte_dist_wdt","T2star_acc_L", "T2star_acc_R", "T2star_amyg_L","T2star_amyg_R", "T2star_caud_L", "T2star_caud_R",
                                "T2star_hipp_L",	"T2star_hipp_R","T2star_palli_L","T2star_palli_R", "T2star_put_L","T2star_put_R", "T2star_tha_L", "T2star_tha_R")
View(file_cor)
#Associations Heatmap: self correlation
COR_comp <- cor(file_cor,file_cor, method="pearson")
file_cor$sex <- as.numeric(file_cor$sex)
file_cor$age <- as.numeric(file_cor$age)
rs <- corr.test(COR_comp, method="spearman")
CorrPlot <- cor.plot(COR_comp,numbers=TRUE,colors=TRUE,main="Blood-T2* correlation",zlim=c(-1,1),n.legend=10,cex.axis=1,cex=0.75,stars=TRUE)

testRes = cor.mtest(COR_comp, conf.level = 0.95)

pAdj <- p.adjust(c(testRes[[1]]), method = "bonferroni")
resAdj <- matrix(pAdj, ncol = dim(testRes[[1]])[1])

rownames(resAdj) <- rownames(testRes$p)
colnames(resAdj) <- colnames(testRes$p)

corrplot(COR_comp, p.mat = resAdj, sig.level = 0.05, method='color', diag=FALSE, addCoef.col ='black', number.cex = 1.0, insig='blank')
par(mar=c(5,6,4,1)+.1)
#DO a regression
x <- file_cor$Hct_percent
y <- file_cor$T2star_amyg_L
x_plus <- x#+file_cor$age+file_cor$sex
relation <- lm(y~x_plus)
# Plot the chart.
plot(x_plus,y,col = "blue",main = "T2star_amyg_L & Hct_percent",
     abline(relation),xlab = "Hct_percent",ylab = "T2star_amyg_L",)
View(file_cor)
# DO regression over all areas
for (names_ in colnames(file_cor)) {
  print(names_)
  if (startsWith(names_,"T2star")) {
    x <- file_cor$hb_concent
    y <- file_cor[ , names_]
    relation <- lm(y~x+file_cor$sex+file_cor$age)
    summary(relation)
    # Plot the chart.
    plot(y,x,col=rgb(red=0.0, green=0.0, blue=1.0, alpha=0.2),main = sprintf("Linear regression: %s & Haemoglobin concentration",names_), abline(lm(x~y)),xlab = "hb concentration",ylab = names_,cex.axis=2.5,cex.lab=2.5,cex.main=2.5)
  }}
summary(relation)

# DO regression over all blood measures
for (names_ in colnames(file_cor)) {
  print(names_)
  if (!startsWith(names_,"T2star")) {
    y <- file_cor$T2star_hipp_L
    x <- file_cor[ , names_]
    relation <- lm(y~x+file_cor$age+file_cor$sex)
    # Plot the chart.
    plot(y,x,col=rgb(red=0.0, green=0.0, blue=1.0, alpha=0.2),main = sprintf("%s & T2star_hipp_L",names_),
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

############################################END STAT ANALYSIS#############################################################################

############################################SWI IMAGE LIST#############################################################################

# Create script to only have table entries where a images exists
img_list <- list.files(path="../../SWI_images")
swi_imgs <- data.frame(matrix(unlist(img_list), nrow=length(img_list), byrow=TRUE))
colnames(swi_imgs) <- c('SWI')
swi_df <- data.frame(do.call('rbind', strsplit(as.character(swi_imgs$SWI),'_')))
swi_df <- subset(swi_df, select = c(X1))
colnames(swi_df) <- c('ID')
swi_df$ID <- as.integer(swi_df$ID)

full_file <- read.csv("final_brain_vol_info.csv")
View(full_file)

h <- hist(full_file$hb_concent, breaks=100)
perc <- quantile(full_file$hb_concent, c(0,.1, .2, .3, .4, .5, .6, .7, .8, .9, 1)) # for 10 groups
perc <- quantile(full_file$hb_concent, c(0, 1/3, 2/3 , 1)) # for 3 groups
cuts <- cut(h$breaks, perc)

# plot the histogram with the colours
png("histo_10cl.png", res = 1200, width = 9, height = 6, units = "in")
plot(h, col=c("green","red","blue","orange","cyan","white","black","yellow","pink","gray")[cuts],main="Hb concentration in 10 percentiles",
     xlab="Hb concentration",cex.axis=1.5, cex.lab=1.3, cex.main=1.5)
dev.off()

#plot histogram
y <- full_file$f.25881.2.0
x <- full_file$mean_corp_hb
plot(y,x,abline(lm(x~y)))
full_file <- rename(full_file,'ID' = 'f.eid')
swi_joint <- merge(x = swi_df, y = full_file, by = "ID")
View(swi_joint)
hist(swi_joint$erythrocyte_dist_wdt, breaks=60)

median(swi_joint$age)
sd(swi_joint$age)
min(swi_joint$hb_concent)
max(swi_joint$hb_concent)
table(swi_joint$sex)

write.csv(swi_joint,'swi_brain_vol_info.csv')

###Dementia file
dementia_file <- rename(dementia_file,'ID' = 'f.eid')
dementia_file <- rename(dementia_file,'dementia_report' = 'f.42018.0.0')
dementia_file <- rename(dementia_file,'dementia_source' = 'f.42019.0.0')
dementia_file <- rename(dementia_file,'ad_report' = 'f.42020.0.0')
dementia_file <- rename(dementia_file,'ad_source' = 'f.42021.0.0')
dementia_file[is.na(dementia_file)] <- -1
View(dementia_file)

### Additional file scan and negative controls
full_file_small <- read.csv("../../UKB_brain_mri_raw/ukb51917_T2_MRIvar_filt_v3.tab", sep="\t")
View(full_file_small)
full_file_small <- rename(full_file_small,'ID' = 'f.eid')
# CORRECTION FOR SCAN MEASURES
reduce_file <- full_file_small %>% select("ID",
                                    "f.25756.2.0",	##	Scanner lateral (X) brain position	T1 structural brain MRI
                                    "f.25758.2.0",	##	Scanner longitudinal (Z) brain position	T1 structural brain MRI
                                    "f.25759.2.0",	##	Scanner table position	T1 structural brain MRI
                                    "f.25757.2.0",	##	Scanner transverse (Y) brain position
                                    "f.25738.2.0",  ##  T1 SWI image difference
                                    "f.25000.2.0",  ##  head volumne
                                    "f.4990.2.0",   ##  fluid intelligence
                                    ## negative controls
                                    "f.1747.2.0",#,  ## hair colour
                                    "f.54.2.0") ## assessment centre
                                    #"f.796.2.0")   ## distance to work
file_fin <- rename(reduce_file, 'Scan_lat_X' = "f.25756.2.0", 'Scan_long_Z' = "f.25758.2.0", 'Scan_table_pos' = "f.25759.2.0",
                   'Scan_trans_Y' = "f.25757.2.0", 'T1_SWI_diff' = "f.25738.2.0", "fluid_int"="f.4990.2.0",
                   'head_vol' = "f.25000.2.0", 'hair_col' = "f.1747.2.0", 'assess_centre'='f.54.2.0')#, 'dis_to_work' = "f.769.2.0")
swi_file <- read.csv('../swi_brain_vol_info.csv')

# merge swi file and additional info
swi_joint <- merge(x = swi_file, y = file_fin, by = "ID",x.all=TRUE)
swi_joint <- subset(swi_joint, select = -c(X.1))
# merge swi file and dementia info
swi_joint <- merge(x = swi_joint, y = dementia_file, by = "ID",x.all=TRUE)
View(swi_joint)

# save new info file
write.csv(swi_joint,'swi_brain_vol_info_additional.csv')
swi_joint <- read.csv('swi_brain_vol_info_additional.csv')
View(swi_joint)

# Check age distribution of both sexes
median(swi_joint$age[swi_joint$sex==1])
sd(swi_joint$age[swi_joint$sex==1])
median(swi_joint$age[swi_joint$sex==0])
sd(swi_joint$age[swi_joint$sex==0])

# check values of interest
table(swi_joint$assess_centre)
swi_joint$fluid_int[is.na(swi_joint$fluid_int)] <- 0
table(swi_joint$fluid_int)
table(swi_joint$dementia_source)
View(swi_joint[ which(swi_joint$dementia_source != -1) ,])


hist(swi_joint$hb_concent, breaks=100, xlab = "Hb concentration", main = "Distribution of hb concentration")
hist(swi_joint$mean_corp_hb, breaks=100, xlab = "Mean corpuscular hb", main = "Distribution of mean corpuscular hb")
hist(swi_joint$Hct_percent, breaks=100, xlab = "Hematocrit concentration", main = "Distribution of hematocrit concentration")

y <- full_file$f.25881.2.0
x <- full_file$mean_corp_hb
plot(y,x,abline(lm(x~y)))
full_file <- rename(full_file,'ID' = 'f.eid')
swi_joint <- merge(x = swi_df, y = full_file, by = "ID")


### Additional file cognition
swi_file_plus <- read.csv('swi_brain_vol_info_additional.csv')
full_file_cog <- read.csv("/home/jsammet/mnt_ox/UKB_brain_mri_raw/ukb51917_cog-demo_var_all_ver4.tab", sep="\t")
cognition_file <- full_file_cog %>% dplyr:: select(starts_with('f.eid') | 
                                                    starts_with('f.20016.'))
cognition_file <- rename(cognition_file,'ID' = 'f.eid', 'fluid_int_base' = 'f.20016.0.0','fluid_int_v2' = 'f.20016.2.0')
View(cognition_file)

swi_joint_cogn <- merge(x = swi_file_plus, y = cognition_file, by = "ID",x.all=TRUE)
swi_joint_cogn <- subset(swi_joint, select = -c(X.1))
View(swi_joint_cogn)
table(swi_joint_cogn$fluid_int_v2, useNA = "always")
swi_joint_cogn$fluid_int_v2[is.na(swi_joint$fluid_int_v2)] <- -1
table(swi_joint_cogn$fluid_int_v2, useNA = "always")
write.csv(swi_joint_cogn,'swi_brain_vol_info_cognition.csv')

swi_cogn <- read.csv('swi_brain_vol_info_cognition.csv')
table(swi_cogn$fluid_int_v2)
mean(swi_cogn$age)
sd(swi_cogn$age)

############################################T2* IMAGE LIST#############################################################################

# Create script to only have table entries where a images exists
img_list <- list.files(path="../../T2star_images")
t2_imgs <- data.frame(matrix(unlist(img_list), nrow=length(img_list), byrow=TRUE))
colnames(t2_imgs) <- c('T2star')
t2_df <- data.frame(do.call('rbind', strsplit(as.character(t2_imgs$T2star),'_')))
t2_df <- subset(t2_df, select = c(X1))
colnames(t2_df) <- c('ID')
t2_df$ID <- as.integer(t2_df$ID)

# Read full data file
full_file <- read.csv("../../UKB_brain_mri_raw/ukb51917_T2_MRIvar_RBC-IDP_all.tab", sep="\t")
# get rid of unwanted columns for better overview
reduce_file <- full_file %>% select('f.eid','f.31.0.0', "f.21003.2.0",
                                    "f.30050.2.0",	##	Mean corpuscular haemoglobin
                                    "f.30060.2.0",	##	Mean corpuscular haemoglobin concentration
                                    "f.30040.2.0",	##	Mean corpuscular volume
                                    "f.30030.2.0",	##	Haematocrit percentage
                                    "f.30020.2.0",	##	Haemoglobin concentration
                                    "f.30010.2.0",	##	Red blood cell (erythrocyte) count
                                    "f.30070.2.0",	##	Red blood cell (erythrocyte) distribution widt
                                  #  matches("^f.258.*.2.0$"),	##	Volume of grey matter areas
                                  #  matches("^f.259.*.2.0$"),	##	Volume of grey matter areas
                                  #  matches("^f.257.*.2.0$"), ##	Volume of grey matter areas
                                  #  matches("^f.2503.*.2.0$"), ##	Median T2star areas
                                  #  matches("^f.2502.*.2.0$"), ##	Median T2star areas
                                  #  matches("^f.42018.*.2.0$"),
                                    "f.25756.2.0",	##	Scanner lateral (X) brain position	T1 structural brain MRI
                                    "f.25758.2.0",	##	Scanner longitudinal (Z) brain position	T1 structural brain MRI
                                    "f.25759.2.0",	##	Scanner table position	T1 structural brain MRI
                                    "f.25757.2.0",	##	Scanner transverse (Y) brain position
                                    ) ## assessment centre
# file_noNA <- reduce_file[complete.cases(reduce_file), ] 
# Renaming for understandability
file_fin <- rename(reduce_file,'ID' = 'f.eid', 'sex'='f.31.0.0','age'="f.21003.2.0","mean_corp_hb"="f.30050.2.0", "mean_corp_hb_concent"= "f.30060.2.0",
                   "mean_corp_vol" = "f.30040.2.0", "Hct_percent"="f.30030.2.0",	"hb_concent"="f.30020.2.0", "erythrocyte_cnt"="f.30010.2.0",
                   "erythrocyte_dist_wdt"="f.30070.2.0",
                #   "T2star_acc_L" = "f.25038.2.0", "T2star_acc_R" = "f.25039.2.0", "T2star_amyg_L" = "f.25036.2.0",
                #   "T2star_amyg_R" = "f.25037.2.0", "T2star_caud_L" = "f.25028.2.0", "T2star_caud_R" = "f.25029.2.0",	##	Median T2star in caudate (right)
                #   "T2star_hipp_L" = "f.25034.2.0",	"T2star_hipp_R" = "f.25035.2.0","T2star_palli_L" = "f.25032.2.0",	##	Median T2star in pallidum (left)
                #   "T2star_palli_R" = "f.25033.2.0", "T2star_put_L" = "f.25030.2.0",	##	Median T2star in putamen (left)
                #   "T2star_put_R" = "f.25031.2.0", "T2star_tha_L" = "f.25026.2.0", "T2star_tha_R" = "f.25027.2.0",
                   "scan_lat"="f.25756.2.0","scan_long"="f.25758.2.0","scan_tab"="f.25759.2.0","scan_trans"="f.25757.2.0")

full_file_control <- read.csv("../../UKB_brain_mri_raw/ukb51917_T2_MRIvar_filt_v3.tab", sep="\t")
# CORRECTION FOR SCAN MEASURES
reduce_ctrl_file <- full_file_control %>% select("f.eid",
                                          "f.25738.2.0",  ##  T1 SWI image difference
                                          "f.25000.2.0",  ##  head volumne
                                          "f.4990.2.0",   ##  fluid intelligence
                                          ## negative controls
                                          "f.1747.2.0",#,  ## hair colour
                                          "f.54.2.0") ## assessment centre
#"f.796.2.0")   ## distance to work
file_ctrl_fin <- rename(reduce_ctrl_file, 'ID' = 'f.eid', 'T1_SWI_diff' = "f.25738.2.0", "fluid_int"="f.4990.2.0",
                   'head_vol' = "f.25000.2.0", 'hair_col' = "f.1747.2.0", 'assess_centre'='f.54.2.0')#, 'dis_to_work' = "f.769.2.0")

### Additional file cognition
full_file_cog <- read.csv("/home/jsammet/mnt_ox/UKB_brain_mri_raw/ukb51917_cog-demo_var_all_ver4.tab", sep="\t")
cognition_file <- full_file_cog %>% dplyr:: select(starts_with('f.eid') | 
                                                     starts_with('f.20016.'))
cognition_file <- rename(cognition_file,'ID' = 'f.eid', 'fluid_int_base' = 'f.20016.0.0','fluid_int_v2' = 'f.20016.2.0')

t2_joint_v1 <- merge(x = t2_df, y = file_fin, by = "ID", all.x = TRUE)
t2_joint_v2 <- merge(x = t2_joint_v1, y = file_ctrl_fin, by = "ID", all.x = TRUE)
t2_joint <- merge(x = t2_joint_v2, y = cognition_file, by = "ID", all.x = TRUE)
View(t2_joint)
write.csv(t2_joint,'t2_brain_vol_info_complete.csv')
