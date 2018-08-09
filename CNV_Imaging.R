CNV = read.table("Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.tab", sep="\t", header = T)
library(dplyr)
head(CNV)
rownames(CNV) = CNV$Gene.Symbol
CNV$Gene.Symbol = NULL
tablex = apply(CNV, 2, sum) 
summary(CNV)
table(CNV)
df =data.frame(sample_id=character(),
           Homo_zygote_gain=numeric(),
           Hemi_zygote_gain=numeric(),
           NoChange=numeric(),
           Hemi_zygote_loss=numeric(),
           Homo_zygote_loss=numeric()
           )

for (col in colnames(CNV)) {
  no_change = sum(CNV[col]==0)
  Homo_zygote_gain = sum(CNV[col]==2)
  Hemi_zygote_gain = sum(CNV[col]==1)
  Homo_zygote_loss = sum(CNV[col]==-2)
  Hemi_zygote_loss = sum(CNV[col]==-1)
  
  df[nrow(df)+1,] = list(col,Homo_zygote_gain, Hemi_zygote_gain, no_change, Hemi_zygote_loss, Homo_zygote_loss)
}
df$sample_id = colnames(CNV)
max(df$NoChange)
sumc =summary(df$NoChange)
sumc[2]
sumc[5]
df$type_status = df$NoChange > sumc[2] & df$NoChange < sumc[5]
df$HoGain = df$Homo_zygote_gain>median(df$Homo_zygote_gain)
df$HeGain = df$Hemi_zygote_gain>median(df$Hemi_zygote_gain)
df$HeLoss = df$Hemi_zygote_loss>median(df$Hemi_zygote_loss)
df$HoLoss = df$Homo_zygote_loss>median(df$Homo_zygote_loss)
df$CNV_Changed = df$HoGain  + df$HeGain + df$HeLoss+df$HoLoss
df$Gain = df$HoGain  + df$HeGain >0
df$Loss = df$HoLoss  + df$HeLoss >0
df$CNV_D = df$Gain + df$Loss
df$CNV_Status = 0
df$CNV_Status[df$Gain==1 & df$CNV_D==1] = 1
df$CNV_Status[df$Loss==1 & df$CNV_D==1] = 2
df$CNV_Status[df$CNV_D==2 & df$Loss_ov_Gain==F] = 3
df$CNV_Status[df$CNV_D==2 & df$Loss_ov_Gain==T] = 4
df$Gain_n = df$HoGain + df$HeGain
df$Loss_n = df$HoLoss + df$HeLoss
df$Loss_ov_Gain = (df$Homo_zygote_loss+df$Homo_zygote_loss)> (df$Homo_zygote_gain+df$Hemi_zygote_gain)
df$Loss_eq_Gain = (df$Homo_zygote_loss+df$Homo_zygote_loss)== (df$Homo_zygote_gain+df$Hemi_zygote_gain)
df$Condition = rowSums(df[,c("Homo_zygote_gain","Hemi_zygote_gain", "Homo_zygote_loss","Hemi_zygote_loss")])
df$Condition = df$Condition / nrow(CNV)
x = df$Condition
cuts <- quantile(x, c(0, 0.25,0.50, 0.75,1))
QuantileThreshold = cut(x, cuts, include.lowest = TRUE)
df$QuantileThreshold <-QuantileThreshold
df$Cluster <- as.numeric(df$QuantileThreshold) -1
table(df$Cluster,df$QuantileThreshold)
write.table(df,"sample_cnv.csv", sep=",", row.names = F)
summary(df)

t.test(df$gai)
table(df$CNV_Status)
table(df$CNV_Changed, df$Loss )
summary(df)



df$NoChange = NULL
View(df)
df$type_status = 0
argmax(df, rows = TRUE)

df$type_status = colnames(df)[apply(df,1,which.max)]
table(df$type_status)

  
sum(CNV>0)
  
table(CNV$TCGA.2A.A8VT.01)
data %>% group_by()