##----------------------------------
##  Code Version 1.0
##  This breast cancer code for Random Forest embeded in the ensemble-based framework.
##  Ensemble_RF
##  Created by Zi-Yi Yang 
##  Modified by Zi-Yi Yang on July 23, 2019
##  Concact: yangziyi091100@163.com
##----------------------------------

library(randomForest)
library(caret)
##--------------
# 1. Load data
##--------------
source("D:/Ziyi/School/PMO/12.Multi_omics/7.MSPL-code/3-Breast/Function_performance.R")
load("D:/Ziyi/School/PMO/12.Multi_omics/7.MSPL-code/3-Breast/1-data/1-datatrainTestDatasetsNormalized.RDATA")

## Preparing data
train_data <- list("mrna" = mrnaTest0,
                   "mirna" = mirnaTest0,
                   "meth" = methTest0)
colnames(train_data$mrna) <- paste("mrna", colnames(train_data$mrna), sep = "_")
colnames(train_data$mirna) <- paste("mirna", colnames(train_data$mirna), sep = "_")
colnames(train_data$meth) <- paste("meth", colnames(train_data$meth), sep = "_")
train_group <- pam50Test0$Call


test_data <- list("mrna" = mrnaTrain0,
                  "mirna" = mirnaTrain0,
                  "meth" = methTrain0)
colnames(test_data$mrna) <- paste("mrna", colnames(test_data$mrna), sep = "_")
colnames(test_data$mirna) <- paste("mirna", colnames(test_data$mirna), sep = "_")
colnames(test_data$meth) <- paste("meth", colnames(test_data$meth), sep = "_")
test_group <- pam50Train0$Call

train_label <-as.numeric(train_group)
test_label <-as.numeric(test_group)

##--------------
# 2.EnsembleRF
##--------------
ensemblePanels <- lapply(train_data, function(i){
  RF.colon <- randomForest(x = i, y = factor(train_label), importance = TRUE, ntree = 8)
})

ensembleValiPredction <- mapply(function(fit,x){
  valprediction <- predict(fit,x)
}, fit = ensemblePanels, x = train_data)

ensembleTestPredction <- mapply(function(fit,x){
  tsprediction <- predict(fit,x)
}, fit = ensemblePanels, x = test_data)

valprediction <- evaluate.ensemble(ensembleValiPredction, train_label)
tsprediction <- evaluate.ensemble(ensembleTestPredction, test_label)

ensembleCoef.idx <- list()
ensembleCoef.name <- list()

for(i in 1:length(ensemblePanels)){
  ensembleCoef.idx[[i]] <- which(ensemblePanels[[i]]$importance[,4]!=0)
  ensembleCoef.name[[i]] <- colnames(train_data[[i]])[ensembleCoef.idx[[i]]]
}

##----------------------------
# 3. Evaluate the performance
##----------------------------
conMatrix.train <- confusionMatrix(as.factor(train_label),as.factor(valprediction))
conMatrix.test <- confusionMatrix(as.factor(test_label),as.factor(tsprediction),mode = "prec_recall")

perf.EnsembleRF <- list("Feature" = ensembleCoef.name,
                        "Feature.idx" = ensembleCoef.idx,
                        "Perf.Train" = conMatrix.train,
                        "Perf.Test" = conMatrix.test)

