##----------------------------------
##  Code Version 1.0
##  This breast cancer code for Random Forest embeded in the concatenation-based framework.
##  Concate_RF
##  Created by Zi-Yi Yang 
##  Modified by Zi-Yi Yang on July 23, 2019
##  Concact: yangziyi091100@163.com
##----------------------------------

library(caret)
library(randomForest)
##--------------
# 1. Load data
##--------------
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
# 2. Concatenation_RF
##--------------
combined_train_data <- do.call(cbind, train_data)
combined_test_data <- do.call(cbind, test_data)

net <- randomForest(x = combined_train_data, y = factor(train_label), 
                    importance = TRUE, ntree = 10)
valprediction <- predict(net, combined_train_data)
tsprediction <- predict(net, combined_test_data)
coef.idx <- which(net$importance[,4]!=0)
coef.name <- colnames(combined_train_data)[coef.idx]

##----------------------------
# 3. Evaluate the performance
##----------------------------
conMatrix.train <- confusionMatrix(as.factor(train_label),as.factor(valprediction))
conMatrix.test <- confusionMatrix(as.factor(test_label),as.factor(tsprediction),mode = "prec_recall")

perf.ConcatenationRF <- list("Feature" = coef.name,
                             "Feature.idx" = coef.idx,
                             "Perf.Train" = conMatrix.train,
                             "Perf.Test" = conMatrix.test)
