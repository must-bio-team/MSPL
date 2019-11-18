##------------------------------------------
##  Code Version 1.0
##  This breast cancer data code for multimodal self-paced learning for multiomics data analysis.
##  MSPL-multiclass
##  Created by Zi-Yi Yang 
##  Modified by Zi-Yi Yang on July 23, 2019
##  Concact: yangziyi091100@163.com
##------------------------------------------

## Load library
library(Matrix)
library(tseries)
library(glmnet)
library(caret)
library(ncvreg)
library(pROC)
library(ROCR)
library(ggplot2)

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

##-----------------
## Step 1 : Initialization parameters
##-----------------
View_num = 3
iter_num = 100
gamma = 0.02
lambda = c(0.001, 0.001, 0.001)
lambda.update = 0.06

## setting selected sample number in each iteration
sample.select <- list()
sample.add <- list()
Times <- 50         #approximate iteration times

for(i in 1: length(unique(train_label))){
  sample.select[[i]] <- 4
  sample.add[[i]] <- ceiling(length(which(train_label==i))/Times)
}


valpredmatrix = list()
valprobmatrix = list()
evlpredmatrix = list()
evlprobmatrix = list()
coefmatrix = list()
nonzerocoefmatrix = list()
coef_idx = list()
coef_coef = list()
coef_name = list()
selectedidx = list()

loss = matrix(0, nrow = length(train_label), ncol = View_num)
v_iter = matrix(0, nrow = length(train_label), ncol = View_num)

for(iter in 1:iter_num) {
  valpredmatrix[[iter]] = matrix(0, nrow = length(train_label), ncol = View_num)
  valprobmatrix[[iter]] = matrix(0, nrow = length(train_label), ncol = View_num)
  evlpredmatrix[[iter]] = matrix(0, nrow = length(test_label), ncol = View_num)
  evlprobmatrix[[iter]] = matrix(0, nrow = length(test_label), ncol = View_num)
  coefmatrix[[iter]] =  list()
  nonzerocoefmatrix[[iter]] = matrix(0, nrow = 1, ncol = View_num)
}

val_labels <- matrix(1, nrow = length(train_label), ncol = 3)
evl_labels <- matrix(1, nrow = length(test_label), ncol = 3)
valmaps <- replicate(iter_num,0)
evlmaps <- replicate(iter_num,0)

##-------------------------------------
## Step 2.1: Initialization classifier
##-------------------------------------
for(i in 1:View_num){
  cvfit<-cv.glmnet(x = train_data[[i]],
                   y = train_label,
                   alpha = 0.9,
                   family = "multinomial",
                   type.multinomial="grouped")
  valpredmatrix[[1]][,i] <- as.numeric(predict(cvfit,train_data[[i]],type="class",s="lambda.min"))
  valprobmatrix[[1]][,i] <- apply(predict(cvfit,train_data[[i]],type="response",s="lambda.min"),1,max)
}

##-----------------------
## Step 2.2: Optimization
##-----------------------
for (iter in 1:iter_num){

  if(length(unlist(selectedidx)) == (length(train_label)*View_num)){break}
  cat("Starting the ",iter,"-th iteration.\n", sep = "")
  
  for(j in 1:View_num){
    
    # update v_view
    dev_prob <- valprobmatrix[[iter]]
    v_iter = mvselfpace.rank.multiclass(dev_prob = dev_prob,
                                        true_label = train_label,
                                        lambda = lambda, 
                                        gamma = gamma,
                                        v_iter = v_iter,
                                        View_id = j,
                                        View_num = View_num,
                                        sample_select = sample.select)
    
    for(i in 1:View_num){
      selectedidx[[i]] = which(v_iter[,i]==1)
    }
    
    # update w_view Logistic with Lasso or Elasitc net
    train.idx = selectedidx[[j]]
    
    cvfit<-cv.glmnet(x = data.matrix(train_data[[j]][train.idx,]),
                     y = data.matrix(train_label[train.idx]),
                     alpha = 0.9,
                     family = "multinomial",
                     type.multinomial = "grouped") 
    
    valprediction <- as.numeric(predict(cvfit, train_data[[j]], type = "class", s = "lambda.min"))
    val.prob <- apply(predict(cvfit,train_data[[j]], type = "response", s = "lambda.min"), 1, max)
    
    tsprediction <- as.numeric(predict(cvfit,test_data[[j]], type = "class", s = "lambda.min"))
    test.prob <- apply(predict(cvfit,test_data[[j]],type = "response", s = "lambda.min"), 1, max)
    
    coefprediction <- predict(cvfit, type = "coefficients", s = "lambda.min")  # coef
    coef.idx <- which(coefprediction$`1`[-1]!=0)
    coef.name <- rownames(coefprediction$`1`)[coef.idx]
    coef.number <- length(coef.idx)
    
    valpredmatrix[[iter]][,j] = valprediction
    valprobmatrix[[iter]][,j] = val.prob
    evlpredmatrix[[iter]][,j] = tsprediction
    evlprobmatrix[[iter]][,j] = test.prob
    
    coefmatrix[[iter]][[j]] = coefprediction
    nonzerocoefmatrix[[iter]][,j] = coef.number
  }
  
  #evaluate the training and test error
  val_loss <- sum((valprobmatrix[[iter]] - val_labels)^2)
  evl_loss <- sum((evlprobmatrix[[iter]] - evl_labels)^2)
  valmaps[iter] <- val_loss
  evlmaps[iter] <- evl_loss
  
  # update lambda and valpredmatrix for next iteriation
  lambda= lambda.update + lambda
  for(i in 1:length(sample.select)){
    sample.select[[i]] <- sample.select[[i]] + sample.add[[i]]
  }
  
  valprobmatrix[[iter+1]]=valprobmatrix[[iter]]

}

##----------------------------------------------------
# Step 3: Find the run with the best valudation map
##----------------------------------------------------
## best results ##
best.iter <- which(valmaps == min(valmaps[1:length(which(valmaps!=0))]))
best_valperf <- valpredmatrix[[best.iter]]
best_evlperf <- evlpredmatrix[[best.iter]]
best_coef <- coefmatrix[[best.iter]]
best_numcoef <- nonzerocoefmatrix[[best.iter]]


## record label
final_val_label <- calculate.final.label(pred_label = best_valperf, true_label = train_label)
final_evl_label <- calculate.final.label(pred_label = best_evlperf, true_label = test_label)

## record selected features
for(i in 1:View_num){
  coef_idx[[i]] <- which(best_coef[[i]]$`1`!=0)[-1]
  coef_coef[[i]] <- best_coef[[i]]$`1`[coef_idx[[i]]]
  coef_name[[i]] <- rownames(best_coef[[i]]$`1`)[coef_idx[[i]]]
  
}
coef.mRNA <- cbind(coef_name[[1]], coef_coef[[1]])
coef.miRNA <- cbind(coef_name[[2]], coef_coef[[2]])
coef.meth <- cbind(coef_name[[3]], coef_coef[[3]])

## record results
MVSPL.result <- list("val.label" = final_val_label, 
                     "evl.label" = final_evl_label,
                     "coef.mRNA" = coef.mRNA, 
                     "coef.miRNA" = coef.miRNA,
                     "coef.meth" = coef.meth)

##-------------------------------------------
## Step 4: Evaluate the best performance
##-------------------------------------------
conMatrix.train <- confusionMatrix(as.factor(train_label),
                                   as.factor(final_val_label),
                                   mode = "prec_recall")
conMatrix.test <- confusionMatrix(as.factor(test_label),
                                  as.factor(final_evl_label),
                                  mode = "prec_recall")

perf.MVSPL <- list("Coef.mRNA" = coef.mRNA,
                   "Coef.miRNA" = coef.miRNA,
                   "Coef.meth" = coef.meth,
                   "Perf.Train" = conMatrix.train,
                   "Perf.Test" = conMatrix.test)

