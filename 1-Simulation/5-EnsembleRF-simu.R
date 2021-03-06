setwd("D:/Ziyi/School/PMO/12.Multi_omics/7.MSPL-code/1-Simulation")

##----------------------------------
##  Code Version 1.0
##  This simulation code for Randomforest embeded in the ensemble-based framework.
##  Ensemble_RF
##  Created by Zi-Yi Yang 
##  Modified by Zi-Yi Yang on July 23, 2019
##  Concact: yangziyi091100@163.com
##----------------------------------


##---------------------
## load libraries
##---------------------
library(amritr)
library(glmnet)
library(Matrix)
library(tseries)
library(caret)
library(ncvreg)
library(pROC)
library(ROCR)
library(ggplot2)
library(randomForest)

source("Function_simu_performance.R")

##--------------------------------
## 1. Generate simulation data
##--------------------------------
J <- 3 # number of omics
p <- c(2000,500,1500)  # number of features in each omics
sam_num <- 1500
train_num <- c(100,150,200) # sample numbers of training
test_num <- 100  # sample numbers of testing
para_correlation <- c(0)  # c(0,0.2,0.4,0.6,0.8)
para_noise <- c(0,0.4,0.8)
S <- 3  # index of train number  (100,150,200)
N <- 3  # index of noise number (0,0.4,0.8)

## Initalize ##
beta.matrix <- list()
data.train <- list()
data.test <- list()
sample.train <- list()
sample.test <- list()
y_train <- NULL
y_test <- NULL
label.matrix.train <- matrix(0, nrow = sam_num, ncol = J)
label.matrix.test <- matrix(0, nrow = sam_num, ncol = J)

for(i in 1:J){
  data.train[[i]] <- list()
  data.test[[i]] <- list()
  sample.train[[i]] <- list()
  sample.test[[i]] <- list()
}

## Inital beta matrix ##
beta1 <- c(1.5,-1.4,2,-1.6,2.3,2.1,-1.8,2.3,1.9,-1.6,rep(0,1990))
beta2 <- c(rep(0,5),1.4,-1.1,-1.8,1.5,2.1,-1.9,2.3,-1.6,1.8,rep(0,500-14))
beta3 <- c(rep(0,10),1.1,2,2.3,-1.5,-1.9,1.7,-1.2,-1.6,rep(0,1500-18))

beta.matrix <- list("beta1" = beta1,
                    "beta2" = beta2,
                    "beta3" = beta3)

## Inital pool of X ##
for(i in 1:J){
  sample.train[[i]] <- matrix(rnorm(sam_num * p[i], mean = 0, sd = 1), 
                              sam_num, p[i])
  sample.test[[i]] <- matrix(rnorm(sam_num * p[i], mean = 0, sd = 1), 
                             sam_num, p[i])
}

## Select sample with the sample label in all omics ##
for(i in 1:J){
  ## Pool of training
  noise_rnorm <- rnorm(sam_num,mean = 0, sd = 4)
  l.train <- sample.train[[i]] %*% beta.matrix[[i]] + para_noise[N] * noise_rnorm
  prob.train <- exp(l.train)/(1+exp(l.train))
  prob.train[which(prob.train>0.5)] <- 1
  prob.train[which(prob.train<=0.5)] <- 0
  label.matrix.train[,i] <- prob.train
  ## Pool of testing
  l.test <- sample.test[[i]] %*% beta.matrix[[i]]
  prob.test <- exp(l.test)/(1+exp(l.test))
  prob.test[which(prob.test>0.5)] <- 1
  prob.test[which(prob.test<=0.5)] <- 0
  label.matrix.test[,i] <- prob.test
  
}

sam0_train.idx <- which(apply(label.matrix.train, 1, sum)==0)
sam1_train.idx <- which(apply(label.matrix.train, 1, sum)==3)
sam0_test.idx <- which(apply(label.matrix.test, 1, sum)==0)
sam1_test.idx <- which(apply(label.matrix.test, 1, sum)==3)

select0.train.idx <- sample(x = sam0_train.idx, size = train_num[S]/2)
select1.train.idx <- sample(x = sam1_train.idx, size = train_num[S]/2)
select0.test.idx <- sample(x = sam0_test.idx, size = test_num/2)
select1.test.idx <- sample(x = sam1_test.idx, size = test_num/2)

select.train.idx <- c(select0.train.idx,select1.train.idx)
select.test.idx <- c(select0.test.idx,select1.test.idx)

##------------------------------
## generate simulation data
##------------------------------

for(i in 1:J){
  data.train[[i]] <- sample.train[[i]][select.train.idx,]
  y_train <- c(rep(0,train_num[S]/2),rep(1,train_num[S]/2)) 
  data.test[[i]] <- sample.test[[i]][select.test.idx,]
  y_test <- c(rep(0,test_num/2),rep(1,test_num/2)) 
}


##-----------------
# 2. Ensemble Randomforest
##-----------------

ensemblePanels <- lapply(data.train, function(i){
  RF.colon <- randomForest(x = i, y = factor(y_train), importance = TRUE, ntree = 30)
})

ensembleValiPredction <- mapply(function(fit,x){
  valprediction <- predict(fit,x)
}, fit = ensemblePanels, x = data.train)

ensembleTestPredction <- mapply(function(fit,x){
  tsprediction <- predict(fit,x,type="response",s="lambda.min")
}, fit = ensemblePanels, x = data.test)

select.feature.idx <- list()
select.feature.value <- list()
select.feature.name <- list()

for(i in 1:length(ensemblePanels)){
  select.feature.idx[[i]] <- which(ensemblePanels[[i]]$importance[,4]!=0)
  select.feature.value[[i]] <- ensemblePanels[[i]]$importance[which(ensemblePanels[[i]]$importance[,4]!=0),4]
  select.feature.name[[i]] <- colnames(data.train[[i]])[select.feature.idx[[i]]]
}

ensemblefeature <- list("feature.idx" = select.feature.idx,
                        "feature.vaule" = select.feature.value,
                        "feature.name" = select.feature.name)

results <- list("valpredmatrix" = ensembleValiPredction, "evlpredmatrix" = ensembleTestPredction, 
                "feature" = ensemblefeature, "ensenmblePanels" = ensemblePanels)

##--------------
# 3. Performance
##--------------
para.runs = list()
para.runs = evaluate.EnsembleRF.performance(results, y_train, y_test, beta.matrix)

