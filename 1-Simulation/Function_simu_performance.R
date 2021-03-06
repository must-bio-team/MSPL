evaluate.ConcatenationEN.performance <- function(results, train_label, test_label, beta){
  
  valiperformance = matrix(0, nrow=1, ncol=5)  #maps on the validation set
  testiperformance = matrix(0, nrow=1, ncol=5)   #maps on the test set
  betaperformance = matrix(0, nrow=1, ncol=3)
  
  colnames(valiperformance) = c("accuracy","sensitivity","specificity","recall","AUC") 
  colnames(testiperformance) = c("accuracy","sensitivity","specificity","recall","AUC")
  colnames(betaperformance) = c("accuracy","sensitivity","specificity")
  
  valiperformance[1,] = evaluate.ConcatenationEN.map(results$valpredmatrix,train_label)
  testiperformance[1,] = evaluate.ConcatenationEN.map(results$evlpredmatrix,test_label)
  betaperformance[1,] = evaluate.beta.ConcatenationEN.map(results$Coef,beta)
  
  return(list("valiperformance" = valiperformance, "testiperformance" = testiperformance, "betaperformance" = betaperformance))
}

evaluate.ConcatenationEN.map <- function(predictlabels,truelabels){
  pre_label_final <- as.integer(predictlabels>0.5)
  
  # calculate AUC
  pred1 <- prediction(pre_label_final, truelabels)
  perf1 <- performance(pred1, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred1, measure = "auc")
  auc <- auc@y.values[[1]]
  
  TN = sum((1 - pre_label_final)*(1 - truelabels)) # A:TN
  FP = sum(pre_label_final*(1 - truelabels)) # B:FP
  FN = sum((1 - pre_label_final)*truelabels) # C:FN
  TP = sum(pre_label_final*truelabels) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  return(perf)
}

evaluate.beta.ConcatenationEN.map <- function(coef,beta){
  Coef <- coef[-1]
  Coefidx = which(Coef!=0)
  betaidx = which(beta!=0)
  
  Coeftrans = replicate(length(Coef),0)
  betatrans = replicate(length(beta),0)
  
  Coeftrans[Coefidx] = 1
  betatrans[betaidx] = 1
  
  TN = sum((1 - Coeftrans)*(1 - betatrans)) # A:TN
  FP = sum(Coeftrans*(1 - betatrans)) # B:FP
  FN = sum((1 - Coeftrans)*betatrans) # C:FN
  TP = sum(Coeftrans*betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}



evaluate.EnsembleRF.performance <- function(results, train_label, test_label, beta){
  
  valiperformance = matrix(0, nrow=1, ncol=5)  #maps on the validation set
  testiperformance = matrix(0, nrow=1, ncol=5)   #maps on the test set
  betaperformance = matrix(0, nrow=1, ncol=3)
  colnames(valiperformance) = c("accuracy","sensitivity","specificity","recall","AUC") 
  colnames(testiperformance) = c("accuracy","sensitivity","specificity","recall","AUC")
  colnames(betaperformance) = c("accuracy","sensitivity","specificity")
  
  valiperformance[1,] = evaluate.EnsembleRF.map(results$valpredmatrix,train_label)
  testiperformance[1,] = evaluate.EnsembleRF.map(results$evlpredmatrix,test_label)
  betaperformance[1,] = evaluate.beta.EnsembleRF.map(results$feature,beta)
  
  return(list("valiperformance" = valiperformance, "testiperformance" = testiperformance, "betaperformance" = betaperformance))
}

evaluate.EnsembleRF.map <- function(predictlabels,truelabels){
  
  ## initial predict matirx
  Prelabel_matrix = matrix(0, nrow = length(truelabels), ncol = dim(predictlabels)[2])
  colnames(Prelabel_matrix) <- colnames(predictlabels)
  
  for(i in 1:length(truelabels)){
    for(j in 1:dim(predictlabels)[2]){
      Prelabel_matrix[i,j] <- as.integer(predictlabels[i,j]>0.5)
    }
  }
  
  ## voting
  prelable_final = matrix(0, nrow = length(truelabels), ncol = 1)
  for(i in 1:length(truelabels)){
    freq_table <- as.data.frame(table(Prelabel_matrix[i,]))
    vote_index <- which.max(freq_table$Freq)
    prelable_final[i] <- as.numeric(as.character(freq_table$Var1[vote_index]))
  }
  
  # calculate AUC
  pred1 <- prediction(prelable_final, truelabels)
  perf1 <- performance(pred1, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred1, measure = "auc")
  auc <- auc@y.values[[1]]
  
  TN = sum((1 - prelable_final)*(1 - truelabels)) # A:TN
  FP = sum(prelable_final*(1 - truelabels)) # B:FP
  FN = sum((1 - prelable_final)*truelabels) # C:FN
  TP = sum(prelable_final*truelabels) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  return(perf)
}



evaluate.beta.EnsembleRF.map <- function (coef, beta){
  
  Coeftrans <- list()
  betatrans <- list()
  
  for(i in 1:length(beta)){
    Coeftrans[[i]] <- replicate(length(beta[[i]]),0)
    betatrans[[i]] <- replicate(length(beta[[i]]),0)
    betaidx <- which(beta[[i]]!=0) 
    Coeftrans[[i]][coef$feature.idx[[i]]] <- 1  ## model beta
    betatrans[[i]][betaidx] <- 1  ## True beta
  }
  
  combined_coef <-c(Coeftrans[[1]],Coeftrans[[2]],Coeftrans[[3]])
  combined_beta <-c(betatrans[[1]],betatrans[[2]],betatrans[[3]])
  
  
  Coefidx = which(combined_coef!=0)
  Betaidx = which(combined_beta!=0)
  
  coeftrans = replicate(length(combined_coef),0)
  Betatrans = replicate(length(combined_beta),0)
  
  coeftrans[Coefidx] = 1
  Betatrans[Betaidx] = 1
  
  TN = sum((1 - coeftrans)*(1 - Betatrans)) # A:TN
  FP = sum(coeftrans*(1 - Betatrans)) # B:FP
  FN = sum((1 - coeftrans)*Betatrans) # C:FN
  TP = sum(coeftrans*Betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}


evaluate.concatenationRF.performance  <- function(confusion.mat, true_lable, predict_label){
  tp <- confusion.mat[2,2]
  tn <- confusion.mat[1,1]
  fp <- confusion.mat[2,1]
  fn <- confusion.mat[1,2]
  
  Accuracy <- (tp + tn)/(tp + tn + fp + fn)
  Sensitivity <- tp/(tp + fn)
  Specificity <- tn/(tn + fp)
  Recall = tp/(tp + fp)
  
  predict_label <- as.factor(predict_label)
  true_lable <- as.numeric(true_lable)
  predict_label <- as.numeric(predict_label)
  pred <- prediction(predict_label, true_lable)
  perf <- performance(pred, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred, measure = "auc")
  AUC <- auc@y.values[[1]]
  
  perf <- c(Accuracy, Sensitivity, Specificity, Recall, AUC)
  return(perf)
  
}



evaluate.beta.concatenationRF.performance <- function(coef, beta){
  
  combined_beta <- c(beta[[1]],beta[[2]],beta[[3]])
  
  Coefidx <- coef
  Betaidx <- which(combined_beta!=0)
  
  coeftrans = replicate(length(combined_beta),0)
  Betatrans = replicate(length(combined_beta),0)
  
  coeftrans[Coefidx] = 1
  Betatrans[Betaidx] = 1
  
  TN = sum((1 - coeftrans)*(1 - Betatrans)) # A:TN
  FP = sum(coeftrans*(1 - Betatrans)) # B:FP
  FN = sum((1 - coeftrans)*Betatrans) # C:FN
  TP = sum(coeftrans*Betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
  
}


evaluate.EnsembleEN.performance <- function(results, train_label, test_label, beta){
  
  valiperformance = matrix(0, nrow=1, ncol=5)  #maps on the validation set
  testiperformance = matrix(0, nrow=1, ncol=5)   #maps on the test set
  betaperformance = matrix(0, nrow=1, ncol=3)
  colnames(valiperformance) = c("accuracy","sensitivity","specificity","recall","AUC") 
  colnames(testiperformance) = c("accuracy","sensitivity","specificity","recall","AUC")
  colnames(betaperformance) = c("accuracy","sensitivity","specificity")
  
  valiperformance[1,] = evaluate.EnsembleEN.map(results$valpredmatrix,train_label)
  testiperformance[1,] = evaluate.EnsembleEN.map(results$evlpredmatrix,test_label)
  betaperformance[1,] = evaluate.beta.EnsembleEN.map(results$Coef,beta)
  
  return(list("valiperformance" = valiperformance, "testiperformance" = testiperformance, "betaperformance" = betaperformance))
}

evaluate.EnsembleEN.map <- function(predictlabels,truelabels){
  
  ## initial predict matirx
  Prelabel_matrix = matrix(0, nrow = length(truelabels), ncol = dim(predictlabels)[2])
  colnames(Prelabel_matrix) <- colnames(predictlabels)
  
  for(i in 1:length(truelabels)){
    for(j in 1:dim(predictlabels)[2]){
      Prelabel_matrix[i,j] <- as.integer(predictlabels[i,j]>0.5)
    }
  }
  
  ## voting
  prelable_final = matrix(0, nrow = length(truelabels), ncol = 1)
  for(i in 1:length(truelabels)){
    freq_table <- as.data.frame(table(Prelabel_matrix[i,]))
    vote_index <- which.max(freq_table$Freq)
    prelable_final[i] <- as.numeric(as.character(freq_table$Var1[vote_index]))
  }
  
  # calculate AUC
  pred1 <- prediction(prelable_final, truelabels)
  perf1 <- performance(pred1, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred1, measure = "auc")
  auc <- auc@y.values[[1]]
  
  TN = sum((1 - prelable_final)*(1 - truelabels)) # A:TN
  FP = sum(prelable_final*(1 - truelabels)) # B:FP
  FN = sum((1 - prelable_final)*truelabels) # C:FN
  TP = sum(prelable_final*truelabels) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  return(perf)
}

evaluate.beta.EnsembleEN.map <- function (coef, beta){
  
  for(i in 1:length(coef)){
    coef[[i]] <- coef[[i]][-1]
  }
  combined_coef <- c(coef[[1]],coef[[2]],coef[[3]])
  combined_beta <- c(beta[[1]],beta[[2]],beta[[3]])
  
  Coefidx = which(combined_coef!=0)
  Betaidx = which(combined_beta!=0)
  
  coeftrans = replicate(length(combined_coef),0)
  Betatrans = replicate(length(combined_beta),0)
  
  coeftrans[Coefidx] = 1
  Betatrans[Betaidx] = 1
  
  TN = sum((1 - coeftrans)*(1 - Betatrans)) # A:TN
  FP = sum(coeftrans*(1 - Betatrans)) # B:FP
  FN = sum((1 - coeftrans)*Betatrans) # C:FN
  TP = sum(coeftrans*Betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}


evaluate.DIABLO.performance <- function(confusion.mat, true_lable, predict_label){
  tp <- confusion.mat[2,2]
  tn <- confusion.mat[1,1]
  fp <- confusion.mat[2,1]
  fn <- confusion.mat[1,2]
  
  Accuracy <- (tp + tn)/(tp + tn + fp + fn)
  Sensitivity <- tp/(tp + fn)
  Specificity <- tn/(tn + fp)
  Recall = tp/(tp + fp)
  
  predict_label <- as.factor(predict_label)
  true_lable <- as.numeric(true_lable)
  predict_label <- as.numeric(predict_label)
  pred <- prediction(predict_label, true_lable)
  perf <- performance(pred, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred, measure = "auc")
  AUC <- auc@y.values[[1]]
  
  perf <- c(Accuracy, Sensitivity, Specificity, Recall, AUC)
  return(perf)
  
}


evaluate.beta.DIABLO.performance <- function(coef = Coef.idx, beta = beta.matrix){
  
  array_mrna <- replicate(length(beta[[1]]),0)
  array_mirna <- replicate(length(beta[[2]]),0)
  array_cpg <- replicate(length(beta[[3]]),0)
  array_mrna[as.numeric(coef[[1]])] <- 1
  array_mirna[as.numeric(coef[[2]])] <- 1
  array_cpg[as.numeric(coef[[3]])] <- 1
  
  combined_coef <- c(array_mrna,array_mirna,array_cpg)
  combined_beta <- c(beta[[1]],beta[[2]],beta[[3]])
  
  Coefidx = which(combined_coef!=0)
  Betaidx = which(combined_beta!=0)
  
  coeftrans = replicate(length(combined_coef),0)
  Betatrans = replicate(length(combined_beta),0)
  
  coeftrans[Coefidx] = 1
  Betatrans[Betaidx] = 1
  
  TN = sum((1 - coeftrans)*(1 - Betatrans)) # A:TN
  FP = sum(coeftrans*(1 - Betatrans)) # B:FP
  FN = sum((1 - coeftrans)*Betatrans) # C:FN
  TP = sum(coeftrans*Betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}


selfpaced.sim.single <- function(X_train, Y_train, X_test, Y_test, lambda, uplambda, Iter_num) {
  
  #the list storing the result for each iteration
  valpredmatrix = list()
  evlpredmatrix = list()
  coefmatrix = list()
  nonzerocoefmatrix = list()
  iters.V_idx = list()
  valmaps <- replicate(Iter_num,0)
  evlmaps <- replicate(Iter_num,0)
  
  #the starting values
  cvfit<-cv.glmnet(X_train,Y_train,alpha=1,family="binomial",type.measure = "class") 
  valpred <- predict(cvfit,X_train,type="response",s="lambda.min")
  
  v.idx = selfpace.sim.rank(dev_decval = valpred, dev_labels = Y_train, lambda = lambda)
  this.training.vidx = v.idx
  iters.vidx = list()	
  
  for(iter in 1:Iter_num) {
    if(length(this.training.vidx) == length(Y_train)){break}
    cat("Starting the ",iter,"-th iteration.\t", sep = "")
    iters.vidx[[iter]] = this.training.vidx
    
    # glmnet (Lasso, Elastic net & L2)
    cvfit<-cv.glmnet(data.matrix(X_train[this.training.vidx,]),
                     Y_train[this.training.vidx],
                     alpha=1,
                     family="binomial",
                     type.measure = "class")
    valprediction <- predict(cvfit,X_train,type="response",s="lambda.min")
    trprediction = valprediction
    tsprediction <- predict(cvfit,X_test,type="response",s="lambda.min")
    coefprediction = as.vector(coef(cvfit,s="lambda.min")[-1])
    numbernonzerocoef = length(which(coefprediction!=0))
    
    #self-paced learning
    selectedidx = selfpace.sim.rank(dev_decval = trprediction, dev_labels = Y_train, lambda)
    this.training.vidx = selectedidx
    cat("Select ", length(selectedidx), " samples.\t", sep = "")
    
    #change the parameter accoding to the step size
    lambda = lambda + uplambda
    
    #store the prediction for this iteration
    coefmatrix[[iter]]= coefprediction
    nonzerocoefmatrix[[iter]] = numbernonzerocoef
    valpredmatrix[[iter]] = as.numeric(valprediction)
    evlpredmatrix[[iter]] = tsprediction
    iters.V_idx[[iter]] = selectedidx
    
    #evaluate the training and test error
    val_loss <- sum((valpredmatrix[[iter]] - Y_train)^2)
    evl_loss <- sum((evlpredmatrix[[iter]] - Y_test)^2)
    valmaps[iter] <- val_loss
    evlmaps[iter] <- evl_loss
    
    cat("Finish the ",iter,"-th iteration.\n", sep = "")
  }
  
  results <- list("valpredmatrix" = valpredmatrix, 
                  "evlpredmatrix" = evlpredmatrix, 
                  "valmaps" = valmaps,
                  "evlmaps" = evlmaps,
                  "itervidx" = iters.V_idx, 
                  "Coef"=coefmatrix, 
                  "NumbernonzeroCoef"=nonzerocoefmatrix)
  return(results)
}


selfpace.sim.rank <- function(dev_decval, dev_labels, lambda) {
  #calculate the loss
  loss = (dev_decval-dev_labels)^2	#squared error
  #loss = 1/(1+e^(-1*loss))			#logistic
  
  posidx = which(dev_labels==1)	#postive id mapping
  negidx = which(dev_labels==0)	#negative id mapping
  
  #calculate pos_lambda1 and neg_lambda2 according to the rank
  pos_lambda = sort(loss[posidx,1])[min(length(posidx), lambda)]
  neg_lambda = sort(loss[negidx,1])[min(length(negidx), lambda)]
  
  #it is like first sorting sampled based on the metric and then select top lambda1_rank
  if(length(unique(loss[posidx]))!=1){
    selectedposidx <- posidx[which(loss[posidx,1] <= pos_lambda)]
  }else{
    selectedposidx <- sample(posidx, size = min(lambda, length(posidx)), replace = FALSE)
  }
  if(length(unique(loss[negidx]))!=1){
    selectednegidx <- negidx[which(loss[negidx,1] <= neg_lambda)]
  }else{
    selectednegidx <- sample(negidx, size = min(lambda, length(negidx)), replace = FALSE)
  }
  
  #selectedposidx <- posidx[which(loss[posidx,1] <= pos_lambda)]
  #selectednegidx <- negidx[which(loss[negidx,1] <= neg_lambda)]
  
  selecedidx = c(selectedposidx, selectednegidx)
  
  return(selecedidx)
}


evaluate.ConcatenationSPL.performance <- function(results, train_label, test_label, beta){
  
  valiperformance = matrix(0, nrow=1, ncol=5)  #maps on the validation set
  testiperformance = matrix(0, nrow=1, ncol=5)   #maps on the test set
  betaperformance = matrix(0, nrow=1, ncol=3)
  
  colnames(valiperformance) = c("accuracy","sensitivity","specificity","recall","AUC") 
  colnames(testiperformance) = c("accuracy","sensitivity","specificity","recall","AUC")
  colnames(betaperformance) = c("accuracy","sensitivity","specificity")
  
  valiperformance[1,] = evaluate.ConcatenationSPL.map(results$valpredmatrix,train_label)
  testiperformance[1,] = evaluate.ConcatenationSPL.map(results$evlpredmatrix,test_label)
  betaperformance[1,] = evaluate.beta.ConcatenationSPL.map(results$Coef,beta)
  
  return(list("valiperformance" = valiperformance, "testiperformance" = testiperformance, "betaperformance" = betaperformance))
}

evaluate.ConcatenationSPL.map <- function(predictlabels,truelabels){
  pre_label_final <- as.integer(predictlabels>0.5)
  
  # calculate AUC
  pred1 <- prediction(pre_label_final, truelabels)
  perf1 <- performance(pred1, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred1, measure = "auc")
  auc <- auc@y.values[[1]]
  
  TN = sum((1 - pre_label_final)*(1 - truelabels)) # A:TN
  FP = sum(pre_label_final*(1 - truelabels)) # B:FP
  FN = sum((1 - pre_label_final)*truelabels) # C:FN
  TP = sum(pre_label_final*truelabels) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  return(perf)
}

evaluate.beta.ConcatenationSPL.map <- function(coef,beta){
  Coef <- coef
  Coefidx = which(Coef!=0)
  betaidx = which(beta!=0)
  
  Coeftrans = replicate(length(Coef),0)
  betatrans = replicate(length(beta),0)
  
  Coeftrans[Coefidx] = 1
  betatrans[betaidx] = 1
  
  TN = sum((1 - Coeftrans)*(1 - betatrans)) # A:TN
  FP = sum(Coeftrans*(1 - betatrans)) # B:FP
  FN = sum((1 - Coeftrans)*betatrans) # C:FN
  TP = sum(Coeftrans*betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}


selfpaced.EN.sim.single <- function(X_train, Y_train, lambda, uplambda, Iter_num) {
  
  #the list storing the result for each iteration
  cvfitmatrix = list()
  valmaps <- replicate(Iter_num,0)
  
  #the starting values
  cvfit<-cv.glmnet(X_train,Y_train,alpha=1,family="binomial",type.measure = "class") 
  valpred <- predict(cvfit,X_train,type="response",s="lambda.min")
  
  this.training.vidx = selfpace.sim1.rank(dev_decval = valpred, dev_labels = Y_train, lambda = lambda)
  
  for(iter in 1:Iter_num){
    
    if(length(this.training.vidx) == length(Y_train)){break}
    cat("Starting the ",iter,"-th iteration.\t", sep = "")
    # glmnet (Lasso, Elastic net & L2)
    cvfit<-cv.glmnet(data.matrix(X_train[this.training.vidx,]),
                     Y_train[this.training.vidx],
                     alpha=1,
                     family="binomial",
                     type.measure = "class")
    valprediction <- predict(cvfit,X_train,type="response",s="lambda.min")
    
    #self-paced learning
    selectedidx = selfpace.sim1.rank(dev_decval = valprediction, dev_labels = Y_train, lambda)
    this.training.vidx = selectedidx
    cat("Select ", length(selectedidx), " samples.\t", sep = "")
    
    #change the parameter accoding to the step size
    lambda = lambda + uplambda
    
    #store the prediction for this iteration
    cvfitmatrix[[iter]] <- cvfit
    
    #evaluate the training and test error
    val_loss <- sum((valprediction - Y_train)^2)
    valmaps[iter] <- val_loss
    
    cat("Finish the ",iter,"-th iteration.\n", sep = "")
  }
  
  ## best results ##
  best.iter <- which(valmaps == min(valmaps[1:length(cvfitmatrix)]))
  best.cvfit <- cvfitmatrix[[best.iter]]
  
  return(best.cvfit)
}


selfpace.sim1.rank <- function(dev_decval, dev_labels, lambda){
  #calculate the loss
  loss = (dev_decval-dev_labels)^2	#squared error
  #loss = 1/(1+e^(-1*loss))			#logistic
  
  posidx = which(dev_labels==1)	#postive id mapping
  negidx = which(dev_labels==0)	#negative id mapping
  
  #calculate pos_lambda1 and neg_lambda2 according to the rank
  pos_lambda = sort(loss[posidx,1])[min(length(posidx), lambda)]
  neg_lambda = sort(loss[negidx,1])[min(length(negidx), lambda)]
  
  #it is like first sorting sampled based on the metric and then select top lambda1_rank
  if(length(unique(loss[posidx]))!=1){
    selectedposidx <- posidx[which(loss[posidx,1] <= pos_lambda)]
  }else{
    selectedposidx <- sample(posidx, size = min(lambda, length(posidx)), replace = FALSE)
  }
  if(length(unique(loss[negidx]))!=1){
    selectednegidx <- negidx[which(loss[negidx,1] <= neg_lambda)]
  }else{
    selectednegidx <- sample(negidx, size = min(lambda, length(negidx)), replace = FALSE)
  }
  selecedidx = c(selectedposidx, selectednegidx)
  
  return(selecedidx)
}

evaluate.EnsembleSPL.performance <- function(results, train_label, test_label, beta){
  
  valiperformance = matrix(0, nrow=1, ncol=5)  #maps on the validation set
  testiperformance = matrix(0, nrow=1, ncol=5)   #maps on the test set
  betaperformance = matrix(0, nrow=1, ncol=3)
  colnames(valiperformance) = c("accuracy","sensitivity","specificity","recall","AUC") 
  colnames(testiperformance) = c("accuracy","sensitivity","specificity","recall","AUC")
  colnames(betaperformance) = c("accuracy","sensitivity","specificity")
  
  valiperformance[1,] = evaluate.EnsembleSPL.map(results$valpredmatrix,train_label)
  testiperformance[1,] = evaluate.EnsembleSPL.map(results$evlpredmatrix,test_label)
  betaperformance[1,] = evaluate.beta.EnsembleSPL.map(results$Coef,beta)
  
  return(list("valiperformance" = valiperformance, "testiperformance" = testiperformance, "betaperformance" = betaperformance))
}

evaluate.EnsembleSPL.map <- function(predictlabels,truelabels){
  
  ## initial predict matirx
  Prelabel_matrix = matrix(0, nrow = length(truelabels), ncol = dim(predictlabels)[2])
  colnames(Prelabel_matrix) <- colnames(predictlabels)
  
  for(i in 1:length(truelabels)){
    for(j in 1:dim(predictlabels)[2]){
      Prelabel_matrix[i,j] <- as.integer(predictlabels[i,j]>0.5)
    }
  }
  
  ## voting
  prelable_final = matrix(0, nrow = length(truelabels), ncol = 1)
  for(i in 1:length(truelabels)){
    freq_table <- as.data.frame(table(Prelabel_matrix[i,]))
    vote_index <- which.max(freq_table$Freq)
    prelable_final[i] <- as.numeric(as.character(freq_table$Var1[vote_index]))
  }
  
  # calculate AUC
  pred1 <- prediction(prelable_final, truelabels)
  perf1 <- performance(pred1, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred1, measure = "auc")
  auc <- auc@y.values[[1]]
  
  TN = sum((1 - prelable_final)*(1 - truelabels)) # A:TN
  FP = sum(prelable_final*(1 - truelabels)) # B:FP
  FN = sum((1 - prelable_final)*truelabels) # C:FN
  TP = sum(prelable_final*truelabels) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  return(perf)
}

evaluate.beta.EnsembleSPL.map <- function (coef, beta){
  
  combined_coef <- c(coef[[1]],coef[[2]],coef[[3]])
  combined_beta <- c(beta[[1]],beta[[2]],beta[[3]])
  
  Coefidx = which(combined_coef!=0)
  Betaidx = which(combined_beta!=0)
  
  coeftrans = replicate(length(combined_coef),0)
  Betatrans = replicate(length(combined_beta),0)
  
  coeftrans[Coefidx] = 1
  Betatrans[Betaidx] = 1
  
  TN = sum((1 - coeftrans)*(1 - Betatrans)) # A:TN
  FP = sum(coeftrans*(1 - Betatrans)) # B:FP
  FN = sum((1 - coeftrans)*Betatrans) # C:FN
  TP = sum(coeftrans*Betatrans) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  perf <- c(accuracy,sensitivity,specificity)
  
  return(perf)
}
