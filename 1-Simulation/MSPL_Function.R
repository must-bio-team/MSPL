mvselfpace.rank <- function(dev_decval, dev_labels, v_iter, View_id, lambda, gamma, View_num, num_add) {
  
  # initialize the loss function
  loss = matrix(0, nrow = length(dev_labels), ncol = View_num)
  #calculate the loss
  for(m in 1: View_num){
    if(m != View_id){
      loss[,m] = (dev_decval[,m] - dev_labels)^2    #squared error
    }else{
      next;
    }
  }
  
  # Update v(View_num-j)
  for(m in 1:View_num){
    if(m != View_id){
      for(i in 1:length(dev_labels)){
        if(loss[i,m] < lambda[m] + gamma * (sum(v_iter[i,])-v_iter[i,m])){
          v_iter[i,m] = 1
        }else{
          v_iter[i,m] = 0
        }
      }
    }
  }
  
  # Update vj
  loss[,View_id] = (dev_decval[,View_id] - dev_labels)^2
  for(i in 1:length(dev_labels)){
    if(loss[i,View_id] < lambda[View_id] + gamma * (sum(v_iter[i,])-v_iter[i,View_id])){
      v_iter[i,View_id] = 1
    }else{
      v_iter[i,View_id] = 0
    }
  }
  
  ## sort sample
  selectedposidx = list()
  selectednegidx = list()
  selecedidx = list()
  pos_lambda = matrix(0, nrow = 1, ncol = View_num)
  neg_lambda = matrix(0, nrow = 1, ncol = View_num)
  V_iter = matrix(0, nrow = length(dev_labels), ncol = View_num)
  posidx = which(dev_labels==1)	#postive id mapping
  negidx = which(dev_labels==0)	#negative id mapping
  
  #calculate pos_lambda and neg_lambda according to the rank
  for(i in 1:View_num){
    
    if(length(which(v_iter[posidx,i]==1))!=0){  # a certain class has samples
      pos_lambda[i] = sort(loss[posidx,i][which(v_iter[posidx,i]==1)])[min(length(posidx), num_add, length(which(v_iter[posidx,i]==1)))]
    }
    if(length(which(v_iter[negidx,i]==1))!=0){
      neg_lambda[i] = sort(loss[negidx,i][which(v_iter[negidx,i]==1)])[min(length(negidx), num_add, length(which(v_iter[negidx,i]==1)))]
    }
    
    if(length(unique(loss[posidx,i]))!=1){  # select samples
      selectedposidx[[i]] <- intersect(posidx[which(v_iter[posidx,i] == 1)],   ## v_iter = 1 && loss is small
                                       posidx[which(loss[posidx,i] <= pos_lambda[i])])
    }else{
      selectedposidx[[i]] <- sample(posidx, size = min(num_add, length(posidx)), replace = FALSE)
    }
    
    if(length(unique(loss[negidx,i]))!=1){
      selectednegidx[[i]] <- intersect(negidx[which(v_iter[negidx,i] == 1)],
                                       negidx[which(loss[negidx,i] <= neg_lambda[i])])
    }else{
      selectednegidx[[i]] <- sample(negidx, size = min(num_add, length(negidx)), replace = FALSE)
    }
    
    
    selecedidx[[i]] = c(selectedposidx[[i]], selectednegidx[[i]])
    V_iter[selecedidx[[i]],i] = 1
  }
  cat("The ",View_id, "-th modality select ",length(selecedidx[[i]]),"samples.\n", sep = "")
  #return the result
  return(V_iter)
}


evaluate.performance <- function(pred_label, true_label, View_num){
  num <- length(true_label)
  
  final_pred_label <- vector(mode = "numeric", length = num)
  
  for(i in 1:num){
    pre_label <- pred_label[i,]
    loss_1 <- 0
    loss_0 <- 0
    for(j in 1:View_num){
      loss_1 <- loss_1 + (pre_label[j] - 1)^2
      loss_0 <- loss_0 + (pre_label[j] - 0)^2
    }
    if(loss_1 >= loss_0){
      final_pred_label[i] <- 0
    }else{
      final_pred_label[i] <- 1
    }
  }
  
  # calculate AUC
  pred <- prediction(final_pred_label, true_label)
  perf <- performance(pred, measure = "tpr", x.measure = "fpr")
  auc <- performance(pred, measure = "auc")
  auc <- auc@y.values[[1]]
  
  # calculate accuracy, sensitivity, specificity
  TN = sum((1 - final_pred_label)*(1 - true_label)) # A:TN
  FP = sum(final_pred_label*(1 - true_label)) # B:FP
  FN = sum((1 - final_pred_label)*true_label) # C:FN
  TP = sum(final_pred_label*true_label) # D:TP
  accuracy = (TP + TN)/(TN + TP + FP + FN)
  sensitivity = TP/(TP + FN)
  specificity = TN/(TN + FP)
  recall = TP/(TP + FP)
  perf <- c(accuracy,sensitivity,specificity,recall,auc)
  
  return(perf)
  
}

evaluate.beta.performance <- function(pred_beta, true_beta){
  
  for(i in 1:length(pred_beta)){
    pred_beta[[i]] <- pred_beta[[i]][-1]
  }
  
  combined_coef <- c(pred_beta[[1]],pred_beta[[2]],pred_beta[[3]])
  combined_beta <- c(true_beta[[1]],true_beta[[2]],true_beta[[3]])
  
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


