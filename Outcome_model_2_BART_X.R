

library(SuperLearner)
library(glmnet)
library(purrr)
library(boot)
library(invgamma)
library(coda)
library(MASS)
library(purrr)
library(boot)
library(extraDistr)
library(mnormt)
library(BayesTree)


BART_ICE_X <- function(dat){
  ###############
  #Global Setting
  ###############
  mc_num <- 4000
  burnin<- 3000
  mc_num_stage_1 <- mc_num
  # the 10th draw is kept
  keepevery_stage2 <- 10
  keepevery_stage1 <- 100
  
  n=dim(dat)[1]
  num_trees <- 50
  sigdf_val <- 3
  sigquant_val <- 0.9
  
  #sigdf_val <- 10
  #sigquant_val <- 0.75
  
  ############################# Stage 2 ###################################
  X = as.matrix(cbind(dat$A1,dat$A2,dat$X_11,dat$X_21,dat$X_3,dat$X_12,dat$X_22))   
  y = dat$Y
  # dat <- as.data.frame(cbind(L1,A1,L2,A2,Y))
  # dat <- as.data.frame(cbind(L1,A1,L2,A2,X_11,X_21,X_3,X_12,X_22,Y))
  
  newdata <- dat
  newdata$a2_0 <- 0
  newdata$a2_1 <- 1
  X_a2_0 <- as.matrix(newdata[,c("A1","a2_0","X_11","X_21","X_3","X_12","X_22")])
  #X_a2_1 <- as.matrix(newdata[,c("A1","a2_1","L1","L2")])
  X_a2_1 <- as.matrix(newdata[,c("A1","a2_1","X_11","X_21","X_3","X_12","X_22")])
  # Initialization
  newdata$Y_a2_0 <- 0
  newdata$Y_a2_1 <- 0
  newdata0 <- newdata[,c(1:7,9,11)]
  colnames(newdata0)[c(8,9)] <- c("a2","Y_hat")
  newdata1 <- newdata[,c(1:7,10,12)]
  colnames(newdata1)[c(8,9)] <- c("a2","Y_hat")
  # change to long format
  # Obtain D_aug 
  data_aug <- as.data.frame(rbind(newdata0,newdata1))
  #View(data_aug)
  # X.test for prediction
  X_test <- as.matrix(cbind(data_aug$A1,data_aug$a2,data_aug$X_11,data_aug$X_21,data_aug$X_3,
                            data_aug$X_12,data_aug$X_22))
  #View(X_test)
  ## Prior info
  # k: 2.000000
  # degrees of freedom in sigma prior: 3 (nu) variable_name = sigdf
  # quantile in sigma prior: 0.900000 (q) variable_name = sigquant
  # power and base for tree prior: 2.000000 0.950000
  
  ndpost_num_stage_2 = mc_num - burnin
  bartFit = bart(x.train=X,y.train=y,x.test=X_test,ntree=num_trees, sigdf=sigdf_val,
                 sigquant=sigquant_val, ndpost=ndpost_num_stage_2,
                 nskip=burnin, keepevery=keepevery_stage2, verbose=FALSE) 
  n_preserved_stage2 <- dim(bartFit$yhat.test)[1]
  
  ATE_list <- c()
  
  
  
  for(i in 1:n_preserved_stage2){
    
    data_aug$Y_hat <- c(bartFit$yhat.test[i,]+rnorm(length(bartFit$yhat.test[i,]),
                                                    mean=0,sd=bartFit$sigma[i]))
    
    ################################################# Stage 1 ###########
    X <- as.matrix(cbind(data_aug$A1,data_aug$a2,data_aug$X_11,data_aug$X_21,data_aug$X_3))
    y <- data_aug$Y_hat
    
    newdata_aug <- dat
    newdata_aug$a1_0 <- 0
    newdata_aug$a1_1 <- 1
    newdata_aug$a2_0 <- 0
    newdata_aug$a2_1 <- 1
    X_a1_0_a2_0 <- as.matrix(newdata_aug[,c("a1_0","a2_0","X_11","X_21","X_3")])
    X_a1_1_a2_0 <- as.matrix(newdata_aug[,c("a1_1","a2_0","X_11","X_21","X_3")])
    X_a1_0_a2_1 <- as.matrix(newdata_aug[,c("a1_0","a2_1","X_11","X_21","X_3")])
    X_a1_1_a2_1 <- as.matrix(newdata_aug[,c("a1_1","a2_1","X_11","X_21","X_3")])
    X_test_1 = as.matrix(rbind(X_a1_0_a2_0,X_a1_1_a2_0,X_a1_0_a2_1,X_a1_1_a2_1))
    
    ndpost_num_stage_1 = mc_num_stage_1 - burnin
    bartFit_1 = bart(x.train=X,y.train=y,x.test=X_test_1,ntree=num_trees,sigdf=sigdf_val,
                     sigquant=sigquant_val, ndpost=ndpost_num_stage_1, nskip=burnin, 
                     keepevery=keepevery_stage1, verbose=FALSE) 
    
    Y_hat_stage1 <- bartFit_1$yhat.test
    n_preserved_stage1 <- dim(Y_hat_stage1)[1]
    
    for(j in 1:n_preserved_stage1){
      Y_stage1 <- Y_hat_stage1[j,]
      new_ATE <- mean(Y_stage1[(3*n+1):(4*n)]) - mean(Y_stage1[1:n])
      ATE_list <- c(ATE_list, new_ATE)
    }
    
  }
  
  return(ATE_list)
}

M = 100  # number of datasets simulated: 50~500
n = 500
true_ATE_Zhu_2023 <- 0.5093185
est_ATE <- c()
sd_ATE <- c()
coverage_num <- 0
upper_CI <- 0
lower_CI <- 0
ATE_list <- c()

for (i in 1:M){
  seed_num <- i+1000
  set.seed(seed_num)
  # At time t = 1, simulate for each individual baseline covariates X_11, X_21, X_3
  X_11 <- rnorm(n,0,1)
  X_21 <- rnorm(n,0,1)
  X_3 <- abs(rnorm(n,0,1))
  
  # Generate treatment A_1
  A0 <- rep(0, n)
  pi1 <- (exp(-A0+ (1*X_11-0.5*X_21+0.25*X_3) + (-1/2)))/(1+exp(-A0+(1*X_11-0.5*X_21+0.25*X_3)+(-1/2)))
  A1 <- rbern(n, prob = pi1)
  
  # At time t = 2, simulate for each individual their time-varying confounders X_12, X_22
  U2 <- 2+(2*A1 - 1)/3
  
  Z12 <- rnorm(n,0,1)
  Z22 <- rnorm(n,0,1)
  
  X_12 <- Z12*U2
  X_22 <- Z22*U2
  
  # Generate treatment A_2
  pi2 <- (exp(-A1 + (1*X_12-0.5*X_22) + (-1/2)^2))/(1 + exp(-A1+(1*X_12-0.5*X_22)+(-1/2)^2))
  A2 <- rbern(n, prob = pi2)
  
  
  Y <- -1-(3/(1+exp(-6*(X_11+X_21+X_3+X_12+X_22)))) + 2*X_12*X_22 +
    1.8*(X_11+X_21+X_3)*A1 - (2/(1+exp(-4*(X_12+X_22))))*A2 +rnorm(n,0,sd=0.8)
  
  # Combine the dataset
  dat <- as.data.frame(cbind(A1,A2,X_11,X_21,X_3,X_12,X_22,Y))
  
  
  ATE_list <- BART_ICE_X(dat)
  est_ATE[i] <- mean(ATE_list)
  sd_ATE[i] <- sd(ATE_list)
  upper_CI <- quantile(ATE_list, 0.975)
  lower_CI <- quantile(ATE_list, 0.025)
  
  print("-----------Seed Number------------------")
  print(seed_num)
  
  c <- 0
  
  if((true_ATE_Zhu_2023 <= upper_CI) & (true_ATE_Zhu_2023 >= lower_CI)){
    coverage_num <- coverage_num+1
    c <- 1
  }
  
  print("-----------Individual Coverage------------------")
  print(c)
  print("--------------Individual ATE---------------")
  print(est_ATE[i])
  print("-----------------------------")
  
}

# Percentage Bias
bias_BART <- mean(est_ATE)-true_ATE_Zhu_2023
percentage_bias_BART <- abs(bias_BART/true_ATE_Zhu_2023)
paste("(BART_Zhu_X) Percentage Bias: ", percentage_bias_BART)

# Credible Interval Coverage: the proportion of all datasets for which the 95% credible intervals contained the true ATE
credible_interval_coverage_BART <- coverage_num/M
paste("(BART_Zhu_X) Credible Interval Coverage: ", credible_interval_coverage_BART)

# RMSE
BART_RMSE <- sqrt(mean((est_ATE - true_ATE_Zhu_2023)^2))
paste("(BART_Zhu_X) RMSE: ", BART_RMSE)

# MCSE
MCSE <- sd(est_ATE)
paste("(BART_Zhu_X) MCSE: ", MCSE)

# AESE
AESE <- mean(sd_ATE)
paste("(BART_Zhu_X) AESE: ", AESE)


