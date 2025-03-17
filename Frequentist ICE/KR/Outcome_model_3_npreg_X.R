# Tan_npreg
print("freq_npreg_Tan_X")

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
library(np)



freq_npreg_ICE_X <- function(dat){
  
  n=dim(dat)[1]
 
  ############################# Stage 2 ###################################
  # Step 1: Create the training matrix
  X <- as.data.frame(cbind(dat$A1, dat$A2, dat$X_11, dat$X_21, dat$X_3, dat$X_12, dat$X_22))
  colnames(X) <- c("A1", "A2", "X_11", "X_21", "X_3", "X_12", "X_22")
  X$A1 <- as.factor(X$A1)
  X$A2 <- as.factor(X$A2)
  y <- dat$Y
  bw <- npregbw(xdat = X, ydat = y,cxkertype = "gaussian", uxkertype = "aitchisonaitken") 
  
  # Step 4: Prepare prediction datasets
  newdata <- dat
  newdata$a2_0 <- 0
  newdata$a2_1 <- 1
  
  # Replace A2 with a2_0 and a2_1 values for predictions
  newdata_a2_0 <- newdata
  newdata_a2_0$A2 <- 0
  
  newdata_a2_1 <- newdata
  newdata_a2_1$A2 <- 1
  
  # Step 5: Create prediction matrices for a2_0 and a2_1
  X_a2_0 <- as.data.frame(newdata_a2_0[, c("A1", "A2", "X_11", "X_21", "X_3", "X_12", "X_22")])
  X_a2_1 <- as.data.frame(newdata_a2_1[, c("A1", "A2", "X_11", "X_21", "X_3", "X_12", "X_22")])
  combined_dataset <- rbind(X_a2_0, X_a2_1)
  colnames(combined_dataset) <- colnames(X)
  combined_dataset$A1 <- as.factor(combined_dataset$A1)
  combined_dataset$A2 <- as.factor(combined_dataset$A2)
  
  model <- npreg(bws = bw, txdat = X, tydat = y, exdat=combined_dataset)
  
  # Step 6: Make predictions
  predictions_a2_0 <- (model$mean)[1:n]
  predictions_a2_1 <- (model$mean)[(n+1):(2*n)]
  
  print(head(predictions_a2_0))
  print(head(predictions_a2_1))
  
  # find Y_a2_0
  newdata$Y_a2_0 <- predictions_a2_0 
  
  # find Y_a2_1
  newdata$Y_a2_1 <- predictions_a2_1
    
  # Create augumented dataset
  newdata0 <- newdata[,c(1:7,9,11)]
  colnames(newdata0)[c(8,9)] <- c("a2","Y_hat")
  newdata1 <- newdata[,c(1:7,10,12)]
  colnames(newdata1)[c(8,9)] <- c("a2","Y_hat")
  data_aug <- as.data.frame(rbind(newdata0,newdata1))
  
  ################################################# Stage 1 ###########
 
  
  X <- as.data.frame(cbind(data_aug$A1,data_aug$a2,data_aug$X_11,data_aug$X_21,data_aug$X_3))
  colnames(X) <- c("A1", "A2", "X_11", "X_21", "X_3")
  X$A1 <- as.factor(X$A1)
  X$A2 <- as.factor(X$A2)
  y <- data_aug$Y_hat
  bw <- npregbw(xdat = X, ydat = y, cxkertype = "gaussian", uxkertype = "aitchisonaitken")
  
  
  ########################################### Sample y missing #####################
  newdata_aug <- dat
  newdata_aug$a1_0 <- 0
  newdata_aug$a1_1 <- 1
  newdata_aug$a2_0 <- 0
  newdata_aug$a2_1 <- 1
  X_a1_0_a2_0 <- as.data.frame(cbind(newdata_aug[,c("a1_0","a2_0","X_11","X_21","X_3")]))
  X_a1_1_a2_1 <- as.data.frame(cbind(newdata_aug[,c("a1_1","a2_1","X_11","X_21","X_3")]))
  colnames(X_a1_0_a2_0) <- colnames(X)
  colnames(X_a1_1_a2_1) <- colnames(X)
  
  
  combined_dataset <- rbind(X_a1_0_a2_0, X_a1_1_a2_1)
  combined_dataset$A1 <- as.factor(combined_dataset$A1)
  combined_dataset$A2 <- as.factor(combined_dataset$A2)

  model <- npreg(bws = bw, txdat = X, tydat = y, exdat=combined_dataset)

  predictions_a1_0_a2_0 <- (model$mean)[1:n]
  predictions_a1_1_a2_1 <- (model$mean)[(n+1):(2*n)]
  

  ATE_est_val <- mean(predictions_a1_1_a2_1) - mean(predictions_a1_0_a2_0)
  

  print(head(predictions_a1_0_a2_0))
  print(head(predictions_a1_1_a2_1))
  # print(length(predictions_a1_0_a2_0))
  # print(length(predictions_a1_1_a2_1))
  # print(ATE_est_val)
  # print(predictions_a1_1_a2_1 == predictions_a1_0_a2_0)
  # stop("stop for debugging")
  
  return(ATE_est_val)
}



num_boot <- 100
M = 100  # number of datasets simulated: 50~500
n = 500
true_ATE_Tan <- 2.104496
est_ATE <- c()
sd_ATE <- c()
coverage_num <- 0
upper_CI <- 0
lower_CI <- 0
ATE_list <- c()

for (i in 1:M){
  seed_num <- 1000+i
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
  
  Y <- 0.5+0.1*A1 + 0.7*sin(A1)+ 0.4*sin(A2)+0.2*X_11 + 0.3*X_21^2 + 0.7*sin(X_3) + 0.2*X_11*X_21 +
    0.9*sqrt(abs(X_11*X_3))+0.8*log(abs(X_12*X_22*X_3+A2))+rnorm(n,0,sd=2.5)
  
  # Collect the propensity score and combine the dataset
  dat <- as.data.frame(cbind(A1,A2,X_11,X_21,X_3,X_12,X_22,Y))
  
  # upper_CI <- quantile(ATE_list, 0.975)
  # lower_CI <- quantile(ATE_list, 0.025)
  
  est_ATE[i] <- freq_npreg_ICE_X(dat)
  print("-----------Seed Number------------------")
  print(seed_num)
  print("--------------Individual ATE---------------")
  print(est_ATE[i])
  print("-----------------------------")
  
  # Perform bootstrap resampling
  # for (j in 1:num_boot) {
  #   resampled_data <- dat[sample(1:nrow(dat), size = nrow(dat), replace = TRUE), ]
  #   ATE_list[j] <- freq_npreg_ICE_X(resampled_data)
  # }
  # sd_ATE[i] <- sd(ATE_list)
  # # Confidence interval
  # lower_CI <- est_ATE[i] - 1.96 * sd_ATE[i]
  # upper_CI <- est_ATE[i] + 1.96 * sd_ATE[i]
  # c <- 0
  # 
  # if((true_ATE_Tan <= upper_CI) & (true_ATE_Tan >= lower_CI)){
  #   coverage_num <- coverage_num+1
  #   c <- 1
  # }
  # 
  # print("-----------Individual Coverage------------------")
  # print(c)
  # 
  # ATE_list <- c()
}



# Percentage Bias
bias_npreg <- mean(est_ATE)-true_ATE_Tan
percentage_bias_npreg <- abs(bias_npreg/true_ATE_Tan)
paste("(freq_npreg_Tan_X) Percentage Bias: ", percentage_bias_npreg)

# MCSD
MCSD <- sd(est_ATE)
paste("(freq_npreg_Tan_X) MCSD: ", MCSD)

# RMSE
npreg_RMSE <- sqrt(mean((est_ATE - true_ATE_Tan)^2))
paste("(freq_npreg_Tan_X) RMSE: ", npreg_RMSE)


#credible_interval_coverage_npreg <- coverage_num/M
#paste("(freq_npreg_Tan_X) Credible Interval Coverage: ", credible_interval_coverage_npreg)






