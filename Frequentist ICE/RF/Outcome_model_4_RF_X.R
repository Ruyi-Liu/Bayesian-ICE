# Hu_RF
print("freq_RF_Hu_X")

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
library(randomForest)


freq_RF_ICE_X <- function(dat){
  
  n=dim(dat)[1]
  
  ############################# Stage 2 ###################################
  # Step 1: Create the training dataset
  X <- dat[, c("A1", "A2", "X_11", "X_21", "X_3", "X_12", "X_22")] 
  y <- dat$Y
  
  # Updated: Fit the Random Forest model
  model <- randomForest(X, y, ntree = 500)
  
  # Step 4: Prepare prediction datasets
  newdata <- dat
  newdata$a2_0 <- 0
  newdata$a2_1 <- 1
  
  # Replace A2 with a2_0 and a2_1 values for predictions
  newdata_a2_0 <- newdata
  newdata_a2_0$A2 <- 0
  
  newdata_a2_1 <- newdata
  newdata_a2_1$A2 <- 1
  
  predictions_a2_0 <- predict(model, newdata = newdata_a2_0[, colnames(X)])
  predictions_a2_1 <- predict(model, newdata = newdata_a2_1[, colnames(X)])
  

  # find Y_a2_0
  newdata$Y_a2_0 <- predictions_a2_0 
  
  # find Y_a2_1
  newdata$Y_a2_1 <- predictions_a2_1 
  # Create augumented dataset
  newdata0 <- newdata[,c(1:7,9,11)]
  colnames(newdata0)[c(8,9)] <- c("a2","Y_hat")
  newdata1 <- newdata[,c(1:7,10,12)]
  colnames(newdata1)[c(8,9)] <- c("a2","Y_hat")
  # change to long format
  # Obtain D_aug 
  data_aug <- as.data.frame(rbind(newdata0,newdata1))
  ################################################# Stage 1 ###########
  X <- data_aug[, c("A1", "a2", "X_11", "X_21", "X_3")]
  y <- data_aug$Y_hat
  model_1 <- randomForest(X, y, ntree = 500, mtry = 3)
  ########################################### Sample y missing #####################
  newdata_aug <- dat
  newdata_aug$a1_0 <- 0
  newdata_aug$a1_1 <- 1
  newdata_aug$a2_0 <- 0
  newdata_aug$a2_1 <- 1
  
  data_a1_0_a2_0 <- newdata_aug[, c("a1_0", "a2_0", "X_11", "X_21", "X_3")]
  data_a1_1_a2_1 <- newdata_aug[, c("a1_1", "a2_1", "X_11", "X_21", "X_3")]
  colnames(data_a1_0_a2_0) <- colnames(X)
  colnames(data_a1_1_a2_1) <- colnames(X)
  
  predictions_a1_0_a2_0 <- predict(model_1, newdata = data_a1_0_a2_0)
  predictions_a1_1_a2_1 <- predict(model_1, newdata = data_a1_1_a2_1)
  
  ATE_est_val <- mean(predictions_a1_1_a2_1) - mean(predictions_a1_0_a2_0)
  
  #stop("stop for debugging")  
  return(ATE_est_val)
}



num_boot <- 100
M = 100  # number of datasets simulated: 50~500
n = 500
true_ATE_Hu <- -11.02395
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
  
  Y <- ifelse(A1==1 & A2==1, -10-5*sin(pi*X_11*X_3)+2/(1+exp(-X_22)),
              ifelse(A1==1 & A2==0, -6+3*(X_11^2)-5*sin(X_21)+4/(1+exp(-X_3))+2*X_12-3*X_22,
                     ifelse(A1==0 & A2==1, -4+1*X_11^2-2*sin(X_3)+2/(1+exp(-X_22)),
                            7.5-1/(1+exp(-X_21))+1*sin(X_3)-1*X_12^2+2*X_12-1*X_22^2)))
  
  Y <- Y + rnorm(n,0,sd=15)
  
  # Collect the propensity score and combine the dataset
  dat <- as.data.frame(cbind(A1,A2,X_11,X_21,X_3,X_12,X_22,Y))
  
  # upper_CI <- quantile(ATE_list, 0.975)
  # lower_CI <- quantile(ATE_list, 0.025)
  
  est_ATE[i] <- freq_RF_ICE_X(dat)
  print("-----------Seed Number------------------")
  print(seed_num)
  print("--------------Individual ATE---------------")
  print(est_ATE[i])
  print("-----------------------------")
  
  # Perform bootstrap resampling
  for (j in 1:num_boot) {
    resampled_data <- dat[sample(1:nrow(dat), size = nrow(dat), replace = TRUE), ]
    ATE_list[j] <- freq_RF_ICE_X(resampled_data)
  }
  sd_ATE[i] <- sd(ATE_list)
  # Confidence interval
  lower_CI <- est_ATE[i] - 1.96 * sd_ATE[i]
  upper_CI <- est_ATE[i] + 1.96 * sd_ATE[i]
  c <- 0

  if((true_ATE_Hu <= upper_CI) & (true_ATE_Hu >= lower_CI)){
    coverage_num <- coverage_num+1
    c <- 1
  }

  print("-----------Individual Coverage------------------")
  print(c)
  
  ATE_list <- c()
}



# Percentage Bias
bias_RF <- mean(est_ATE)-true_ATE_Hu
percentage_bias_RF <- abs(bias_RF/true_ATE_Hu)
paste("(freq_RF_Hu_X) Percentage Bias: ", percentage_bias_RF)

# MCSD
MCSD <- sd(est_ATE)
paste("(freq_RF_Hu_X) MCSD: ", MCSD)

# RMSE
RF_RMSE <- sqrt(mean((est_ATE - true_ATE_Hu)^2))
paste("(freq_RF_Hu_X) RMSE: ", RF_RMSE)

#Credible Interval Coverage: the proportion of all datasets for which the 95% credible intervals contained the true ATE
credible_interval_coverage_RF <- coverage_num/M
paste("(freq_RF_Hu_X) Credible Interval Coverage: ", credible_interval_coverage_RF)











