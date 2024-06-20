

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



# BLR
BLR_ICE_X <- function(dat){
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
  
  ############################# Stage 2 ###################################
  X = cbind(1,dat$A1,dat$A2,dat$X_11,dat$X_21,dat$X_3,dat$X_12,dat$X_22)   
  #X = cbind(1,dat$A1,dat$A2,dat$L1,dat$L2) 
  y=dat$Y
  
  ###############
  #Prior Settings
  ###############
  # delta_2
  mean_delta_2_prior <- rep(0,dim(X)[2])
  cov_delta_2_prior <- 10000*diag(dim(X)[2])
  # sigma_2^2
  a<-0.001
  b<-0.001
  
  ############################
  #Parameters
  ############################
  delta_2 <- matrix(0,nrow=mc_num,ncol=dim(X)[2])
  sigma_2 <- rep(0, times=mc_num)
  
  ###############
  #Initial Values
  ###############
  delta_2[1,] <- rep(1,dim(X)[2])
  sigma_2[1] <- 1
  
  Sigma_1_result <- c()
  Delta_1_result <- matrix(nrow=0,ncol=6)
  
  for(i in 2:mc_num){
    ######################################################################
    #Update delta_2
    ######################################################################
    mean_delta_2<- solve(solve(cov_delta_2_prior) + ((1/sigma_2[i-1])*(t(X)%*%X)))%*%
      (solve(cov_delta_2_prior) %*% mean_delta_2_prior + ((1/sigma_2[i-1])*(t(X)%*%y)))
    
    cov_delta_2<-solve(solve(cov_delta_2_prior) + ((1/sigma_2[i-1])*(t(X)%*%X)))
    
    delta_2[i,]<-rmnorm(n=1,mean=mean_delta_2,varcov=cov_delta_2)
    
    ################################
    #Update sigma_2
    ################################
    sigma_2[i] <- rinvgamma(1,n/2+a,t(dat$Y-as.matrix(X)%*%as.vector(delta_2[i,]))%*%
                              (dat$Y-as.matrix(X)%*%as.vector(delta_2[i,]))/2+b)
    
  }
  
  #### Create new dataframe with sample size 2N using stage 2 posterior
  rows_to_keep <- seq(from = burnin+keepevery_stage2, to = mc_num, by = keepevery_stage2)
  parameters <- as.data.frame(delta_2[rows_to_keep,])
  sigma_2_sub <- sigma_2[rows_to_keep]
  
  for(i in 1:length(sigma_2_sub)){
    newdata <- dat
    newdata$a2_0 <- 0
    newdata$a2_1 <- 1
    X_a2_0 <- as.matrix(cbind(1,newdata[,c("A1","a2_0","X_11","X_21","X_3","X_12","X_22")]))
    X_a2_1 <- as.matrix(cbind(1,newdata[,c("A1","a2_1","X_11","X_21","X_3","X_12","X_22")]))
    # find Y_a2_0
    newdata$Y_a2_0 <- c(X_a2_0%*%t((parameters[i,])) +
                          rnorm((dim(newdata)[1]),mean=0,sd=sqrt(sigma_2_sub[i])))
    
    # find Y_a2_1
    newdata$Y_a2_1 <- c(X_a2_1%*%t((parameters[i,])) +
                          rnorm((dim(newdata)[1]),mean=0,sd=sqrt(sigma_2_sub[i])))
    
    # Create augumented dataset
    newdata0 <- newdata[,c(1:7,9,11)]
    colnames(newdata0)[c(8,9)] <- c("a2","Y_hat")
    newdata1 <- newdata[,c(1:7,10,12)]
    colnames(newdata1)[c(8,9)] <- c("a2","Y_hat")
    # change to long format
    # Obtain D_aug 
    data_aug <- as.data.frame(rbind(newdata0,newdata1))
    
    ################################################# Stage 1 ###########
    X <- as.matrix(cbind(1,data_aug$A1,data_aug$a2,data_aug$X_11,
                         data_aug$X_21,data_aug$X_3))
    y <- data_aug$Y_hat
    
    ###############
    #Prior Settings
    ###############
    
    
    a<-0.001
    b<-0.001
    
    mean_delta_1_prior <- rep(0,dim(X)[2])
    cov_delta_1_prior <- 10000*diag(dim(X)[2])
    
    
    ############################
    #Parameters
    ############################
    delta_1 <- matrix(0,nrow=mc_num_stage_1,ncol=dim(X)[2])
    sigma_1 <- rep(0, times=mc_num_stage_1)
    
    ###############
    #Initial Values
    ###############
    
    delta_1[1,] <- rep(1,dim(X)[2])
    sigma_1[1] <- 1
    
    for(j in 2:mc_num_stage_1){
      ######################################################################
      #Update delta_2
      ######################################################################
      
      mean_delta_1 <- solve(solve(cov_delta_1_prior) +
                              ((1/sigma_1[j-1])*(t(X)%*%X)))%*%
        (solve(cov_delta_1_prior) %*% mean_delta_1_prior +
           ((1/sigma_1[j-1])*(t(X)%*%y)))
      
      cov_delta_1<-solve(solve(cov_delta_1_prior) + ((1/sigma_1[j-1])*(t(X)%*%X)))
      
      delta_1[j,]<-rmnorm(n=1,mean=mean_delta_1,varcov=cov_delta_1)
      
      ################################
      #Update sigma_1
      ################################
      sigma_1[j] <- rinvgamma(1,n+a,t(y-as.matrix(X)%*%as.vector(delta_1[j,]))%*%
                                (y-as.matrix(X)%*%as.vector(delta_1[j,]))/2+b)
      
    }
    
    rows_to_keep_1 <- seq(from = burnin+keepevery_stage1, to = mc_num_stage_1, by = keepevery_stage1)
    for (index_keep in rows_to_keep_1){
      Sigma_1_result <- c(Sigma_1_result,sigma_1[index_keep])
      Delta_1_result <- rbind(Delta_1_result, delta_1[index_keep,])
    }
  }
  
  
  ########################################### Sample y missing #####################
  # define linear model 1
  newdata_aug <- dat
  newdata_aug$a1_0 <- 0
  newdata_aug$a1_1 <- 1
  newdata_aug$a2_0 <- 0
  newdata_aug$a2_1 <- 1
  parameters_1 <- as.data.frame(Delta_1_result)
  #colnames(parameters_1) <- c("delta_1_0","delta_1_1","delta_1_2","delta_1_3")
  sigma_1_sub <- Sigma_1_result
  X_a1_0_a2_0 <- as.matrix(cbind(1,newdata_aug[,c("a1_0","a2_0","X_11","X_21","X_3")]))
  X_a1_1_a2_0 <- as.matrix(cbind(1,newdata_aug[,c("a1_1","a2_0","X_11","X_21","X_3")]))
  X_a1_0_a2_1 <- as.matrix(cbind(1,newdata_aug[,c("a1_0","a2_1","X_11","X_21","X_3")]))
  X_a1_1_a2_1 <- as.matrix(cbind(1,newdata_aug[,c("a1_1","a2_1","X_11","X_21","X_3")]))
  
  res_a1_0_a2_0 <- list()
  res_a1_1_a2_0 <- list()
  res_a1_0_a2_1 <- list()
  res_a1_1_a2_1 <- list()
  ATE_list <- c()
  for(i in 1:length(sigma_1_sub)){
    ############# Remove error terms in the last step ###############
    res_a1_0_a2_0[[i]] <- c(X_a1_0_a2_0%*%t((parameters_1[i,]))) 
    # find Y_a1_1_a2_0
    res_a1_1_a2_0[[i]] <- c(X_a1_1_a2_0%*%t((parameters_1[i,])))
    # find Y_a1_0_a2_1
    res_a1_0_a2_1[[i]] <- c(X_a1_0_a2_1%*%t((parameters_1[i,]))) 
    # find Y_a1_1_a2_1
    res_a1_1_a2_1[[i]] <- c(X_a1_1_a2_1%*%t((parameters_1[i,])))
    ATE_list[i] <- mean(c(X_a1_1_a2_1%*%t((parameters_1[i,])))) - mean(c(X_a1_0_a2_0%*%t((parameters_1[i,]))))
  }
  #print(length(ATE_list))
  return(ATE_list)
}



M = 100  # number of datasets simulated: 50~500
n = 500
true_ATE_Hu_2021 <- -11.02395
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
  
  # Combine the dataset
  dat <- as.data.frame(cbind(A1,A2,X_11,X_21,X_3,X_12,X_22,Y))
  
  
  ATE_list <- BLR_ICE_X(dat)
  est_ATE[i] <- mean(ATE_list)
  sd_ATE[i] <- sd(ATE_list)
  upper_CI <- quantile(ATE_list, 0.975)
  lower_CI <- quantile(ATE_list, 0.025)
  
  print("-----------Seed Number------------------")
  print(seed_num)
  
  c <- 0
  
  if((true_ATE_Hu_2021 <= upper_CI) & (true_ATE_Hu_2021 >= lower_CI)){
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
bias_BLR <- mean(est_ATE)-true_ATE_Hu_2021
percentage_bias_BLR <- abs(bias_BLR/true_ATE_Hu_2021)
paste("(BLR_Hu_X) Percentage Bias: ", percentage_bias_BLR)

# Credible Interval Coverage: the proportion of all datasets for which the 95% credible intervals contained the true ATE
credible_interval_coverage_BLR <- coverage_num/M
paste("(BLR_Hu_X) Credible Interval Coverage: ", credible_interval_coverage_BLR)

# RMSE
BLR_RMSE <- sqrt(mean((est_ATE - true_ATE_Hu_2021)^2))
paste("(BLR_Hu_X) RMSE: ", BLR_RMSE)

# MCSE
MCSE <- sd(est_ATE)
paste("(BLR_Hu_X) MCSE: ", MCSE)

# AESE
AESE <- mean(sd_ATE)
paste("(BLR_Hu_X) AESE: ", AESE)








