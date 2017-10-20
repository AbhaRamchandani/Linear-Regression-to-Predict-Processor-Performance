# ML Autumn 2015: 
# Submitted by Group : Three Musketeers
# rabha@student.ethz.ch
# rsridhar@student.ethz.ch
# xandeepv@student.ethz.ch

library(MASS)

# N-fold Cross-Validation with Ridge Regression
crossval <- function(data, lambda=0, ngroup=nrow(data)) {

  n <- nrow(data)  # No of rows (samples)
  r <- ncol(data)  # No of columns (variables)
  resid <- rep(0, ngroup)  # Prepare vector for storing residuals
  
  # Compute indices of CV folds
  splits <- round(seq(0, n, len=ngroup+1))
  indices <- lapply(1:ngroup, function(i) ((splits[i]+1):splits[i+1]))
  
  # Add intercept to design matrix
  data <- cbind(rep(1, n), data)

  # Loop over folds
  for (i in 1:ngroup) {
  
    # Fit on all folds except i-th (and excluding intercept column)
    fit <- lm.ridge(V15 ~ ., data=data[-indices[[i]],-1], lambda=lambda)

    # Predict on i-th fold and compute residual
    resid[i] <- mean(( data[indices[[i]],r+1] - as.matrix(data[indices[[i]],1:r]) %*% matrix(coef(fit), r, 1) )^2)
  }
  
  # Return CV RSS
  return(mean(resid))
}

# Optimal (for lambda) GCV error with ridge regression
gcv <- function(pred, y) {

  # Put data in data frame
  data <- as.data.frame(cbind(pred, y))
  colnames(data)[ncol(data)] <- "V15"
  
  # Prepare lambda grid: powers of 2
  lambda <- 2^seq(from=-10, to=10, length.out=21)
  
  # Compute GCV values for lambda grid
  res <- lm.ridge(V15 ~ ., data=data, lambda=lambda)
  #res <- optim(par=lambda[which.min(res$GCV)], fn=function(lambda) lm.ridge(V15 ~ ., data=data, lambda=abs(lambda))$GCV)
  
  # Return optimal GCV error
  return(min(res$GCV))
}

# Greedy forward feature extraction
greedyFeatures <- function(pred, y, nfeat) {

  # Possible feature transformations
  trans <- c(function(x) x^2,
             function(x) sqrt(abs(x)),
             function(x) log(abs(x)),
             function(x) x,
             function(x) x^3,
             function(x) x*log(abs(x)),
             function(x) x*x*log(abs(x)),
             function(x) x*x*log(abs(x))*log(abs(x)))
  trans.names <- c("sq",
                   "sqrtabs",
                   "logabs",
                   "Id",
                   "cubic",
                   "xlogabs",
                   "xxlogabs",
                   "xxlogabsq")
  P <- length(trans)
  
  # List for storing (indices of) feature transformations
  feat.trans <- list()

  # Add nfeat features
  for (rep in 1:nfeat) {
 
    
    p <- ncol(pred)
    #del <- array(0, dim=p)
    sadd <- array(0, dim=c(p,P))
    cadd <- array(0, dim=c(p,P,p))
  
    # Loop over already existing features
    for (i in 1:p) {
      
       # Loop over possible transformations
       for (j in 1:P) {
         
         # Single addition
         newpred <- trans[[j]](pred[,i])
         sadd[i,j] <- gcv(cbind(pred, newpred), y)
         
         # Cross additions
         for (k in 1:p) {        
           cadd[i,j,k] <- gcv(cbind(pred, newpred*pred[,k]), y)
         }
       }
    }
    
    # Check which operation was best
    if (min(sadd) < min(cadd)) {  # Single addition
      
      # Find indices of best single addition
      ind <- arrayInd(which.min(sadd), dim(sadd))
      
      # Update predictor matrix
      pred <- cbind(pred, trans[[ind[2]]](pred[,ind[1]]))
      predname <- paste(trans.names[ind[2]], "(", colnames(pred)[ind[1]], ")", sep="")
      colnames(pred)[ncol(pred)] <- predname
      
      # Store in feature transformation list
      feat.trans[[rep]] <- c(ind[1],ind[2],0)
      
      print(paste(rep, " Adding feature", predname))
    
      print(paste("Current Score:", min(sadd)))
    }
    else {  # Cross addition
    
      # Find indices of best single addition
      ind <- arrayInd(which.min(cadd), dim(cadd))
      
      # Update predictor matrix
      pred <- cbind(pred, trans[[ind[2]]](pred[,ind[1]]) * pred[,ind[3]])
      predname <- paste(trans.names[ind[2]], "(", colnames(pred)[ind[1]], ")*", colnames(pred)[ind[3]], sep="")
      colnames(pred)[ncol(pred)] <- predname
      
      # Store in feature transformation list
      feat.trans[[rep]] <- c(ind[1],ind[2],ind[3])
     
      print(paste(rep, " Adding feature", predname))
      print(paste("Current Score:", min(cadd)))
    }
  }

  return(list(pred=pred, feat.trans=feat.trans))
}

# Transform features according to transformation list for prediction
transformFeatures <- function(pred, feat.trans) {
  
  # Possible feature transformations
  trans <- c(function(x) x^2,
             function(x) sqrt(abs(x)),
             function(x) log(abs(x)),
             function(x) x,
             function(x) x^3,
             function(x) x*log(abs(x)),
             function(x) x*x*log(abs(x)),
             function(x) x*x*log(abs(x))*log(abs(x)))

  # Loop over new features
  for (index in 1:length(feat.trans)) {
  
    # Recover transformation indices (see greedyFeatures)
    i <- feat.trans[[index]][1]
    j <- feat.trans[[index]][2]
    k <- feat.trans[[index]][3]
    
    # Add new features
    if (k != 0) {  # Cross Addition
      pred <- cbind(pred, trans[[j]](pred[,i])*pred[,k])
    }
    else {  # Single Addition
      pred <- cbind(pred, trans[[j]](pred[,i]))
    }
  }
  
  return(pred)
}



# Read in data
#data <- read.table("train.csv", sep=",")
data <- read.csv("C:\\Users\\XANDEEP\\Dropbox\\Autumn2015\\ML\\Project1\\train.csv",sep=",",header=T)

# Add features
res <- greedyFeatures(data[,-ncol(data)], data[,ncol(data)], 11)
feat.trans <- res$feat.trans
data <- cbind(as.data.frame(res$pred), data.frame(V15=data[,15]))

# Optimize lambda with GCV
lambda <- 2^seq(from=-15, to=15, length.out=51)  # Global search: on grid of powers of 2
res <- lm.ridge(V15 ~ ., data=data, lambda=lambda)
res <- optim(par=lambda[which.min(res$GCV)], fn=function(lambda) lm.ridge(V15 ~ ., data=data, lambda=abs(lambda))$GCV)  # Do local optimization starting with best value from global seach

# Compute CVRMSE with optimal lambda
print(sqrt(crossval(data, lambda=res$par))/mean(data$V15))  #0.2199245 for 8 Params and 0.214981 for 13 params

# Fit on full training dataset with optimal lambda
fit <- lm.ridge(V15 ~ ., data=data, lambda=res$par)

# Predict on Validation data
#data.valid <- read.table("handout/validation.csv", sep=",")
data.valid <- read.csv("C:\\Users\\XANDEEP\\Dropbox\\Autumn2015\\ML\\Project1\\validate.csv", sep=",",header=T)
data.valid <- transformFeatures(data.valid, feat.trans)
pred.valid <- cbind(1, as.matrix(data.valid)) %*% matrix(coef(fit), ncol(data), 1)
write.table(pred.valid, file="C:\\Users\\XANDEEP\\Dropbox\\Autumn2015\\ML\\Project1\\threeM_final009.csv", row.names=FALSE, col.names=FALSE)
fit
