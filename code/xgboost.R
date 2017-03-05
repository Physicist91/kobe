# xgboost
# https://www.kaggle.com/brandao/kobe-bryant-shot-selection/xgboost-in-r-kobe-bryant-benchmark/code

setwd("~/kobe")

library(xgboost)
library(data.table)
library(Matrix)
library(caret)

# load the data
completeData <- as.data.frame(fread("data.csv", header = T, stringsAsFactors = T))

# Data cleaning
completeData$shot_distance[completeData$shot_distance>45] <- 45
dropped_features <- c("seconds_remaining", "minutes_remaining", "team_name", "team_id", "game_event_id",
                      "game_id", "matchup", "lon", "lat", "game_date")
cat_features <- c("action_type", "combined_shot_type", "period", "season", "shot_type",
                  "shot_zone_area", "shot_zone_basic", "shot_zone_range",
                  "opponent")
for(col in cat_features){
  completeData[,col] <- as.character(completeData[,col])
}


################## Feature engineering #####################
############################################################
completeData$time_remaining <- completeData$minutes_remaining*60+completeData$seconds_remaining;
completeData$last5secs <- (completeData$time_remaining < 5) * 1
completeData$home <- grepl('vs',completeData$matchup) * 1
completeData$month <- substr(completeData$game_date, 6, 7)
completeData$year <- substr(completeData$game_date, 1, 4)


# One-hot encoding
# trainM<-data.matrix(train, rownames.force = NA)
ohe_features <- c('action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
                  'shot_zone_area','shot_zone_basic','shot_zone_range', 'month', 'year',
                  'opponent')
dummies <- dummyVars(as.formula(paste0('~ ',paste(ohe_features, collapse=" + "))), data=completeData) #based on cat_features
df_ohe <- as.data.frame(predict(dummies, newdata=completeData))
names(df_ohe) <- make.names(colnames(df_ohe))
completeData <- cbind(completeData[,!names(completeData) %in% ohe_features], df_ohe)


###########################################################
###########################################################

# split data into train and test set
train<-subset(completeData, !is.na(completeData$shot_made_flag))
test<-subset(completeData, is.na(completeData$shot_made_flag))

# remove id as it does not provide any predictive power
test.id <- test$shot_id
train$shot_id <- NULL
test$shot_id <- NULL

train <- train[,!names(train) %in% dropped_features]
test <- test[,!names(test) %in% dropped_features]



#################################################################
################### ALGORITHMS ##################################
#################################################################

# 1. XGBOOST TREE

train.y = train$shot_made_flag
trainM <- sparse.model.matrix(shot_made_flag ~ .-1, data=train)

pred <- rep(0,nrow(test))
dtrain <- xgb.DMatrix(data=trainM, label=train.y, missing = NaN)
watchlist <- list(trainM=dtrain)

set.seed(1984)
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.035,
                max_depth           = 4,
                subsample           = 0.8,
                colsample_bytree    = 0.9
)

clf <- xgb.cv(  params              = param, 
                data                = dtrain, 
                nrounds             = 1500, 
                verbose             = 1,
                watchlist           = watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
)

bestRound <- which.min( as.matrix(clf)[,3] )
cat("Best round:", bestRound,"\n")
cat("Best result:",min(as.matrix(clf)[,3]),"\n")



### Parameter Tuning
# depths <- c(4, 5, 6, 7, 8, 9, 10)
# etas <- c(0.1, 0.5, 0.01, 0.001)
# subsamples <- c(0.4, 0.5, 0.7, 0.8, 0.9)
# colsamples <- c(0.4, 0.5, 0.7, 0.8, 0.9)
# #for(depth in depths){
# #  param$max_depth <- depth
# #for(eta in etas){
# #  param$eta <- eta
# #for(subsample in subsamples){
# #  param$subsample <- subsample
# for(colsample in colsamples){
#   param$colsample_bytree <- colsample
#   clf <- xgb.cv(  params              = param, 
#                   data                = dtrain, 
#                   nrounds             = 1500, 
#                   verbose             = 0,
#                   watchlist           = watchlist,
#                   maximize            = FALSE,
#                   nfold               = 3,
#                   early.stop.round    = 10,
#                   print.every.n       = 10
#   )
#   bestRound <- which.min( as.matrix(clf)[,3] )
#   #cat("Depth:", depth,"\n")
#   #cat("Subsample:",subsample,"\n")
#   #cat("Eta:",eta,"\n")
#   cat("Colsample_bytree:", colsample,"\n")
#   cat("Best round:", bestRound,"\n")
#   cat("Best result:",min(as.matrix(clf)[,3]),"\n")
# }




clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = bestRound,
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

test$shot_made_flag <- -1
testM <- sparse.model.matrix(shot_made_flag ~.-1, data=test)
preds <- predict(clf, testM)
submission <- data.frame(shot_id=test.id, shot_made_flag=preds)
write.csv(submission, "basicXGBoost.csv", row.names = F)


# 2. XGBLinear

param <- list(  objective           = "binary:logistic", 
                booster             = "gblinear",
                eval_metric         = "logloss",
                lambda = 1,
                lambda_bias = 1,
                aplha = 3
)
set.seed(1984)
clf <- xgb.cv(  params              = param, 
                data                = dtrain, 
                nrounds             = 1500, 
                verbose             = 1,
                watchlist           = watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
)

bestRound <- which.min( as.matrix(clf)[,3] )
cat("Best round:", bestRound,"\n")
cat("Best result:",min(as.matrix(clf)[,3]),"\n")


### Hyperparameter tuning
linear_params <- 1:5
df <- data.frame(alpha=integer(),lambda=integer(),lambda_bias=integer())
for(lambda in linear_params){
  param$lambda <- lambda
  for(lambda_bias in linear_params){
    param$lambda_bias <- lambda_bias
    for(alpha in linear_params){
      param$alpha <- alpha
      set.seed(1984)
      clf <- xgb.cv(  params              = param, 
                      data                = dtrain, 
                      nrounds             = 1500, 
                      verbose             = 0,
                      watchlist           = watchlist,
                      maximize            = FALSE,
                      nfold               = 3,
                      early.stop.round    = 10,
                      print.every.n       = 1
      )
      bestRound <- which.min( as.matrix(clf)[,3] )
      df <- rbind(df, data.frame(alpha=alpha, lambda=lambda,lambda_bias=lambda_bias))
      cat("Best round:", bestRound,"\n")
      cat("Best result:",min(as.matrix(clf)[,3]),"\n")
    }
  }
}

# 3. Random Forest
# xgboost
# https://www.kaggle.com/brandao/kobe-bryant-shot-selection/xgboost-in-r-kobe-bryant-benchmark/code

setwd("~/kobe")

library(xgboost)
library(data.table)
library(Matrix)
library(caret)

# load the data
completeData <- as.data.frame(fread("data.csv", header = T, stringsAsFactors = T))

# Data cleaning
completeData$shot_distance[completeData$shot_distance>45] <- 45
dropped_features <- c("seconds_remaining", "minutes_remaining", "team_name", "team_id", "game_event_id",
                      "game_id", "matchup", "lon", "lat", "game_date")
cat_features <- c("action_type", "combined_shot_type", "period", "season", "shot_type",
                  "shot_zone_area", "shot_zone_basic", "shot_zone_range",
                  "opponent")
for(col in cat_features){
  completeData[,col] <- as.character(completeData[,col])
}


################## Feature engineering #####################
############################################################
completeData$time_remaining <- completeData$minutes_remaining*60+completeData$seconds_remaining;
completeData$last5secs <- (completeData$time_remaining < 5) * 1
completeData$home <- grepl('vs',completeData$matchup) * 1
completeData$month <- substr(completeData$game_date, 6, 7)
completeData$year <- substr(completeData$game_date, 1, 4)


# One-hot encoding
# trainM<-data.matrix(train, rownames.force = NA)
ohe_features <- c('action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
                  'shot_zone_area','shot_zone_basic','shot_zone_range', 'month', 'year',
                  'opponent')
dummies <- dummyVars(as.formula(paste0('~ ',paste(ohe_features, collapse=" + "))), data=completeData) #based on cat_features
df_ohe <- as.data.frame(predict(dummies, newdata=completeData))
names(df_ohe) <- make.names(colnames(df_ohe))
completeData <- cbind(completeData[,!names(completeData) %in% ohe_features], df_ohe)


###########################################################
###########################################################

# split data into train and test set
train<-subset(completeData, !is.na(completeData$shot_made_flag))
test<-subset(completeData, is.na(completeData$shot_made_flag))

# remove id as it does not provide any predictive power
test.id <- test$shot_id
train$shot_id <- NULL
test$shot_id <- NULL

train <- train[,!names(train) %in% dropped_features]
test <- test[,!names(test) %in% dropped_features]



#################################################################
################### ALGORITHMS ##################################
#################################################################

# 1. XGBOOST TREE

train.y = train$shot_made_flag
trainM <- sparse.model.matrix(shot_made_flag ~ .-1, data=train)

pred <- rep(0,nrow(test))
dtrain <- xgb.DMatrix(data=trainM, label=train.y, missing = NaN)
watchlist <- list(trainM=dtrain)

set.seed(1984)
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.035,
                max_depth           = 4,
                subsample           = 0.8,
                colsample_bytree    = 0.9
)

clf <- xgb.cv(  params              = param, 
                data                = dtrain, 
                nrounds             = 1500, 
                verbose             = 1,
                watchlist           = watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
)

bestRound <- which.min( as.matrix(clf)[,3] )
cat("Best round:", bestRound,"\n")
cat("Best result:",min(as.matrix(clf)[,3]),"\n")



### Parameter Tuning
# depths <- c(4, 5, 6, 7, 8, 9, 10)
# etas <- c(0.1, 0.5, 0.01, 0.001)
# subsamples <- c(0.4, 0.5, 0.7, 0.8, 0.9)
# colsamples <- c(0.4, 0.5, 0.7, 0.8, 0.9)
# #for(depth in depths){
# #  param$max_depth <- depth
# #for(eta in etas){
# #  param$eta <- eta
# #for(subsample in subsamples){
# #  param$subsample <- subsample
# for(colsample in colsamples){
#   param$colsample_bytree <- colsample
#   clf <- xgb.cv(  params              = param, 
#                   data                = dtrain, 
#                   nrounds             = 1500, 
#                   verbose             = 0,
#                   watchlist           = watchlist,
#                   maximize            = FALSE,
#                   nfold               = 3,
#                   early.stop.round    = 10,
#                   print.every.n       = 10
#   )
#   bestRound <- which.min( as.matrix(clf)[,3] )
#   #cat("Depth:", depth,"\n")
#   #cat("Subsample:",subsample,"\n")
#   #cat("Eta:",eta,"\n")
#   cat("Colsample_bytree:", colsample,"\n")
#   cat("Best round:", bestRound,"\n")
#   cat("Best result:",min(as.matrix(clf)[,3]),"\n")
# }




clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = bestRound,
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

test$shot_made_flag <- -1
testM <- sparse.model.matrix(shot_made_flag ~.-1, data=test)
preds <- predict(clf, testM)
submission <- data.frame(shot_id=test.id, shot_made_flag=preds)
write.csv(submission, "basicXGBoost.csv", row.names = F)


# 2. XGBLinear

param <- list(  objective           = "binary:logistic", 
                booster             = "gblinear",
                eval_metric         = "logloss",
                lambda = 1,
                lambda_bias = 1,
                aplha = 3
)
set.seed(1984)
clf <- xgb.cv(  params              = param, 
                data                = dtrain, 
                nrounds             = 1500, 
                verbose             = 1,
                watchlist           = watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
)

bestRound <- which.min( as.matrix(clf)[,3] )
cat("Best round:", bestRound,"\n")
cat("Best result:",min(as.matrix(clf)[,3]),"\n")


### Hyperparameter tuning
linear_params <- 1:5
df <- data.frame(alpha=integer(),lambda=integer(),lambda_bias=integer())
for(lambda in linear_params){
  param$lambda <- lambda
  for(lambda_bias in linear_params){
    param$lambda_bias <- lambda_bias
    for(alpha in linear_params){
      param$alpha <- alpha
      set.seed(1984)
      clf <- xgb.cv(  params              = param, 
                      data                = dtrain, 
                      nrounds             = 1500, 
                      verbose             = 0,
                      watchlist           = watchlist,
                      maximize            = FALSE,
                      nfold               = 3,
                      early.stop.round    = 10,
                      print.every.n       = 1
      )
      bestRound <- which.min( as.matrix(clf)[,3] )
      df <- rbind(df, data.frame(alpha=alpha, lambda=lambda,lambda_bias=lambda_bias))
      cat("Best round:", bestRound,"\n")
      cat("Best result:",min(as.matrix(clf)[,3]),"\n")
    }
  }
}

# 3. Random Forest

library(randomForest)
trControl <- trainControl(method="cv",number=5,verboseIter=TRUE)
rf <- train(shot_made_flag ~ ., data=train, method="rf",metric="logLoss")






