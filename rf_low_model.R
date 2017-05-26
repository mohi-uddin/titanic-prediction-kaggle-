rm(list=ls())

# Load Packages
library(caret)
library(xgboost)

path = "D:/mohi/kaggle work/kaggle/titanic"
setwd(path)

# Read Data
train <- read.csv("train.csv",na.strings=c(""," ","NA"))
test <- read.csv("test.csv",na.strings=c(""," ","NA"))
# Test whether data is successfully loaded
names(train)

table(is.na(Train))
sapply(train, function(x) sum(is.na(x)))
train<- train[,-c(11,9,4)]
str(train)

names(test)
sapply(test, function(x) sum(is.na(x)))
test<- test[,-c(10,8,3)]

library(data.table)
#impute missing values by class
#do not make dummy variables, they are not helping
imp <- impute(train, classes = list(factor = imputeMode(), integer = imputeMedian(),numeric=imputeMedian()))
imp1 <- impute(test, classes = list(factor = imputeMode(), integer = imputeMedian(), numeric=imputeMedian()))

train <- imp$data
test <- imp1$data

setDT(train)
setDT(test)


# Encoding sex, as 0,1 -------------------------------------

train$Sex[1:50]
train[,Sex := as.integer(as.factor(Sex))-1]
test[,Sex := as.integer(as.factor(Sex))-1]

# One Hot Encoding --------------------------------------------------------

train_mod <- train[,.(Embarked)]
test_mod <- test[,.(Embarked)]
train_mod[1:50]
# train_mod[is.na(train_mod)] <- "-1"
# test_mod[is.na(test_mod)] <- "-1"
train_ex[1:50]
train_ex <- model.matrix(~.+0, data = train_mod)
test_ex <- model.matrix(~.+0, data = test_mod)

train1 <- as.data.table(train_ex)
test1 <- as.data.table(test_ex)

new_train <- cbind(train, train1)
new_test <- cbind(test, test1)

new_train[,c("Embarked") := NULL]
new_test[,c("Embarked") := NULL]
View(test)
train = new_train
test = new_test

setDF(train)
setDF(test)
str(test)
test$Survived <- sample(c("0","1"),size = 418,replace = T)

test$Survived = as.numeric(test$Survived)
submit = test$PassengerId
test$PassengerId = NULL
train$PassengerId = NULL
train$Survived = NULL
test$Survived = NULL

#######################################33##
train$Survived = as.factor(train$Survived)
train$Sex = as.factor(train$Sex)
test$Sex = as.factor(test$Sex)
train$Pclass = as.factor(train$Pclass)
test$Pclass = as.factor(test$Pclass)
train$EmbarkedC = as.factor(train$EmbarkedC)
train$EmbarkedQ = as.factor(train$EmbarkedQ)
train$EmbarkedS = as.factor(train$EmbarkedS)
test$EmbarkedC = as.factor(test$EmbarkedC)
test$EmbarkedQ = as.factor(test$EmbarkedQ)
test$EmbarkedS = as.factor(test$EmbarkedS)
str(train)
str(test)
###########################################33
y=train$Survived
y=as.factor(y)
model_xgb_1 <- xgboost(data=as.matrix(train),label=as.matrix(y),cv=5,objective="binary:logistic",nrounds=500,max.depth=10,eta=0.1,colsample_bytree=0.5,seed=235,metric="auc",importance=1)

pred <- predict(model_xgb_1, as.matrix(test))
head(pred)
submit <- data.frame(submit, "Purchase" = pred)
write.csv(submit, "submit.csv", row.names=F)


#create a task
trainTask <- makeClassifTask(data = train,target = "Survived")
testTask <- makeClassifTask(data = test, target = "Survived")


#normalize the variables
trainTask <- normalizeFeatures(trainTask,method = "standardize")
testTask <- normalizeFeatures(testTask,method = "standardize")
trainTask <- dropFeatures(task = trainTask,features = c("PassengerId"))

getParamSet("classif.randomForest")

#create a learner
rf <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
  importance = TRUE
)

#set tunable parameters
#grid search to find hyperparameters
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#let's do random search for 50 iterations
rancontrol <- makeTuneControlRandom(maxit = 50L)

#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = trainTask, par.set = rf_param, control = rancontrol, measures = acc)
#cv accuracy
rf_tune$y#best parameters
rf_tune$x

#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rforest <- train(rf.tree, trainTask)
getLearnerModel(t.rpart)

#make predictions
rfmodel <- predict(rforest, testTask)
rfmodel
#submission file
submit <- data.frame( PassengerId= test$PassengerId, Survived = rfmodel$data$response)
write.csv(submit, "submit_titanic_pred.csv",row.names = F)