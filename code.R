setwd("C:\\R Programming\\abzooba")
data<-read.csv("Training_Data.csv")
test<-read.csv("Test_Data.csv")

##Converting factor variable to numeric(by one hot encoding/dummy variables)
library(dummies)
library(ROSE)


dummyVarData<-dummy.data.frame(data[,-11])
dummyVarData$Adherence<-data$Adherence


##Data is not balanced
##So training on a balanced data set.(selecting equal no of outcomes)
table(data$Adherence)
under<-ovun.sample(Adherence~.,data=dummyVarData,method="under",N=108780)$data


## Exploratory Data Analysis

dummyVarData$Adherence<-as.numeric(dummyVarData$Adherence)-1

library(corrplot)
correlation<-cor((dummyVarData))
corrplot(correlation)

##Clearly patient_id has no correlation with a any variable 
##Age has positive correlation with Diabetes and HyperTension
##Diabetes has positive relation with HyperTension
##Smoke has positive correlation with Alcoholism


g<-ggplot(data,aes(x=Age,fill=Adherence))+geom_histogram(stat="bin",bins = 15)         
print(g)

##People with Age group of 10-40 seems to have highest percentage of Adherence as Yes


g<-ggplot(data,aes(x=Prescription_period,fill=Adherence))+geom_histogram(stat="bin",bins = 15)         
print(g)

##People with Prescription_period less than or equal to 30 seems to have highest percentage of Adherence as Yes


##Feature Engineering
under$isPrescription_periodLT30<-as.factor(ifelse(under$Prescription_period<=31,"Yes","No"))
under$ageGroups<-cut(under$Age,breaks = c(0,10,40,60,80,120),include.lowest = TRUE)
levels(under$ageGroups)<-c("Young","Adult","Old","VeryOld","ExtremelyOld")


test$isPrescription_periodLT30<-as.factor(ifelse(test$Prescription_period<=31,"Yes","No"))
test$ageGroups<-cut(test$Age,breaks = c(0,10,40,60,80,120),include.lowest = TRUE)
levels(test$ageGroups)<-c("Young","Adult","Old","VeryOld","ExtremelyOld")



library(dplyr)
explore<- under %>% group_by(isPrescription_periodLT30) %>% summarise(mean(as.numeric(Adherence)))

explore<- under %>% group_by(ageGroups) %>% summarise(mean(as.numeric(Adherence)))

library(caTools)
library(caret)
library(ROSE)

train<-dummy.data.frame(under[,-12])
train$Adherence<-under$Adherence
train$patient_id<-NULL

test<-dummy.data.frame(test)


cor(as.numeric(train$Adherence)-1,train[,-18])


splits<-sample.split(train$Adherence,SplitRatio = 0.8)
validate<-subset(train,splits==FALSE)
train<-subset(train,splits==TRUE)

trControl<-trainControl(method="cv",number=5)


##Training with rpart
modelrpart<-train(Adherence~.,data=train,method="rpart",trControl=trControl)
predictionrpartTrain<-predict(modelrpart,train,type="prob")
predictionrpartTrain<-predictionrpartTrain[,2]

predictionrpartValidation<-predict(modelrpart,validate,type="prob")
predictionrpartValidation<-predictionrpartValidation[,2]

predictionrpartTest<-predict(modelrpart,test,type="prob")
predictionrpartTest<-predictionrpartTest[,2]


table(predictionrpartValidation>0.5,validate$Adherence)
##Accuracy of ~89.5%

library(party)
##Training with ctree
modelctree<-train(Adherence~.,data=train,method="ctree",trControl=trControl)
predictionctreeTrain<-predict(modelctree,train,type="prob")
predictionctreeTrain<-predictionctreeTrain[,2]

predictionctreeValidation<-predict(modelctree,validate,type="prob")
predictionctreeValidation<-predictionctreeValidation[,2]

predictionctreeTest<-predict(modelctree,test,type="prob")
predictionctreeTest<-predictionctreeTest[,2]

table(predictionctreeValidation>0.5,validate$Adherence)


##Training using gbm
modelgbm<-train(Adherence~.,data=train,method="gbm",trControl=trControl)
predictiongbmTrain<-predict(modelgbm,train,type="prob")
predictiongbmTrain<-predictiongbmTrain[,2]

predictiongbmValidation<-predict(modelgbm,validate,type="prob")
predictiongbmValidation<-predictiongbmValidation[,2]

predictiongbmTest<-predict(modelgbm,test,type="prob")
predictiongbmTest<-predictiongbmTest[,2]

table(predictiongbmValidation>0.5,validate$Adherence)


## Training using bayesian model
modelbayes<-train(Adherence~.,data=train,method="bayesglm",trControl=trControl)
predictionbayesTrain<-predict(modelbayes,train,type="prob")
predictionbayesTrain<-predictionbayesTrain[,2]

predictionbayesValidation<-predict(modelbayes,validate,type="prob")
predictionbayesValidation<-predictionbayesValidation[,2]

predictionbayesTest<-predict(modelbayes,test,type="prob")
predictionbayesTest<-predictionbayesTest[,2]

table(predictionbayesValidation>0.5,validate$Adherence)




##Training using XgBoost
library("xgboost")
library("Matrix")
data_variables <- as.matrix(train[,-18])
data_label <- as.numeric(train[,"Adherence"])-1
data_matrix <- xgb.DMatrix(data = data_variables, label = data_label)


xgb_params <- list(booster = "gbtree", objective = "binary:logistic",
                   eta=0.1, max_depth=10)
xgbcv <- xgb.cv( params = xgb_params, data = data_matrix, nrounds = 100, nfold = 10, showsd = T, 
                 stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)


nround    <- xgbcv$best_iteration # number of XGBoost rounds
cv.nfold  <- 10

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
bst_model <- xgb.train(params = xgb_params,
                       data = data_matrix,
                       nrounds = nround)

predictionsXGBoostTrain<-predict(bst_model,data_variables)

test_matrix<-xgb.DMatrix(data = as.matrix(validate[,-18]))
predictionsXGBoostValidation<-predict(bst_model,test_matrix)
table(predictionsXGBoostValidation>0.5,validate$Adherence)

patient_id<-test$patient_id
test$patient_id<-NULL
predictionsXGBoostTest<-predict(bst_model,as.matrix(test))


kmeans<-kmeans(train[,-18],centers = 2)
cluster<-kmeans$cluster

library(flexclust)
kcca<-kcca(train[,-18],k=2)
clusterValidate<-predict(kcca,validate[,-18])
clusterTest<-predict(kcca,test)

stackedtrain<-data.frame(rpart=predictionrpartTrain,ctree=predictionctreeTrain,gbm=predictiongbmTrain,xgboost=predictionsXGBoostTrain,bayes=predictionbayesTrain,group=cluster-1,target=train$Adherence)
stackedvalidation<-data.frame(rpart=predictionrpartValidation,ctree=predictionctreeValidation,gbm=predictiongbmValidation,xgboost=predictionsXGBoostValidation,bayes=predictionbayesValidation,group=clusterValidate-1,target=validate$Adherence)
stackedtest<-data.frame(rpart=predictionrpartTest,ctree=predictionctreeTest,gbm=predictiongbmTest,xgboost=predictionsXGBoostTest,bayes=predictionbayesTest,group=clusterTest-1)


modelnnet<-train(target~.,data=stackedtrain,method="nnet",trControl=trControl)
predictionsValidate<-predict(modelnnet,stackedvalidation,type="prob")
predictionsValidate<-predictionsValidate[,2]
##Accuracy of ~90.8 on validation set

predictionsTest<-predict(modelnnet,stackedtest,type="prob")
predictionsTest<-predictionsTest[,2]
adherence<-ifelse(predictionsTest>0.5,"Yes","No")

df<-data.frame(patient_id=patient_id,adherence=adherence,probability=predictionsTest)
write.csv(df,"output.csv",row.names = FALSE)