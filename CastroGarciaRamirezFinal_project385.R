#############################################################################
#Setting work directory and reading in data files
#############################################################################
#Castro,Brian
#Garcia, Isay
#Ramirez, Joshua
#setwd("/Users/Icee/Documents/Stat 385/Project")
setwd('V:/STAT 385')
Dfirst=read.csv(file="Dfirst.csv") 
Dsecond=read.csv(file="Dsecond.csv")
Dthird=read.csv(file="Dthird.csv")

#############################################################################
#Test
#############################################################################

#Training and Testing data sets
Train<- Dfirst[1:4000, ]
Test <- Dfirst[4001:5893, ]

Train$target <- as.numeric(as.factor(Train$target))
Test$target <- as.numeric(as.factor(Test$target))
Train <- Train[,-c(1)]
Test <- Test[,-c(1)]

library(olsrr)

#Regression model using target as the dependent variable and the 12 features as independent variables
regressionmodel <- lm(Train$target ~ Train$android.sensor.accelerometer.mean 
                      + Train$android.sensor.accelerometer.min
                      + Train$android.sensor.accelerometer.max
                      + Train$android.sensor.accelerometer.std
                      + Train$android.sensor.gyroscope.mean
                      + Train$android.sensor.gyroscope.min
                      + Train$android.sensor.gyroscope.max
                      + Train$android.sensor.gyroscope.std
                      + Train$sound.mean
                      + Train$sound.min
                      + Train$sound.max
                      + Train$sound.std)

#regressionmodel <- lm(Train$target~.,data=Train)
#?lm
#regressionmodelTest <- lm(Train$target ~ .)

#Forward regression using p-values
ForwardFit <- ols_step_forward_p(regressionmodel, penter = 0.05, details = TRUE) 
ForwardFit

#Backward regression using p-values
BackwardFit <- ols_step_backward_p(regressionmodel, prem = 0.05, details = TRUE) 
BackwardFit

#Stepwise regression using p-values
Stepwisefit <- ols_step_both_p(regressionmodel, pent = 0.05, prem = 0.05)
Stepwisefit

#Subset regression for model comparison
Modelcompare <- ols_step_best_subset(regressionmodel)
Modelcompare

####################################################
#Graphs- Ranking Feature Variables
####################################################

library(tidyverse)
varImportance <- data.frame(Variables = ForwardFit$predictors, 
                            Importance = ForwardFit$aic)
View(varImportance)
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#graphing the ranks
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables',title = 'Foward Fit AIC') +
  coord_flip()

########################################################################
#Creates a new data frame that contains all of the feature variables as well as their binary classification
NewDfirstTrain <- Train[,-c(1,7,9,11)]
NewDfirstTest <- Test[,-c(1,7,9,11)]

library(rpart)
library(rpart.plot)
treefit <- rpart(NewDfirstTrain$target~., data = NewDfirstTrain, method = "class", 
                 parms = list(split = "information"), 
                 control = rpart.control(minsplit = 10, cp = 0.01))
printcp(treefit)
treefit$split
#Information Tree
prp(treefit,type = 2, extra = 1)
plotcp(treefit)

#Grow the tree to its entirety by setting the minimum number of sample data points in each node to be 2 and 
#cost-complexity to be 0.
treefit1 <- rpart(as.factor(NewDfirstTrain$target)~., data = NewDfirstTrain, 
                  control = rpart.control(minsplit = 2, minbucket = 1, cp=0))

#Post-pruning the tree so that the cross-validation error rate is minimized.
bestcp <- treefit1$cptable[which.min(treefit1$cptable[,"xerror"]),"CP"]
tree.pruned <- prune(treefit1, cp = bestcp)
prp(tree.pruned, type = 2, extra = 1)
printcp(tree.pruned)
plotcp(tree.pruned)

#Predicting results
predvalsTree = predict(tree.pruned, newdata = NewDfirstTest, type = "class") 
truthTree = NewDfirstTest$target
results <- table(truthTree, predvalsTree)
results
#.76 accuracy

###############################################################################
#Third data set with forward, backward, stepwise selection
###############################################################################

#Training and Testing data sets
Train<- Dthird[1:4000, ]
Test <- Dthird[4001:5893, ]

Train$target <- as.numeric(as.factor(Train$target))
Test$target <- as.numeric(as.factor(Test$target))
Train <- Train[,-c(1)]
Test <- Test[,-c(1)]

#Regression model
regressionmodel <- lm(Train$target~.,data=Train)

#Forward regression using p-values
ForwardFit <- ols_step_forward_p(regressionmodel, penter = 0.05, details = TRUE) 
ForwardFit
#Backward regression using p-values
BackwardFit <- ols_step_backward_p(regressionmodel, prem = 0.05, details = TRUE) 
BackwardFit

#Stepwise regression using p-values
Stepwisefit <- ols_step_both_p(regressionmodel, pent = 0.05, prem = 0.05)
Stepwisefit

#Subset regression for model comparison (dataset to big. Will not run)
#Modelcompare <- ols_step_best_subset(regressionmodel)
#Modelcompare

####################################################################
#Forward AIC Ranking 
####################################################################
varImportance <- data.frame(Variables = ForwardFit$predictors, 
                            Importance = ForwardFit$aic)
View(varImportance)
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
#graphing the ranks
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables',title = 'Foward Fit AIC') +
  coord_flip()

#################################################################
#Creates a new data frame that contains all of the feature variables as well as their binary classification
#################################################################

NewDthirdTrain <- Train[,-c(10,12,14:17,20,21,25,29:31,35)]
NewDthirdTest <- Test[,-c(10,12,14:17,20,21,25,29:31,35)]

library(rpart)
library(rpart.plot)
treefit <- rpart(NewDthirdTrain$target~., data = NewDthirdTrain, method = "class", 
                 parms = list(split = "information"), 
                 control = rpart.control(minsplit = 10, cp = 0.01))

printcp(treefit)
treefit$split

#Information Tree
prp(treefit,type = 2, extra = 1)
plotcp(treefit)

#Grow the tree to its entirety by setting the minimum number of sample data points in each node to be 2 and 
#cost-complexity to be 0.
treefit1 <- rpart(as.factor(NewDthirdTrain$target)~., data = NewDthirdTrain, 
                  control = rpart.control(minsplit = 2, minbucket = 1, cp=0))

#Post-pruning the tree so that the cross-validation error rate is minimized.
bestcp <- treefit1$cptable[which.min(treefit1$cptable[,"xerror"]),"CP"]
tree.pruned <- prune(treefit1, cp = bestcp)
prp(tree.pruned, type = 2, extra = 1)
printcp(tree.pruned)
plotcp(tree.pruned)

#Predicting results
predvalsTree = predict(tree.pruned, newdata = NewDthirdTest, type = "class") 
truthTree = NewDthirdTest$target
results <- table(truthTree, predvalsTree)
results
#.85 accuracy

######################################################################
#SVM (All vs One Approach)
######################################################################
getwd()
setwd("C:/Users/tito0/Desktop")
Dfirst=read.csv(file="Dfirst.csv")
Dsecond=read.csv(file="Dsecond.csv")
Dthird=read.csv(file="Dthird.csv")

VechOrNot <- 0
for(i in 1:5893){
  if (Dthird[i,38] == 'Still'){
    VechOrNot[i] <- 0
  }
  else if(Dthird[i,38] == 'Walking')
  {
    VechOrNot[i] <- 0
  }
  else
    VechOrNot[i] <-1
}
#c(2:3,4:5,10:21,30:32,39)]
Dthird$VechOrNot<-as.factor(VechOrNot)
data<-Dthird[,c(2:3,4:5,10:21,30:32,39)]
train<-data[1:4000,]
test<-data[4001:5893,]

library("e1071")
svmmodel=svm(VechOrNot~., data=train, kernel="radial",scale=F,cost=1300)
tobj<-tune.svm(VechOrNot~., data=train, cost=c(25:30)*50,gamma=c(.1:1))
predvals=predict(svmmodel, newdata=test) 
truth=test$VechOrNot
fpfn<-table(truth,predvals)
#     predvals
#truth    0    1
#    0  626   89
#    1   84  1094
#.91

WalkOrNot<-0
for(i in 1:5893){
  if (Dthird[i,38] == 'Walking'){
    WalkOrNot[i] <- 1
  }
  else{
    WalkOrNot[i] <- 0
  }
}

Dthird$WalkOrNot=as.factor(WalkOrNot)
train=Dthird[1:4000,]
test=Dthird[4001:5893,]
train=train[,c(2:5,10:21,39)]

library("e1071")
svmmodel=svm(WalkOrNot~., data=train, kernel="radial",scale=F,cost=1300)
tobj<-tune.svm(WalkOrNot~., data=ttest, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=test) 
truth=test$WalkOrNot
fpfn<-table(truth,predvals)
#    predvals
#truth    0    1
#    0 1488   41
#    1   66  312
#.95

Dthird$Walktrain=as.factor(Walktrain)
Dthird=Dthird[,c(2,4,5,10:12,15,16,19:21,40)]

library("e1071")
svmmodel=svm(Walktrain~., data=Dthird, kernel="radial",scale=F,cost=1300)
tobj<-tune.svm(StillOrNot~., data=train, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=NonVehicles) 
truth=NonVehicles$WalkOrNot
fpfn<-table(truth,predvals)
#    predvals
#truth    0    1
#0      311    0
#1       16  299
#.97

class(test$WalkOrNot)="Integer"
test$predvals<-as.integer(predvals)
test$correct<-test$predvals+test$WalkOrNot
TestNotWalk=test[test$correct==2,]

TestNotWalk=TestNotWalk[,-c(39:41)]
Dthird=read.csv(file="Dthird.csv")
TestNotWalk=Dthird[c(as.numeric(rownames(TestNotWalk))),]

train=Dthird[1:4000,]
train=train[which(train$target!='Walking'),]
train=train[,c(2:5,10:13,18:21,26:38)]
test=TestNotWalk
test=test[,c(2:5,10:13,18:21,26:38)]

library("e1071")
svmmodel=svm(target~., data=train,type="C-classification", kernel="radial",scale=F,cost=1250)
tobj<-tune.svm(target~., data=train, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=test) 
truth=test$target
fpfn<-table(truth,predvals)
#          predvals
#truth     Bus Car Still Train Walking
#Bus       309  24     8    25       0
#Car        27 318    15    20       0
#Still       6   6   228    25       0
#Train       9  22     5   345       0
#Walking     2   0     1     0       0
#.86

TrainOrNot <- 0
for(i in 1:5893){
  if (Dthird[i,38] == 'Train'){
    TrainOrNot[i] <- 1
  }
  else{
    TrainOrNot[i] <- 0
  }
}
Dthird$TrainOrNot<-as.factor(TrainOrNot)
Dthird<-Dthird[,-c(38)]
train<-Dthird[1:4000,]
test<-Dthird[4001:5893,]

library("e1071")
svmmodel=svm(TrainOrNot~., data=train, kernel="radial",scale=F,cost=1250)
tobj<-tune.svm(TrainOrNot~., data=train, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=test) 
truth=test$TrainOrNot
fpfn<-table(truth,predvals)
predvals
#truth 0    1
#0   1483   5
#1    200  205
#.89

BusOrNot <- 0
for(i in 1:5893){
  if (Dthird[i,38] == 'Bus'){
    BusOrNot[i] <- 1
  }
  else{
    BusOrNot[i] <- 0
  }
}
Dthird$BusOrNot<-as.factor(BusOrNot)
Dthird<-Dthird[,-c(38)]
train<-Dthird[1:4000,]
test<-Dthird[4001:5893,]

library("e1071")
svmmodel=svm(BusOrNot~., data=train, kernel="radial",scale=F,cost=1250)
tobj<-tune.svm(TrainOrNot~., data=train, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=test) 
truth=test$BusOrNot
fpfn<-table(truth,predvals)
#predvals
#truth 0    1
#  0  1508  6
#  1  214  165
#.88

CarOrNot <- 0
for(i in 1:5893){
  if (Dthird[i,38] == 'Car'){
    CarOrNot[i] <- 1
  }
  else{
    CarOrNot[i] <- 0
  }
}
Dthird$CarOrNot<-as.factor(CarOrNot)
Car<-Dthird[,-c(38)]
# Testing data and training data
train<-Car[1:4000,]
test <- Car[4001:5893,]

library("e1071")
svmmodel=svm(CarOrNot~., data=train, kernel="radial",scale=F,cost=1250)
tobj<-tune.svm(CarOrNot~., data=train, cost=c(25:30)*50)
predvals=predict(svmmodel, newdata=test) 
truth=test$CarOrNot
fpfn<-table(truth,predvals)
#     predvals
#truth  0   1
#  0  1499  0
#  1  256  138
#.86

######################################################################
#SVM(1vs1 Approach)
######################################################################

class(test$VechOrNot)="Integer"
test$predvals<-as.integer(predvals)
test$correct<-test$predvals+test$VechOrNot
TestNonVehicles=test[test$correct==2,]
TestVehicles=test[test$correct==4,]

TestNonVehicles=TestNonVehicles[,-c(20,21,22)]
TestVehicles=test[,-c(20,21,22)]

Dthird=read.csv(file="Dthird.csv")
NonVehicles=Dthird[c(as.numeric(rownames(TestNonVehicles))),-c(39)]
Vehicles=Dthird[c(as.numeric(rownames(TestVehicles))),]

Dthird=read.csv(file="Dthird.csv")
TrainVehicles<-Dthird[1:4000,]
TrainVehicles=TrainVehicles[Dthird$target!='Walking'&Dthird$target!='Still',]
TrainVehicles=TrainVehicles[,c(2:5,10:21,34:38)]
TestVehicles<-Dthird[c(as.numeric(rownames(TestVehicles))),]
test<-TestVehicles[,-c(38)]

library("e1071")
svmmodel=svm(target~., data=TrainVehicles,method="C-classification", kernel="radial",scale=F,cost=1250)
predvals=predict(svmmodel, newdata=test) 
truth=TestVehicles$target
fpfn<-table(truth,predvals)

#SVM
library("e1071")
for(i in 1:4000){
  if(NewDthirdTrain[i,24] == 1){
    NewDthirdTrain$target[i] <- "Bus"
  }
  else if(NewDthirdTrain[i,24] == 2){
    NewDthirdTrain$target[i] <- "Car"
  }
  else if(NewDthirdTrain[i,24] == 3){
    NewDthirdTrain$target[i] <- "Still"
  }
  else if(NewDthirdTrain[i,24] == 4){
    NewDthirdTrain$target[i] <- "Train"
  }
  else{
    NewDthirdTrain$target[i] <- "Walking"
  }
}

for(i in 1:1893){
  if (NewDthirdTest[i,24] == 1){
    NewDthirdTest$target[i] <- "Bus"
  }
  else if (NewDthirdTest[i,24] == 2){
    NewDthirdTest$target[i] <- "Car"
  }
  else if (NewDthirdTest[i,24] == 3){
    NewDthirdTest$target[i] <- "Still"
  }
  else if(NewDthirdTest[i,24] == 4){
    NewDthirdTest$target[i] <- "Train"
  }
  else{
    NewDthirdTest$target[i] <- "Walking"
  }
}

NewDthirdTrain$target <- as.factor(NewDthirdTrain$target)
NewDthirdTest$target <- as.factor(NewDthirdTest$target)

svmmodel=svm(NewDthirdTrain$target~., data=NewDthirdTrain, kernel="radial",scale=F,cost=100,
             type = "C-classification")
tobj=tune.svm(target~., data=NewDthirdTrain, cost=10^c(-2:4))
bestmagnitude=tobj$best.parameters
tobj1=tune.svm(target~., data=NewDthirdTrain, cost=1*c(1:20)/20)
bestpara=tobj1$best.parameters
cv_svmmodel=svm(NewDthirdTrain$target~., data=NewDthirdTrain, kernel="radial",
                scale=F,
                cost=1001,
                type = "C-classification")

prevals <- predict(cv_svmmodel,newdata = NewDthirdTest)
truth <- NewDthirdTest$target
View(fpfn)
fpfn <- table(truth,prevals)
fpfn 

######################################################################
#Implementing Random Forest
######################################################################
train=Dthird[1:4000,]
test=Dthird[4001:5893,]

install.packages('randomForest')
install.packages('caTools')
library(randomForest)
require(caTools)

rf<-randomForest(target~.,data=train)
rf
importance(rf)
importance<-importance(rf)
varImportance<-data.frame(Variables = row.names(importance), 
                          Importance = round(importance[ ,'MeanDecreaseGini'],2))
View(varImportance)
#importance of features 
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#graphing the ranks
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

predvals<-predict(rf, newdata=test)
truth=test$target
fpfn<-table(truth,predvals)
#truth     Bus Car Still Train Walking
#Bus       368   1     1     3       6
#Car        14 360     8     5       7
#Still       2   0   329     9       5
#Train       5   1    12   386       1
#Walking     5   2     3     5     355
#.95






