#Citation for Using Database.
#[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014


library(dataMaid)
library(DataExplorer)
library(janitor)
library(lubridate)
library(Hmisc)
library(corrplot)
library(fastDummies)
library(ggplot2)
library(DMwR)
library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(e1071)
library(ROCR)


#library(smotefamily)
df <- read.csv("bank-full.csv")
set.seed(123)
summary(df)
head(df)
dim(df)
#45211 x 17

length(unique((df$job)))
#12

table(df$job)

table(df$marital)
#3

table(df$default)
# no yes

table(df$housing)
#no yes

table(df$loan)
#no yes

table(df$contact)
# cellular,telephone,unknown 

table(df$poutcome)
#failure,other,success,unknown

table(df$y)
#No-39922 y-5289

length(unique(df$day))
#31

length(unique(df$month))
#12

table(df$marital)
#divorced  married   single 
#5207    27214    12790 

table(df$education)
# primary secondary  tertiary   unknown 
#6851     23202     13301      1857 

table(df$previous)

#Quick Report

create_report(df)

makeDataReport(df, replace = TRUE)


  #Converting categorical data to factors for quick EDA-------------------------

bank_factors <- bank.full

mydata[,names] <- lapply(mydata[,names] , factor)



#['job','marital','education','default','housing','loan','poutcome']

bank_factors$job <- as.factor(bank_factors$job)
bank_factors$marital <- as.factor(bank_factors$marital)
bank_factors$education <- as.factor(bank_factors$education)
bank_factors$default <- as.factor(bank_factors$default)
bank_factors$housing <- as.factor(bank_factors$housing)
bank_factors$loan <- as.factor(bank_factors$loan)
bank_factors$poutcome <- as.factor(bank_factors$poutcome)
bank_factors$y <- as.factor(bank_factors$y)
bank_factors$contact <- as.factor(bank_factors$contact)
bank_factors$day <- as.factor(bank_factors$day) 
bank_factors$month <- as.factor(bank_factors$month)
bank_factors$y <- as.factor(bank_factors$y)
write.csv(bank_factors,'bank_factors.csv')

#plotting bar graphs.
ggplot(bank_factors, aes(x = marital)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = job )) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = education)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = default)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = contact)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = loan)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = poutcome)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = housing)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = y)) + geom_bar(aes(fill= y))
ggplot(bank_factors, aes(x = month)) + geom_bar(aes(fill= y))

ggplot(bank_factors, aes(x = age)) + geom_histogram(bins = 40)

table(bank_factors$y)
#--------------------------------------------------------------------------------
  
boxplot(df$age,ylab = "age")

boxplot(df$duration,ylab = "duration")
boxplot(df$balance,ylab = "balance")

summary(bank_factors$balance)
summary(bank_factors$age)
#---------------------------------
str(bank_factors)
str(training)

#Decision Tree

unique(training$y)

inTrain <- createDataPartition(y = bank_factors$y, p = 0.7, list = FALSE)

training <- df[inTrain,] # The rows selected from inTrain sampling become our training set
testing <- df[-inTrain,]

training$job <- as.factor(training$job)
training$marital <- as.factor(training$marital)
training$education <- as.factor(training$education)
training$default <- as.factor(training$default)
training$housing <- as.factor(training$housing)
training$loan <- as.factor(training$loan)
training$poutcome <- as.factor(training$poutcome)
training$y <- as.factor(training$y)
training$contact <- as.factor(training$contact)
training$day <- as.factor(training$day) 
training$month <- as.factor(training$month)
training$y <- as.factor(training$y)

training <- SMOTE( y ~ ., training, perc.over=100, perc.under = 200)
table(training$y)

tree <- rpart(y ~ ., data = training)

prp(tree, under=TRUE, type=3, varlen = 0, faclen = 0, extra = TRUE)

tree$variable.importance
printcp(tree)
print(tree)

testing$job <- as.factor(testing$job)
testing$marital <- as.factor(testing$marital)
testing$education <- as.factor(testing$education)
testing$default <- as.factor(testing$default)
testing$housing <- as.factor(testing$housing)
testing$loan <- as.factor(testing$loan)
testing$poutcome <- as.factor(testing$poutcome)
testing$y <- as.factor(testing$y)
testing$contact <- as.factor(testing$contact)
testing$day <- as.factor(testing$day) 
testing$month <- as.factor(testing$month)
testing$y <- as.factor(testing$y)

table(testing$y)

tree.pred <- predict(tree, testing, type="class")

confusionMatrix(tree.pred, testing$y)

#Naiye Bayes



fitControl <- trainControl(method = "cv", number = 3)
tuneControl <- data.frame(fL=1, usekernel = FALSE, adjust=1)

x <- training[, 1:16]
z <- training[,17]

bayes <- train(x,y, method = "nb", trControl=fitControl, tuneGrid=tuneControl)
bayes

pred <- predict(bayes, newdata = testing)
pred
confusionMatrix(pred, testing$y)

varImp(bayes)

sample_bank <- sample_n(bank_factors, 1000)
write.csv(sample_bank,'bank_factors_sample.csv')



