# Coursera  - Practical Machine Learning

## Course Project

### Introduction

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how *much* of a particular activity they do, but they rarely quantify *how well they do it*. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>. 



### Code Implementing

#### Data import

The first step of the implementation was to import the data from CSV. One important lesson was to transform empty values ("")  and Excel errors ("#DIV/0!") as NA.

```R
originalTrainingData <- read.csv("pml-training.csv", header = TRUE,
                                 na.strings = c("", " ", "  ", "#DIV/0!"))
originalValidationData <- read.csv("pml-testing.csv", header = TRUE,
                                   na.strings = c("", " ", "  ", "#DIV/0!"))
```

#### Data Split

The labeled data was divided into 80% for training and 20% for testing. The data was also divided into input and output.

```R
trainId <- createDataPartition(originalTrainingData$X, p=0.8, list = FALSE)

trainingInput <- originalTrainingData[trainId,-grep("classe",
                                                    names(originalTrainingData))]
trainingOutput <- originalTrainingData[trainId,grep("classe",
                                                    names(originalTrainingData))]
testingInput <- originalTrainingData[-trainId,-grep("classe",
                                                   names(originalTrainingData))]
testingOutput <- originalTrainingData[-trainId,grep("classe",
                                                   names(originalTrainingData))]
```

#### Inputs Selection

There are 159 inputs in the original dataset. To select the most relevant inputs, six steps were made. Only the training set was used to calculate these steps:

- Remove columns with no repetitions in its values. The goal here is to remove columns like ID.
- Remove columns by name analysis. Some columns are not relevant, i.e. the name of the users, the time stamp.
- Some columns have a very small set of data. Columns with more than 95% of NA were removed.
- Columns with near zero variance were removed.
- Columns with high correlation between each other were removed.
- Columns with low correlation to the output (0.01) were removed.

After these actions, the number of selected inputs was 39.

````R
trainingInput <- data.frame(sapply(trainingInput, as.numeric))

# Manual removal of all different values columns (ids removal). FALSE are removed
removeColumns1 <- (!sapply(trainingInput, function(x) {sum(duplicated(x))})==0)

# Manual removal of non relevant inputs based on its name. FALSE are removed
removeColumns1[c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
                "cvtd_timestamp", "new_window", "num_window")] <- FALSE

# Remove Columns that have more than 95% of NA data
removeColumns1 <- removeColumns1 & (!colSums(is.na(trainingInput))/
                                        nrow(trainingInput)>0.95)

# Apply removal
trainingInput <- trainingInput[, removeColumns1]

# Near zero variance to remove inputs
preprocessNzv <- preProcess(x = trainingInput, method = "nzv")
trainingInput <- predict(preprocessNzv, trainingInput)

# High correlated inputs removal
preprocessCorr <- preProcess(x = trainingInput, method = "corr")
trainingInput <- predict(preprocessCorr, trainingInput)

# Remove inputs with low correlation between inputs and the output (less than 0.01)
corrOutput <- data.frame(matrix(ncol = ncol(trainingInput), nrow = 0))
names(corrOutput) <- names(trainingInput)
for (i in seq(ncol(trainingInput))){
    corrOutput[1,i] <- abs(cor(trainingInput[,i],as.numeric(trainingOutput)))
}
removeColumns2 <- corrOutput>0.01
trainingInput <- trainingInput[, removeColumns2]
````

#### Data normalization

Two methods of normalization were used:

- Center - removes the mean of each columns.
- Scale - divides the data by its standard deviation.

Only the training set was used for this normalization.

````R
preprocessCenter <- preProcess(trainingInput, method = "center")
trainingInput <- predict(preprocessCenter, trainingInput)
testingInput <- predict(preprocessCenter, testingInput)

preprocessScale <- preProcess(trainingInput, method = "scale")
trainingInput <- predict(preprocessScale, trainingInput)
testingInput <- predict(preprocessScale, testingInput)
````

#### Training

Random forest was the selected method for this project. The elapsed time to the test was approximately 6 minutes.

````R
startTime <- Sys.time()
print (paste("Starting training in:", startTime))

rfFit <- train(x = trainingInput, y = trainingOutput, method = "rf",
               trControl = fitControl)

endTime <- Sys.time()
print (paste("Ending training in:", endTime))
print (paste("Elapsed time:", endTime - startTime))
````

#### Testing

To analyze the results, the postResample function from caret was used. The training and testing outputs were calculated and compared with the real values. The results are presented at the Results Section. 

````R
testingInput <- testingInput[, names(trainingInput)]
predictedTrainOutput <- predict(rfFit, trainingInput)
predictedTestOutput <- predict(rfFit, testingInput)

postResample(predictedTrainOutput, trainingOutput)
postResample(predictedTestOutput, testingOutput)
````

#### Validation

To answer the Coursera Quizz, the non labeled data was used to predict its output.

````R
validationInput <- originalValidationData[, -grep("problem_id",
                                                  names(originalValidationData))]
validationInput <- validationInput[, names(trainingInput)]
validationInput <- predict(preprocessCenter, validationInput)
validationInput <- predict(preprocessScale, validationInput)

predictedValidationOutput <- predict(rfFit, validationInput)
````

#### Parallel computing

To improve the speed of the random forest training, a parallel computing method was applied. This is the implemented code.

````R
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
````

### Results

#### Train accuracy

This is the accuracy for the train data:

````R
Accuracy    Kappa 
       1        1 
````

#### Test accuracy

This is the accuracy for the train data:

````R
 Accuracy     Kappa 
0.9949032 0.9935537 
````

#### Predicted values for non labeled data

````R
B A B A A E D B A A B C B A E E A B B B
````

### Conclusion

The observed accuracy for the test data was very good. The result of the non labeled data also showed 100% right answers. 