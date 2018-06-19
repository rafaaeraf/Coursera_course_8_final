setwd("Z:/Coursera/Course_8/final_project")
library(caret)

# Parallel computing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

set.seed(1)

# Section One ---------------------------------
# Data load
originalTrainingData <- read.csv("pml-training.csv", header = TRUE,
                                 na.strings = c("", " ", "  ", "#DIV/0!"))
originalValidationData <- read.csv("pml-testing.csv", header = TRUE,
                                   na.strings = c("", " ", "  ", "#DIV/0!"))

# Section Two ---------------------------------
# Data split
trainId <- createDataPartition(originalTrainingData$X, p=0.8, list = FALSE)

trainingInput <- originalTrainingData[trainId,-grep("classe",
                                                    names(originalTrainingData))]
trainingOutput <- originalTrainingData[trainId,grep("classe",
                                                    names(originalTrainingData))]
testingInput <- originalTrainingData[-trainId,-grep("classe",
                                                   names(originalTrainingData))]
testingOutput <- originalTrainingData[-trainId,grep("classe",
                                                   names(originalTrainingData))]

# Section Three ---------------------------------
# Inputs Selection
# Transform data in numeric format
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

# Section Three ---------------------------------
# Data Normalization
preprocessCenter <- preProcess(trainingInput, method = "center")
trainingInput <- predict(preprocessCenter, trainingInput)
testingInput <- predict(preprocessCenter, testingInput)

preprocessScale <- preProcess(trainingInput, method = "scale")
trainingInput <- predict(preprocessScale, trainingInput)
testingInput <- predict(preprocessScale, testingInput)

# Section Four ---------------------------------
# Run Random Forest
startTime <- Sys.time()
print (paste("Starting training in:", startTime))

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

rfFit <- train(x = trainingInput, y = trainingOutput, method = "rf",
               trControl = fitControl)

endTime <- Sys.time()
print (paste("Ending training in:", endTime))
print (paste("Elapsed time:", endTime - startTime))

# Section Five ---------------------------------
# Testing
testingInput <- testingInput[, names(trainingInput)]
predictedTrainOutput <- predict(rfFit, trainingInput)
predictedTestOutput <- predict(rfFit, testingInput)

postResample(predictedTrainOutput, trainingOutput)
postResample(predictedTestOutput, testingOutput)

# Section Five ---------------------------------
# No labeled data prediction
validationInput <- originalValidationData[, -grep("problem_id",
                                                  names(originalValidationData))]
validationInput <- validationInput[, names(trainingInput)]
validationInput <- predict(preprocessCenter, validationInput)
validationInput <- predict(preprocessScale, validationInput)

predictedValidationOutput <- predict(rfFit, validationInput)