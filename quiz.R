install.packages("caret")
install.packages("kernlab")

library(caret);library(kernlab); data(spam); library(tidyverse)
inTrain <- createDataPartition(y = spam$type,
                               p = 0.75,
                               list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
training
summary(training)
glimpse(training)
glimpse(testing)
M <- abs(cor(training[,-58]))
M
diag(M) <- 0
which(M > 0.8, arr.ind=T)
plot(spam[,34], spam[,32])

install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
data(AlzheimerDisease)


adData = data.frame(diagnosis,predictors)
train = createDataPartition(diagnosis, p = 0.50,list=FALSE)
test = createDataPartition(diagnosis, p = 0.50,list=FALSE)

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]



library(AppliedPredictiveModeling)
library(caret)
library(Hmisc)
library(ggplot2)

# Load and split dataset
data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

# Add index variable to training data
training$Index <- 1:nrow(training)

# Identify all predictor names (exclude outcome and index)
predictors <- setdiff(names(training), c("CompressiveStrength", "Index"))

# Plot CompressiveStrength vs Index colored by each predictor
for (var in predictors) {
        # Convert continuous predictor into factor categories (4 groups)
        training[[paste0(var, "_cut")]] <- cut2(training[[var]], g = 4)
        
        p <- ggplot(training, aes(x = Index, y = CompressiveStrength, color = get(paste0(var, "_cut")))) +
                geom_point(alpha = 0.7, size = 2) +
                labs(
                        title = paste("CompressiveStrength vs. Index (Colored by", var, ")"),
                        x = "Sample Index",
                        y = "Compressive Strength",
                        color = var
                ) +
                theme_minimal()
        
        print(p)
}


# Convert continuous predictors to 4 quantile categories
cut_covariates <- apply(training[, predictors], 2, function(x) cut2(x, g = 4))

# Plot feature matrix
featurePlot(
        x = training$Index,
        y = training$CompressiveStrength,
        plot = "scatter",
        group = cut_covariates[, "Age"] # Change "Age" to any predictor variable name
)


#No 3.

library(AppliedPredictiveModeling)
library(caret)
library(ggplot2)
library(gridExtra)

# Load data and split
data(concrete)
set.seed(1000)
inTrain <- createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training <- mixtures[inTrain, ]

# 1. Inspect zeros count vs non-zeros
cat("Total rows:", nrow(training), "\n")
cat("Zero values count:", sum(training$Superplasticizer == 0), "\n")

# 2. Original Histogram
p1 <- ggplot(training, aes(x = Superplasticizer)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Original Superplasticizer", x = "Superplasticizer", y = "Count") +
  theme_minimal()

# 3. Log(x + 1) Transformed Histogram
p2 <- ggplot(training, aes(x = log(Superplasticizer + 1))) +
  geom_histogram(bins = 30, fill = "darkorange", color = "black") +
  labs(title = "Log(Superplasticizer + 1)", x = "log(Superplasticizer + 1)", y = "Count") +
  theme_minimal()

# Display side-by-side
grid.arrange(p1, p2, ncol = 2)

#No4
library(caret)
library(AppliedPredictiveModeling)

# Set seed and load data
set.seed(3433)
data(AlzheimerDisease)
adData <- data.frame(diagnosis, predictors)

# Split into train/test
inTrain <- createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training <- adData[inTrain, ]
testing  <- adData[-inTrain, ]

# 1. Identify all predictors starting with "IL"
il_cols <- grep("^IL", names(training), value = TRUE)

# 2. Extract subset of IL predictors
il_training <- training[, il_cols]

# 3. Perform PCA targeting 90% variance threshold
pca_proc <- preProcess(il_training, method = "pca", thresh = 0.90)

# 4. Output the number of principal components retained
pca_proc$numComp


#no5
library(caret)
library(AppliedPredictiveModeling)

# 1. Load data and set seed exactly as specified
set.seed(3433)
data(AlzheimerDisease)
adData <- data.frame(diagnosis, predictors)

inTrain <- createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training <- adData[inTrain, ]
testing  <- adData[-inTrain, ]

# 2. Extract predictors starting with "IL" plus outcome variable
il_cols <- grep("^IL", names(training), value = TRUE)
sub_train <- training[, c("diagnosis", il_cols)]
sub_test  <- testing[, c("diagnosis", il_cols)]

# --- Model 1: Non-PCA Logistic Regression ---
set.seed(3433)
model_non_pca <- train(
        diagnosis ~ ., 
        data = sub_train, 
        method = "glm"
)
pred_non_pca <- predict(model_non_pca, newdata = sub_test)
cm_non_pca <- confusionMatrix(pred_non_pca, sub_test$diagnosis)
acc_non_pca <- cm_non_pca$overall["Accuracy"]

# --- Model 2: PCA Logistic Regression (80% variance threshold) ---
set.seed(3433)
model_pca <- train(
        diagnosis ~ ., 
        data = sub_train, 
        method = "glm",
        preProcess = "pca",
        trControl = trainControl(preProcOptions = list(thresh = 0.80))
)
pred_pca <- predict(model_pca, newdata = sub_test)
cm_pca <- confusionMatrix(pred_pca, sub_test$diagnosis)
acc_pca <- cm_pca$overall["Accuracy"]

# --- Display Results ---
cat(sprintf("Non-PCA Accuracy: %.2f\n", round(acc_non_pca, 2)))
cat(sprintf("PCA Accuracy: %.2f\n", round(acc_pca, 2)))


Quiz 3
library(AppliedPredictiveModeling)
library(caret)
library(rpart)
library(rpart.plot)

data(segmentationOriginal)

# 1. Subset the data to a training set and testing set based on the Case variable
inTrain <- segmentationOriginal$Case == "Train"
training <- segmentationOriginal[inTrain, ]
testing <- segmentationOriginal[!inTrain, ]

# 2. Set seed to 125 and fit a CART model with default caret settings
set.seed(125)
modFit <- train(Class ~ ., data = training, method = "rpart")

# Print the final model decision tree structure
print(modFit$finalModel)

# Visualize the decision tree
rpart.plot(modFit$finalModel)


#2
library(pgmm)
library(tree)

# Load and prepare data
data(olive)
olive <- olive[, -1] # Remove Area variable's higher-level grouping column (Region)

# Fit decision tree (Area is numeric here, so tree() treats it as a regression tree)
tree_model <- tree(Area ~ ., data = olive)

# Create newdata using the column means of the dataset
newdata <- as.data.frame(t(colMeans(olive)))

# Predict the Area for newdata
pred <- predict(tree_model, newdata)
print(pred)

#4

library(bestglm)
library(caret)

# Load SAheart from bestglm
data(SAheart)

# Split data into training and testing sets
set.seed(8484)
train <- sample(1:dim(SAheart)[1], size = dim(SAheart)[1] / 2, replace = FALSE)
trainSA <- SAheart[train, ]
testSA  <- SAheart[-train, ]

# Misclassification helper function
missClass <- function(values, prediction) {
        sum(((prediction > 0.5) * 1) != values) / length(values)
}

# Standard glm approach (avoids caret's factor warning)
set.seed(13234)
glm_model <- glm(
        chd ~ age + alcohol + obesity + tobacco + typea + ldl,
        data = trainSA,
        family = "binomial"
)

# Predict probabilities on the response scale
train_pred <- predict(glm_model, newdata = trainSA, type = "response")
test_pred  <- predict(glm_model, newdata = testSA,  type = "response")

# Calculate misclassification rates
cat("Training Set Misclassification:", missClass(trainSA$chd, train_pred), "\n")
cat("Test Set Misclassification:",     missClass(testSA$chd, test_pred), "\n")

#5

library(randomForest)
library(caret)

# Load datasets directly from GitHub sources
vowel.train <- read.csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/vowel.train")
vowel.test  <- read.csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/vowel.test")

# If the first column was imported as row indices (e.g. named 'X' or row numbers), drop it:
if ("X" %in% names(vowel.train)) vowel.train$X <- NULL
if ("X" %in% names(vowel.test))  vowel.test$X  <- NULL

# 1. Convert y to a factor variable in both datasets
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y  <- as.factor(vowel.test$y)

# 2. Set seed to 33833 and fit the randomForest model
set.seed(33833)
rf_model <- randomForest(y ~ ., data = vowel.train)

# 3. Calculate variable importance using caret's varImp function
var_importance <- varImp(rf_model)

# Order variables by importance descending
var_importance[order(-var_importance$Overall), , drop = FALSE]


#Quiz 4
#1
library(caret)
library(gbm)
library(randomForest)

# Load data directly from GitHub mirror (bypassing missing ElemStatLearn package)
vowel.train <- read.csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/vowel.train")
vowel.test  <- read.csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/vowel.test")

# Clean up row index column if present
if ("X" %in% names(vowel.train)) vowel.train$X <- NULL
if ("X" %in% names(vowel.test))  vowel.test$X  <- NULL

# 1. Convert y to a factor variable in both datasets
vowel.train$y <- factor(vowel.train$y)
vowel.test$y  <- factor(vowel.test$y)

# 2. Set seed to 33833
set.seed(33833)

# Fit (1) Random Forest model via caret
mod_rf <- train(y ~ ., data = vowel.train, method = "rf")

# Fit (2) Gradient Boosting (GBM) model via caret (verbose = FALSE suppresses output)
mod_gbm <- train(y ~ ., data = vowel.train, method = "gbm", verbose = FALSE)

# Predict on the test set
pred_rf  <- predict(mod_rf, newdata = vowel.test)
pred_gbm <- predict(mod_gbm, newdata = vowel.test)

# Calculate individual model accuracies
acc_rf  <- confusionMatrix(pred_rf, vowel.test$y)$overall['Accuracy']
acc_gbm <- confusionMatrix(pred_gbm, vowel.test$y)$overall['Accuracy']

# Calculate accuracy strictly on samples where BOTH models agreed
agreed_indices <- (pred_rf == pred_gbm)
acc_agreed     <- confusionMatrix(pred_rf[agreed_indices], vowel.test$y[agreed_indices])$overall['Accuracy']

# Print results
cat("RF Accuracy =", round(acc_rf, 4), "\n")
cat("GBM Accuracy =", round(acc_gbm, 4), "\n")
cat("Agreement Accuracy =", round(acc_agreed, 4), "\n")


#### 2
library(caret)
library(gbm)
library(AppliedPredictiveModeling)
library(randomForest)

# Load data and set initial partition
set.seed(3433)
data(AlzheimerDisease)
adData <- data.frame(diagnosis, predictors)

inTrain  <- createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training <- adData[inTrain, ]
testing  <- adData[-inTrain, ]

# 1. Set seed to 62433 and train individual models
set.seed(62433)
mod_rf  <- train(diagnosis ~ ., data = training, method = "rf")
mod_gbm <- train(diagnosis ~ ., data = training, method = "gbm", verbose = FALSE)
mod_lda <- train(diagnosis ~ ., data = training, method = "lda")

# Predict on testing set with individual models
pred_rf  <- predict(mod_rf, newdata = testing)
pred_gbm <- predict(mod_gbm, newdata = testing)
pred_lda <- predict(mod_lda, newdata = testing)

# Calculate individual accuracies
acc_rf  <- confusionMatrix(pred_rf, testing$diagnosis)$overall['Accuracy']
acc_gbm <- confusionMatrix(pred_gbm, testing$diagnosis)$overall['Accuracy']
acc_lda <- confusionMatrix(pred_lda, testing$diagnosis)$overall['Accuracy']

# 2. Fit stacked model on the training predictions
# Get training set predictions from all base models
pred_rf_train  <- predict(mod_rf, newdata = training)
pred_gbm_train <- predict(mod_gbm, newdata = training)
pred_lda_train <- predict(mod_lda, newdata = training)

# Build stack training data frame
pred_df_train <- data.frame(
        pred_rf = pred_rf_train, 
        pred_gbm = pred_gbm_train, 
        pred_lda = pred_lda_train, 
        diagnosis = training$diagnosis
)

# Train stacked model with random forest
mod_stack <- train(diagnosis ~ ., data = pred_df_train, method = "rf")

# 3. Evaluate stacked model on the test predictions
pred_df_test <- data.frame(
        pred_rf = pred_rf, 
        pred_gbm = pred_gbm, 
        pred_lda = pred_lda
)

pred_stack <- predict(mod_stack, newdata = pred_df_test)
acc_stack  <- confusionMatrix(pred_stack, testing$diagnosis)$overall['Accuracy']

# Print all accuracy scores
cat("RF Accuracy:", round(acc_rf, 2), "\n")
cat("GBM Accuracy:", round(acc_gbm, 2), "\n")
cat("LDA Accuracy:", round(acc_lda, 2), "\n")
cat("Stacked Accuracy:", round(acc_stack, 2), "\n")


####################3
library(AppliedPredictiveModeling)
library(caret)
library(elasticnet)


# Load data and set partition
set.seed(3523)
data(concrete)

inTrain  <- createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training <- concrete[inTrain, ]
testing  <- concrete[-inTrain, ]

# Set seed to 233 and fit the LASSO model
set.seed(233)
mod_lasso <- train(
        CompressiveStrength ~ ., 
        data = training, 
        method = "lasso"
)

# Plot the coefficient paths for the final enet model
plot.enet(mod_lasso$finalModel, xvar = "penalty", use.color = TRUE)

#######4
library(lubridate)
library(forecast)

# 1. Create data folder if it doesn't already exist
if (!dir.exists("data")) {
        dir.create("data")
}

# 2. Download data file if not already downloaded
file_path <- "data/gaData.csv"
if (!file.exists(file_path)) {
        download.file(
                url = "https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv",
                destfile = file_path,
                mode = "wb"
        )
}

# 3. Read dataset
dat <- read.csv(file_path)

# Convert date column to Date class for lubridate functions
dat$date <- ymd(dat$date)

# 4. Split training and testing sets
training <- dat[year(dat$date) < 2012, ]
testing  <- dat[year(dat$date) > 2011, ]

# Create time series object for training set
tstrain <- ts(training$visitsTumblr)

# 5. Fit BATS model on training data
bats_model <- bats(tstrain)

# 6. Forecast for the length of the test period with 95% confidence level
h_periods <- nrow(testing)
bats_forecast <- forecast(bats_model, level = 95, h = h_periods)

# 7. Check how many test points fall within lower and upper 95% bounds
lower_bound <- bats_forecast$lower
upper_bound <- bats_forecast$upper
actuals     <- testing$visitsTumblr

in_bounds <- (actuals >= lower_bound) & (actuals <= upper_bound)
percentage <- mean(in_bounds) * 100

cat("Percentage of test points in 95% interval:", round(percentage, 2), "%\n")



#####5
library(AppliedPredictiveModeling)
library(caret)
library(e1071)

# 1. Load data and create partition
set.seed(3523)
data(concrete)

inTrain  <- createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training <- concrete[inTrain, ]
testing  <- concrete[-inTrain, ]

# 2. Set seed to 325 and fit SVM model using e1071 default settings
set.seed(325)
svm_model <- svm(CompressiveStrength ~ ., data = training)

# 3. Predict on testing set
svm_pred <- predict(svm_model, newdata = testing)

# 4. Calculate RMSE
rmse_val <- sqrt(mean((svm_pred - testing$CompressiveStrength)^2))
cat("Test Set RMSE:", round(rmse_val, 2), "\n")

