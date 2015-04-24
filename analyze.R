#########################################################
# Data@Night #2, Princeton Data Science
# Analyze Reddit Shower Thoughts data.
# 
# Coded by echow@princeton.edu for Data@Night #2, 4/24/15.
# 
#########################################################

#########################################################
# Section 0: import packages
# 
#########################################################

set.seed(42)

# If you're new to RStudio, run the below command (commented so
# remove the "#". This installs the packages that you need
# to run various commands.)
# THE COMMAND:
# install.packages(c("ggplot2", "gridExtra", "wordcloud", "tm",
#                    "RColorBrewer", "e1071", "ngram"))

# We import the 3rd-party packages with various tools we need.
# "data.table" : a package 
lapply(c("ggplot2", "gridExtra", "wordcloud", "tm",
         "RColorBrewer", "e1071", "ngram"), require, character.only=T)

#########################################################
# Section 1: read in data, some preprocessing
#
#########################################################

df.raw <- read.csv("./processed_showerthoughts.csv", header=T)
df.raw$created_utc <- as.POSIXct(df.raw$created_utc, 
                                 origin="1970-01-01 00:00.00 UTC")

#########################################################
# Section 2: exploring the numbers
# 
# Possible questions:
# (1) What explains the shape of these graph(s)?
# (2) What explains differences between these graphs(s)?
# (3) What explains the maximums? The minimums? Do these
# graphs have the same mean/median/standard deviation?
# 
# Hit "Zoom" after this plots to get a higher resolution.
# Important note: these are histograms so parts of the
# plots do NOT necessarily refer to the same datapoints.
#########################################################

plt.score <- qplot(df.raw$score) +
  xlab("What score (higher = better) does a post have?") +
  ylab("How many posts there are") +
  ggtitle("Histogram of score for each post")
plt.ups <- qplot(df.raw$ups) +
  xlab("How many upvotes does a post have?") +
  ylab("How many posts there are") +
  ggtitle("Histogram of upvotes for each post")
plt.downs <- qplot(df.raw$downs) +
  xlab("How many downvotes does a post have?") +
  ylab("How many posts there are") +
  ggtitle("Histogram of downvotes for each post")
plt.comments <- qplot(df.raw$num_comments) +
  xlab("How many comments does a post have?") +
  ylab("How many posts there are") +
  ggtitle("Histogram of # comments for each post")

# Below: arrange the grids in a plot and visualize.
# Takes a few seconds, so comment when not using.
grid.arrange(plt.score, plt.comments, plt.ups, plt.downs, ncol=2)

#########################################################
# Section 3: exploring the words
# 
# Exploratory analysis of the most frequent words (TF-IDF)
# with a wordcloud. I recommend comparing this with 
# the original data.
# 
# Possible questions:
# (1) Why are the the most frequent words what they are?
# (2) What words would you expect to be more frequent,
# but aren't?
# (3) Can you tell, by looking at the dataset, what
# are the most frequent topics? And why, do you think?
#########################################################

corpus <- Corpus(VectorSource(df.raw$title))
corpus <- tm_map(corpus, 
  function(x) removeWords(x, stopwords("english")))
corpus.mat <- as.matrix(TermDocumentMatrix(corpus, 
  control=list(weighting=weightTfIdf)))
v <- sort(rowSums(corpus.mat), decreasing=TRUE)
df.wordfreqs <- data.frame(word=names(v), freq=v)
df.wordfreqs <- df.wordfreqs[1:300,] # truncate
pal <- brewer.pal(8, "PuBuGn")[-(1:2)]
# plot takes a while to load, so keep it commented when don't need it
# Note minimum word frequency is 2. What happens if you change it?
wordcloud(df.wordfreqs$word,df.wordfreqs$freq, 
  scale=c(3,.3), min.freq=2,
  max.words=Inf, random.order=F, rot.per=.15, colors=pal)

#########################################################
# Section 4: some machine learning, and a contest
# 
# Machine learning to predict scores based on the
# shower thought.
#########################################################

# Data needs to be in the right shape.
# You want the matrix of the data to be (n, m) where
# n is the number of observations, and m is the number of 
# different predictor - in our case, each unique word.
# The matrix of the labels is (n, 1). 
# So you predict y as a function of the m variables in x.
x <- t(corpus.mat)
y <- df.raw$score

# Set up train, test sets for x and y.
# Train/test ratio is 80%/20%.
train.ixs <- sample(1:nrow(x), 
  ceiling(nrow(x)*4/5), replace=FALSE)
x.train <- x[train.ixs,]
y.train <- y[train.ixs]
x.test <- x[-train.ixs,]
y.test <- y[-train.ixs]

# Function to calculate Mean Absolute Percentage Error (MAPE).
mape <- function(actual, predicted) {
  return(sum(abs((actual - predicted) / actual)) 
    / length(actual))
}

###### CONTEST HERE
# Fit a Support Vector Machine. Some example parameters given.
# You can specify your model parameters here. See:
# http://www.inside-r.org/node/57517
svm.model <- svm(x, y, # <-- don't change the data!
                ## ADD/CHANGE YOUR PARAMETERS HERE
                 kernel="linear",
                 cost=0.5,
                 epsilon=0.2)

# Calculate the training and test predictive accuracy.
# Just by experimenting with your SVM parameters above 
# (which includes cross-validation), how high accuracy can you 
# get on the testing set?
train.preds <- predict(svm.model, x.train)
test.preds <- predict(svm.model, x.test)
print(sprintf("Training accurracy (by MAPE): %.3f", 
  1 - mape(train.preds, y.train)))
print(sprintf("Testing accuracy (by MAPE): %.3f", 
  1 - mape(test.preds, y.test)))


#########################################################
# Section 5: some extra fun with text generation
# 
# EXTRA FUN: generate new random "shower thoughts"!
# First, we concatenate all the shower thoughts into 
# one giant text string. Then we use the "ngram" package
# to randomly generate new shower thoughts!
#########################################################

## install.packages("ngram") # if you haven't already

all.text <- paste(df.raw$title, collapse=' ')
ngram.model <- ngram(all.text, n=4)
shower.thought.length <- 20
print(sprintf("Just had a shower thought: %s.", 
  babble(ngram.model, shower.thought.length)))
