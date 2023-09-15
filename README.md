## Executive Summary


In this project, I analyzed textual data from two popular Reddit communities, '/r/wallstreetbets' and '/r/CryptoCurrency', with the aim to build models that could accurately classify posts to their originating subreddit. I pulled and cleaned the data, then added new features such as post length, word count, and sentiment scores. I then used histograms to visualize the data and get insights into patterns and differences between the two subreddits. I created two models: a Multinomial Naive Bayes model and a Support Vector Classifier model, each paired with a different text vectorization technique. The Naive Bayes model yielded training and test accuracy scores of 95.5% and 94.4%, respectively. The Support Vector Classifier model outperformed the first, achieving near perfect training accuracy (99.9%) and a test accuracy of 98.4%, which suggests it could be slightly overfitting the training data. In conclusion, I successfully trained models that accurately predict the origin of a Reddit post based on its content, which provides valuable insights into the language and style preferences of each subreddit's community.


## Data Collection: Data Pull

In the initial stage of this project, we use the `praw` library to pull posts from the Reddit API. The process of data collection is carried out using a Python file named `Data Pull`.

```
import pandas as pd
import praw

reddit = praw.Reddit(
    client_id='Your Client ID',
    client_secret='Your Client Secret',
    user_agent='Your User Agent',
    username='Your Username',
    password='Your Password'
)
```
Please replace `'Your Client ID'`, `'Your Client Secret'`, `'Your User Agent'`, `'Your Username'`, and `'Your Password'` with your personal Reddit API credentials.

We define a function `combine_data(posts, label)` to combine and format data that has been pulled from the Reddit API. This function returns a list of tuples, each containing the UTC timestamp of a post's creation, its title, its text content, and its subreddit of origin.

We then utilize this function to pull posts from two specific subreddits: 'wallstreetbets' and 'CryptoCurrency'. We gather different categories of posts: new, hot, top, and controversial. These are limited to 1000 posts each.

Finally, we convert the gathered data into a pandas DataFrame, remove duplicate posts, and concatenate the data from both subreddits.

```python
df_pull = pd.concat([df_pull_wsb,df_pull_cryp])
df_pull = df_pull.drop_duplicates()
```

After the above operation, `df_pull` contains all the unique posts from both subreddits, ready for the next stages of the project.

Understood, let's rewrite the data cleaning section focusing on providing more detailed explanations and less code snippets:


## Data Cleaning

The next critical step of my project involves cleaning the collected data to ensure optimal results in the following stages.

First, I address any missing values in the 'text' column by filling them with empty strings, ensuring that no data is lost when combining text and title fields later.

Next, I combine the 'title' and 'text' columns into a single column, 'title_and_text'. This new field consolidates the main content of the posts, which will be crucial for the model to identify common patterns and themes.

To make the text data easier to process, I remove unnecessary characters and alphanumeric words from 'title_and_text'. These include newline characters and alphanumeric combinations that may not contribute to the understanding of the overall text sentiment or content.

To provide more informative features for the model, I create new columns to represent the length and word count of the text and title. These features could potentially reveal patterns linked to the subreddit of origin. For example, posts from one subreddit may consistently have longer titles or texts than the other.

In preparation for model training, I map the 'subreddit' column values to numerical format: I assign '1' to 'wallstreetbets' and '0' to 'CryptoCurrency'.

Lastly, I utilize the sentiment analysis capability of the `SentimentIntensityAnalyzer` from the NLTK library to extract the sentiment score of the 'text', 'title', and 'title_and_text' fields. This transformation results in new columns: 'composite_sentiment_text', 'composite_sentiment_title', and 'composite_sentiment_title_and_text', which hold the sentiment scores. Such features can be beneficial for the model to understand the overall sentiment of the posts.

After these operations, my data is cleaned and enriched with new features, ready for the next stage: Exploratory Data Analysis (EDA) and model training.

## Data Visualization

Visualizing the data is an essential step that provides me with a clear understanding of the features and helps me identify patterns, trends, or outliers.

To visualize various aspects of my dataset, I create a function `hist_plot(data, column)`, which generates histograms for a specified column from my dataset. This function overlays the data distributions from the 'wallstreetbets' and 'CryptoCurrency' subreddits to facilitate a straightforward comparison.

Using the `hist_plot` function, I generate histograms for five different columns:

1. **Title Length (`title_len`)**: This plot shows the distribution of the lengths of post titles in each subreddit. The distribution of lengths can hint at any differing patterns in how users title their posts in each subreddit.

2. **Text Length (`text_len`)**: Similar to title length, this plot represents the lengths of the post contents. Differences here could highlight unique content length preferences or trends in each subreddit.

3. **Title Word Count (`title_word_count`)**: This plot shows the distribution of the number of words in the titles of each subreddit's posts. It gives insight into the verbosity or conciseness of the users when forming their post titles.

4. **Text Word Count (`text_word_count`)**: This histogram represents the number of words in the post contents, which, along with the text length, can demonstrate the depth of discussion in each subreddit.

5. **Composite Sentiment of Title and Text (`composite_sentiment_title_and_text`)**: This plot shows the sentiment analysis scores for the combination of title and text. It gives an overview of the emotional context of the posts in each subreddit.

These visualizations assist in understanding the data and can reveal key insights that will inform model selection and feature importance during the next stages of the project.

## Modelling

With my data cleaned, explored, and visualized, I move on to the modelling stage. For the problem at hand, I decide to use a classification model to identify the origin of each post.

Here are the steps I take in my modelling process:

1. **Set X and y variables**: I choose 'title_and_text' (the combined text from the post title and post content) as my predictor variable (X), and 'subreddit' (indicating which subreddit a post belongs to) as my target variable (y).

2. **Train-Test Split**: To avoid overfitting and accurately evaluate my model's performance, I split my dataset into a training set and a test set. This allows me to train my model on a portion of the data and then test it on unseen data.

3. **Define a Pipeline**: I use a pipeline to streamline my modelling process. My pipeline consists of a CountVectorizer (which converts the text data into a matrix of token counts) and a Multinomial Naive Bayes classifier (which is often effective for text classification problems).

4. **Parameter Tuning**: To optimize my model's performance, I perform GridSearchCV to tune the hyperparameters of my pipeline. This technique systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. The parameters I adjust include `max_df`, `min_df`, `ngram_range`, `max_features`, `strip_accents`, and `alpha` for smoothing in the Naive Bayes model.

5. **Model Fitting**: I fit the model to the training data and use the best parameters identified by GridSearchCV. I find that the best parameters include a maximum document frequency of 0.5, minimum document frequency of 2, ngram range of 1, max features of 10,000, ASCII strip accents, and alpha of 0.12575.

6. **Model Evaluation**: I evaluate my model's performance by calculating its accuracy on both the training set and the test set. My model achieves a training accuracy of 0.955 and a test accuracy of 0.944, indicating a high degree of accuracy and suggesting that it generalizes well to unseen data.

7. **Confusion Matrix**: I further assess the model's performance by generating a confusion matrix for the predictions on the test set. The confusion matrix provides a summary of correct and incorrect predictions, broken down by each category.

Overall, the model I build performs well in classifying subreddit posts, demonstrating both high accuracy and a good ability to generalize to unseen data.


## Second Model

For comparative purposes, I decide to create a second model to see how it fares against my initial model. For this model, I choose a pipeline that includes the TfidfVectorizer and a Support Vector Classifier (SVC). 

Here are the steps I take in creating my second model:

1. **Define a Second Pipeline**: This pipeline also includes two steps: First, the TfidfVectorizer, which transforms the text data into a matrix of TF-IDF features. Then, I use an SVC, which is a type of SVM classifier, known for its effectiveness in high dimensional spaces, which makes it suitable for text data.

2. **Parameter Tuning**: Similar to the first model, I use GridSearchCV to fine-tune the parameters for my pipeline. The parameters I tune for the TfidfVectorizer include `max_df`, `max_features`, `min_df`, and `stop_words`. For the SVC, I tune parameters such as `C` (regularization parameter), `kernel`, and `random_state`.

3. **Model Fitting**: I fit the second model to the training data using the best parameters identified from the grid search. The best parameters include a maximum document frequency of 0.7, maximum features of 4000, minimum document frequency of 2, no stop words, an SVC C value of 2.50075, an 'rbf' kernel, and a random state of 42.

4. **Model Evaluation**: For the second model, the training score is nearly perfect at 0.999, and the test score is also high at 0.984. This suggests that the second model might be slightly overfitting to the training data compared to the first model, but it's still performing very well on the unseen test data.

5. **Confusion Matrix**: I also generate a confusion matrix for the second model's predictions on the test data, this provides a visual way to examine the performance of the model in terms of correctly and incorrectly classified instances.

Overall, the second model shows a higher performance than the first model, although it might be overfitting a bit more to the training data. Both models are strong performers, and choosing between them would depend on the specific needs and constraints of the task at hand.
