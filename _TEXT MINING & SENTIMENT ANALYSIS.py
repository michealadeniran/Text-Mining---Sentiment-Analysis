#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib seaborn wordcloud nltk


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
from  wordcloud import wordcloud
import nltk
nltk.download(['stopwords',
               'punkt',
               'wordnet',
               'omw-1.4',
               'vader_lexicon'])


# In[3]:


import sklearn as sk


# In[4]:


stop_words =nltk.corpus.stopwords.words('english')
print(stop_words)


# In[5]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

print(sentiment.polarity_scores("This move is great"))
print(sentiment.polarity_scores("This move is not great"))


# In[6]:


British_Airways = pd.read_csv('British_Airway_Review.csv')


# In[7]:


British_Airways.describe()


# In[8]:


British_Airways.shape


# In[9]:


British_Airways.head()


# In[10]:


British_Airways['reviews'] = British_Airways['reviews'].str.replace('✅ Trip Verified', '')
British_Airways['reviews'] = British_Airways['reviews'].str.replace('\|', '')
British_Airways


# In[11]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()
sia = SentimentIntensityAnalyzer() 




    


# In[12]:


British_Airways['compound'] = [sia.polarity_scores(review)['compound'] for review in British_Airways['reviews']]
British_Airways['neg'] = [sia.polarity_scores(review)['neg'] for review in British_Airways['reviews']]
British_Airways['neu'] = [sia.polarity_scores(review)['neu'] for review in British_Airways['reviews']]
British_Airways['pos'] = [sia.polarity_scores(review)['pos'] for review in British_Airways['reviews']]


# In[13]:


British_Airways.head()


# In[14]:


British_Airways[['compound', 'neg', 'neu', 'pos']].describe()


# In[15]:


sns.histplot(British_Airways['compound'])


# In[16]:


sns.histplot(British_Airways['pos'])


# In[17]:


sns.histplot(British_Airways['neg'])


# In[18]:


sns.histplot(British_Airways['neu'])


# In[19]:


#negative reviews

(British_Airways['compound']<0).groupby(British_Airways['recommended']).sum()


# In[20]:


import pandas as pd



# Calculate the percentage of negative reviews for each review
percent_negative = (British_Airways['compound'] <= 0).groupby(British_Airways['recommended']).mean() * 100

# Sort the values by the percentage of negative reviews
percent_negative = percent_negative.sort_values(ascending=False)

# Display the result
print(percent_negative)


# In[21]:


# Apply text preprocessing to the 'processed_review' column
British_Airways['Processed_reviews'] = British_Airways['reviews']

# Create subsets based on seat type and sentiment
positive_reviews = British_Airways['reviews'][British_Airways['compound'] > 0].tolist()
negative_reviews = British_Airways['reviews'][British_Airways['compound'] <= 0].tolist()




# In[22]:


positive_reviews = British_Airways['reviews'][British_Airways['compound'] > 0].tolist()
negative_reviews = British_Airways['reviews'][British_Airways['compound'] <= 0].tolist()


# In[23]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[24]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


negative_reviews = [word for review in British_Airways['reviews'][British_Airways['compound'] <= 0].dropna() for word in review.split()]

# Generate the word clouds
wordcloud_negative = WordCloud(background_color='white').generate(' '.join(negative_reviews))

# Display the word clouds using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Reviews')
plt.axis('off')

plt.show()



# In[25]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 'reviews' is the column containing text data
positive_reviews = [word for review in British_Airways['reviews'][British_Airways['compound'] > 0].dropna() for word in review.split()]

# Generate the word clouds
wordcloud_positive = WordCloud(background_color='white').generate(' '.join(positive_reviews))

# Display the word clouds using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Reviews')
plt.axis('off')


# In[26]:


from nltk import FreqDist
from nltk.tokenize import word_tokenize
positive_reviews = ' '.join(British_Airways['reviews'][British_Airways['compound'] > 0].dropna())

# Tokenize the combined positive reviews
positive_tokens = word_tokenize(positive_reviews)

# Create a frequency distribution for positive reviews
freq_dist_positive = FreqDist(positive_tokens)

# Display the most common words in positive reviews
print(freq_dist_positive.most_common(10))


# In[27]:


from nltk import FreqDist
from nltk.tokenize import word_tokenize

negative_reviews = ' '.join(British_Airways['reviews'][British_Airways['compound'] <= 0].dropna())
negative_tokens = word_tokenize(negative_reviews)

# Create a frequency distribution for negative reviews
freq_dist_negative = FreqDist(negative_tokens)

# Display the most common words in negative reviews
print(freq_dist_negative.most_common(10))


# In[31]:


freq_dist_negative.plot(30)


# In[33]:


freq_dist_positive.plot(30)


# In[ ]:




