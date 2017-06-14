# Udacity Machine Learning Engineer

## Lesson9:Numpy & Pandas Tutorial

```
# numpy
numpy.mean(numbers)
numpy.median(numbers)
numpy.std(numbers)

array_1 = np.array([1, 2, 3], float)
array_2 = np.array([[6], [7], [8]], float)
print np.mean(array_1)
print np.mean(array_2)
print np.dot(array_1, array_2)
```

```
# pandas
d = {'name': Series(['Braund', 'Cummings', 'Heikkinen', 'Allen'], index=['a', 'b', 'c', 'd']),
     'age': Series([22, 38, 26, 35], index=['a', 'b', 'c', 'd']),
     'fare': Series([7,25, 71.83, 8.05], index=['a', 'b', 'c', 'd']),
     'survived?': Series([False, True, True, False), index=['a', 'b', 'c', 'd'])}
     
df = DataFrame(d)
series = pd.Series(['Dave', 'Cheng-Han', 'Udacity', 42, -1789710578])
series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],
                       index=['Instructor', 'Curriculum Manager',
                              'Course Number', 'Power Level'])

series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],
                       index=['Instructor', 'Curriculum Manager',
                              'Course Number', 'Power Level'])
print series['Instructor']
print series[['Instructor', 'Curriculum Manager', 'Course Number']]

cuteness = pd.Series([1, 2, 3, 4, 5], index=['Cockroach', 'Fish', 'Mini Pig',
                                                 'Puppy', 'Kitten'])
print cuteness > 3
print cuteness[cuteness > 3]

data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)

data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print football.dtypes
print football.describe()
print football.head()
print football.tail()

data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print football.iloc[[0]]
print ""
print football.loc[[0]]
print ""
print football[3:5]
print ""
print football[football.wins > 10]
print ""
print football[(football.wins > 10) & (football.team == "Packers")]

d = {'one': Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = DataFrame(d)
df.apply(numpy.mean)

df['one'].map(lambda x: x>=1) # only for the series
df.applymap(lambda x: x>= 1) # for the whole dataframe

a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5, 6]
numpy.dot(a, b)
```

`lambda x: x>= 1` will take an input x and return x>=1, or a boolean that equals True or False.

In this example, `map()` and `applymap()` create a new Series or DataFrame by applying the lambda function to each element. Note that map() can only be used on a Series to return a new Series and applymap() can only be used on a DataFrame to return a new DataFrame.


```
from pandas import DataFrame, Series
import numpy


def avg_medal_count():
    '''
    Compute the average number of bronze medals earned by countries who 
    earned at least one gold medal.  
    
    Save this to a variable named avg_bronze_at_least_one_gold. You do not
    need to call the function in your code when running it in the browser -
    the grader will do that automatically when you submit or test it.
    
    HINT-1:
    You can retrieve all of the values of a Pandas column from a 
    data frame, "df", as follows:
    df['column_name']
    
    HINT-2:
    The numpy.mean function can accept as an argument a single
    Pandas column. 
    
    For example, numpy.mean(df["col_name"]) would return the 
    mean of the values located in "col_name" of a dataframe df.
    '''


    countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

    gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
    bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]
    
    olympic_medal_counts = {'country_name':Series(countries),
                            'gold': Series(gold),
                            'silver': Series(silver),
                            'bronze': Series(bronze)}
    olympic_medal_counts_df = DataFrame(olympic_medal_counts)
    
    # YOUR CODE HERE
    avg_medal_count = olympic_medal_counts_df[['gold', 'silver', 'bronze']].apply(numpy.mean)
    return avg_medal_count
```


```
from pandas import DataFrame, Series
import numpy


def avg_medal_count():
    '''
    Compute the average number of bronze medals earned by countries who 
    earned at least one gold medal.  
    
    Save this to a variable named avg_bronze_at_least_one_gold. You do not
    need to call the function in your code when running it in the browser -
    the grader will do that automatically when you submit or test it.
    
    HINT-1:
    You can retrieve all of the values of a Pandas column from a 
    data frame, "df", as follows:
    df['column_name']
    
    HINT-2:
    The numpy.mean function can accept as an argument a single
    Pandas column. 
    
    For example, numpy.mean(df["col_name"]) would return the 
    mean of the values located in "col_name" of a dataframe df.
    '''


    countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

    gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
    bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]
    
    olympic_medal_counts = {'country_name':Series(countries),
                            'gold': Series(gold),
                            'silver': Series(silver),
                            'bronze': Series(bronze)}
    olympic_medal_counts_df = DataFrame(olympic_medal_counts)
    
    # YOUR CODE HERE
    metal_counts = olympic_medal_counts_df[['gold', 'silver', 'bronze']]
    points = numpy.dot(metal_counts, [4, 2, 1])
    olympic_points = {'country_name': Series(countries), 'points': Series(points)}
    olympic_points_df = DataFrame(olympic_points)
    
    # An alternate solution using only pandas: 
    # df['points'] = df[['gold','silver','bronze']].dot([4, 2, 1]) olympic_points_df = df[['country_name','points']]
    
    return olympic_points_df
```

[goog resources](https://bitbucket.org/hrojas/learn-pandas)


## Lession10: Intro to Scikit-Learn

```
from sklearn import datasets # sklearn comes with a variety of preloaded datasets 
from sklearn import metrics # calculate how well our model is doing
from sklearn.linear_model import LinearRegression

# Load the dataset
housing_data = datasets.load_boston()
linear_regression_model = LinearRegression()

# Next, we can fit our Linear Regression model on our feature set to make predictions for our labels(the price of the houses). Here, housing_data.data is our feature set and housing_data.target are the labels we are trying to predict.

linear_regression_model.fit(housing_data.data, housing_data.target)
predictions = linear_regression_model.predict(housing_data.data)

# Lastly, we want to check how our model does by comparing our predictions with the actual label values. Since this is a regression problem, we will use the r2 score metric. You will learn about the various classification and regression metrics in future lessons.

score = metrics.r2_score(housing_data.target, predictions)
```

### Install Scikit-Learn
If you've accidentally installed version v0.18 through pip, not to worry! Use the command below to downgrade your scikit-learn version to v0.17:

`pip install --upgrade 'scikit-learn>=0.17,<0.18'`
If you are using the Anaconda distribution of Python and have scikit-learn installed as version v0.18, you can also use the command below to downgrade your scikit-learn version to v0.17:

`conda -c install scikit-learn=0.17`



































