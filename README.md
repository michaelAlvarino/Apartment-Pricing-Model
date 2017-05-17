
# Apartment Pricing Model
I am going to be looking for a new one-bedroom apartment pretty soon. Given that I just finished a course in Machine Learning and a semester of computer science, my first reaction was whether or not I could find data to try and model the rental market in my area. That should be simple right?

## Finding Data
This was much more difficult than I expected. Once I started to google "Hoboken, NJ Rent Data" it was pretty clear there was no obvious data set or survey that I could get for free. My next best bet was to use the US Census data. Specifically, I used the American Housing Survey (AHS) 2015 National Public Use File, and the AHS 2015 Summary Tables. Both can be found [here](https://www.census.gov/programs-surveys/ahs.html).

## Understanding The Data
The first task was to understand my dataset, and what all the values mean. Luckily for me, Hudson County NJ is included in the New York data, as specified [here](https://www.census.gov/programs-surveys/ahs/data/interactive/docs/2013%20v%202015%20Metro%20Areas.pdf). It also meant going through the Summary Tables and deciding which fields I wanted to include in my model. The fields had to be things I could estimate after visiting an apartment and plug back into my model. If you look at the ```get_descriptions()``` function in HousingData.py you'll see exactly which features I included and their description.

## Cleaning The Data
Next I needed to understand how individual features behaved to see if features or rows should be removed. Specifically, I wanted to understand the continuous features like rent price and square footage. I found that the maximum rent was $10,600 which seemed much too high, so I simply removed those points. The same problem arose with people who paid $0 in rent. Alternately, I'm considering leaving the maximum rent field untouched and changing the zeros to the reported monthly mortgage for those same rows, if the mortgage is reported. I also found that unreported square footages were marked as -6, more fields that I just removed from the data set. 

## Training A Model
Finally, I needed to train a model. I decided to begin with a simple linear model and leave a more complicated model for the future. In order to account for non-linear relationships between my predicted value (rent) and features, I trained the linear model 7 times on the polynomial sets 0 - 6. I then chose the model which scored best on my testing data. Because I was using polynomial features I needed to punish overfitting and therefore used a Bayesian linear model. Bayesian linear models protect from overfitting by introducing a prior distribution on the model parameters. The log likelihood then ends up having some value lambda/2 * weights^2 subtracted from it, thereby penalizing your likelihood for many, or large weights. Specifically, I decided to use the BayesianRidge model provided by sklearn, and the default distribution parameters, with normalized data. I saw no reason to change the default distribution parameters yet because really I had no prior intuition on their distribution.

## Results
This model succeeded pretty well in spite of only using ~600 of the ~2000 New York values reported. The score and model could be improved by following through on what was mentioned in Understanding the data, or maybe using a neural network (next project :).   

## Questions
1. How could I go about deciding better values for my prior parameters than the defaults?
2. Should I try higher degree polynomials?
3. Would a model other than Bayesian Ridge work better? Why?
4. Is it worth normalizing the data in this case? When would one NOT normalize the data?

Please contact me if you think there's something I could improve or am misunderstanding. Specifically I am worried I am misunderstanding the score value returned by the model on testing data. Sklearn documentation says R^2 is defined as (1 - u/v), but also says that the best possible score is 1.0. I am measuring distance from 1.0, but worry that this is already being done in the 'definition' of the score.
