#######################
 Commands to run code: 
#######################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perceptron, adaline, and sgd binary 
# classifiers on iris dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python main.py perceptron iris.data
python main.py adaline iris.data
python main.py sgd iris.data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perceptron, adaline, and sgd binary 
# classifiers on car.data dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python main.py perceptron car.data`
python main.py adaline car.data
python main.py sgd car.data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Multiclass classifier (One-Vs-Rest) 
# on iris and car.data datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python main.py ovr iris.data
python main.py ovr car.data 

# This should work for any data set, because I have
# not hardcoded any lengths or attribut names. However
# read further below to see cases this wouldn't work.

########
# Note #
########

Each classifier will first first print the error (or cost) of
each of the n iterations. Then it will test the trained model
and print the results of that. 

Also, each file has a comment at the top containing my in-class 
notes on that classifier, to show where I am getting the main 
structure of my code. 

I got car.data from the required website in the assignment. However, 
I manually converted the string values to integer values, and then m
anually converted very large integer values to smaller values. The
latter was done because otherwise I would get overflow errors while
calculating the sum squared error. Though this data file has been changed
by me, it still represents the same data.

Because of this, I can't guarantee this will work 'out of the box' with
another data set (though if the data set was cleaned as I did with car.data
it should work)