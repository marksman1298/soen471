import os
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StringType, MapType
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import size, abs, col,  row_number, udf
from all_states import all_states
'''
INTRODUCTION

With this assignment you will get a practical hands-on of frequent 
itemsets and clustering algorithms in Spark. Before starting, you may 
want to review the following definitions and algorithms:
* Frequent itemsets: Market-basket model, association rules, confidence, interest.
* Clustering: kmeans clustering algorithm and its Spark implementation.

DATASET

We will use the dataset at 
https://archive.ics.uci.edu/ml/datasets/Plants, extracted from the USDA 
plant dataset. This dataset lists the plants found in US and Canadian 
states.

The dataset is available in data/plants.data, in CSV format. Every line 
in this file contains a tuple where the first element is the name of a 
plant, and the remaining elements are the states in which the plant is 
found. State abbreviations are in data/stateabbr.txt for your 
information.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x, y: '\n'.join([x, y]))
    return a + '\n'

def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None


'''
PART 1: FREQUENT ITEMSETS

Here we will seek to identify association rules between states to 
associate them based on the plants that they contain. For instance, 
"[A, B] => C" will mean that "plants found in states A and B are likely 
to be found in state C". We adopt a market-basket model where the 
baskets are the plants and the items are the states. This example 
intentionally uses the market-basket model outside of its traditional 
scope to show how frequent itemset mining can be used in a variety of 
contexts.
'''

def data_frame(filename, n):
    '''
    Write a function that returns a CSV string representing the first 
    <n> rows of a DataFrame with the following columns,
    ordered by increasing values of <id>:
    1. <id>: the id of the basket in the data file, i.e., its line number - 1 (ids start at 0).
    2. <plant>: the name of the plant associated to basket.
    3. <items>: the items (states) in the basket, ordered as in the data file.

    Return value: a CSV string. Using function toCSVLine on the right 
                  DataFrame should return the correct answer.
    Test file: tests/test_data_frame.py
    '''
    spark = init_spark()
    # lines = spark.read.csv(filename)
    # lines = lines.withColumn("states", concat(array(lines.columns[1:])))
    # result = lines.select("_c0", "states")
    # result = result.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("_c0"))-1)
    # result = result.select("row_num", "_c0", "states")

    # lines = spark.read.load(filename, format="text")
    # lines = lines.withColumn("c0", split(lines["value"], ",").getItem(0)).withColumn("states", split(lines["value"], ","))
    # lines = lines.withColumn("state", slice(col("states"), 2, 69)) #disaster 
    # lines = lines.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("c0"))-1)
    # result = lines.select("row_num", "c0", "state")

    lines = spark.read.load(filename, format="text").rdd
    plant_df = lines.map(lambda row: row.value.split(","))
    plant_df = plant_df.map(lambda x: Row(plant=x[0], states=x[1:])).toDF()
    plant_df = plant_df.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("plant"))-1)
    result = plant_df.select("row_num", "plant", "states")
    return toCSVLine(result.limit(n))

def frequent_itemsets(filename, n, s, c):
    '''
    Using the FP-Growth algorithm from the ML library (see 
    http://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html), 
    write a function that returns the first <n> frequent itemsets 
    obtained using min support <s> and min confidence <c> (parameters 
    of the FP-Growth model), sorted by (1) descending itemset size, and 
    (2) descending frequency. The FP-Growth model should be applied to 
    the DataFrame computed in the previous task. 
    
    Return value: a CSV string. As before, using toCSVLine may help.
    Test: tests/test_frequent_items.py
    '''
    spark = init_spark()

    lines = spark.read.load(filename, format="text").rdd
    plant_df = lines.map(lambda row: row.value.split(","))
    plant_df = plant_df.map(lambda x: Row(plant=x[0], states=x[1:])).toDF()
    plant_df = plant_df.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("plant"))-1)
    result = plant_df.select("row_num", "plant", "states")

    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(result)
    return toCSVLine(model.freqItemsets.sort([size("items"), "freq"], ascending=[False, False]).limit(n))
     

def association_rules(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that returns the 
    first <n> association rules obtained using min support <s> and min 
    confidence <c> (parameters of the FP-Growth model), sorted by (1) 
    descending antecedent size in association rule, and (2) descending 
    confidence.

    Return value: a CSV string.
    Test: tests/test_association_rules.py
    '''
    spark = init_spark()
    lines = spark.read.load(filename, format="text").rdd
    plant_df = lines.map(lambda row: row.value.split(","))
    plant_df = plant_df.map(lambda x: Row(plant=x[0], states=x[1:])).toDF()
    plant_df = plant_df.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("plant"))-1)
    result = plant_df.select("row_num", "plant", "states")

    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(result)
    return toCSVLine(model.associationRules.select("antecedent", "consequent", "confidence").sort([size("antecedent"), "confidence"], ascending=[False, False]).limit(n))

def interests(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that computes 
    the interest of association rules (interest = |confidence - 
    frequency(consequent)|; note the absolute value)  obtained using 
    min support <s> and min confidence <c> (parameters of the FP-Growth 
    model), and prints the first <n> rules sorted by (1) descending 
    antecedent size in association rule, and (2) descending interest.

    Return value: a CSV string.
    Test: tests/test_interests.py
    '''

    #Frequency -> Probability of the consequent 
    spark = init_spark()
    lines = spark.read.load(filename, format="text").rdd
    plant_df = lines.map(lambda row: row.value.split(","))
    plant_df = plant_df.map(lambda x: Row(plant=x[0], states=x[1:])).toDF()
    plant_df = plant_df.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("plant"))-1)
    result = plant_df.select("row_num", "plant", "states")
    total = result.count() 

    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(result)
    return toCSVLine(model.associationRules.join(model.freqItemsets, model.associationRules.consequent == model.freqItemsets.items).withColumn("interest", abs(model.associationRules.confidence - (model.freqItemsets.freq/total))).sort([size("antecedent"), "interest"], ascending=[False, False]).select("antecedent", "consequent", "confidence", "items", "freq", "interest").limit(n))

'''
PART 2: CLUSTERING

We will now cluster the states based on the plants that they contain.
We will reimplement and use the kmeans algorithm. States will be 
represented by a vector of binary components (0/1) of dimension D, 
where D is the number of plants in the data file. Coordinate i in a 
state vector will be 1 if and only if the ith plant in the dataset was 
found in the state (plants are ordered alphabetically, as in the 
dataset). For simplicity, we will initialize the kmeans algorithm 
randomly.

An example of clustering result can be visualized in states.png in this 
repository. This image was obtained with R's 'maps' package (Canadian 
provinces, Alaska and Hawaii couldn't be represented and a different 
seed than used in the tests was used). The classes seem to make sense 
from a geographical point of view!
'''

def data_preparation(filename, plant, state):
    '''
    This function creates an RDD in which every element is a tuple with 
    the state as first element and a dictionary representing a vector 
    of plant as a second element:
    (name of the state, {dictionary})

    The dictionary should contains the plant names as keys. The 
    corresponding values should be 1 if the plant occurs in the state 
    represented by the tuple.

    You are strongly encouraged to use the RDD created here in the 
    remainder of the assignment.

    Return value: True if the plant occurs in the state and False otherwise.
    Test: tests/test_data_preparation.py
    '''
    spark = init_spark()
    fin_dict = data_prep(filename, spark)

    return fin_dict[state][plant]

def data_prep(filename, spark):
    
    #plant vector [1, 0, 0] roses are present, tulips dandilions are not
    #tuple (name of the state, dictionary{})
    # ex: (mtl, {roses: 1, tulips: 0, dandilions: 0})
    
    lines = spark.read.load(filename, format="text").rdd
    plant_df = lines.map(lambda row: row.value.split(","))
    plant_df = plant_df.map(lambda x: Row(plant=x[0], states=x[1:])).toDF()
    plant_df = plant_df.withColumn("row_num", row_number().over(Window.partitionBy().orderBy("plant"))-1)
    result = plant_df.select("row_num", "plant", "states")
    
    def udf_func(plant, i):
        states_plant = {}
        for state in i:
            states_plant[state] = {plant}
        return states_plant
    

    state_plant_dic = udf(lambda p, i: udf_func(p, i), MapType(StringType(), StringType(), False))
    result = result.withColumn("plant_dict", state_plant_dic(col("plant"), col("states")))
    dic = {}

    def merge_dict(state_dict):
        for k, v in state_dict.items():
            s = dic.get(k, set())
            s.add(v[1:-1])
            dic[k] = s

    for row in result.collect():
        merge_dict(row["plant_dict"])
    plants = set(result.rdd.map(lambda x: x[1]).collect())
    fin_dict = {}
    for key, value in dic.items():
        fin_dict[key] = dict.fromkeys(value, 1)
        dne = dict.fromkeys(plants - value, 0)
        fin_dict[key].update(dne)
    return fin_dict



def distance2(filename, state1, state2):
    '''
    This function computes the squared Euclidean
    distance between two states.
    
    Return value: an integer.
    Test: tests/test_distance.py
    '''

    spark = init_spark()
    fin_dict = data_prep(filename, spark)
    
    distance = 0
    for k, v in fin_dict[state1].items():
        distance += (v - fin_dict[state2][k]) ** 2
    return distance

def init_centroids(k, seed):
    '''
    This function randomly picks <k> states from the array in answers/all_states.py (you
    may import or copy this array to your code) using the random seed passed as
    argument and Python's 'random.sample' function.

    In the remainder, the centroids of the kmeans algorithm must be
    initialized using the method implemented here, perhaps using a line
    such as: `centroids = rdd.filter(lambda x: x[0] in
    init_states).collect()`, where 'rdd' is the RDD created in the data
    preparation task.

    Note that if your array of states has all the states, but not in the same
    order as the array in 'answers/all_states.py' you may fail the test case or
    have issues in the next questions.

    Return value: a list of <k> states.
    Test: tests/test_init_centroids.py
    '''
    random.seed(seed)
    return random.sample(all_states, k)

def first_iter(filename, k, seed):
    '''
    This function assigns each state to its 'closest' class, where 'closest'
    means 'the class with the centroid closest to the tested state
    according to the distance defined in the distance function task'. Centroids
    must be initialized as in the previous task.

    Return value: a dictionary with <k> entries:
    - The key is a centroid.
    - The value is a list of states that are the closest to the centroid. The list should be alphabetically sorted.

    Test: tests/test_first_iter.py
    '''
    centroids = init_centroids(k, seed)
    spark = init_spark()
    fin_dict = data_prep(filename, spark)
    state_rdd = spark.sparkContext.parallelize(([(k, v) for k, v in fin_dict.items()]))
    state_rdd = state_rdd.filter(lambda x: x[0] in centroids).map(lambda x:x[1]).collect()
    return {}

def kmeans(filename, k, seed):
    '''
    This function:
    1. Initializes <k> centroids.
    2. Assigns states to these centroids as in the previous task.
    3. Updates the centroids based on the assignments in 2.
    4. Goes to step 2 if the assignments have not changed since the previous iteration.
    5. Returns the <k> classes.

    Note: You should use the list of states provided in all_states.py to ensure that the same initialization is made.
    
    Return value: a list of lists where each sub-list contains all states (alphabetically sorted) of one class.
                  Example: [["qc", "on"], ["az", "ca"]] has two 
                  classes: the first one contains the states "qc" and 
                  "on", and the second one contains the states "az" 
                  and "ca".
    Test file: tests/test_kmeans.py
    '''
    spark = init_spark()
    return []
