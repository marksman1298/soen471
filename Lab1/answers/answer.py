import csv
import os
import sys
# Spark imports
import dask.dataframe
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc
# Dask imports
import dask.bag as db
import dask.dataframe as df  # you can use Dask bags or dataframes
from csv import reader

'''
INTRODUCTION

The goal of this assignment is to implement a basic analysis of textual 
data using Apache Spark (http://spark.apache.org) and 
Dask (https://dask.org). 
'''

'''
DATASET

We will study a dataset provided by the city of Montreal that contains 
the list of trees treated against the emerald ash borer 
(https://en.wikipedia.org/wiki/Emerald_ash_borer). The dataset is 
described at 
http://donnees.ville.montreal.qc.ca/dataset/frenes-publics-proteges-injection-agrile-du-frene 
(use Google translate to translate from French to English). 

We will use the 2015 and 2016 data sets available in directory `data`.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

#Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: "\n".join([x,y]))
    return a + "\n"

def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None

'''
Plain PYTHON implementation

To get started smoothly and become familiar with the assignment's 
technical context (Git, GitHub, pytest, GitHub actions), we will implement a 
few steps in plain Python.
'''

#Python answer functions
def count(filename):
    '''
    Write a Python (not DataFrame, nor RDD) script that prints the number of trees (non-header lines) in
    the data file passed as first argument.
    Test file: tests/test_count.py
    Note: The return value should be an integer
    '''
    
    # ADD YOUR CODE HERE
    num_rows = -1
    for _ in open(filename, 'r', encoding='utf8', errors='ignore'):
        num_rows += 1
    return num_rows

def parks(filename):
    '''
    Write a Python (not DataFrame, nor RDD) script that prints the number of trees that are *located in a park*.
    To get the park location information, have a look at the *Nom_parc* column (name of park).
    Test file: tests/test_parks.py
    Note: The return value should be an integer
    '''

    # ADD YOUR CODE HERE
    with open(filename, 'r') as csvfile:
        datafile = csv.reader(csvfile)
        trees_in_park = [
            row for row in datafile
            if row[6]
        ]
    return len(trees_in_park)-1

def uniq_parks(filename):
    '''
    Write a Python (not DataFrame, nor RDD) script that prints the list of unique parks where trees
    were treated. The list must be ordered alphabetically. Every element in the list must be printed on
    a new line.
    Test file: tests/test_uniq_parks.py
    Note: The return value should be a string with one park name per line
    '''

    # ADD YOUR CODE HERE
    with open(filename, 'r', encoding="ISO-8859-1") as csvfile:
        datafile = csv.reader(csvfile)
        trees_in_park = [
            row[6] for row in datafile
            if row[6]
        ]
        sorted_parks = sorted(list(set(trees_in_park[1:])))
        park_str = "".join([park+"\n" for park in sorted_parks])
        return park_str

def uniq_parks_counts(filename):
    '''
    Write a Python (not DataFrame, nor RDD) script that counts the number of trees treated in each park
    and prints a list of "park,count" pairs in a CSV manner ordered
    alphabetically by the park name. Every element in the list must be printed
    on a new line.
    Test file: tests/test_uniq_parks_counts.py
    Note: The return value should be a CSV string
          Have a look at the file *tests/list_parks_count.txt* to get the exact return format.
    '''

    # ADD YOUR CODE HERE
    with open(filename, 'r', encoding="ISO-8859-1") as csvfile:
        datafile = csv.reader(csvfile)
        trees_in_park = [
            row[6] for row in datafile
            if row[6]
        ]
        tree_count = {}
        for park in trees_in_park[1:]:
            if park not in tree_count:
                tree_count[park] = 1
            else:
                tree_count[park] += 1
        tree_tuples = [(k, v) for k, v in tree_count.items()]
        sorted_parks = sorted(tree_tuples)
        fmt_park = "".join("{},{}\n".format(*i) for i in sorted_parks)
        return fmt_park

def frequent_parks_count(filename):
    '''
    Write a Python (not DataFrame, nor RDD) script that prints the list of the 10 parks with the
    highest number of treated trees. Parks must be ordered by decreasing
    number of treated trees and by alphabetical order when they have similar number.
    Every list element must be printed on a new line.
    Test file: tests/test_frequent_parks_count.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/frequent.txt* to get the exact return format.
    '''

    # ADD YOUR CODE HERE
    with open(filename, 'r', encoding="ISO-8859-1") as csvfile:
        datafile = csv.reader(csvfile)
        trees_in_park = [
            row[6] for row in datafile
            if row[6]
        ]
        tree_count = {}
        for park in trees_in_park[1:]:
            if park not in tree_count:
                tree_count[park] = 1
            else:
                tree_count[park] += 1
        tree_tuples = [(k, v) for k, v in tree_count.items()]
        sorted_parks = sorted(tree_tuples, key=lambda x: (-x[1], x[0]))
        fmt_park = "".join("{},{}\n".format(*i) for i in sorted_parks[:10])
        return fmt_park

def intersection(filename1, filename2):
    '''
    Write a Python (not DataFrame, nor RDD) script that prints the alphabetically sorted list of
    parks that had trees treated both in 2016 and 2015. Every list element
    must be printed on a new line.
    Test file: tests/test_intersection.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/intersection.txt* to get the exact return format.
    '''

    # ADD YOUR CODE HERE

    with open(filename1, 'r', encoding="ISO-8859-1") as f1, open(filename2, 'r', encoding="ISO-8859-1") as f2:
        csv_r1 = csv.reader(f1)
        csv_r2 = csv.reader(f2)
        trees_in_park1 = [
            row[6] for row in csv_r1
            if row[6]
        ]
        trees_in_park2 = [
            row[6] for row in csv_r2
            if row[6]
        ]
        tree_intersection = set.intersection(set(trees_in_park1[1:]), set(trees_in_park2[1:]))
        sorted_trees = sorted(tree_intersection)
        trees = "".join([park + "\n" for park in sorted_trees])
        return trees



'''
SPARK RDD IMPLEMENTATION

You will now have to re-implement all the functions above using Apache 
Spark's Resilient Distributed Datasets API (RDD, see documentation at 
https://spark.apache.org/docs/latest/rdd-programming-guide.html). 
Outputs must be identical to the ones obtained above in plain Python. 
However, all operations must be re-implemented using the RDD API, you 
are not allowed to simply convert results obtained with plain Python to 
RDDs (this will be checked). Note that the function *toCSVLine* in the 
HELPER section at the top of this file converts RDDs into CSV strings.
'''

# RDD functions

def count_rdd(filename):
    '''
    Write a Python script using RDDs that prints the number of trees
    (non-header lines) in the data file passed as first argument.
    Test file: tests/test_count_rdd.py
    Note: The return value should be an integer
    '''

    spark = init_spark()
    parks_rdd = spark.sparkContext.textFile(filename)
    return parks_rdd.count() - 1


def parks_rdd(filename):
    '''
    Write a Python script using RDDs that prints the number of trees that are *located in a park*.
    To get the park location information, have a look at the *Nom_parc* column (name of park).
    Test file: tests/test_parks_rdd.py
    Note: The return value should be an integer
    '''

    spark = init_spark()

    parks_rdd = spark.read.csv(filename, header=True, mode="DROPMALFORMED").rdd
    return parks_rdd.map(lambda x: x[6]).filter(lambda x: x is not None).count()


def uniq_parks_rdd(filename):
    '''
    Write a Python script using RDDs that prints the list of unique parks where
    trees were treated. The list must be ordered alphabetically. Every element
    in the list must be printed on a new line.
    Test file: tests/test_uniq_parks_rdd.py
    Note: The return value should be a CSV string
    '''

    spark = init_spark()
    parks_rdd = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1").rdd

    parks_lst = parks_rdd.map(lambda x: x[6]).filter(lambda x: x is not None).distinct().sortBy(lambda x: x).collect()
    fmt_park = "".join([park+"\n" for park in parks_lst])
    return fmt_park

def uniq_parks_counts_rdd(filename):
    '''
    Write a Python script using RDDs that counts the number of trees treated in
    each park and prints a list of "park,count" pairs in a CSV manner ordered
    alphabetically by the park name. Every element in the list must be printed
    on a new line.
    Test file: tests/test_uniq_parks_counts_rdd.py
    Note: The return value should be a CSV string
          Have a look at the file *tests/list_parks_count.txt* to get the exact return format.
    '''

    spark = init_spark()
    tree_rdd = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1").rdd

    park_rdd = tree_rdd.map(lambda x: x[6]).filter(lambda x: x is not None)
    park_value = park_rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).sortByKey()
    return toCSVLine(park_value)

def frequent_parks_count_rdd(filename):
    '''
    Write a Python script using RDDs that prints the list of the 10 parks with
    the highest number of treated trees. Parks must be ordered by decreasing
    number of treated trees and by alphabetical order when they have similar
    number.  Every list element must be printed on a new line.
    Test file: tests/test_frequent_parks_count_rdd.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/frequent.txt* to get the exact return format.
    '''

    spark = init_spark()

    tree_rdd = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1").rdd
    park_rdd = tree_rdd.map(lambda x: x[6]).filter(lambda x: x is not None)
    park_value = park_rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False)
    freq_parks = park_value.take(10)
    print(freq_parks)

    fmt_park = "".join("{},{}\n".format(*i) for i in freq_parks)
    return fmt_park

def intersection_rdd(filename1, filename2):
    '''
    Write a Python script using RDDs that prints the alphabetically sorted list
    of parks that had trees treated both in 2016 and 2015. Every list element
    must be printed on a new line.
    Test file: tests/test_intersection_rdd.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/intersection.txt* to get the exact return format.
    '''

    spark = init_spark()
    tree_rdd1 = spark.read.csv(filename1, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1").rdd
    tree_rdd2 = spark.read.csv(filename2, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1").rdd
    park_rdd1 = tree_rdd1.map(lambda x: x[6]).filter(lambda x: x is not None)
    park_rdd2 = tree_rdd2.map(lambda x: x[6]).filter(lambda x: x is not None)
    trees_in_common = park_rdd1.intersection(park_rdd2).sortBy(lambda x: x[0]).collect()
    return "".join([park+"\n" for park in trees_in_common])

'''
SPARK DATAFRAME IMPLEMENTATION

You will now re-implement all the tasks above using Apache Spark's 
DataFrame API (see documentation at 
https://spark.apache.org/docs/latest/sql-programming-guide.html). 
Outputs must be identical to the ones obtained above in plain Python. 
Note: all operations must be re-implemented using the DataFrame API, 
you are not allowed to simply convert results obtained with the RDD API 
to Data Frames. Note that the function *toCSVLine* in the HELPER 
section at the top of this file also converts DataFrames into CSV 
strings.
'''

# DataFrame functions

def count_df(filename):
    '''
    Write a Python script using DataFrames that prints the number of trees
    (non-header lines) in the data file passed as first argument.
    Test file: tests/test_count_df.py
    Note: The return value should be an integer
    '''

    spark = init_spark()

    parks_dataframe = spark.read.csv(filename, header=True, mode="DROPMALFORMED")
    return parks_dataframe.count()

def parks_df(filename):
    '''
    Write a Python script using DataFrames that prints the number of trees that are *located in a park*.
    To get the park location information, have a look at the *Nom_parc* column (name of park).
    Test file: tests/test_parks_df.py
    Note: The return value should be an integer
    '''

    spark = init_spark()
    parks_dataframe = spark.read.csv(filename, header=True, mode="DROPMALFORMED")
    return parks_dataframe.select("Nom_parc").where("Nom_parc IS NOT NULL").count()

def uniq_parks_df(filename):
    '''
    Write a Python script using DataFrames that prints the list of unique parks
    where trees were treated. The list must be ordered alphabetically. Every
    element in the list must be printed on a new line.
    Test file: tests/test_uniq_parks_df.py
    Note: The return value should be a CSV string
    '''

    spark = init_spark()
    parks_dataframe = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1")
    temp = parks_dataframe.select("Nom_parc").where("Nom_parc IS NOT NULL").distinct().orderBy("Nom_parc")
    return toCSVLine(temp)

def uniq_parks_counts_df(filename):
    '''
    Write a Python script using DataFrames that counts the number of trees
    treated in each park and prints a list of "park,count" pairs in a CSV
    manner ordered alphabetically by the park name. Every element in the list
    must be printed on a new line.
    Test file: tests/test_uniq_parks_counts_df.py
    Note: The return value should be a CSV string
          Have a look at the file *tests/list_parks_count.txt* to get the exact return format.
    '''

    spark = init_spark()
    parks_dataframe = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1")
    temp = parks_dataframe.select("Nom_parc").where("Nom_parc IS NOT NULL").groupBy("Nom_parc").count().orderBy("Nom_parc")
    return toCSVLine(temp)

def frequent_parks_count_df(filename):
    '''
    Write a Python script using DataFrames that prints the list of the 10 parks
    with the highest number of treated trees. Parks must be ordered by
    decreasing number of treated trees and by alphabetical order when they have
    similar number.  Every list element must be printed on a new line.
    Test file: tests/test_frequent_parks_count_df.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/frequent.txt* to get the exact return format.
    '''

    spark = init_spark()
    parks_dataframe = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1")
    temp = parks_dataframe.select("Nom_parc").where("Nom_parc IS NOT NULL").groupBy("Nom_parc").count().sort(["count", "Nom_parc"], ascending=[False, True])
    return toCSVLine(temp.limit(10))

def intersection_df(filename1, filename2):
    '''
    Write a Python script using DataFrames that prints the alphabetically
    sorted list of parks that had trees treated both in 2016 and 2015. Every
    list element must be printed on a new line.
    Test file: tests/test_intersection_df.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/intersection.txt* to get the exact return format.
    '''

    spark = init_spark()

    trees_df1 = spark.read.csv(filename1, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1")
    trees_df2 = spark.read.csv(filename2, header=True, mode="DROPMALFORMED", encoding="ISO-8859-1")

    parks_df1 = trees_df1.select("Nom_parc").where("Nom_parc IS NOT NULL")
    parks_df2 = trees_df2.select("Nom_parc").where("Nom_parc IS NOT NULL")

    return toCSVLine(parks_df1.intersect(parks_df2).orderBy("Nom_parc"))

'''
DASK IMPLEMENTATION (bonus)

You will now re-implement all the tasks above using Dask (see 
documentation at http://docs.dask.org/en/latest). Outputs must be 
identical to the ones obtained previously. Note: all operations must be 
re-implemented using Dask, you are not allowed to simply convert 
results obtained with the other APIs.
'''

# Dask functions

def count_dask(filename):
    '''
    Write a Python script using Dask that prints the number of trees
    (non-header lines) in the data file passed as first argument.
    Test file: tests/test_count_dask.py
    Note: The return value should be an integer
    '''
    trees = df.read_csv(filename)
    return len(trees)

def parks_dask(filename):
    '''
    Write a Python script using Dask that prints the number of trees that are *located in a park*.
    To get the park location information, have a look at the *Nom_parc* column (name of park).
    Test file: tests/test_parks_dask.py
    Note: The return value should be an integer
    '''

    trees = df.read_csv(filename, dtype = {"Nom_parc": str})
    parks =  trees.dropna(how="all", subset=["Nom_parc"])
    return len(parks)
    

def uniq_parks_dask(filename):
    '''
    Write a Python script using Dask that prints the list of unique parks
    where trees were treated. The list must be ordered alphabetically. Every
    element in the list must be printed on a new line.
    Test file: tests/test_uniq_parks_dask.py
    Note: The return value should be a CSV string
    '''
    trees = df.read_csv(filename, dtype = {"Nom_parc": str},  encoding="ISO-8859-1")
    parks =  trees.dropna(how="all", subset=["Nom_parc"]).drop_duplicates(subset=["Nom_parc"]).sort_values("Nom_parc")
    return "".join([park+"\n" for park in parks.Nom_parc])

#pytest tests/test_uniq_parks_counts_dask.py
def uniq_parks_counts_dask(filename):
    '''
    Write a Python script using Dask that counts the number of trees
    treated in each park and prints a list of "park,count" pairs in a CSV
    manner ordered alphabetically by the park name. Every element in the list
    must be printed on a new line.
    Test file: tests/test_uniq_parks_counts_dask.py
    Note: The return value should be a CSV string
          Have a look at the file *tests/list_parks_count.txt* to get the exact return format.
    '''

    trees = df.read_csv(filename, dtype = {"Nom_parc": str},  encoding="ISO-8859-1")
    uniq = trees["Nom_parc"].dropna().value_counts(sort=False).compute()  
    return "".join([park+"," + str(uniq[park]) + "\n" for park in uniq.index])

def frequent_parks_count_dask(filename):
    '''
    Write a Python script using Dask that prints the list of the 10 parks
    with the highest number of treated trees. Parks must be ordered by
    decreasing number of treated trees and by alphabetical order when they have
    similar number.  Every list element must be printed on a new line.
    Test file: tests/test_frequent_parks_count_dask.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/frequent.txt* to get the exact return format.
    '''

    trees = df.read_csv(filename, dtype = {"Nom_parc": str},  encoding="ISO-8859-1")
    uniq = trees["Nom_parc"].dropna().value_counts().compute() 
    return "".join([park+"," + str(uniq[park]) + "\n" for park in uniq.index[:10]])

def intersection_dask(filename1, filename2):
    '''
    Write a Python script using Dask that prints the alphabetically
    sorted list of parks that had trees treated both in 2016 and 2015. Every
    list element must be printed on a new line.
    Test file: tests/test_intersection_dask.py
    Note: The return value should be a CSV string.
          Have a look at the file *tests/intersection.txt* to get the exact return format.
    '''
    
    trees1 = df.read_csv(filename1, dtype = {"Nom_parc": str, 'No_Civiq': 'object'}, encoding="ISO-8859-1")
    trees2 = df.read_csv(filename2, dtype = {"Nom_parc": str, 'No_Civiq': 'object'}, encoding="ISO-8859-1")  

    parks1 = trees1.dropna(how="all", subset=["Nom_parc"]).drop_duplicates(subset=["Nom_parc"])
    parks2 = trees2.dropna(how="all", subset=["Nom_parc"]).drop_duplicates(subset=["Nom_parc"])
    inter_parks = parks1.Nom_parc.to_frame().merge(parks2.Nom_parc.to_frame()).compute()
    inter_parks = inter_parks.sort_values("Nom_parc")
    return "".join([park+"\n" for park in inter_parks["Nom_parc"]])