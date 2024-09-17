import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import nltk
from nltk.stem import SnowballStemmer
import pandas as pd

stemmer = SnowballStemmer("english")
tokenizer = nltk.RegexpTokenizer(r"\w+")

def transform_name(product_name):
    # use nltk tokenizer to remove punctuations
    return " ".join(list(map(lambda wd : stemmer.stem(wd), tokenizer.tokenize(product_name.lower()))))


# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

general.add_argument("--category_depth", default=100, type=int, help="The maximal category depth to parse, the larger this value the more granularity.")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate
category_depth = args.category_depth
category_data = []

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                    if len(child.find('categoryPath')) < category_depth:
                        cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                    else:
                        cat = child.find('categoryPath')[category_depth - 1][0].text
                    
                    # Replace newline chars with spaces so fastText doesn't complain
                    name = child.find('name').text.replace('\n', ' ')
                    category_data.append(dict([('category', cat), ('name', transform_name(name))]))

    category_data_df = pd.DataFrame(category_data)

    if min_products > 0:
        category_data_df = category_data_df.groupby('category').filter(lambda c: len(c) >= min_products)
    
    for _, row in category_data_df.iterrows():
        output.write("__label__%s %s\n" % (row['category'], row['name']))

