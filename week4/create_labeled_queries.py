import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

def convert_query(query):
    # use nltk tokenizer to remoe punctuations
    return " ".join(list(map(lambda wd : stemmer.stem(wd), tokenizer.tokenize(query.lower()))))

def category_rollup(category, category_chain):
    if category not in cat_parent_dict:
        print("{} does not have parent".format(category))
        return category_chain

    parent = cat_parent_dict[category]
    if parent == root_category_id:
        return category_chain

    new_chain = category_chain.copy()
    new_chain.append(parent)
    total_count = query_count.loc[query_count['category'].isin(new_chain), 'count'].sum()
    
    if total_count >= min_queries:
        return new_chain
    else:
        return category_rollup(parent, new_chain)
        

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])

parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
df['query'] = df['query'].apply(lambda q: convert_query(q))
print('### Finished query nomralization')

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
query_count = df.groupby('category').size().reset_index(name='count')

low_query_category = query_count.loc[query_count['count'] < min_queries]['category']
cat_parent_dict = dict(zip(parents_df.category, parents_df.parent))
# print(len(low_query_category))

rollup_dict = {}
for cat in low_query_category:
    cat_count = query_count.loc[query_count['category'] == cat, 'count'].sum()
    category_chain = [cat]
    
    # get the rollup chain where the last element is the final category, all previous categories need to be replaced with this category
    category_rollup_chain = category_rollup(cat, category_chain)
    chain_count = query_count.loc[query_count['category'].isin(category_rollup_chain), 'count'].sum()
    
    # construct the category rollup dictionary
    for child_cat in category_rollup_chain[:-1]:
        rollup_dict[child_cat] = category_rollup_chain[-1]

print('### Finished rollup dictionary construction')

# inplace roll up categories
df.replace({'category': rollup_dict}, inplace=True)
print('### Finished inplace category rollup')

# new_count = df.groupby('category').size().reset_index(name='count')
# print(len(new_count.loc[query_count['count'] < min_queries]['category']))

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
