class Node:
    ''' Class for nodes in decision tree '''
    def __init__(self, feature_name='', feature_value=None):
        self.feature_name = feature_name # head if dummy head node, otherwise is feature or label value
        self.feature_value = feature_value # head if dummy head node, otherwise is feature or label value
        self.children = None # list of nodes

def print_tree(node, i):
    ''' Helper function that prints decision tree.
        Inputs:
            * node: root node of decision tree to print
            * i: number of tabs to offset each layer of decision tree
                 when printing.
    '''
    if node.children is None:
        tabs = i * '\t'
        print(tabs + str(node.feature_value))
    else:
        tabs = i * '\t'
        print(tabs + str(node.feature_name) + ':' + str(node.feature_value))
        for child in node.children:
            print_tree(child, i + 1)

"""
HELPER FUNCTIONS
"""
random.seed(1234)

def get_children_from_fvals(df, f):
    ''' Inputs:
            * df: dataframe containing data
            * f: name of current feature being considered
        Output:
            * List of child nodes from feature values for f
    '''
    fvals = df[f].unique()
    return [Node(f, v) for v in fvals]

def get_df_num_rows(df):
    ''' Returns number of rows in dataframe '''
    return len(df)

def get_df_subset(df, f, fval):
    ''' Inputs:
            * df: dataframe containing data
            * f: name of current feature being considered
            * fval: one value that f can take on
        Output:
            * Dataframe comprising rows of df for which f's value is fval
    '''
    df_fval = df[ df[f] == fval ]
    num_fval = get_df_num_rows(df_fval)
    return df_fval, num_fval

def get_df_num_labels(df, label):
    ''' Inputs:
            * df: dataframe containing data
            * label: column name in df to use as label
        Output:
            * Dictionary, where keys are label value in df_val and values the
            number of rows in df with that label
    '''
    num_label = df[label].value_counts().to_dict()
    return num_label

def information_gain(df, features, label):
    ''' Inputs:
            * df: dataframe containing data
            * features: current features to consider
            * label: column name in df to use as label
        Output:
            * Feature name on which to split, i.e., feature with the maximum
              information gain
    '''

    # How many rows in dataframe are there? Each row is an instance
    num_instances = get_df_num_rows(df)
    min_entropy = None
    split_on = None

    # Iterate through each feature/category to determine
    # which gives maximum information gain
    for f in features:
        sum_entropy = 0

        # Get all values of the current feature to split the dataset
        for fval in df[f].unique():

            # Get rows of dataframe for which f's value is fval
            df_fval, num_fval = get_df_subset(df, f, fval)

            # Get counts for each possible label value in df_val
            num_label_fval = get_df_num_labels(df_fval, label)

            # Calculate entropy
            entropy = 0
            for _, num_label in num_label_fval.items():
                prob = num_label/num_fval
                entropy += - (prob)*np.log2(prob)
            sum_entropy += (num_fval/num_instances) * entropy

        # Get feature with minimum entropy sum
        # because this feaure has the maximum info gain
        if min_entropy is None or sum_entropy < min_entropy:
            min_entropy = sum_entropy
            split_on = f

    return split_on

"""

ID3 Algorithm

"""

root_node = Node('root')

def ID3_build_tree(df, features, label, parent, max_depth):
    ''' Inputs
            * df: dataframe containing data
            * features: current features to consider
            * label: column name in df to use as label
            * parent: of type Node
        Output
            * a decision tree
    '''

    #####
    # Todo: update this function to stop once max depth hit
    #####
    depth_count = max_depth
    # if all features have been used, return most popular label
    # if there is only one label, also return
    if len(features) == 0 or len( df[label].unique() ) == 1 or depth_count == 0:

        # Get most frequent label using mode
        leaf = df[label].mode()[0]

        # Set children to be leaf node containing most frequent label
        parent.children = [Node(label, leaf)]

    # Otherwise, continue to recurse down the tree
    else:

        # Get feature with the most info gain
        feature_to_split_on = information_gain(df, features, label)

        # Values from that feature become the children of the previous node
        parent.children = get_children_from_fvals(df, feature_to_split_on)

        # Remove the feature we just split on so we don't try to split on it again
        new_features = features.copy()
        new_features.remove(feature_to_split_on)

        # Recursively call build_tree on each child node
        for child in parent.children:
            ID3_build_tree(df[ df[feature_to_split_on] == child.feature_value ],
                    new_features, label, child,max_depth-1)
        depth_count -= 1

def ID3_decision_tree(df, features, label, max_depth=5, random_subspace = None):
    ''' Inputs
            * df: dataframe containing data
            * features: list of current features to consider
            * label: column name in df to use as label
            * max_depth: max depth of tree
            * random_subspace: number of random features to be chosen
        Output
            * dtree: root node of trained decision tree
    '''
    # get the number of columns of the dataframe
    _, n_col = df.shape
    # get the indeces of the columns excluding target column
    n_col_indeces = list(range(n_col-1))
    # check if random
    if random_subspace != None:
      n_col_indeces = random.sample(population = n_col_indeces, k = random_subspace)
    # initialize empty list of random features
    random_features = []
    # append randomly chosen features to the random_features list
    for index in n_col_indeces:
      random_features.append(features[index])

    dtree = Node('root', '')
    ID3_build_tree(df, random_features, label, dtree, max_depth)
    return dtree

def ID3_get_prediction(tree, example):
    ''' Input:
    * tree: a decision tree
    * example: example from the datafreame
    Output:
    * a label
    '''
  if tree.children is None:
    # print("one")
    return tree.feature_value
  else:
    for child in tree.children:
      fval = example[child.feature_name]
      # print(fval)
      # print("child.feature name:" + child.feature_name)
      if child.feature_value == fval:
        # print("two")
        return ID3_get_prediction(child,example)
      elif child.children is None:
        # print("three")
        return ID3_get_prediction(child,example)

def ID3_decision_tree_all(tree, df):
    '''Input:
    * tree: a decision tree
    * df: a data frame
    Output:
    * predictions: a list of predictions for each example using a decision tree
    '''
  # get number of examples
  num_examples = df.shape[0]
  # initialize empty list of predictions
  predictions = []
  # loop through all of the examples
  for example in range(num_examples):
    # get prediction for one example
    pred = ID3_decision_tree_prediction(tree,df.iloc[example])
    # append to list of predictions
    predictions.append(pred)
  return predictions
