from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#training_data =  [
#    ['Green', 3, 'Apple'],
#    ['Yellow', 3, 'Apple'],
 #   ['Red', 1, 'Grape'],
 #   ['Red', 1, 'Grape'],
 #   ['Yellow', 3, 'Lemon'],
#]

df = pd.read_csv('iris.csv', header = None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
#print(t.true_branch.true_branch.class_label)
#p = print_leaf(classify(training_data[0], t))
#print(p)

#inner = getInnerNodes(t)
#print(inner)
#leaf = getLeafNodes(t)
#print(leaf)
#p = computeAccuracy(lst,t)
#print(p)
#print("********** Leaf nodes ****************")
#leaves = getLeafNodes(t)
#print(len(leaves))
#for leaf in leaves:
#    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
#print("********** Non-leaf nodes ****************")
#innerNodes = getInnerNodes(t)
#for inner in innerNodes:
#    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

#trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
#train = trainDF.values.tolist()
#test = testDF.values.tolist()

#t = build_tree(train, header)
#print("*************Tree before pruning*******")
#print_tree(t)
#acc = computeAccuracy(test, t)
#print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
#t_pruned = prune_tree(t, [26, 11, 5])

#print("*************Tree after pruning*******")
#print_tree(t_pruned)
#acc = computeAccuracy(train, t)
#print("Accuracy on train = " + str(acc))
