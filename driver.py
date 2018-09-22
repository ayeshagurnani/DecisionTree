from DecisionTree import *
import pandas as pd
from sklearn import model_selection

#We have used checked the accuracy on two datasets.
header_tic = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Class'] #First one is the tic_tac_toe dataset.
header_leaves = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class'] #Another is the iris dataset.
header = header_tic
df = pd.read_csv('tic_tac_toe.csv', header=None, names= header)
lst = df.values.tolist()
#t = build_tree(lst, header)
#print_tree(t)

trainDF, testDF = model_selection.train_test_split(df, test_size=0.3, random_state=1)
train = trainDF.values.tolist()
test = testDF.values.tolist()

tree = build_tree(train, header)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(tree)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(tree)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))


print("*************Tree before pruning*******")
#print_tree(tree)
acc = computeAccuracy(test, tree)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(tree, [0])
#print_tree(t_pruned)
print("*************Tree after pruning*******")
#print_tree(t_pruned)
acc1 = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc1))
