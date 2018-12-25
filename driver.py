from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['Age','Year of Operation',
  '#Positive Auxillary nodes',
  'Survival status']
df = pd.read_csv('http://mlr.cs.umass.edu/ml/machine-learning-databases/haberman/haberman.data', header=None, names=header)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

pruneList = [] #Prune list to select all the nodes 1 level above the leaf

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    pruneList.append((leaf.id-1)//2) # Adding all the ids of nodes 1 level above leaf
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## Pruning strategey is to prune all the nodes 1 level above the leaf nodes
t_pruned = prune_tree(t, pruneList)


print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
