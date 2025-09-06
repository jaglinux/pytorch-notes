import math
def softmax(values):
    exp_list = [math.exp(value) for value in values]
    exp_list_sum = sum(exp_list)
    return [value/exp_list_sum  for value in exp_list]

values = [2, 4, 5, 3]
result = softmax(values)
print("input to softmax ", values)
print("softmax output ", result, sum(result))
maxi = max(result)
max_index = result.index(maxi)
print("Max index is ", max_index)
print("Max in original list is ", values[max_index])

'''
input to softmax  [2, 4, 5, 3]
softmax output  [0.03205860328008499, 0.23688281808991013, 0.6439142598879724, 0.08714431874203257] 1.0
Max index is  2
Max in original list is  5
'''
