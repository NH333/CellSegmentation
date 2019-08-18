list1 = ['1','2','3']
list2 = ['a','b','c']
list3 = ['A','B','C']
count = 141
list4 = []
d = {}
for i in range(0, len(list1)):
    d[list1[i]]=(list2[i], list3[i]);

for i in range(count):
    list4.append(('%d')%(i+1))

print(d)