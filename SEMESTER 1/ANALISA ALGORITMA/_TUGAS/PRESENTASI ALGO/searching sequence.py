# def pencarian_sequence (dlist,item):
#     row = 0
#     found = False

#     while row < len(dlist) and not found:
#         if dlist[row] ==item:
#             found = True
#         else:
#             row = row + 1
            
#     return found, row
    
# teslis=input("silahakan masukan data")
   
# print(pencarian_sequence(teslis,4))

def pencarian_sequence(dlist, item):
    row = 0
    found = False

    while row < len(dlist) and not found:
        if dlist[row] == item:
            found = True
        else:
            row += 1
            
    return found, row

# Input list manually
teslis = input("Enter a list of numbers separated by spaces: ")
if '' not in teslis:
    print("warning,Input mesto ada koma")
    exit()
teslis.split()
teslis = [int(x) for x in teslis]  # Convert input strings to integers

# Input item to search
item = int(input("Enter the number to search: "))

# Call the function with manual input
print(pencarian_sequence(teslis, item))


     
# print("hi")
# def Sequential_Search(dlist, item):

#     pos = 0
#     found = False
    
#     while pos < len(dlist) and not found:
#         if dlist[pos] == item:
#             found = True
#         else:
#             pos = pos + 1
    
#     return found, pos

# print(Sequential_Search([11,23,58,31,56,77,43,12,65,19],31))