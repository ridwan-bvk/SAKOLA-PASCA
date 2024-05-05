def pencarian_sequence(dlist, item):
    found = False
    row = None  # Set row to None to indicate that no index is found

    for index, value in enumerate(dlist):
        if value == item:
            found = True
            row = index
            break  # Exit the loop after the element is found

    return found, row

# Input list secara manual
teslis = input("Enter a list of items separated by spaces/comma: ")

# Check if there are spaces or commas as separators
if ' ' in teslis:
    delimiter = ' '
elif ',' in teslis:
    delimiter = ','
else:
    print("Warning: Input should be separated by spaces or commas.")
    exit()

# Split the input into elements
teslis = teslis.split(delimiter)

# Input item to search
item = input("Enter the item to search: ")

# Call the pencarian_sequence function with manual input
found, index = pencarian_sequence(teslis, item)

if found:
    print(f"Item '{item}' found at index: {index}")
else:
    print("Item not found in the list.")
