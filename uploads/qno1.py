def sum_of_elements(arr):
    total = 0
    for num in arr:   # Loop runs n times → O(n)
        total += num
    return total

# Example usage
arr = [1, 2, 3, 4, 5]
print("Sum:", sum_of_elements(arr))