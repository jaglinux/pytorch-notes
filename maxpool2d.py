def maxpool2d(matrix, pool_size):
    result = []
    row_pool, col_pool = pool_size
    for i in range(0, len(matrix)-row_pool+1, row_pool):
        window_max = []
        for j in range(0, len(matrix[0])-col_pool+1, col_pool):
            window = [row[j:j+col_pool]  for row in matrix[i:i+row_pool]]
            maxi = 0
            for row in window:
                maxi = max(maxi, max(row))
            window_max.append(maxi)
        result.append(window_max)
    return result

matrix = [
    [1, 3, 2, 1],
    [4, 6, 5, 2],
    [7, 8, 9, 3],
    [1, 2, 3, 4]
]
result = maxpool2d(matrix, pool_size=(2,2))
print(result)  # Output: [[6, 5], [8, 9]]
