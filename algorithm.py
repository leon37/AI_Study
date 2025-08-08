from collections import defaultdict
from typing import List


class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        ret = 0
        n = len(fruits)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ret += fruits[i][j]

        dp2, dp3 = [[0 for _ in range(n)] for _ in range(n)], [[0 for _ in range(n)] for _ in range(n)]
        dp2[0][n-1] = fruits[0][n-1]
        dp3[n-1][0] = fruits[n-1][0]

        for i in range(1, n):
            for j in range(1, i+2):
                row, col = i, n-j
                if row >= col:
                    continue
                dp2[row][col] = max(dp2[row-1][col], dp2[row-1][col-1])+fruits[row][col]
                if col < n-1:
                    dp2[row][col] = max(dp2[row][col], dp2[row-1][col+1]+fruits[row][col])

        for j in range(1, n):
            for i in range(1, j+2):
                row, col = n-i, j
                if col >= row:
                    continue
                dp3[row][col] = max(dp3[row][col-1], dp3[row-1][col-1])+fruits[row][col]
                if row < n-1:
                    dp3[row][col] = max(dp3[row][col], dp3[row+1][col-1]+fruits[row][col])

        ret += max(dp2[n-2][n-1], dp2[n-2][n-2])+max(dp3[n-1][n-2], dp3[n-2][n-2])
        return ret

if __name__ == '__main__':
    s = Solution()
    print(s.maxCollectedFruits([[1,2,3,4],[5,6,8,7],[9,10,11,12],[13,14,15,16]]))

