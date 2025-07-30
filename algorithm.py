from collections import defaultdict
from typing import List


class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        max_value = 0
        subset_count = 0
        for right in range(len(nums)):
            cur_value = 0
            for left in range(right+1):
                cur_value = cur_value | nums[left]
                if cur_value > max_value:
                    max_value = cur_value
                    subset_count = 1
                elif cur_value == max_value:
                    subset_count += 1
        return subset_count




if __name__ == '__main__':
    s = Solution()
    print(s.countMaxOrSubsets([3, 1]))
