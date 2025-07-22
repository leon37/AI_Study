from collections import defaultdict
from typing import List

class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        ret = nums[0]
        left, right = 0, 0
        record = {}
        curSum = 0
        for right in range(0, len(nums)):
            while left < right and nums[right] in record and record[nums[right]] >= 1:
                curSum -= nums[left]
                record[nums[left]] -= 1
                left += 1
            curSum += nums[right]
            if nums[right] not in record:
                record[nums[right]] = 1
            else:
                record[nums[right]] += 1
            ret = max(ret, curSum)
        return ret


if __name__ == '__main__':
    s = Solution()
    print(s.maximumUniqueSubarray(nums=[4,2,4,5,6]))
