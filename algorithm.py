from collections import defaultdict
from typing import List


class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        record = defaultdict(int)
        ret = 0
        for fruit in fruits:
            find = False
            for j in range(len(baskets)):
                if fruit <= baskets[j] and not record[j]:
                    record[j] = True
                    find = True
                    break
            if not find:
               ret += 1
        return ret


