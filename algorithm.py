from collections import defaultdict
from typing import List


class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        opt, opt1 = '', ''
        point = {'ab': x, 'ba': y}
        if x > y:
            opt = 'ab'
            opt1 = 'ba'
        else:
            opt = 'ba'
            opt1 = 'ab'
        stack = []
        ret = 0
        for i in range(len(s)):
            if len(stack) > 0 and stack[-1] + s[i] == opt:
                ret += point.get(opt, 0)
                stack.pop(-1)
            else:
                stack.append(s[i])

        stk = []
        for c in stack:
            if len(stk) > 0 and stk[-1] + c == opt1:
                ret += point.get(opt1, 0)
                stk.pop(-1)
            else:
                stk.append(c)
        return ret


if __name__ == '__main__':
    s = Solution()
    print(s.maximumGain("cdbcbbaaabab", 4, 5))

# aabbaaxybbaabb
# abaaxybbaabb 4
# aaxybbaabb 9
# aaxybabb 14
