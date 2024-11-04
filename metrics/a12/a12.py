class Rx:
    """has the nums of a treatment, its name and rank"""

    def __init__(self, lst):
        self.rx, self.lst = lst[0], lst[1:]
        self.mean = sum(self.lst) / len(self.lst)
        self.rank = 0

    def __repr__(self):
        return 'rank #%s %s at %s' % (self.rank, self.rx, self.mean)


def a12s(lst, rev=True, enough=0.66):
    """sees if lst[i+1] has rank higher than lst[i]
        TC comment:
        Input:
            lst - a list of compared lists (compared methods, such as oob, our); for the compared lists,
            the first item is the method name (oob), the other item are the compared values (gmean).
            rev - if True, larger value is better; if False, smaller value is better
            enough - used to divide rank (when effect size reach this value, it will be divided to a new group)
        Output:
            The output of this method will rank the compared methods with its average value.
        Note:
            The effect size will not be output, it is calculated on a12 method.
    """
    lst = [Rx(one) for one in lst]
    lst = sorted(lst, key=lambda x: x.mean, reverse=rev)
    one = lst[0]
    rank = one.rank = 1
    for two in lst[1:]:
        if a12(one.lst, two.lst, rev) > enough: rank += 1
        two.rank = rank
        one = two
    return lst


def a12(lst1, lst2, rev=True):
    """how often is x in lst1 more than y in lst2?
        TC comment:
        Input:
            lst1,lst2 - Two compared method
            rev - if True, calculate how often is lst1 > lst2; if False, calculate how often is lst1 < lst2
        Output:
            This method returns the effect size.
    """
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif rev and x > y:
                more += 1
            elif not rev and x < y:
                more += 1
    return (more + 0.5 * same) / (len(lst1) * len(lst2))


def fromFile(f="a12.dat", rev=True, enough=0.66):
    """utility for reading sample data from disk"""
    import re
    cache = {}
    num, space = r'^\+?-?[0-9]', r'[ \t\n]+'
    for line in open(f):
        line = line.strip()
        if line:
            for word in re.split(space, line):
                if re.match(num, word[0]):
                    cache[now] += [float(word)]
                else:
                    now = word
                    cache[now] = [now]
    return a12s(cache.values(), rev, enough)


if __name__ == "__main__":
    """a12s example"""
    a = ["oob", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = ["our", 0, 0.9, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    c = ["test", 0, 0, 0, 0, 0, 1.1, 1.1, 1.1, 1.1, 1.1]
    a12_list = [a, c, b]
    print(a12s(a12_list))

    """a12 example"""
    a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = [0, 0.9, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    print(a12(a, b))
