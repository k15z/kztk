"""
Detect idea reuse / plagiarism in a sequence of documents.
"""
import difflib
import multiprocessing as mp

def _max(arr):
    arr = list(arr) # maybe not a good idea?
    return max(arr) if len(arr) > 0 else 0

def _mean(arr):
    arr = list(arr) # maybe not a good idea?
    return sum(arr) / len(arr) if len(arr) > 0 else 0.0

class Plagiarist(object):
    """
    Given a sequence of documents, produce scores indicating how much "plagiarism" (i.e. how many 
    ideas were borrowed) from each of the preceding documents. Note that getting a high score 
    doesn't really mean there was plagiarism since citations, etc. are not analyzed.
    """

    def observe(self, document):
        raise NotImplementedError()

def score(matcher, document, top_N):
    matcher.set_seq1(document)
    lengths = [n for i, j, n in matcher.get_matching_blocks()]
    if len(lengths) > top_N:
        lengths = list(sorted(lengths))[-top_N:] # top N substrings
    return {
        "total": sum(lengths),
        "average": _mean(lengths)
    }

class LCSPlagiarist(Plagiarist):
    """
    Find the N longest common substrings in previous documents and compute max/mean statistics.

    Example:
        from kztk.plagiarist import LCSPlagiarist

        lcsp = LCSPlagiarist()
        print(lcsp.observe("hello world"))
        print(lcsp.observe("hello world 2")) # high score
        print(lcsp.observe("this is a novel idea")) # low score
    """

    def __init__(self, top_N=1, aggregate=True):
        self.top_N = top_N
        self.aggregate = aggregate
        self.matchers = []
        self.pool = mp.Pool()

    def observe(self, document):
        results = []
        for matcher in self.matchers:
            results.append(self.pool.apply_async(score, (matcher, document, self.top_N)))

        scores = [res.get() for res in results]
        self.matchers.append(difflib.SequenceMatcher(b=document))

        if self.aggregate:
            return {
                "max_total": _max(map(lambda x: x["total"], scores)),
                "max_average": _max(map(lambda x: x["average"], scores)),
                "mean_total": _mean(map(lambda x: x["total"], scores)),
                "mean_average": _mean(map(lambda x: x["average"], scores))
            }
        return scores
