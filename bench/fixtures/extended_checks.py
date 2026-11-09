"""Additional synthetic coding tasks; validation uses the retained fixture runner."""

TASKS = [
    ('interval_merge',
     'Implement merge_intervals(intervals). Return sorted merged closed intervals as '
     'a list of tuples. Merge overlaps and touching endpoints. Accept any iterable '
     'of pairs, reject start > end with ValueError, preserve the input. Empty input '
     'returns []. Standard library only. Return a complete Python module.',
     '''items=[(5,7),(1,3),(3,5),(10,10)]; original=items[:]
assert merge_intervals(iter(items))==[(1,7),(10,10)]
assert items==original
assert merge_intervals([])==[]
assert merge_intervals([(2,2),(2,2)])==[(2,2)]
try:
    merge_intervals([(3,1)])
except ValueError:
    pass
else:
    raise AssertionError("reversed interval accepted")'''),
    ('dependency_order',
     'Implement dependency_order(dependencies), taking a dict mapping task names '
     'to iterable prerequisite names. Include prerequisites absent as dict keys. '
     'Return a topological ordering, always choosing the lexicographically smallest '
     'currently ready name. Deduplicate edges, reject cycles with ValueError, '
     'do not mutate inputs. Empty mapping returns []. Standard library only. '
     'Return a complete Python module.',
     '''d={"deploy":["test","build","test"],"test":["build"],"lint":[]}
assert dependency_order(d)==["build","lint","test","deploy"]
assert d["deploy"]==["test","build","test"]
assert dependency_order({})==[]
assert dependency_order({"a":iter(["b"])})==["b","a"]
for bad in ({"a":["a"]},{"a":["b"],"b":["a"]}):
    try:
        dependency_order(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle accepted")'''),
    ('json_merge_patch',
     'Implement merge_patch(target, patch) with JSON merge-patch semantics: a '
     'non-dict patch replaces the target; a dict patch treats a non-dict target '
     'as {}, deletes keys whose patch value is None, and recursively merges '
     'other values. Never mutate or alias mutable input containers in the result, '
     'including untouched target branches and replacement lists. Inputs contain '
     'only JSON types. Standard library only, implement copying locally without '
     'imports. Return a complete Python module.',
     '''target={"a":{"x":1,"y":2},"keep":[{"v":3}],"drop":4}
patch={"a":{"x":None,"z":5},"drop":None,"new":[1,2]}
result=merge_patch(target,patch)
assert result=={"a":{"y":2,"z":5},"keep":[{"v":3}],"new":[1,2]}
result["keep"][0]["v"]=99; result["new"].append(3)
assert target["keep"]==[{"v":3}] and patch["new"]==[1,2]
assert target["a"]=={"x":1,"y":2} and target["drop"]==4
assert merge_patch([1],{"x":1})=={"x":1}
assert merge_patch({"x":1},None) is None
assert merge_patch({}, {"missing":None})=={}'''),
]
