#!/usr/bin/env python3
import sys
import h5py

def updateAttr(obj, attrName="name"):
    if attrName not in obj.attrs:
        return False, None, None
    oldVal = obj.attrs[attrName]
    # Handle bytes or str
    if isinstance(oldVal, bytes):
        oldStr = oldVal.decode("utf-8", errors="ignore")
    else:
        oldStr = str(oldVal)

    newStr = oldStr.replace("YDBVector(", "YDBVectorSingle(")
    if newStr == oldStr:
        return False, oldStr, newStr

    obj.attrs[attrName] = newStr
    return True, oldStr, newStr

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.h5>", file=sys.stderr)
        sys.exit(2)

    filePath = sys.argv[1]
    changedCount = 0
    triedCount = 0

    with h5py.File(filePath, "r+") as f:
        # Try root first
        didChange, oldStr, newStr = updateAttr(f, "name")
        if didChange:
            triedCount += 1
            changedCount += 1
            print(f'[root]/attrs["name"]:\n  OLD: {oldStr}\n  NEW: {newStr}')
        elif oldStr is not None:
            triedCount += 1
            print(f'[root]/attrs["name"] unchanged:\n  VAL: {oldStr}')

        # If root didn’t have it or didn’t change, scan all objects for "name" attrs
        def visitor(name, obj):
            nonlocal changedCount, triedCount
            didChange, o, n = updateAttr(obj, "name")
            if o is not None:
                triedCount += 1
                if didChange:
                    changedCount += 1
                    print(f'/{name}/attrs["name"]:\n  OLD: {o}\n  NEW: {n}')
                else:
                    print(f'/{name}/attrs["name"] unchanged:\n  VAL: {o}')

        # Only walk if we didn’t already change root or if you want to catch others too.
        # We’ll walk regardless to catch any additional occurrences.
        f.visititems(visitor)

    if triedCount == 0:
        print('No "name" attributes found.', file=sys.stderr)
        sys.exit(1)

    if changedCount == 0:
        print('Found "name" attribute(s), but nothing required changing.')
    else:
        print(f"Updated {changedCount} attribute(s).")

if __name__ == "__main__":
    main()
