# minimum implementation of merkle commitment scheme on *list of raw bytes*, uses sha256
import hashlib
import math
from typing import Optional


class LeafNode:

    def __init__(self, file: bytes, idx: int, isPad: bool) -> None:
        sha256 = hashlib.sha256()
        sha256.update(b"leaf:")
        sha256.update(file)
        self.preimage = file
        self.data = sha256.digest()
        self.Proof: list = []
        self.idx = idx
        self.isPad = isPad
        self.parent: InternalNode | None = None

    def __str__(self) -> str:
        return str(self.data)



class InternalNode:

    def __init__(self, left: "LeafNode | InternalNode", right: "LeafNode | InternalNode", idx: int) -> None:
        self.data: bytes = b""
        self.left = left
        self.right = right
        self.parent: InternalNode | None = None
        self.idx = idx
        left.parent = self
        right.parent = self

    def __str__(self) -> str:
        return str(self.data)

    def updateHash(self) -> bytes:
        if not (self.left and self.right):
            raise ValueError("Internal node must have both left and right children")
        sha256 = hashlib.sha256()
        sha256.update(b"node:")
        sha256.update(self.left.data)
        sha256.update(self.right.data)
        self.data = sha256.digest()
        return self.data


class MerkleTree:

    def __init__(self, files: list[bytes]) -> None:
        data = files
        self.numFiles = len(files)
        height = math.ceil(math.log(len(files),2))
        self.h = height
        padlen = (2**height)-len(files)
        data += [b"\x00"] * padlen

        self.leafs = []
        for i in range(len(data)):
            self.leafs.append(LeafNode(data[i] ,i ,i >= self.numFiles))

        self.commit()


    def commit(self) -> bytes:

        current: list[LeafNode | InternalNode] = list(self.leafs)
        for h in range(self.h):
            parents: list[LeafNode | InternalNode] = []
            for i in range(0,len(current),2):
                n = InternalNode(current[i],current[i+1],i//2)
                n.updateHash()
                parents.append(n)
            current = parents
        self.root = current[0]

        return self.root.data


    def appendLeaf(self, newFile: bytes) -> tuple[int, bytes]:
        #return idx and updated merkle root.
        idx = 0
        if self.numFiles < len(self.leafs):
            #padding available.
            idx = self.numFiles
            if not self.leafs[idx].isPad:
                raise ValueError(f"Expected padding node at index {idx}")
            self.leafs[idx] = LeafNode(newFile,idx,False)

        else:
            idx = len(self.leafs)
            self.leafs.append(LeafNode(newFile ,idx ,False))
            #padd the tree to nearest power of 2
            height = math.ceil(math.log(len(self.leafs),2))
            self.h = height
            padlen = (2**height)-len(self.leafs)
            for i in range(1,padlen+1):
                self.leafs.append(LeafNode(b"\x00",idx+i,True))

        self.numFiles += 1
        r = self.commit()
        return (idx,r)

    def updateLeaf(self, idx: int, update: bytes) -> tuple[int, bytes]:
        if idx >= self.numFiles:
            raise IndexError(f"Index {idx} out of range (numFiles={self.numFiles})")
        self.leafs[idx] = LeafNode(update,idx,False)
        r = self.commit()
        return (idx,r)

    def deleteLeaf(self, idx: int) -> None:
        # can be done by using nullifier b"\x00" and queue indexing, not implemented
        # the file system does not support delete for now
        pass

    def open(self, idx: int) -> tuple[list, bytes]:
        current: LeafNode | InternalNode = self.leafs[idx]
        path: list[bytes] = []
        while current.parent:
            if current.idx % 2 == 0:
                #left
                path.append(current.parent.right.data)
            else:
                #right
                path.append(current.parent.left.data)
            current = current.parent
        self.leafs[idx].Proof = path
        return path, self.leafs[idx].preimage

    def verify(self, idx: int, proof: list) -> None:
        #user should implement merkle verify themselves to check the validity of the merkel proof
        pass
