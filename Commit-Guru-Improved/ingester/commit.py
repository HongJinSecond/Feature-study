import copy

class Commit:
    def __init__(self) -> None:
        self.hash_code=''
        self.parent_hash=None
        self.file_dict=None
        self.Is_merge=False
        self.Is_branch_start=False
        #   This list records the subsequent nodes of this node
        self.followCommit=[]
        #   To count how many copy file it has, when copy count equals the len of followCommit
        #that means all the children of this node have already gained the data, that means, this
        #node can be delete!
        self.copy_count=0
        #   this property record whether the node can be delete



    def get_dict(self)->dict:
    #   I am not sure whether this operation would introduce the memory overflow or not?
    #   I am comssuing a method to reduce and recicle the memory.
        if self.Is_branch_start==True:
            return copy.deepcopy(self.file_dict)
        else:
            return self.file_dict
        
    
    def update_dict(self,file_dict:dict):
        """
        pass the file dictionary to its child node.
        """
        if self.copy_count==len(self.followCommit):
            file_dict.clear()
            return
        if self.Is_merge==True:
            self.file_dict=copy.deepcopy(file_dict)
        else:
            self.file_dict=file_dict

    def release_memory(self):
        """
        release the memory, you can execute this function whenever u want.
        """
        self.copy_count+=1
        if self.copy_count==len(self.followCommit):
            if self.Is_branch_start:
                self.file_dict.clear()
            self.file_dict=None
