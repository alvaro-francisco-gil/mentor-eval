from enum import Enum  
from typing import List, Tuple  
  
class MentorEvalDataset(Enum):  
    ASAP = "asap"  
    ASAP2 = "asap2"  
    MOHLER = "mohler"  
  
class MentorEvalTask:  
    def __init__(self, dataset: MentorEvalDataset, exercise_set: int):  
        self.dataset = dataset  
        self.exercise_set = exercise_set  
        self.value = f"{dataset.value}_exercise_set_{exercise_set}"  
      
    def __str__(self):  
        return self.value  
      
    def __repr__(self):  
        return f"MentorEvalTask({self.dataset.value}, {self.exercise_set})"  
  
class MentorEvalTasks:  
    @staticmethod  
    def get_all_asap_tasks() -> List[MentorEvalTask]:  
        return [MentorEvalTask(MentorEvalDataset.ASAP, i) for i in range(1, 9)]  
      
    @staticmethod  
    def get_all_asap2_tasks() -> List[MentorEvalTask]:  
        return [MentorEvalTask(MentorEvalDataset.ASAP2, i) for i in range(1, 8)]  
      
    @staticmethod  
    def get_all_tasks() -> List[MentorEvalTask]:  
        # Combine all supported datasets' tasks
        return MentorEvalTasks.get_all_asap_tasks() + MentorEvalTasks.get_all_asap2_tasks()  