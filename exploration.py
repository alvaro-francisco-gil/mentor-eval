from mentoreval import MentorEvalBenchmark, MentorEvalTasks, MentorEvalTask, MentorEvalDataset  
from deepeval.models import GPTModel  
  
# Evaluate specific exercise sets  
specific_tasks = [  
    MentorEvalTask(MentorEvalDataset.ASAP, 1),  
    MentorEvalTask(MentorEvalDataset.ASAP, 2),  
    MentorEvalTask(MentorEvalDataset.ASAP2, 1)  
]  
  
benchmark = MentorEvalBenchmark(  
    tasks=specific_tasks,  
    n_problems_per_task=3,  
    verbose_mode=True  
)  

"""  
# Or evaluate all ASAP tasks  
benchmark_asap = MentorEvalBenchmark(  
    tasks=MentorEvalTasks.get_all_asap_tasks(),  
    use_test_set=True  
)  
"""
  
model = GPTModel(model="gpt-4")  
score = benchmark.evaluate(model)