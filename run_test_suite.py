import checklist
from checklist.test_suite import TestSuite
import pickle
import dill

with open('squad_suite.pkl', 'rb') as pickle_file:
    tests = pickle.load(pickle_file)

print(tests)

# suite_path = 'release_data/squad/squad_suite.pkl'
# suite = TestSuite.from_file(suite_path)

# print("BUILT SUITE")
# pred_path = 'eval_results/eval_outputv100t100/predictions.jsonl'
# suite.run_from_file(pred_path, overwrite=True, file_format='pred_only')
# suite.visual_summary_table()

