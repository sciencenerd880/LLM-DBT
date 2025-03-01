"""
USes DeepEval framework to synthetically generate data from documents
and store in local and also pushed to cloud (refer to confident AI dashboard)
"""


from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from deepeval.dataset import EvaluationDataset


import pandas as pd

"""
Initialization
"""
DOCUMENT_PATHS = ["././data/test_data/fy2025_budget_statement.pdf"]
synthesizer = Synthesizer()
DATASET_ALIAS = "FY25_BUDGET_SPEECH_GOLDENS"

""""
Generation of goldens and samples
"""
# refer to this link for detailed customization https://docs.confident-ai.com/docs/synthesizer-generate-from-docs
synthesizer.generate_goldens_from_docs(
    document_paths = DOCUMENT_PATHS,
    include_expected_output=True,
    context_construction_config=ContextConstructionConfig(
        context_quality_threshold=0.5, #default is 0.5
        max_contexts_per_document=20 #default is 3, determines total goldens = max contexts per doc * 2 goldens per context
    )
)
print(synthesizer.synthetic_goldens)
print()
df = synthesizer.to_pandas()
print(df)
# df.to_csv("././data/syn_data.csv")

""""
Save the data locally 
"""
synthesizer.save_as(
    file_type='json',
    directory='././data/synthetic_data'
    )

""""
Save the data on cloud / Confident AI @ https://app.confident-ai.com/project/cm7k4hv8d5q5k7advs35m4pl5/datasets/cm7njfvv90frki7dqdd3cg7a3?pageNumber=1&pageSize=50
"""
dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
dataset.push(alias=DATASET_ALIAS
             )