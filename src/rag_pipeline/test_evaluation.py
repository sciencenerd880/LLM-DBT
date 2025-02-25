'''
reference: https://docs.confident-ai.com/guides/guides-rag-evaluation
'''

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval import evaluate



def test_answer_relevancy():
    # initialize the metric
    ARM = AnswerRelevancyMetric(threshold=0.8)
    # CPM = ContextualPrecisionMetric(threshold=0.5)
    # CRM = ContextualRelevancyMetric(threshold=0.5)
    # defines a test case 
    test_case = LLMTestCase(
        input="what if these shoes doesnt fit",
        # actual_output="we offer a 30 day full refund at no extra cost.",
        actual_output="hi i like genai",
        retrieval_context = ["All the customers are eligilble for a 30 day full refund at no cost"]
    )
    evaluate([test_case], 
                [ARM])