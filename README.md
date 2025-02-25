# Multi-Agent Intelligent LLM for Tackling Business-Pitches
CS606 Generative AI with LLMs

## Setup Instructions

1. Create a new conda environment:
    ```sh
    conda create -n sharktank python=3.10
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Activate the conda environment:
    ```sh
    conda activate sharktank
    ```

4. Create a [.env](http://_vscodecontentref_/1) file at the root directory with the following content:
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    ```

## Running the Inference Script 
0. Make the necessary changes in terms of model, refer to this link https://docs.litellm.ai/docs/providers/text_completion_openai, https://docs.aimlapi.com/api-overview/model-database/text-models?utm_source=aimlapi&utm_medium=github&utm_campaign=integration

Some of the models defined at src/config.py includes "gpt-4", "o1-mini", "o1-preview", "gpt-3.5-turbo"
1. Run the main script:
    ```sh
    python src/main.py
    ```


## Running the RAG process 
1. Run the 'indexer.py' to split the desired pdf into chunks and generate the embeddings into the specified embedding model and store into milvus db collection:
    ```sh
    python src/rag_pipeline/indexer.py
    ```

2. To get the top k results from the milvus client collection based on query, run the following:
    ```sh
    python src/rag_pipeline/retriever.py
    ```


## Starting the Evaluation using DeepEval framework
 1. On your terminal, the UI will be opened after running the following command:
    ```sh
    deepeval login
    ```
2. Sign in accordingly as per recommendation to to use work email for login access. 
3. Paste your API Key: XXXXX. You should see the following messages
    ```sh
🎉🥳 Congratulations! You've successfully logged in! 🙌 
You're now using DeepEval with Confident AI. Follow our quickstart tutorial 
here: https://docs.confident-ai.com/confident-ai/confident-ai-introduction
    ```
4. Run the sample test case via:
    ```sh
deepeval test run src/rag_pipeline/test_evaluation.py
    ```

Note: If you have not set the OpenAI key, then export it on your terminal via: 
    ```sh
export OPENAI_API_KEY=XXXX
    ```

5. This is the expected via of the evaluation:

```sh

Evaluating 1 test case(s) in parallel: |█|100% (1/1) [Time Taken: 00:03,  3.78
.Running teardown with pytest sessionfinish...

============================ slowest 10 durations ============================
3.81s call     src/rag_pipeline/test_evaluation.py::test_answer_relevancy

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
1 passed, 3 warnings in 3.82s
                                 Test Results                                 
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃                ┃                ┃                ┃        ┃ Overall        ┃
┃ Test case      ┃ Metric         ┃ Score          ┃ Status ┃ Success Rate   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ test_answer_r… │                │                │        │ 100.0%         │
│                │ Answer         │ 1.0            │ PASSED │                │
│                │ Relevancy      │ (threshold=0.… │        │                │
│                │                │ evaluation     │        │                │
│                │                │ model=gpt-4o,  │        │                │
│                │                │ reason=The     │        │                │
│                │                │ score is 1.00  │        │                │
│                │                │ because the    │        │                │
│                │                │ actual output  │        │                │
│                │                │ flawlessly     │        │                │
│                │                │ addresses the  │        │                │
│                │                │ question       │        │                │
│                │                │ regarding shoe │        │                │
│                │                │ fit, with no   │        │                │
│                │                │ irrelevant     │        │                │
│                │                │ statements     │        │                │
│                │                │ distracting    │        │                │
│                │                │ from the       │        │                │
│                │                │ topic.         │        │                │
│                │                │ Excellent      │        │                │
│                │                │ job!,          │        │                │
│                │                │ error=None)    │        │                │
│ Note: Use      │                │                │        │                │
│ Confident AI   │                │                │        │                │
│ with DeepEval  │                │                │        │                │
│ to analyze     │                │                │        │                │
│ failed test    │                │                │        │                │
│ cases for more │                │                │        │                │
│ details        │                │                │        │                │
└────────────────┴────────────────┴────────────────┴────────┴────────────────┘
Total estimated evaluation tokens cost: 0.0037350000000000005 USD
✓ Tests finished 🎉! View results on 
https://app.confident-ai.com/project/cm7k4hv8d5q5k7advs35m4pl5/evaluation/test-runs/cm7k5
119t00d6qyy0o7ywn80r/test-cases.

```