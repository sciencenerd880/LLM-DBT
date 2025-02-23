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
