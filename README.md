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

## Running the script
1. Run the main script:
    ```sh
    python src/main.py
    ```