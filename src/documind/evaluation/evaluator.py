import json
import logging
from pathlib import Path


import sys
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.documind.config import GEMINI_API_KEY
from src.documind.agent.graph import agent_executor


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_evaluation():
    eval_file = Path("data/eval_set.json")
    if not eval_file.exists():
        logger.error("Evaluation dataset not found!")
        return

    with open(eval_file, "r") as f:
        raw_dataset = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    num_questions = len(raw_dataset)
    logger.info("\n=== STARTING RAGAS EVALUATION ===")
    logger.info("1. Generating Agent responses and extracting ChromaDB contexts...\n")
    
    for i, item in enumerate(raw_dataset):
        question = item["question"]
        truth = item["ground_truth"]
        
        logger.info(f"[Test {i+1}/{num_questions}] Question: {question}")
        
        
        config = {"configurable": {"thread_id": f"ragas_eval_user_{i}"}}
        result = agent_executor.invoke({"messages": [("user", question)]}, config=config)
        
        
        raw_content = result["messages"][-1].content
        agent_answer = "\n".join(block.get("text", "") for block in raw_content if isinstance(block, dict)) if isinstance(raw_content, list) else str(raw_content)
        
        
        context_list = []
        for msg in result["messages"]:
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "search_documents":
                
                chunks = str(msg.content).split("\n\n---\n\n")
                context_list.extend(chunks)
        
      
        if not context_list:
            context_list = ["No documents retrieved."]

        questions.append(question)
        answers.append(agent_answer)
        contexts.append(context_list)
        ground_truths.append(truth)

    
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(data_dict)

    
    logger.info("\n2. Handing data over to Ragas for mathematical grading...")
    
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        api_key=GEMINI_API_KEY, 
        temperature=0.0
    )
    
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", 
        google_api_key=GEMINI_API_KEY
    )


    evaluation_result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )


    df = evaluation_result.to_pandas()
    
    logger.info("\n=== RAGAS REPORT CARD ===")
    logger.info(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string())
    
    logger.info("\n=== AVERAGE METRICS ===")
    logger.info(evaluation_result)

if __name__ == "__main__":
    run_evaluation()
