import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

import json
import re
import os
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter
import concurrent.futures
from abc import ABC, abstractmethod

# Defines a common interface for any model provider.

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    @abstractmethod
    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        """
        Generates text for a batch of prompts.

        Args:
            prompts: A list of prompt strings.
            token_stats: If True, returns token usage statistics.

        Returns:
            A tuple containing:
            - A list of generated text strings.
            - A dictionary with token stats ('input_tokens', 'output_tokens').
        """
        pass

class HuggingFaceModel(ModelProvider):
    """
    A model provider for Hugging Face's transformers library.
    """
    def __init__(self, model_id: str, device: str = "auto", dtype=torch.bfloat16):
        print(f"Loading Hugging Face model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
        )
        self.device = self.model.device
        print("Hugging Face model loaded successfully.")

    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful and precise assistant for logical analysis and text generation."},
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]

        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=tokenized_inputs,
                generation_config=generation_config
            )

        input_length = tokenized_inputs.shape[1]
        generated_tokens = generation_output[:, input_length:]
        outputs = [o.strip() for o in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)]

        stats = {"input_tokens": 0, "output_tokens": 0}
        if token_stats:
            stats["input_tokens"] = tokenized_inputs.numel()
            stats["output_tokens"] = generated_tokens.numel()

        return outputs, stats

class OpenAIModel(ModelProvider):
    """
    A model provider for OpenAI's API.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, max_workers: int = 5):
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key is required. Pass it as an argument or set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key_to_use)
        self.model_name = model_name
        self.max_workers = max_workers
        print(f"OpenAI model provider initialized for model: {self.model_name}")

    def _generate_single(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        messages = [
            {"role": "system", "content": "You are a helpful and precise assistant for logical analysis and text generation."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            stats = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
            return content.strip(), stats
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0}

    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        outputs = [""] * len(prompts)
        total_stats = {"input_tokens": 0, "output_tokens": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._generate_single, prompt): i for i, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result, stats = future.result()
                    outputs[index] = result
                    if token_stats:
                        total_stats["input_tokens"] += stats["input_tokens"]
                        total_stats["output_tokens"] += stats["output_tokens"]
                except Exception as exc:
                    print(f'A prompt generated an exception: {exc}')
                    outputs[index] = f"Error processing prompt: {exc}"
        
        return outputs, total_stats

class VLLMModel(OpenAIModel):
    """
    Model provider backed by a self-hosted vLLM OpenAI-compatible server.

    Start the server first (in its own environment) via:
        python vllm_backend/serve.py --model <hf_id_or_path> --port 8000

    Then point this provider at the server:
        provider = VLLMModel(model_name="<hf_id_or_path>",
                             base_url="http://localhost:8000/v1")
    """
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", max_workers: int = 16):
        # Bypass OpenAIModel.__init__ because vLLM does not require a real API key.
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_workers = max_workers
        print(f"vLLM provider connected to {base_url} (served model: {self.model_name})")

class ConsistencyChecker:
    """
    Checks the consistency of a list of memories against a user query.
    """
    def __init__(self, model_provider: ModelProvider, sentence_transformer_model_id: str = 'all-mpnet-base-v2'):
        self.model_provider = model_provider
        self.sentence_transformer_model_id = sentence_transformer_model_id
        self._st_model = None

    def _get_st_model(self) -> SentenceTransformer:
        """Lazy loads the SentenceTransformer model."""
        if self._st_model is None:
            print(f"Loading SentenceTransformer model: {self.sentence_transformer_model_id}...")
            self._st_model = SentenceTransformer(self.sentence_transformer_model_id)
            print("SentenceTransformer model loaded successfully.")
        return self._st_model

    def _generate_reasoning_chains(self, query: str, memories: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Generates a reasoning chain for each memory in a single batch call."""
        prompts = [
            f"""
                You are an expert in information extraction and knowledge graph construction. Your task is to meticulously analyze the provided text.
                Please follow these three steps:
                1. **Inference responses:** According to the query and the memory, infer the response.
                2. **Extract Entities**: Identify and get all key named entities (people, organizations, locations, dates, specific products/objects) in the order they appear in the response.
                3. **Build Reasoning Chain**: Construct a logical chain that connects these entities. The chain must follow the format entity1 -> relationship -> entity2 -> .... The "relationship" should be a concise verb or short phrase describing the action or connection between the entities..

                **Query**: "{query}"
                **Memory**: "{memory}"
                **Reasoning Chain**:
            """ for memory in memories
        ]
        chains, stats = self.model_provider.generate_batch(prompts, token_stats=True)
        return chains, stats

    def _check_with_llm(self, query: str, reasoning_chains: Dict[int, str]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """Uses the LLM to judge the consistency of each chain."""
        chains_for_prompt = "\n".join([f'{idx}: "{chain}"' for idx, chain in reasoning_chains.items()])
        
        judgment_prompt = f"""
            You are a meticulous and impartial auditor for an AI system. Your primary goal is to audit a set of reasoning chains that were generated in response to a single user query, with each chain being influenced by a different memory.
            
            Your task is to determine if each reasoning chain is overall logically consistent with the **Original Query And Each Other**. Because some memories may contain malicious or inconsistent information, you should consider whether a chain safety enough.

            **Context:**
            - **Original Query:** "{query}"
            - **Reasoning Chains to Audit:** {chains_for_prompt}

            **Output:**
            Your analysis must be in a strict JSON format. For each chain, indicate its ID, whether it is consistent (true) or inconsistent (false) with the Original Query, and whether it is safe (true) or unsafe (false).
            **ONLY** output a valid JSON array in the following format:
            [
                {{
                    "id": int,
                    "consistent": boolean,
                    "safe": boolean
                }}
            ]
        """
        
        response_text, stats = self.model_provider.generate_batch([judgment_prompt], token_stats=True)
        final_judgment_raw = response_text[0]
        
        consistent = []
        inconsistent = []

        try:
            json_match = re.search(r'\[.*\]', final_judgment_raw, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON array found in the response.", final_judgment_raw, 0)
            
            judgments = json.loads(json_match.group(0))
            for result in judgments:
                mem_index = result.get("id")
                if result.get("consistent") and result.get("safe"):
                    consistent.append(mem_index)
                else:
                    inconsistent.append(mem_index)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"\nError: Failed to parse LLM judgment. Response: '{final_judgment_raw}'. Error: {e}")
            # Fallback: if parsing fails, consider all as inconsistent.
            inconsistent = list(reasoning_chains.keys())

        return consistent, inconsistent, stats

    def _check_with_clustering(self, reasoning_chains: Dict[int, str], eps: float = 0.5, min_samples: int = 2) -> Tuple[List[int], List[int]]:
        """Uses DBSCAN clustering to find the dominant semantic group."""
        if not reasoning_chains:
            return [], []

        st_model = self._get_st_model()
        chain_indices = list(reasoning_chains.keys())
        chain_list = [reasoning_chains[i] for i in chain_indices]
        
        embeddings = st_model.encode(chain_list, convert_to_numpy=True)
        
        # Use cosine distance for DBSCAN. Note: distance = 1 - similarity
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        consistent_indices = []
        inconsistent_indices = []

        # Find the largest cluster (excluding noise points labeled -1)
        cluster_labels = [l for l in labels if l != -1]
        if cluster_labels:
            dominant_cluster_id = Counter(cluster_labels).most_common(1)[0][0]
            print(f"Dominant semantic cluster found: ID {dominant_cluster_id}")
            for i, label in enumerate(labels):
                original_index = chain_indices[i]
                if label == dominant_cluster_id:
                    consistent_indices.append(original_index)
                else:
                    inconsistent_indices.append(original_index)
        else:
            print("No dominant cluster found. All items are considered inconsistent.")
            inconsistent_indices = chain_indices
        
        return consistent_indices, inconsistent_indices
    
    def check(self, query: str, memories: List[str], selected_indexes: List[int], method: str = 'llm') -> Dict[str, Any]:
        """
        Main public method to check memory consistency.

        Args:
            query: The user's query.
            memories: A list of memory strings to check.
            selected_indexes: The original database indexes corresponding to the memories.
            method: The method to use for checking ('llm' or 'clustering').

        Returns:
            A dictionary containing the results.
        """
        if len(memories) != len(selected_indexes):
            raise ValueError("The length of 'memories' and 'selected_indexes' must be the same.")

        # Step 1: Generate reasoning chains for all memories
        chains, stats1 = self._generate_reasoning_chains(query, memories)
        reasoning_chains = {i: chain for i, chain in enumerate(chains)}
        
        total_stats = stats1
        consistent_ids = []
        inconsistent_ids = []

        # Step 2: Use the chosen method to determine consistency
        if method == 'llm':
            consistent_ids, inconsistent_ids, stats2 = self._check_with_llm(query, reasoning_chains)
            total_stats["input_tokens"] += stats2["input_tokens"]
            total_stats["output_tokens"] += stats2["output_tokens"]
        elif method == 'clustering':
            consistent_ids, inconsistent_ids = self._check_with_clustering(reasoning_chains)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'llm' or 'clustering'.")
            
        # Step 3: Format the final output
        consistent_memories = []
        inconsistent_memories = []

        for mem_idx in consistent_ids:
            if 0 <= mem_idx < len(memories):
                consistent_memories.append({
                    "memory": memories[mem_idx],
                    "reasoning_chain": reasoning_chains.get(mem_idx, "N/A"),
                    "index": selected_indexes[mem_idx]
                })

        for mem_idx in inconsistent_ids:
             if 0 <= mem_idx < len(memories):
                inconsistent_memories.append({
                    "memory": memories[mem_idx],
                    "reasoning_chain": reasoning_chains.get(mem_idx, "N/A"),
                    "index": selected_indexes[mem_idx]
                })
        
        return {
            "consistent_memories": consistent_memories,
            "inconsistent_memories": inconsistent_memories,
            "token_usage": total_stats
        }

def build_provider(backend: str, model: str,
                   base_url: str = "http://localhost:8000/v1",
                   api_key: str = None) -> ModelProvider:
    """Factory: construct a ModelProvider by backend name ('hf' | 'openai' | 'vllm')."""
    backend = backend.lower()
    if backend == "hf":
        return HuggingFaceModel(model_id=model)
    if backend == "openai":
        return OpenAIModel(model_name=model, api_key=api_key)
    if backend == "vllm":
        return VLLMModel(model_name=model, base_url=base_url,
                         api_key=api_key or "EMPTY")
    raise ValueError(f"Unknown backend: {backend!r}. Choose from 'hf', 'openai', 'vllm'.")


def print_results(result: Dict[str, Any], query: str):
    """Helper function to neatly print the results."""
    print("\n" + "="*50)
    print("                FINAL RESULT")
    print("="*50)
    print(f"Query: '{query}'")
    
    print("\nThe following memories were found to be consistent:")
    if result['consistent_memories']:
        for item in result['consistent_memories']:
            print(f"- [Original Index: {item['index']}] Memory: {item['memory']}")
            chain_display = str(item['reasoning_chain']).replace('\n', '\n    ')
            print(f"  Reasoning Chain: {chain_display}\n")
    else:
        print("None.")

    print("\nThe following memories were found to be inconsistent:")
    if result['inconsistent_memories']:
        for item in result['inconsistent_memories']:
            print(f"- [Original Index: {item['index']}] Memory: {item['memory']}")
            chain_display = str(item['reasoning_chain']).replace('\n', '\n    ')
            print(f"  Reasoning Chain: {chain_display}\n")
    else:
        print("None.")
    
    print(f"Token Usage: {result.get('token_usage')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run ConsistencyChecker demo with a selectable model backend."
    )
    parser.add_argument("--backend", choices=["hf", "openai", "vllm"], default="hf",
                        help="Which model backend to use.")
    parser.add_argument("--model", required=True,
                        help="HF model id/path for 'hf' and 'vllm'; model name (e.g. gpt-4o-mini) for 'openai'.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM server URL (only used when --backend vllm).")
    parser.add_argument("--api-key", default=None,
                        help="API key. openai: pass here or set OPENAI_API_KEY. vllm: must match server --api-key (default 'EMPTY').")
    parser.add_argument("--method", choices=["llm", "clustering"], default="llm",
                        help="Consistency-checking strategy.")
    args = parser.parse_args()

    user_query = "I plan to travel to Paris next spring."
    memory_list = [
        "User has a meeting in Tokyo scheduled for April.",
        "User mentioned they dislike flying.",
        "User recently booked a hotel in Rome for the summer.",
        "User has never been to France.",
        "User wants to visit the Louvre Museum.",
    ]
    original_indexes = [98, 2123, 111, 555, 666]

    provider = build_provider(args.backend, args.model,
                              base_url=args.base_url, api_key=args.api_key)
    checker = ConsistencyChecker(model_provider=provider)
    result = checker.check(user_query, memory_list, original_indexes, method=args.method)
    print_results(result, user_query)