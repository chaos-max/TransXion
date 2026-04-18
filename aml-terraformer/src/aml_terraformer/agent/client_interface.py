"""LLM client interface."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate completion from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Completion text
        """
        pass


class DummyLLMClient(LLMClient):
    """Dummy LLM client for testing (always returns stop)."""

    def complete(self, prompt: str) -> str:
        """Return stop decision.

        Args:
            prompt: Input prompt

        Returns:
            JSON string with stop decision
        """
        return '{"tool":"stop","args":{},"rationale":"Dummy client"}'


class OpenAILLMClient(LLMClient):
    """OpenAI API client (requires openai package and API key)."""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model name

        Raises:
            ImportError: If openai package not installed
            ValueError: If API key not provided
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for OpenAILLMClient. Install with: pip install openai")

        if api_key is None:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Generate completion using OpenAI API.

        Args:
            prompt: Input prompt

        Returns:
            Completion text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class AnthropicLLMClient(LLMClient):
    """Anthropic Claude API client (requires anthropic package and API key)."""

    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (if None, will try to get from environment)
            model: Model name

        Raises:
            ImportError: If anthropic package not installed
            ValueError: If API key not provided
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required for AnthropicLLMClient. Install with: pip install anthropic")

        if api_key is None:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Generate completion using Anthropic API.

        Args:
            prompt: Input prompt

        Returns:
            Completion text
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return message.content[0].text.strip()


class DeepSeekLLMClient(LLMClient):
    """DeepSeek API client (uses OpenAI-compatible API)."""

    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        """Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (if None, will try to get from environment)
            model: Model name (default: deepseek-chat)

        Raises:
            ImportError: If openai package not installed
            ValueError: If API key not provided
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for DeepSeekLLMClient. Install with: pip install openai")

        if api_key is None:
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        # DeepSeek uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def complete(self, prompt: str) -> str:
        """Generate completion using DeepSeek API.

        Args:
            prompt: Input prompt

        Returns:
            Completion text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class LocalQwenLLMClient(LLMClient):
    """Local Qwen model client using transformers.

    This client loads a local Qwen model (e.g., Qwen3-1.7B) and runs inference
    on your machine without API calls.

    Example:
        >>> client = LocalQwenLLMClient(
        ...     model_path="/path/to/Qwen3-1.7B",
        ...     device="cuda",  # or "cpu"
        ...     max_new_tokens=512
        ... )
        >>> response = client.complete("Your prompt here")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize local Qwen model.

        Args:
            model_path: Path to local model directory (e.g., /path/to/Qwen3-1.7B)
            device: Device to use ("cuda", "cpu", or "cuda:0", etc.)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            load_in_8bit: Load model in 8-bit precision (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit precision (requires bitsandbytes)

        Raises:
            ImportError: If transformers or torch not installed
            FileNotFoundError: If model_path does not exist
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for LocalQwenLLMClient. "
                "Install with: pip install transformers torch"
            )

        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        print(f"Loading local Qwen model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model with optional quantization
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
        }

        if load_in_8bit:
            print("Loading in 8-bit precision...")
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            print("Loading in 4-bit precision...")
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            # Standard loading
            load_kwargs["dtype"] = torch.float16 if "cuda" in device else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Move to device if not using device_map
        if not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(device)

        self.model.eval()
        print(f"Model loaded on {device}")

    def complete(self, prompt: str) -> str:
        """Generate completion using local model.

        Args:
            prompt: Input prompt

        Returns:
            Completion text
        """
        import torch

        # Format prompt for Qwen (add system message if needed)
        messages = [
            {"role": "system", "content": "You are an AI assistant that outputs only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                formatted_prompt = f"<|im_start|>system\nYou are an AI assistant that outputs only valid JSON.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Manual formatting for Qwen
            formatted_prompt = f"<|im_start|>system\nYou are an AI assistant that outputs only valid JSON.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Remove prompt
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()


def create_llm_client(
    provider: str,
    api_key: str = None,
    model: str = None,
    lora_path: str = None,
    **kwargs
) -> LLMClient:
    """Factory function to create LLM client.

    Args:
        provider: Provider name ("openai", "anthropic", "deepseek", "local", "local-lora")
        api_key: API key (not needed for "local" or "local-lora")
        model: Model name or path (for local, this is the base model path)
        lora_path: Path to LoRA checkpoint (only for "local-lora")
        **kwargs: Additional arguments passed to client constructor

    Returns:
        LLMClient instance

    Example:
        >>> # API-based
        >>> client = create_llm_client("deepseek", api_key="sk-xxx")
        >>>
        >>> # Local model
        >>> client = create_llm_client(
        ...     "local",
        ...     model="/path/to/Qwen3-1.7B",
        ...     device="cuda"
        ... )
        >>>
        >>> # Local model with LoRA
        >>> client = create_llm_client(
        ...     "local-lora",
        ...     model="/path/to/Qwen2.5-7B",
        ...     lora_path="/path/to/checkpoint-50",
        ...     device="cuda"
        ... )
    """
    if provider == "openai":
        return OpenAILLMClient(
            api_key=api_key,
            model=model or "gpt-4"
        )
    elif provider == "anthropic":
        return AnthropicLLMClient(
            api_key=api_key,
            model=model or "claude-3-sonnet-20240229"
        )
    elif provider == "deepseek":
        return DeepSeekLLMClient(
            api_key=api_key,
            model=model or "deepseek-chat"
        )
    elif provider == "local":
        if model is None:
            raise ValueError("model path required for local provider")
        return LocalQwenLLMClient(
            model_path=model,
            **kwargs
        )
    elif provider == "local-lora":
        if model is None:
            raise ValueError("base model path required for local-lora provider")
        if lora_path is None:
            raise ValueError("lora_path required for local-lora provider")
        return _create_local_lora_client(model, lora_path, **kwargs)
    elif provider == "dummy":
        return DummyLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, anthropic, deepseek, local, local-lora, dummy")


def _create_local_lora_client(base_model_path: str, lora_path: str, device: str = "cuda", **kwargs) -> LocalQwenLLMClient:
    """Create LocalQwenLLMClient with LoRA adapter loaded.

    This follows the pattern from evaluate_detection_rate.py.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA checkpoint directory
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        LocalQwenLLMClient instance with LoRA adapters loaded
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    print(f"Loading LoRA model...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA checkpoint: {lora_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # Load base model
    load_in_8bit = kwargs.pop('load_in_8bit', False)
    load_in_4bit = kwargs.pop('load_in_4bit', False)

    load_kwargs = {
        "pretrained_model_name_or_path": base_model_path,
        "trust_remote_code": True,
    }

    if load_in_8bit:
        print("  Loading in 8-bit precision...")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        print("  Loading in 4-bit precision...")
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if "cuda" in device else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Move to device if not using device_map
    if not (load_in_8bit or load_in_4bit):
        model = model.to(device)

    model.eval()
    print(f"  LoRA model loaded successfully")

    # Create a custom client object that wraps the LoRA model
    client = LocalQwenLLMClient.__new__(LocalQwenLLMClient)
    client.model = model
    client.tokenizer = tokenizer
    client.device = device
    client.model_path = base_model_path
    client.max_new_tokens = kwargs.get('max_new_tokens', 512)
    client.temperature = kwargs.get('temperature', 0.7)
    client.top_p = kwargs.get('top_p', 0.9)
    client.do_sample = kwargs.get('do_sample', True)

    return client
