#!/Users/lisihao/mlx-env/bin/python3
"""
ThunderOMLX 8B Inference Demo
Runs on mac mini with mlx_lm
"""
import sys
import time

def main():
    prompt = "用 5 字介绍 macOS"
    
    print("============================================", file=sys.stderr)
    print(f"ThunderOMLX 8B Demo", file=sys.stderr)
    print(f"Prompt: {prompt}", file=sys.stderr)
    print(f"Model: mlx-community/Qwen3-8B-4bit", file=sys.stderr)
    print(f"Host: {sys.argv[0] if len(sys.argv) > 1 else 'unknown'}", file=sys.stderr)
    print("============================================", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        
        print("[Loading model...]", file=sys.stderr)
        sys.stderr.flush()
        
        model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
        
        print("[Generating response...]", file=sys.stderr)
        sys.stderr.flush()
        
        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=20,
        )
        elapsed = time.time() - start
        
        print("", file=sys.stderr)
        print(f"[Generated in {elapsed:.1f}s]", file=sys.stderr)
        print("============================================", file=sys.stderr)
        sys.stderr.flush()
        
        # Output only the response (for easy grep verification)
        print(response.strip())
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
