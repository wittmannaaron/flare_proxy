import os
import sys
import json
import argparse
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from flare.engine import FlareProcessor
from retrievers.chroma import ChromaRetriever
from llm.anthropic_client import AnthropicClient
from utils.logging import init_logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FLARE Proxy Server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Load environment variables
load_dotenv()

# Initialize logging
log_path = os.getenv("LOG_PATH")
if not log_path:
    print("Error: LOG_PATH environment variable is not set", file=sys.stderr)
    sys.exit(1)

logger, traffic_logger = init_logging(log_path, args.debug)

# Initialize global variables for components
llm_client = None
retriever = None
flare_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    try:
        # Initialize components
        logger.info("Initializing FLARE Proxy components...")
        
        global llm_client, retriever, flare_processor
        
        llm_client = AnthropicClient()
        logger.debug("Initialized Anthropic client")
        
        retriever = ChromaRetriever()
        logger.debug("Initialized ChromaDB retriever")
        await retriever.initialize()
        logger.debug("ChromaDB connection established")
        
        flare_processor = FlareProcessor(
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            max_retrieval_rounds=int(os.getenv("MAX_RETRIEVAL_ROUNDS", "3"))
        )
        logger.debug("Initialized FLARE processor")
        
        logger.info("All components initialized successfully")
        yield
        
    finally:
        logger.info("Shutting down FLARE Proxy server")
        if retriever:
            await retriever.cleanup()
        logger.info("Cleanup completed successfully")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="FLARE Proxy",
    description="A FLARE (Forward-Looking Active REtrieval) proxy for AnythingLLM",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint to test connectivity"""
    logger.info("Received health check request")
    return {"status": "ok", "timestamp": str(datetime.now())}

@app.get("/v1/models")
async def list_models():
    """List available models endpoint"""
    logger.info("Received models list request")
    return {
        "data": [{
            "id": "claude-3-sonnet-20240229",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "anthropic",
            "permission": [],
            "root": "claude-3-sonnet-20240229",
            "parent": None
        }]
    }

@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def completions(request: Request):
    """LM Studio compatible completions endpoint"""
    try:
        logger.info("Received completion request")
        
        # Parse request data
        data = await request.json()
        logger.debug("Received request data:")
        logger.debug(f"- Model: {data.get('model')}")
        logger.debug(f"- Temperature: {data.get('temperature')}")
        logger.debug(f"- Stream: {data.get('stream')}")
        
        # Handle messages
        messages = data.get("messages", [])
        logger.debug("\nMessage history:")
        for idx, msg in enumerate(messages):
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Message {idx + 1}:")
            logger.debug(f"Role: {msg.get('role')}")
            logger.debug(f"Content:\n{msg.get('content', '')}")
            logger.debug('='*80)
            
        if not messages:
            prompt = data.get("prompt", "")
            if not prompt:
                logger.warning("Request received with no prompt or messages")
                raise HTTPException(status_code=400, detail="No prompt or messages provided")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        
        # Process request
        parameters = {
            "model": "claude-3-sonnet-20240229",
            "temperature": data.get("temperature", float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))),
            "max_tokens": data.get("max_tokens", int(os.getenv("DEFAULT_MAX_TOKENS", "4096")))
        }
        
        # Log complete traffic in debug mode
        if traffic_logger:
            traffic_logger.log_request("AnythingLLM", json.dumps({"messages": messages, "parameters": parameters}, indent=2))
        
        logger.info("Processing request through FLARE engine")
        response = await flare_processor.process_message(
            messages=messages,
            parameters=parameters,
            traffic_logger=traffic_logger
        )
        
        # Handle response
        is_chat = request.url.path == "/v1/chat/completions"
        stream = data.get("stream", False)
        
        if stream:
            return StreamingResponse(
                generate_stream(response, parameters, is_chat),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # Format response
        api_response = format_response(response, parameters, is_chat)
        
        # Log response
        logger.info("Successfully processed completion request")
        logger.debug("\nAPI Response:")
        logger.debug(json.dumps(api_response, indent=2))
        
        # Log complete traffic in debug mode
        if traffic_logger:
            traffic_logger.log_response("FLARE-Proxy", json.dumps(api_response, indent=2))
        
        return api_response
        
    except HTTPException as he:
        logger.warning(f"HTTP error in chat completion: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def format_response(response, parameters, is_chat):
    """Format API response based on request type"""
    if is_chat:
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content[0].text if response.content else ""
                },
                "finish_reason": response.stop_reason or "stop"
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            }
        }
    else:
        return {
            "id": response.id,
            "object": "text_completion",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "text": response.content[0].text if response.content else "",
                "index": 0,
                "finish_reason": response.stop_reason or "stop"
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            }
        }

async def generate_stream(response, parameters, is_chat):
    """Generate streaming response"""
    # Initial response
    if is_chat:
        yield "data: " + json.dumps({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": response.content[0].text if response.content else ""
                },
                "finish_reason": None
            }]
        }) + "\n\n"
    else:
        yield "data: " + json.dumps({
            "id": response.id,
            "object": "text_completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "text": response.content[0].text if response.content else "",
                "index": 0,
                "finish_reason": None
            }]
        }) + "\n\n"
    
    # Final response
    if is_chat:
        yield "data: " + json.dumps({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": response.stop_reason or "stop"
            }]
        }) + "\n\n"
    else:
        yield "data: " + json.dumps({
            "id": response.id,
            "object": "text_completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": parameters["model"],
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": response.stop_reason or "stop"
            }]
        }) + "\n\n"
    
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3128"))
    
    # Configure Uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_config=log_config,
        log_level="debug" if args.debug else "info"
    )
