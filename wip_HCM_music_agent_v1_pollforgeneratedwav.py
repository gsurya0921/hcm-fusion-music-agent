#!/usr/bin/env python3
import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator
from unittest import result

import dotenv
import gradio as gr
import mirdata
import numpy as np
import librosa

# Anthropic SDK
from anthropic import AsyncAnthropic

# MCP Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("HCM_MusicAgent")

# =========================
# CONFIG
# =========================
dotenv.load_dotenv("/home/ganesh/fubar/music_agent_v1/.env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = os.getenv("AGENT_MODEL", "claude-3-5-haiku-20241022")
if not MODEL_NAME or not isinstance(MODEL_NAME, str):
    MODEL_NAME = "claude-3-5-haiku-20241022"
    
log.info(f"Using model: {MODEL_NAME}")

# Anthropic async client
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# MCP server
#GEN_SERVER_CMD = ["python", "/home/ganesh/fubar/music_agent_v1/mcp_servers/gen_server.py"]
GEN_SERVER_CMD = ["python", "/home/ganesh/fubar/music_agent_v1/mcp_servers/musicgen_chromaHCM_genserver.py"]
# Throttling
AGENT_GATE = asyncio.Semaphore(int(os.getenv("AGENT_CONCURRENCY", "2")))
MAX_RETRIES = 2

# Raag database path
RAAG_DB_PATH = "/home/ganesh/fubar/music_agent_v1/genre_db_v1/raag_lookup.json"
HCM_DATA_HOME = "/home/ganesh/fubar/music_agent_v1/genre_db_v1"

INSTRUCTIONS = (
    "You are an agentic music assistant creating Hindustani classical fusion music in layers.\n"
    "You MUST use the available MCP tools to complete the task.\n\n"
    "Workflow:\n"
    "1) Layer 1: User provides a raag name. The system retrieves authentic Hindustani classical clips for that raag.\n"
    "2) Layer 2: Call generate_layer2_music() with the Layer 1 raag foundation to create fusion music.\n"
    "3) Provide the final output path as: OUTPUT_PATH: <path>\n\n"
    "Important: Layer 1 is handled by the RAG system automatically. You only need to call the Layer 2 generation tool.\n"
)

# =========================
# RAAG DATABASE MANAGER
# =========================
class RaagDatabase:
    def __init__(self, data_home: str):
        self.data_home = Path(data_home)
        self.raag_db = {}
        self.dataset = None
        
    def initialize(self):
        """Initialize the Saraga Hindustani dataset and build raag lookup"""
        log.info("Initializing Hindustani Classical Music dataset...")
        
        # Load mirdata dataset
        self.dataset = mirdata.initialize('saraga_hindustani', data_home=str(self.data_home))
        
        # Check if database JSON exists
        db_file = self.data_home / 'raag_lookup.json'
        if db_file.exists():
            log.info(f"Loading existing raag database from {db_file}")
            with open(db_file, 'r') as f:
                self.raag_db = json.load(f)
            log.info(f"Loaded {len(self.raag_db)} raagas from database")
        else:
            log.info("Building raag database from scratch...")
            self._build_database()
            self._save_database(db_file)
    
    def _build_database(self):
        """Build raag lookup dictionary from dataset"""
        for track_id in self.dataset.track_ids:
            track = self.dataset.track(track_id)
            
            try:
                # Get raag name from metadata
                raag_info = track.metadata.get('raags', [{}])[0]
                raag_name = raag_info.get('name', '').lower().strip()
                
                if not raag_name:
                    continue
                
                # Initialize list for this raag
                if raag_name not in self.raag_db:
                    self.raag_db[raag_name] = []
                
                # Add track info
                self.raag_db[raag_name].append({
                    'audio_path': track.audio_path,
                    'track_id': track_id,
                    'artist': track.metadata.get('artists', [{}])[0].get('name', 'Unknown')
                })
                
            except Exception as e:
                log.warning(f"Error processing track {track_id}: {e}")
                continue
        
        log.info(f"Built database with {len(self.raag_db)} raagas")
        for raag, clips in list(self.raag_db.items())[:5]:
            log.info(f"  {raag}: {len(clips)} clips")
    
    def _save_database(self, filepath: Path):
        """Save raag database to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.raag_db, f, indent=2)
        log.info(f"Saved raag database to {filepath}")
    
    def get_available_raagas(self):
        """Get list of available raag names"""
        return sorted(list(self.raag_db.keys()))
    
    def retrieve_raag_clips(self, raag_query: str, n_clips: int = 3):
        """Retrieve audio clips for a specific raag"""
        raag_query = raag_query.lower().strip()
        
        if raag_query not in self.raag_db:
            # Try fuzzy matching
            from difflib import get_close_matches
            matches = get_close_matches(raag_query, self.raag_db.keys(), n=1, cutoff=0.6)
            if matches:
                raag_query = matches[0]
                log.info(f"Fuzzy matched '{raag_query}' to '{matches[0]}'")
            else:
                return None, f"Raag '{raag_query}' not found. Available: {', '.join(self.get_available_raagas()[:10])}"
        
        clips = self.raag_db[raag_query][:n_clips]
        return clips, None
    
    def combine_clips(self, clips: list, method='weighted_mean'):
        """Combine multiple audio clips into one"""
        audio_arrays = []
        
        for clip_info in clips:
            # Handle both dict and string formats
            if isinstance(clip_info, dict):
                audio_path = clip_info['audio_path']
            else:
                audio_path = clip_info

            try:
                audio, sr = librosa.load(audio_path, offset=10, sr=32000, duration=30, mono=True)
                audio_arrays.append(audio)
                log.info(f"Loaded clip: {Path(audio_path)}, duration: {len(audio)/sr:.2f}s")
            except Exception as e:
                log.warning(f"Error loading {audio_path}: {e}")
                continue
        
        if not audio_arrays:
            return None
        
        # Pad to same length
        max_len = max(len(a) for a in audio_arrays)
        padded = [np.pad(a, (0, max_len - len(a))) for a in audio_arrays]
        
        if method == 'weighted_mean':
            # Equal weights for simplicity
            weights = np.ones(len(padded)) / len(padded)
            combined = np.average(padded, axis=0, weights=weights)
        elif method == 'concatenate':
            combined = np.concatenate(padded)
        else:
            combined = np.mean(padded, axis=0)
        
        return combined

# Global raag database instance
raag_db = RaagDatabase(HCM_DATA_HOME)

# =========================
# MCP CLIENT MANAGER
# =========================
class MCPClientManager:
    def __init__(self):
        self.sessions = {}
        self.stdio_contexts = {}
        self.tools = []
    
    async def connect_server(self, name: str, command: list[str]):
        """Connect to an MCP server via stdio"""
        server_params = StdioServerParameters(
            command=command[0],
            args=command[1:] if len(command) > 1 else []
        )
        
        stdio_context = stdio_client(server_params)
        read, write = await stdio_context.__aenter__()
        self.stdio_contexts[name] = stdio_context
        
        session = ClientSession(read, write)
        await session.__aenter__()
        
        await session.initialize()
        tools_list = await session.list_tools()
        
        self.sessions[name] = session
        
        # Convert MCP tools to Claude tool format
        for tool in tools_list.tools:
            raw_tool_name = f"{name}_{tool.name}"
            sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_tool_name)[:128]
            
            claude_tool = {
                "name": sanitized_name,
                "description": tool.description or f"Tool {tool.name} from {name} server",
                "input_schema": tool.inputSchema
            }
            self.tools.append(claude_tool)
            log.info(f"Added tool: {sanitized_name}")
        
        log.info(f"Connected to {name} server with {len(tools_list.tools)} tools")
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the appropriate MCP server"""
        parts = tool_name.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool name format: {tool_name}")
        
        server_name, tool_method = parts
        session = self.sessions.get(server_name)
        
        if not session:
            raise ValueError(f"Server {server_name} not connected")
        
        result = await session.call_tool(tool_method, arguments)
        return result
    
    async def cleanup(self):
        """Close all MCP sessions"""
        for name, session in list(self.sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                log.warning(f"Error closing session {name}: {e}")
        
        for name, stdio_ctx in list(self.stdio_contexts.items()):
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except Exception as e:
                log.warning(f"Error closing stdio context {name}: {e}")
        
        self.sessions.clear()
        self.stdio_contexts.clear()

# =========================
# BUILD AGENT
# =========================
async def build_agent_stdio():
    """Create MCP client manager"""
    manager = MCPClientManager()
    await manager.connect_server("GEN", GEN_SERVER_CMD)
    return manager

# =========================
# STREAMING RUNNER
# =========================
async def stream_once_with_stdio(
    raag_name: str,
    instrument: str,
    tempo: int,
    style: str,
    knob: float,
    approval_feedback: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    
    log.info(f"=== NEW REQUEST: raag={raag_name}, instrument={instrument}, tempo={tempo}, style={style} ===")
    
    # Layer 1: Retrieve raag clips
    yield {"type": "log", "text": f"Layer 1: Retrieving clips for Raag {raag_name}...\n"}
    
    clips, error = raag_db.retrieve_raag_clips(raag_name, n_clips=3)
    
    if error:
        yield {"type": "error", "text": error}
        return
    
    yield {"type": "log", "text": f"Found {len(clips)} clips for Raag {raag_name}\n"}
    
    # Combine clips
    yield {"type": "log", "text": "Combining clips...\n"}
    layer1_audio = raag_db.combine_clips(clips)
    
    if layer1_audio is None:
        yield {"type": "error", "text": "Failed to combine audio clips"}
        return
    
    # Save Layer 1 audio temporarily
    layer1_path = Path("/home/ganesh/fubar/music_agent_v1/layer1_raag_foundation.wav")
    import soundfile as sf
    sf.write(str(layer1_path), layer1_audio, 32000)
    
    yield {"type": "log", "text": f"Layer 1 foundation saved to {layer1_path}\n"}
    yield {"type": "log", "text": f"Layer 2: Generating fusion music with {instrument}, {tempo} BPM, {style} style...\n"}
    
    # Create MCP manager
    mcp_manager = await build_agent_stdio()
    
    # Build user task for Claude
    user_task = (
        f"Generate Layer 2 fusion music:\n"
        f"- Layer 1 raag foundation: {layer1_path}\n"
        f"- Instrument: {instrument}\n"
        f"- Tempo: {tempo} BPM\n"
        f"- Style: {style}\n"
        f"- Knob: {knob}\n"
        f"Use generate_layer2_music tool and provide OUTPUT_PATH."
    )
    
    if approval_feedback:
        user_task += f"\n- Feedback: {approval_feedback}\n"

    async def _do_stream():
        messages = [{"role": "user", "content": user_task}]
        text_buf = []
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            log.info(f"=== Iteration {iteration}/{max_iterations} ===")
            
            iteration_text = []
            has_tool_calls = False
            final_message = None
            
            try:
                async with anthropic_client.messages.stream(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    system=INSTRUCTIONS,
                    messages=messages,
                    tools=mcp_manager.tools,
                    tool_choice={"type": "auto"},
                    temperature=0,
                ) as stream:
                    async for text in stream.text_stream:
                        if text:
                            text_buf.append(text)
                            iteration_text.append(text)
                            yield {"type": "log", "text": text}
                    
                    final_message = await asyncio.wait_for(
                        stream.get_final_message(), 
                        timeout=120.0
                    )
            except Exception as e:
                log.error(f"Error in stream: {e}")
                yield {"type": "error", "text": str(e)}
                return
            
            if not final_message:
                yield {"type": "error", "text": "No response from Claude"}
                return
            
            # Check for tool calls
            has_tool_calls = any(block.type == "tool_use" for block in final_message.content)
            
            if not has_tool_calls:
                combined = "".join(text_buf).strip()
                m = re.search(r"OUTPUT_PATH:\s*([^\n\r]+)", combined)
                
                if m:
                    output_path = m.group(1).strip()
                    yield {"type": "final", "output_path": output_path}
                else:
                    yield {"type": "error", "text": "No OUTPUT_PATH found"}
                return
            
            # Process tool calls
            messages.append({"role": "assistant", "content": final_message.content})
            tool_results = []
            
            for block in final_message.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    
                    # ADD THIS
                    log.info(f"==== TOOL CALL DEBUG ====")
                    log.info(f"Tool name: {tool_name}")
                    log.info(f"Tool input: {json.dumps(tool_input, indent=2)}")
                    log.info(f"========================")
        
                    yield {"type": "log", "text": f"\nCalling {tool_name}...\n"}
                    
                    # # Calculate timeout based on requested duration
                    requested_seconds = tool_input.get('seconds', 10.0)
                    timeout = max(180.0, requested_seconds * 10)  # At least 3 min, or 10x generation time
        
                    # try:
                    #     result = await asyncio.wait_for(
                    #         mcp_manager.call_tool(tool_name, tool_input),
                    #         timeout=timeout
                    #     )

                    # Don't wait for result - assume success after timeout
                    try: 
                        result_task = asyncio.create_task(mcp_manager.call_tool(tool_name, tool_input))
                        try:
                            result = await asyncio.wait_for(result_task, timeout=30)
                            # If we get here within 5s, process normally
                            log.info("Got result within timeout")
                            if hasattr(result, "content"):
                                result_content = result.content
                                if isinstance(result_content, list) and len(result_content) > 0:
                                    result_text = result_content[0].text if hasattr(result_content[0], "text") else str(result_content[0])
                                else:
                                    result_text = str(result_content)
                            else:
                                result_text = str(result)
                        
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result_text
                            })
                        except asyncio.TimeoutError:
                            # Long-running operation - assume it will succeed
                            log.info("MCP Client timed out. Waiting for file to appear...")
                            result_task.cancel()
                            output_path = "/home/ganesh/fubar/music_agent_v1/generated_layer2.wav"

                            # Poll for file existence
                            max_wait = 180  # 3 minutes
                            for i in range(max_wait):
                                if os.path.exists(output_path):
                                    log.info(f"File appeared after {i} seconds")
                                    result_text = json.dumps({"output_path": output_path})
                                    break
                                await asyncio.sleep(1)
                                if i % 10 == 0:
                                    yield {"type": "log", "text": "."}
                            else:
                                raise TimeoutError("File never appeared")
                            # Manually construct expected result
                            result_text = json.dumps({"output_path": "/home/ganesh/fubar/music_agent_v1/generated_layer2.wav"}) 
                            # Wait a bit for generation to actually complete
                            #await asyncio.sleep(120)  # Wait 2 minutes
                            # Append assumed success result 
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result_text
                            })  
                            yield {"type": "log", "text": f"Assuming tool {tool_name} succeeded with output {result_text}\n"}
                        else:
                            yield {"type": "log", "text": f"Tool {tool_name} finished\n"}     
                    except Exception as e:
                        log.error(f"Error calling tool {tool_name}: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
                        yield {"type": "error", "text": f"Tool failed: {str(e)}"}
            
            messages.append({"role": "user", "content": tool_results})
            await asyncio.sleep(0.01)
        
        yield {"type": "error", "text": "Max iterations reached"}

    async with AGENT_GATE:
        try:
            async for item in _do_stream():
                yield item
        finally:
            await mcp_manager.cleanup()

# =========================
# GRADIO UI
# =========================
class WorkflowState:
    def __init__(self):
        self.current_output = None
        self.raag_name = None
        self.instrument = None
        self.tempo = None
        self.style = None
        self.knob = None
        self.stage = "initial"

workflow_state = WorkflowState()

async def ui_generate_stream(raag_name, instrument, tempo, style, knob):
    """Generate fusion music from raag"""
    workflow_state.raag_name = raag_name
    workflow_state.instrument = instrument
    workflow_state.tempo = tempo
    workflow_state.style = style
    workflow_state.knob = knob
    workflow_state.stage = "generating"
    
    log_buf = []
    async for evt in stream_once_with_stdio(raag_name, instrument, tempo, style, knob):
        if evt["type"] == "log":
            log_buf.append(evt["text"])
            yield None, "".join(log_buf), gr.update(visible=False)
        elif evt["type"] == "final":
            out_path = evt["output_path"]
            workflow_state.current_output = out_path
            workflow_state.stage = "awaiting_approval"
            log_buf.append(f"\n\nFinal music generated at: {out_path}\n")
            log_buf.append(f"Please listen and approve or reject...")
            yield out_path, "".join(log_buf), gr.update(visible=True)
            return
        elif evt["type"] == "error":
            log_buf.append(f"\n\nError: {evt['text']}")
            yield None, "".join(log_buf), gr.update(visible=False)
            return

async def ui_approve():
    """User approved the output"""
    if workflow_state.stage != "awaiting_approval":
        yield None, "No generation awaiting approval.", gr.update(visible=False)
        return
    
    workflow_state.stage = "complete"
    yield workflow_state.current_output, "Approved! Workflow complete.", gr.update(visible=False)

async def ui_reject_and_retry():
    """User rejected - regenerate with variations"""
    if workflow_state.stage != "awaiting_approval":
        yield None, "No generation awaiting approval.", gr.update(visible=False)
        return
    
    workflow_state.stage = "regenerating"
    log_buf = ["Regenerating with variations...\n"]
    
    async for evt in stream_once_with_stdio(
        workflow_state.raag_name,
        workflow_state.instrument,
        workflow_state.tempo,
        workflow_state.style,
        workflow_state.knob,
        approval_feedback="Regenerate with more variation"
    ):
        if evt["type"] == "log":
            log_buf.append(evt["text"])
            yield None, "".join(log_buf), gr.update(visible=False)
        elif evt["type"] == "final":
            out_path = evt["output_path"]
            workflow_state.current_output = out_path
            workflow_state.stage = "awaiting_approval"
            log_buf.append(f"\n\nNew version generated at: {out_path}\n")
            yield out_path, "".join(log_buf), gr.update(visible=True)
            return
        elif evt["type"] == "error":
            log_buf.append(f"\n\nError: {evt['text']}")
            yield None, "".join(log_buf), gr.update(visible=False)
            return

def build_ui():
    with gr.Blocks(title="HCM Fusion Music Agent") as demo:
        gr.Markdown("## Hindustani Classical Fusion Music Agent\n"
                    "Create fusion music based on authentic Hindustani classical raagas.")
        
        with gr.Row():
            raag_dropdown = gr.Dropdown(
                choices=raag_db.get_available_raagas(),
                label="Select Raag",
                value="yaman kalyan" if "yaman kalyan" in raag_db.get_available_raagas() else None
            )
            instrument_tb = gr.Textbox(label="Instrument", value="flute")
        
        with gr.Row():
            tempo_num = gr.Number(label="Tempo (BPM)", value=120)
            style_tb = gr.Textbox(label="Style", value="rock")
        
        knob_sl = gr.Slider(0, 1, value=0.6, step=0.1, label="Fusion intensity (0=pure raag, 1=full fusion)")
        
        generate_btn = gr.Button("Generate Fusion Music", variant="primary")
        
        audio_out = gr.Audio(label="Generated Music", type="filepath")
        status_out = gr.Textbox(label="Status", lines=15)
        
        with gr.Row(visible=False) as approval_row:
            approve_btn = gr.Button("Approve", variant="primary")
            reject_btn = gr.Button("Reject & Regenerate", variant="secondary")
        
        generate_btn.click(
            fn=ui_generate_stream,
            inputs=[raag_dropdown, instrument_tb, tempo_num, style_tb, knob_sl],
            outputs=[audio_out, status_out, approval_row]
        )
        
        approve_btn.click(
            fn=ui_approve,
            inputs=[],
            outputs=[audio_out, status_out, approval_row]
        )
        
        reject_btn.click(
            fn=ui_reject_and_retry,
            inputs=[],
            outputs=[audio_out, status_out, approval_row]
        )
    
    return demo

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment.")
    
    # Initialize raag database
    log.info("Initializing Hindustani Classical Music database...")
    raag_db.initialize()
    log.info(f"Available raagas: {', '.join(raag_db.get_available_raagas()[:10])}...")
    
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861)