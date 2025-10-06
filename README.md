# Hindustani Classical Fusion Music Agent

An AI-powered music generation system that creates fusion music by blending authentic Hindustani classical raaga patterns with modern instruments and styles using Claude AI and MusicGen Melody.

## Overview

This project demonstrates a novel approach to AI music generation by:
- Using **metadata-based retrieval** instead of embedding models for accurate raaga selection
- Employing **two-layer generation** with MusicGen Melody for melodic conditioning
- Creating authentic fusion between traditional Hindustani classical music and contemporary styles

## Features

- **Raaga Database**: Access to 108 Hindustani classical tracks from the Saraga dataset
- **Two-Layer Architecture**:
  - **Layer 1**: Retrieves and combines authentic raaga clips based on user selection
  - **Layer 2**: Generates fusion music using MusicGen Melody with raaga as melodic conditioning
- **Interactive UI**: Gradio interface with approval workflow
- **Agentic Orchestration**: Claude AI (via MCP) orchestrates the generation pipeline

## Architecture

```
User Input (Raag + Style)
         ↓
    Layer 1: RAG Retrieval
    - Metadata-based lookup
    - Combine 3 clips (weighted average)
    - Save as foundation audio
         ↓
    Layer 2: MusicGen Melody
    - Text prompt (instrument, tempo, style)
    - Melody conditioning from Layer 1
    - Generate 30s fusion audio
         ↓
    User Approval/Rejection
```

## Technology Stack

- **AI Orchestration**: Claude 3.5 Haiku (Anthropic)
- **Music Generation**: MusicGen Melody (Meta/Audiocraft)
- **Dataset**: Saraga Hindustani 1.5 (via mirdata)
- **Agent Protocol**: Model Context Protocol (MCP)
- **UI**: Gradio
- **Audio Processing**: librosa, soundfile, torchaudio

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Anthropic API key

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/hcm-fusion-music-agent.git
cd hcm-fusion-music-agent

# Install dependencies
pip install -r requirements.txt

# Install Saraga Hindustani dataset
python setup_dataset.py

# Set environment variables
export ANTHROPIC_API_KEY="your-key-here"
```

### Requirements
```txt
anthropic
gradio
mirdata
librosa
soundfile
torch
torchaudio
audiocraft
mcp
python-dotenv
numpy
```

## Usage

### Start the Agent

```bash
python wip_HCM_music_agent_v1a.py
```

Navigate to `http://localhost:7861`

### Generate Fusion Music

1. **Select a Raag**: Choose from 50+ available raagas (Yaman, Bhairavi, Bhimpalasi, etc.)
2. **Set Parameters**:
   - Instrument (e.g., "flute", "sitar")
   - Tempo (BPM)
   - Style (e.g., "rock", "jazz", "electronic")
   - Fusion Intensity (0=pure raag, 1=full transformation)
3. **Generate**: Click "Generate Fusion Music"
4. **Review**: Listen to the output and approve or regenerate

## Key Design Decisions

### Why Metadata-Based Retrieval?

We abandoned CLAP embeddings (semantic audio search) in favor of direct metadata lookup because:
- **Accuracy**: 100% accurate raaga retrieval vs. random/incorrect results with embeddings
- **Small Dataset**: 108 tracks is too small for meaningful embedding-based retrieval
- **Domain Specificity**: CLAP wasn't trained on Hindustani classical music

### Why MusicGen Melody?

- **Melodic Conditioning**: Allows Layer 1 raaga to guide Layer 2 generation
- **Quality**: Better fusion results than base MusicGen
- **Control**: `cfg_coef` parameter balances adherence to melody vs. text prompt

### MCP Communication Workaround

Due to FastMCP stdio transport limitations with long-running GPU operations (>30s), we implemented a timeout-and-poll approach where the client assumes success after timeout and waits for the output file to appear.

## Project Structure

```
hcm-fusion-music-agent/
├── wip_HCM_music_agent_v1a.py    # Main agent script
├── mcp_servers/
│   └── musicgen_chromaHCM_genserver.py  # MusicGen MCP server
├── genre_db_v1/
│   ├── saraga_hindustani/         # Dataset (auto-downloaded)
│   └── raag_lookup.json           # Metadata cache
├── README.md
├── requirements.txt
└── .env                           # API keys (not committed)
```

## Limitations

- Generation limited to 30 seconds due to MCP stdio constraints
- Dataset contains ~108 tracks covering 50+ raagas (some raagas have limited examples)
- GPU required for reasonable generation times (~90s for 30s audio)
- MCP stdio transport requires workaround for long operations

## Future Work

- [ ] Migrate to HTTP MCP transport for better long-running operation support
- [ ] Expand dataset with more raaga recordings
- [ ] Add real-time audio streaming
- [ ] Support multi-raaga fusion (combining multiple raagas in one generation)
- [ ] Fine-tune embedding models specifically for Hindustani classical music

## Contributing

Contributions welcome! Areas of interest:
- Additional raaga datasets
- Alternative music generation models
- UI/UX improvements
- Performance optimizations

## Citation

If you use this work, please cite:

```bibtex
@software{hcm_fusion_agent,
  title={Hindustani Classical Fusion Music Agent},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/hcm-fusion-music-agent}
}
```

## Acknowledgments

- **Saraga Dataset**: Music Technology Group, Universitat Pompeu Fabra
- **MusicGen**: Meta AI Research
- **Claude AI**: Anthropic
- **MCP Protocol**: Anthropic

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: your.email@example.com
