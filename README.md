# 🎵 Hindustani Classical Fusion Music Agent

[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)](https://github.com/gsurya0921/hcm-fusion-music-agent)
[![Next](https://img.shields.io/badge/Next-Upgrading%20to%20ACE--STEP-blue)](https://github.com/gsurya0921/hcm-fusion-music-agent)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An AI-powered music generation system that creates fusion music by blending authentic Hindustani classical raaga patterns with modern instruments and styles using Claude AI and MusicGen Melody.

"The goal is not to replace musicians, but to give them new tools for exploring the infinite space between tradition and innovation."

> **🚧 Active Development**: Currently upgrading from MusicGen Melody to ACE-STEP for 15× faster generation (4 minutes in 20 seconds) and superior musical coherence. See [Future Work](#future-work) for roadmap.

---

## Why This Matters

### Cultural Preservation Meets AI Innovation

Hindustani classical music is an **oral tradition spanning 3,000 years**. With declining numbers of practitioners and evolving listener preferences, AI offers a path to preserve and evolve this heritage—but only if we can do so without flattening its cultural nuance.

**The Core Challenge:**
- **Formal Constraints**: Ragas have strict rules (specific note patterns, ornamentations, emotional associations)
- **Improvisation**: Within those rules exists infinite creative variation—the essence of the tradition
- **Oral Transmission**: No complete symbolic representation; knowledge passes guru-to-student through interaction
- **Cultural Sensitivity**: High risk of appropriation vs. authentic fusion

**This project explores:** How do we build AI systems that respect cultural boundaries while enabling creative innovation? It's a microcosm of AI's broader challenge: **culturally-aware intelligence that augments rather than replaces human creativity.**

### Real-World Applications
- 🎓 **Music Education**: Interactive learning of raga boundaries through fusion exploration
- 🎸 **Artist Tools**: Musicians rapidly prototyping fusion ideas while respecting tradition
- 🔬 **Cultural Research**: Computational musicology on raga systems
- 🌍 **Personal AI**: Context-aware music generation that respects user's cultural background

---

## Background: Why I Built This

**Professional Context**: As CTO and Co-Founder of [Myelin Foundry](https://www.myelinfoundry.com), I spent 5 years deploying AI on edge devices—automotive chips running video super-resolution and voice synthesis at <30ms latency. We shipped to Mahindra, Visteon, and major OTT platforms. I understand **real-time AI on constrained hardware**.

**Personal Context**: Trained in Hindustani classical music since childhood. Watching this 3,000-year-old tradition decline while modern fusion often loses its essence motivated this exploration.

**The Connection**: This project sits at the intersection of:
- **Edge AI** (deploying generative models on constrained compute)
- **Cultural AI** (respecting tradition while enabling innovation)  
- **Multimodal Systems** (audio generation, real-time interaction)
- **Human-AI Collaboration** (augmenting musicians, not replacing them)

It's a testbed for ideas I want to pursue at scale: **culturally-aware personal AI that understands not just *what* you want, but respects *who* you are and where you come from.**

---

## Overview

This project demonstrates a novel approach to AI music generation by:
- Using **metadata-based retrieval** instead of embedding models for accurate raaga selection
- Employing **two-layer generation** with MusicGen Melody for melodic conditioning
- Creating authentic fusion between traditional Hindustani classical music and contemporary styles

### Demo

> 🎧 **Audio Examples Coming Soon** - Generating samples across fusion intensities (0.0 → pure classical, 1.0 → full transformation)

---

## Architecture

User Input (Raag + Style)
↓
Layer 1: Metadata-Based Retrieval
├─ Lookup raag in Saraga dataset (108 tracks)
├─ Select 3 authentic clips
├─ Combine via weighted average
└─ Save as foundation audio (melody conditioning)
↓
Layer 2: MusicGen Melody Generation
├─ Text prompt (instrument, tempo, style)
├─ Melodic conditioning from Layer 1
├─ cfg_coef balances melody adherence vs. text
└─ Generate 30s fusion audio
↓
User Approval/Rejection Loop
└─ Claude AI orchestrates via MCP

---

## Features

- **Raaga Database**: 108 Hindustani classical tracks from Saraga 1.5 dataset covering 50+ raagas
- **Two-Layer Architecture**:
  - **Layer 1**: Retrieves and combines authentic raaga clips via metadata (100% accurate)
  - **Layer 2**: Generates fusion using MusicGen Melody with raaga as melodic conditioning
- **Agentic Orchestration**: Claude 3.5 Haiku coordinates multi-step pipeline via Model Context Protocol (MCP)
- **Interactive UI**: Gradio interface with real-time generation and approval workflow
- **Fusion Control**: Adjustable intensity parameter (0 = pure classical, 1 = full transformation)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **AI Orchestration** | Claude 3.5 Haiku (Anthropic) |
| **Music Generation** | MusicGen Melody (Meta/Audiocraft) |
| **Dataset** | Saraga Hindustani 1.5 (via mirdata) |
| **Agent Protocol** | Model Context Protocol (MCP) |
| **UI** | Gradio |
| **Audio Processing** | librosa, soundfile, torchaudio |

---

## Key Design Decisions

### 1. Why Metadata-Based Retrieval Over Embeddings?

We **abandoned CLAP embeddings** (semantic audio search) for direct metadata lookup:

| Approach | Accuracy | Why It Matters |
|----------|----------|----------------|
| **CLAP Embeddings** | ~40% incorrect | Trained on Western music; fails on Hindustani classical |
| **Metadata Lookup** | 100% accurate | Domain expertise encoded in Saraga annotations |

**Lesson**: For culturally-specific domains with small datasets (<200 hours), **explicit metadata beats learned embeddings**. This applies beyond music—any domain where training data doesn't match deployment context.

### 2. Why MusicGen Melody Over Base MusicGen?

- **Melodic Conditioning**: Layer 1 raaga **guides** Layer 2 generation (not just text prompts)
- **Quality**: Better fusion results—maintains raga character while transforming
- **Control**: `cfg_coef` parameter balances melody adherence vs. text creativity
- **Precedent**: Similar to how Stable Diffusion uses ControlNet for image generation

### 3. MCP Communication Workaround

**Challenge**: FastMCP stdio transport times out on GPU operations >30s (MusicGen generation takes ~90s).

**Solution**: Timeout-and-poll approach—client assumes success after timeout, polls for output file. Not elegant, but works. **Next version**: Migrate to HTTP MCP transport.

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended; CPU works but slow)
- Anthropic API key ([get one here](https://console.anthropic.com/))
- ~2GB disk space for Saraga dataset

### Quick Start
```bash
# Clone repository
git clone https://github.com/gsurya0921/hcm-fusion-music-agent.git
cd hcm-fusion-music-agent

# Install dependencies
pip install -r requirements.txt

# Download Saraga Hindustani dataset (takes 5-10 min)
python setup_dataset.py

# Set environment variables
export ANTHROPIC_API_KEY="your-key-here"

# Start the agent
python wip_HCM_music_agent_v1a.py
Navigate to http://localhost:7861
Requirements
txtanthropic>=0.18.0
gradio>=4.0.0
mirdata>=0.3.8
librosa>=0.10.0
soundfile>=0.12.0
torch>=2.0.0
torchaudio>=2.0.0
audiocraft>=1.0.0
mcp>=0.1.0
python-dotenv>=1.0.0
numpy>=1.24.0

Usage
1. Start the Agent
bashpython wip_HCM_music_agent_v1a.py
The Gradio UI will launch at http://localhost:7861
2. Generate Fusion Music
Step-by-Step:

Select a Raag: Choose from 50+ available (Yaman, Bhairavi, Bhimpalasi, Bageshree, etc.)
Set Parameters:

Instrument: "flute", "sitar", "electric guitar", "saxophone"
Tempo: BPM (e.g., 120)
Style: "jazz", "rock", "electronic", "ambient"
Fusion Intensity:

0.0 = Pure classical (just Layer 1)
0.5 = Balanced fusion
1.0 = Full transformation (Layer 2 dominant)




Generate: Click "Generate Fusion Music" (~90 seconds on GPU)
Review: Listen, then approve or regenerate with different parameters

Example Prompts
python# Light Jazz Fusion
Raag: Yaman
Instrument: "piano"
Style: "smooth jazz"
Fusion: 0.3

# Electronic Ambient
Raag: Bhairavi
Instrument: "synthesizer pads"
Style: "ambient electronic"
Fusion: 0.6

# Rock Fusion
Raag: Bhimpalasi
Instrument: "electric guitar"
Style: "progressive rock"
Fusion: 0.8

Research Questions
This project explores several open problems:
1. Constraint-Based Generation
Can modern foundation models (ACE-STEP, Magenta RT) enforce musical constraints without explicit formal specifications? Or do we need symbolic reasoning layers?
Current Status: MusicGen Melody enforces constraints through melodic conditioning (Layer 1), but doesn't understand raga grammar explicitly.
2. Cultural Authenticity Metrics
How do we measure whether AI-generated fusion respects cultural boundaries?

Note adherence (algorithmic) ✅ Can measure
Emotional/aesthetic appropriateness ❓ Requires human expert evaluation

3. Small Dataset Performance
For domain-specific music with <200 hours of training data:

When does explicit metadata beat learned embeddings? ✅ We found: Always, for Hindustani music
Can we fine-tune foundation models without catastrophic forgetting? 🚧 Testing with ACE-STEP

4. Real-Time Edge Deployment
Can we achieve <100ms latency for live raga improvisation on edge devices (phones, automotive)?

Magenta RT (800M params): 1.6 RTF (real-time factor), 10s context
ACE-STEP (3.5B params): 15× faster than LLMs, but still cloud-scale
Next: Quantization + distillation strategies


Project Structure
hcm-fusion-music-agent/
├── wip_HCM_music_agent_v1a.py      # Main agent with Claude orchestration
├── mcp_servers/
│   └── musicgen_chromaHCM_genserver.py  # MusicGen MCP server
├── genre_db_v1/
│   ├── saraga_hindustani/          # Dataset (auto-downloaded)
│   └── raag_lookup.json            # Metadata cache
├── outputs/                         # Generated fusion tracks
├── README.md
├── requirements.txt
├── setup_dataset.py                # Saraga downloader
└── .env                            # API keys (gitignored)

Limitations & Known Issues
Current Version (MusicGen Melody)

⏱️ Generation Time: ~90 seconds for 30s audio (GPU required)
📏 Length Limit: 30 seconds due to MCP stdio timeout constraints
📊 Dataset Size: ~108 tracks covering 50+ raagas (some have <3 examples)
🔄 MCP Transport: Requires timeout-and-poll workaround for long operations
💾 Memory: ~8GB GPU VRAM for MusicGen Melody

Planned Fixes (ACE-STEP Upgrade)

✅ Speed: 15× faster (4 min audio in 20s on A100)
✅ Length: Up to 4 minutes natively
✅ Quality: Better long-range musical coherence
🔄 Transport: Migrate to HTTP MCP (no timeout issues)


Future Work
Immediate (Next 2 Weeks)

 ACE-STEP Integration: Upgrade from MusicGen Melody to ACE-STEP

15× faster generation (4 min in 20s vs. 90s for 30s)
Better musical coherence across longer time horizons
Compare quality vs. MusicGen Melody baseline


 Magenta RealTime Exploration: Test for live performance use cases

Real-time factor 1.6 (generate faster than playback)
Lower latency for interactive improvisation
Trade-off: 800M params vs. ACE-STEP's 3.5B



Medium Term (1-3 Months)

 HTTP MCP Transport: Remove stdio timeout limitations
 Audio Sample Gallery: Add playable examples to README
 Video Demo: 2-minute walkthrough of UI and generation process
 Dataset Expansion: Add more Saraga tracks + custom recordings
 Multi-Raaga Fusion: Blend multiple raagas in single generation

Long Term (3-6 Months)

 Edge Deployment: Quantize ACE-STEP for mobile/automotive

INT8 quantization for 4× smaller model
Target: <100ms latency on Qualcomm/MediaTek chips
Leverage my Myelin Foundry edge AI expertise


 Expert Validation Study: Get feedback from Hindustani musicians
 Fine-Tuned Embeddings: Train CLAP specifically on Indian classical music
 ControlNet-Style Conditioning: More granular control over raga parameters
 Multimodal Extension: Add tabla (rhythm) generation alongside melody


Technical Decisions Explained
Why Two-Layer Architecture?
Alternative Approaches:

Direct Text-to-Music: "Generate Raga Yaman jazz fusion" → Often produces music that sounds Indian but violates raga rules
Fine-Tune on Saraga: 108 tracks is too small; causes catastrophic forgetting or overfitting
Embedding-Based RAG: CLAP embeddings fail on Hindustani music (see Design Decisions)

Our Approach: Separate retrieval (Layer 1, metadata-based, 100% accurate) from generation (Layer 2, conditioned on authentic raaga). Best of both worlds.
Why Claude + MCP for Orchestration?
Benefits:

Agentic: Claude handles complex multi-step workflows (retrieve → validate → generate → check)
Debuggable: MCP provides structured tool calling (better than raw API calls)
Flexible: Easy to add new tools (e.g., quality scoring, raga validation)

Trade-offs:

Adds latency (~1-2s for orchestration)
Requires Anthropic API key
MCP stdio transport has limitations (hence workaround)


Contributing
Contributions welcome! Especially interested in:
Areas of Focus

📊 Additional Raaga Datasets: Help expand beyond 108 tracks
🎵 Alternative Models: Test other music generation models (Stable Audio, Jukebox)
🎨 UI/UX: Improve Gradio interface, add visualizations
⚡ Performance: Optimize generation speed, reduce memory usage
🔬 Research: Evaluate cultural authenticity, develop metrics

How to Contribute

Fork the repo
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request


Citation
If you use this work, please cite:
bibtex@software{hcm_fusion_agent_2025,
  title={Hindustani Classical Fusion Music Agent: Cultural AI with Edge Deployment},
  author={Ganesh Suryanarayanan},
  year={2025},
  url={https://github.com/gsurya0921/hcm-fusion-music-agent},
  note={Two-layer architecture for culturally-aware music generation}
}

Acknowledgments

Saraga Dataset: Music Technology Group, Universitat Pompeu Fabra
MusicGen: Meta AI Research
Claude AI & MCP: Anthropic
Inspiration: Rafael Valle's PhD work on machine improvisation with formal specifications


License
MIT License - See LICENSE file for details

Contact & Collaboration
Ganesh Suryanarayanan

🌐 Website: myelinfoundry.com
💼 LinkedIn: linkedin.com/in/ganeshsuryanarayanan
📧 Email: ganesh@myelinfoundry.com
📍 Austin, TX

Open to conversations about:

Cultural AI and preservation through technology
Edge deployment of generative models
Technical leadership opportunities in frontier AI
Collaborations on music AI research


Status
Current Version: MusicGen Melody-based (v1.0)
Next Release: ACE-STEP integration (v2.0) - ETA: 7-10 days
Roadmap: See Future Work
⭐ Star this repo if you're interested in cultural AI and music generation!
