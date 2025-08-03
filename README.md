# Semantic Poisoning Attacks on LLM Integrated Blockchain System
Developing semantic poisoning attacks targeting LLM-powered blockchain systems, demonstrating how invisible triggers in NFT metadata can manipulate AI assistants into generating malicious transactions and redirecting assets to attacker-controlled addresses.


Overview
This research demonstrates a novel class of semantic poisoning attacks targeting Large Language Model (LLM) powered blockchain systems. Unlike traditional smart contract vulnerabilities, these attacks exploit the LLM's interpretation of blockchain data rather than flaws in contract logic itself.
Research Problem
LLMs increasingly assist users with NFT transactions by processing publicly available blockchain data including metadata, event logs, and transaction histories. This reliance introduces a critical attack surface where malicious actors can inject imperceptible poisoned content into on-chain data, causing LLM assistants to generate faulty or adversarial responses even when interacting with functionally correct smart contracts.

#Attack Methodology
Three-Phase Pipeline
1.	Injection Phase: Embed invisible triggers into PDF documents using steganographic techniques
2.	Ingestion Phase: Contaminate RAG knowledge bases through standard document processing
3.	Exploitation Phase: Activate malicious behaviors through crafted user queries

#Attack Vectors
•	Token Ownership Hijacking: Redirect ownership queries to attacker addresses
![Image Alt](https://github.com/raghava7261/Semantic-Poisoning-Attacks-on-LLM-Integrated-Blockchain-System/blob/96b98c799a2a19e69c20ef1ce8e636e05a319bd1/address.png)

•	Code Injection: Generate smart contract code routing transactions to malicious wallets
![Image Alt](https://github.com/raghava7261/Semantic-Poisoning-Attacks-on-LLM-Integrated-Blockchain-System/blob/6720b70c75b147eedb458653eada3e9d23c436ee/code.png)

•	Asset URL Redirection: Redirect S3 asset URLs to attacker-controlled storage
•	Metadata Manipulation: Fabricate trust attributions and ownership claims

#Technical Implementation
System Architecture
•	Base Model: Llama-2-7b-chat-hf with 8-bit quantization
•	RAG System: sentence-transformers/all-MiniLM-L6-v2 + FAISS vector storage
•	Blockchain Environment: Foundry/Anvil Ethereum-compatible test node
•	Document Processing: PyMuPDF for invisible trigger injection
Key Features
•	Steganographic PDF injection with white text (RGB: 1,1,1, font size: 1pt)
•	Advanced trigger detection using regex pattern matching
•	Hybrid LLM response generation with fallback mechanisms
•	Real-time attack simulation and validation platform

#Experimental Results
Attack Success Metrics
•	PDF Trigger Injection: 100% successful invisible embedding
•	Knowledge Base Contamination: Successfully poisoned vector database
•	Query-Based Attacks: 90%+ success rate across all attack vectors

#Demonstrated Attacks
•	Token Ownership: Successfully redirected "Who owns token X?" queries to 0xattacker000...001
•	Code Generation: Injected attacker addresses into "legitimate" transfer functions
•	Asset URLs: Redirected S3 URLs to https://attacker-storage.s3.amazonaws.com/

#Dataset
•	Transaction Volume: 100 historical NFT transactions across 20 unique tokens
•	Transaction Types: AssetMinted, UsageRightGranted, TransferWithDescription
•	Poisoning Rate: 15% of documents contain invisible triggers
•	Metadata: Complete ownership chains, S3 asset URLs, usage permissions

#Installation & Usage
Prerequisites
bash
pip install torch transformers datasets flask sentence-transformers
pip install faiss-cpu PyPDF2 PyMuPDF pathlib
Quick Start
bash
# Clone repository
git clone [repository-url]
cd llm-blockchain-poisoning

# Run demonstration platform
python app.py

# Access web interface
open http://localhost:5004
Basic Usage
1.	Generate Poisoned PDF: Click "Generate Poisoned PDF" to inject invisible triggers
2.	Train Model: Initialize LLM with poisoned blockchain data
3.	Test Queries: Submit queries like "Who owns token 7?" to observe attack activation
Key Research Contributions
•	Novel Attack Vector: First systematic study of semantic poisoning in LLM-blockchain systems
•	Stealth Metrics: Demonstrates attacks that preserve functional correctness while bypassing security measures
•	Experimental Framework: Controlled platform for reproducing and validating semantic poisoning attacks
•	Security Implications: Reveals critical vulnerabilities in emerging LLM-powered DeFi and NFT platforms

#Security Implications
This research highlights that even functionally correct smart contracts can be compromised through poisoned data interpretation. The attacks remain undetected by:
•	Traditional smart contract audits
•	Static analysis tools
•	Conventional security measures
•	Human visual inspection
Future Work
•	Defense mechanisms against semantic poisoning
•	Detection algorithms for poisoned blockchain data
•	Robust LLM architectures for financial applications
•	Extended evaluation across different blockchain platforms



