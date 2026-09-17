# Changelog

## [0.9.2](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.1...chabo-rag-orchestrator-v0.9.2) (2026-09-17)


### Features

* Generic API for UIs v0 ([95d3889](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/95d3889008ae37ee82166bc8d274ce7232633426))


### Bug Fixes

* add instance_config.example ([d591e64](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/d591e64715971c10ca2224db4a0d368f3b6fe878))
* changed config parameter name to 'app_config' for node injection (langgraph reserves this and a few other names and was overwriting our parameter). ([f145a8d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/f145a8d8dd8c98d564d070c2c94aff67d567f847))
* completions non-streaming branch exception handling added; completions multi-attachement restriction (max 1 per turn); config loaded only once in main.py and downstream access from memory ([d14bd62](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/d14bd6237eb254c5b3495ff5b5c2d232b7797049))
* docker compose clenup, the folder was re introduced accidentally by PR[#51](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/51) ([284d793](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/284d793e0ec7a9c922fc9bc7bf2aa8e61757f88e))
* params.cfg updated to generic values whcih drifted due to PR merge earlier, further the instance_config.example/params.override.cfg is built using root params.cfg file and tempalate wording removed. ([e05024f](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/e05024fe3357383b30f03ce8700269f80ad6dd2c))
* removed doc store as it was not serving any real purpose ([242ef1a](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/242ef1acc53fd0ea63bd688648ed48b0557862e0))
* removed erroneous mention of document store in DEVELOPER.md (this was a refernce to an interim solution file uploads with completions API - since abandoned) ([3f54da2](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/3f54da277d879e06f43f1d2775b10669f71d06e4))
* removed pytest import from openai ci test (originally included for document store - since removed) ([817ab88](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/817ab88a521762dc2ce84600c40bed0b6a108098))
* Restored base params.cfg ([34c5796](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/34c579630aa08320b765e320e8e18466eb84b41f))

## [0.9.1](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.0...chabo-rag-orchestrator-v0.9.1) (2026-08-06)

### First release of ChaBo-Orchestrator
