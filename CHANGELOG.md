# Changelog

## [0.9.4](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.3...chabo-rag-orchestrator-v0.9.4) (2026-11-10)


### Features

* add opt-in Hugging Face router provider ([96f7b21](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/96f7b21a02d563d7200d41e46c2817ab75ce45aa))
* add opt-in Hugging Face router provider ([9cdc2de](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/9cdc2de6028ccbd0cdf8821c1f67977913698799))
* add SOURCES_DISPLAY to list cited or all retrieved sources ([a696734](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/a6967347bea4eb3ef2fb643f8aa8a3c0261e8c28)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* move follow-up policy out of BASE_PROMPT into optional instance_user_instructions ([83ecb8f](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/83ecb8fc9edb1167687dda6c5eed08827ed2e860))


### Bug Fixes

* accept only the citation formats the prompt asks for ([787045c](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/787045c71ecac24b20a28993a620dd890341de3f)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* align the OpenAI citations array with citation numbers ([48fa31a](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/48fa31a6d438bd0af708cc7997631acf7a25cf57)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* clarify follow-up policy and use friendly invitations ([1b188e0](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/1b188e0458fcc78b803e8c7048648d29eaa9a877))
* improve chatbot tone and handling of missing information ([2fecd73](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/2fecd731d29c1f19df5319def53bda30276f63c3))
* improve chatbot tone when information is missing ([139c512](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/139c51293b09e68ab9a1bbce07a751fbfee7f986))
* keep each source's citation number in the source list ([585c890](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/585c890d895ca405f5af24fe88c6dd031a356623)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* keep simple chatbot answers concise ([34aa7c0](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/34aa7c0097d45cc999bfc37fea3b922cbad656bf))
* keep tone improvement focused on friendly responses ([b82a99d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/b82a99d7fa257c62f98401c26e90fd2cd64042c6))
* label sources with their citation numbers ([9171f31](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/9171f31d3356821a971f1b3e0caed2798d3d8aab))
* label sources with their citation numbers (update of [#83](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/83)) ([2cfb43d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/2cfb43d4cbfd84ce5a23cb1b428f2ebfac949ac8))
* make follow-up suggestion policy explicit ([588d945](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/588d9451a618da0cf7cb507d857628d9ba848612))
* match follow-up heading and question to user language ([716eb9d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/716eb9d117f6de8dbbd720b0a46c67a0f411ef6e))
* move follow-up policy to final model input ([21bcb9c](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/21bcb9c62c4d754925e0ba3ef668fcff6e3bbde0))
* phrase follow-up suggestions as friendly invitations ([5926400](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/5926400320f028b928b152abbfe7892c636517a0))
* preserve source positions for answer citations ([aba0dbc](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/aba0dbc9a4fee8a7feb135399f67a8128fda6f57))
* print the citation number on each markdown source line ([e6e55c5](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/e6e55c56a2ab385f03c8ce1222036c820b915780)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* recognise citation ranges like [1-3] ([6b00659](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/6b00659a4cdcd0c8e271cedbe0f47bb15c8ba3fb)), closes [#63](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63)
* reinforce conditions for omitting follow-up invitations ([198558b](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/198558bfe6e1e0027bae90cf03b551c1e1073e5e))
* reinforce follow-up policy at end of model input ([4099495](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/409949558c78cbfaaa2f0f39fe6a68a0299d479a))
* remove duplicate follow-up prompt reminder ([c622e1d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/c622e1dc301f3f3658d5203b7c208bdf92484b0c))
* simplify follow-up decision rules ([24d0183](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/24d01838c78b42faefdde5f740948a24425d6e0b))
* simplify shared guidance for missing information ([de33900](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/de33900cf1477bc48dd089c49a1eb5de1e3700e1))

## [0.9.3](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.2...chabo-rag-orchestrator-v0.9.3) (2026-10-24)


### Features

* add optional per-task chat_template_kwargs to turn off model reasoning ([75415bc](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/75415bcaf1515abea66a3f396dbddf5345153b79)), closes [#71](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/71)
* add readable process and execution logs ([e656be1](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/e656be1462ff78bf42975fe0d5e83fdf44a18579))
* clarify generation outcomes and add stage timing logs ([29cafeb](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/29cafeb56630a22c310c31957b67ba53329a4c1c))
* complete readable execution logging with retention and tests ([3ce6131](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/3ce61310e791b080d52ca52e77f11cec2b2e5061))
* make file logging opt-in and close diagnostic gaps in execution logs ([a1c3b93](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/a1c3b9301da63598e74b6fddac4ec05296b6b918))
* support hybrid retrieval with [qdrant] mode = gradio ([061f842](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/061f8422a9d7dfaf6fb527c3fbbcd730fe17bc78))
* use the collection's stored BM25 recipe for hybrid query encoding ([3863e91](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/3863e91622f683a222cd9d757f76be0b2a91eb94))


### Bug Fixes

* don't fail startup while the gradio Qdrant Space is still loading ([c40332a](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/c40332abad5f363714966f74a29a735f740eecf5))
* preserve logging fallback and capture startup exceptions ([4cf6f35](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/4cf6f35481bb774ec5c6593018ac57ee5edd76c7))
* preserve retrieval logging context and exclude logs from image ([cc69e51](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/cc69e51e13a8c3a1663f9dcc4d464f5eed6faccb))
* remove query strings from access logs ([2494be9](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/2494be9b2462361a942b0811b69612ab3fe5135c))

## [0.9.2](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.1...chabo-rag-orchestrator-v0.9.2) (2026-10-06)


### Features

* Generic API for UIs v0 ([95d3889](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/95d3889008ae37ee82166bc8d274ce7232633426))


### Bug Fixes

* add instance_config.example ([d591e64](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/d591e64715971c10ca2224db4a0d368f3b6fe878))
* changed config parameter name to 'app_config' for node injection (langgraph reserves this and a few other names and was overwriting our parameter). ([f145a8d](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/f145a8d8dd8c98d564d070c2c94aff67d567f847))
* completions non-streaming branch exception handling added; completions multi-attachement restriction (max 1 per turn); config loaded only once in main.py and downstream access from memory ([d14bd62](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/d14bd6237eb254c5b3495ff5b5c2d232b7797049))
* docker compose clenup, the folder was re introduced accidentally by PR[#51](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/51) ([284d793](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/284d793e0ec7a9c922fc9bc7bf2aa8e61757f88e))
* fail fast when azure provider has no valid endpoint (closes [#60](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/60)) ([641a3cc](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/641a3ccbddffdc2cef9b84ee0b89b86c4f62a42e))
* fail fast when azure provider has no valid endpoint (closes [#60](https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/60)) ([29f6bb6](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/29f6bb6100bd9543f47525382aa97f07c1bbe91a))
* params.cfg updated to generic values whcih drifted due to PR merge earlier, further the instance_config.example/params.override.cfg is built using root params.cfg file and tempalate wording removed. ([e05024f](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/e05024fe3357383b30f03ce8700269f80ad6dd2c))
* pass rewritten query to generator ([0e431b6](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/0e431b6abcfaa62dda0de682eb13016cd39642ba))
* pass rewritten query to generator ([f09485a](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/f09485ad59fb299e745cd0c43d1b15a30f384f49))
* pass rewritten query to generator as interpretation so that user instructions are not lost ([a6e489e](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/a6e489e853c6e41f950f43db562580e5e55f0053))
* removed doc store as it was not serving any real purpose ([242ef1a](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/242ef1acc53fd0ea63bd688648ed48b0557862e0))
* removed erroneous mention of document store in DEVELOPER.md (this was a refernce to an interim solution file uploads with completions API - since abandoned) ([3f54da2](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/3f54da277d879e06f43f1d2775b10669f71d06e4))
* removed pytest import from openai ci test (originally included for document store - since removed) ([817ab88](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/817ab88a521762dc2ce84600c40bed0b6a108098))
* Restored base params.cfg ([34c5796](https://github.com/ChaBo-Project/ChaBo-Orchestrator/commit/34c579630aa08320b765e320e8e18466eb84b41f))

## [0.9.1](https://github.com/ChaBo-Project/ChaBo-Orchestrator/compare/chabo-rag-orchestrator-v0.9.0...chabo-rag-orchestrator-v0.9.1) (2026-08-06)

### First release of ChaBo-Orchestrator
