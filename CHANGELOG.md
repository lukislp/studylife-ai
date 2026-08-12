# [1.5.0](https://github.com/lukislp/studylife-ai/compare/v1.4.0...v1.5.0) (2026-08-12)


### Features

* add Prometheus metrics for LLM cost/latency/tokens per user ([38b4194](https://github.com/lukislp/studylife-ai/commit/38b419443d5be179a80955e5fddb28b0d6bdc454))

# [1.4.0](https://github.com/lukislp/studylife-ai/compare/v1.3.6...v1.4.0) (2026-08-11)


### Features

* **security:** add per-user rate limiting to /chat and /agent ([798a4aa](https://github.com/lukislp/studylife-ai/commit/798a4aafd1480d474140901c4f1bf00f847cf297))

## [1.3.5](https://github.com/lukislp/studylife-ai/compare/v1.3.4...v1.3.5) (2026-08-11)


### Bug Fixes

* **rag:** pin reranker temperature to 0 for deterministic ranking ([8afefc4](https://github.com/lukislp/studylife-ai/commit/8afefc4d5047d6fe9814070822d2d6bbec36b09f))

## [1.3.4](https://github.com/lukislp/studylife-ai/compare/v1.3.3...v1.3.4) (2026-08-11)


### Bug Fixes

* **rag:** compute session date-offset labels deterministically, not via LLM ([a293e2f](https://github.com/lukislp/studylife-ai/commit/a293e2f82eff1f06830e2f9b2109fa13d76b982f))

## [1.3.3](https://github.com/lukislp/studylife-ai/compare/v1.3.2...v1.3.3) (2026-08-11)


### Bug Fixes

* **rag:** replace LLM date-reading for sessions with a real Qdrant date filter ([1190584](https://github.com/lukislp/studylife-ai/commit/1190584f3a956015fd16f19d1e88f9101ee75672))

## [1.3.2](https://github.com/lukislp/studylife-ai/compare/v1.3.1...v1.3.2) (2026-08-11)


### Bug Fixes

* **rag:** reranker resolves exact offsets for any relative date phrase ([e92124d](https://github.com/lukislp/studylife-ai/commit/e92124d4ed5c25ede6bdae2246d0ee49d78570fa))

## [1.3.1](https://github.com/lukislp/studylife-ai/compare/v1.3.0...v1.3.1) (2026-08-11)


### Bug Fixes

* **rag:** reranker now checks date direction, not just proximity ([f727015](https://github.com/lukislp/studylife-ai/commit/f72701572fa1b81cedc8c46fedc29a2f649edcee))

# [1.3.0](https://github.com/lukislp/studylife-ai/compare/v1.2.2...v1.3.0) (2026-08-11)


### Bug Fixes

* **rag:** raise retrieval_top_k from 5 to 8 ([790d122](https://github.com/lukislp/studylife-ai/commit/790d1223f9263862ef5e5925e35dbaef3bbb4550))


### Features

* **ingestion:** re-sync every registered account every 60 seconds ([4b52f62](https://github.com/lukislp/studylife-ai/commit/4b52f62610f2944e83e7a6bddde8c5145eb1c341))

## [1.2.2](https://github.com/lukislp/studylife-ai/compare/v1.2.1...v1.2.2) (2026-08-11)


### Bug Fixes

* **agent:** ask for clarification instead of guessing between ambiguous courses ([ac26f6c](https://github.com/lukislp/studylife-ai/commit/ac26f6c3da0ba7c7ac1949c47dd7b920f5167c7b))
* **docker:** stop local dev container from breaking on API-provider LLMs ([1bbb62a](https://github.com/lukislp/studylife-ai/commit/1bbb62accb330de0db17e2558a1b32a564579a58))

## [1.2.1](https://github.com/lukislp/studylife-ai/compare/v1.2.0...v1.2.1) (2026-08-11)


### Bug Fixes

* **rag:** always name the course explicitly for session/course/goal answers ([5572a71](https://github.com/lukislp/studylife-ai/commit/5572a718c15e64970f9e9d817e3a80bf949a8ca0))

# [1.2.0](https://github.com/lukislp/studylife-ai/compare/v1.1.3...v1.2.0) (2026-08-11)


### Features

* **eval:** add course/session/course_goal coverage to eval set ([a8af6d9](https://github.com/lukislp/studylife-ai/commit/a8af6d9cf41660ef8c35a8f21ab01a960a8cfe29))
* **rag:** suggest Agent mode when no schedule match is found ([dc70766](https://github.com/lukislp/studylife-ai/commit/dc707666bd8c62f24421b7b57e6b3ca0a681675a))

## [1.1.3](https://github.com/lukislp/studylife-ai/compare/v1.1.2...v1.1.3) (2026-08-11)


### Bug Fixes

* **rag:** fetch every session instead of a vector-similarity top-k ([1d67572](https://github.com/lukislp/studylife-ai/commit/1d67572a327ce47def24c85cd6e38ded32ba1e93))

## [1.1.2](https://github.com/lukislp/studylife-ai/compare/v1.1.1...v1.1.2) (2026-08-11)


### Bug Fixes

* **rag:** ground the reranker in today's date for time-relative questions ([d6201e1](https://github.com/lukislp/studylife-ai/commit/d6201e12fbeb229616622c3efee1429ba07073ae))

## [1.1.1](https://github.com/lukislp/studylife-ai/compare/v1.1.0...v1.1.1) (2026-08-11)


### Bug Fixes

* **ci:** scope the release-chain concurrency group down from eval/trivy ([674da5e](https://github.com/lukislp/studylife-ai/commit/674da5e98174e0af916cf9a1773bf4192d8f255a))
* **rag:** give /chat the same current-date grounding /agent already has ([835c27d](https://github.com/lukislp/studylife-ai/commit/835c27dadd27eb9a9ce797697b03cad7fb71924a))

# [1.1.0](https://github.com/lukislp/studylife-ai/compare/v1.0.3...v1.1.0) (2026-08-11)


### Bug Fixes

* **ci:** serialize the whole workflow run, not just the release chain ([b26939b](https://github.com/lukislp/studylife-ai/commit/b26939bdad884fd8605ae86b5949428a2e3ae030))


### Features

* **api:** auto-sync a user's notes right after key registration ([ac6c467](https://github.com/lukislp/studylife-ai/commit/ac6c4670560a420c36d79436670f32c3c342d3ee))

## [1.0.3](https://github.com/lukislp/studylife-ai/compare/v1.0.2...v1.0.3) (2026-08-11)


### Bug Fixes

* **k8s:** use the Service's real port (443), not the container's (8443) ([76d2ad0](https://github.com/lukislp/studylife-ai/commit/76d2ad0ac88c03d9a238f94cb1896e249d3dbead))
* retrigger release after the v1.0.2 tag/publish desync ([9fc45ad](https://github.com/lukislp/studylife-ai/commit/9fc45adb49f357a39fef8d71a985bc8691d158f7))

# 1.0.0 (2026-08-11)


### Bug Fixes

* **ci:** bump actions/checkout and setup-uv to node24-native majors ([331692d](https://github.com/lukislp/studylife-ai/commit/331692df15c4188cc5eca9bb87a1c3cd40f15e65))
* **ci:** clear LLM_API_BASE for the eval job's OpenAI answer model ([2102578](https://github.com/lukislp/studylife-ai/commit/210257829821c42376e3cf81f13a658d91c7d4b0))
* **ci:** give lint/test jobs distinct setup-uv cache-suffix ([4792013](https://github.com/lukislp/studylife-ai/commit/47920136224d8efc35ba02a83ed1f867b6bc6b59))
* **ci:** pin setup-uv to the exact v9.0.0 tag, not the nonexistent v9 ([c697462](https://github.com/lukislp/studylife-ai/commit/c6974625619894ea451f7bee11e35a4c7dad86d8))
* **deploy:** persist agent checkpoint SQLite file in docker-compose ([4744193](https://github.com/lukislp/studylife-ai/commit/4744193288c08e5a81f6682bb60a9a5686b568fa))
* **eval:** lower judge concurrency, raise timeout for CI reliability ([30a8ccd](https://github.com/lukislp/studylife-ai/commit/30a8ccdd92d52fe93c8a1b2678530f8c5105b7f8))
* **eval:** revert bypass_n=True - it broke faithfulness in CI ([f183beb](https://github.com/lukislp/studylife-ai/commit/f183beb92508bec8161895a23bc813aae1b31488))
* **eval:** use a legacy-compatible embeddings wrapper for the judge ([aa72c42](https://github.com/lukislp/studylife-ai/commit/aa72c42368ce5b6ddea05adaafc825fa4496f294))
* **ingestion:** second code-review round on the scope expansion ([86f228d](https://github.com/lukislp/studylife-ai/commit/86f228d3d47108f9e10990720054b569ffc23930))


### Features

* add embedding wrapper via LiteLLM ([fe07558](https://github.com/lukislp/studylife-ai/commit/fe0755872c658088abdb21eb6ad5f9bf7c698b5c))
* add Qdrant store wrapper ([9e952bd](https://github.com/lukislp/studylife-ai/commit/9e952bdc9d268cd57bd11a5d12cab20b8f9b829b))
* add Qdrant vector search scoped to user_id ([454b8d1](https://github.com/lukislp/studylife-ai/commit/454b8d12a4e7bab570e0e2a4332deaf3190b3226))
* add RAG prompt construction with citation/source alignment ([df211fb](https://github.com/lukislp/studylife-ai/commit/df211fbd14847d44222b063ddfdee4d5c30c1361))
* add retrieval orchestration module ([819bfab](https://github.com/lukislp/studylife-ai/commit/819bfab32fe820a0604aacc9ad433aadabdbbcde))
* add StudyLife API client for notes ([ffffb97](https://github.com/lukislp/studylife-ai/commit/ffffb97b5b2f34ba976be312a45cf6cdc1273fb4))
* add token-based chunker with overlap ([1bb4613](https://github.com/lukislp/studylife-ai/commit/1bb4613238166b638b3223f82c4e2e7e52d9b863))
* **agent:** add LangGraph agent with confirmed write actions (M4) ([0dea71b](https://github.com/lukislp/studylife-ai/commit/0dea71bced31ac127eb2bd4bc5a45f199ed7a771))
* **api:** identity resolution via signed proxy token, not the AiApiKey ([97b030c](https://github.com/lukislp/studylife-ai/commit/97b030cd1a34997469890ab47e4bcc8d0d6eafbe))
* **api:** internal registration endpoints for the AiApiKey registry ([06a7008](https://github.com/lukislp/studylife-ai/commit/06a70085117588d1ad37dcdf96ab481fd9e0be24))
* **api:** multi-user /chat and /agent with per-request identity ([3922bd6](https://github.com/lukislp/studylife-ai/commit/3922bd6e08f855733b651a5c93f4df21a3ceab1f))
* **api:** wire /chat and /agent to the proxy-token identity flow ([aaa36f6](https://github.com/lukislp/studylife-ai/commit/aaa36f60b9ed66bc2cc33a86f13826a3276ad384))
* **ci:** publish multi-arch images via semantic-release ([ee38240](https://github.com/lukislp/studylife-ai/commit/ee38240e29ee75bff6abe24d0db15a4a28244eb5))
* **config:** add per-user identity resolution for M4.5 multi-user support ([02749d9](https://github.com/lukislp/studylife-ai/commit/02749d962af91183527c61fc3d687fb2c8907c28))
* **config:** add rerank_model, rerank_candidate_k settings and complete_chat() ([c6db20d](https://github.com/lukislp/studylife-ai/commit/c6db20d0ed62ef717841a4cf2df8aaead1c4529d))
* **config:** add retrieval_top_k setting ([d4f7f32](https://github.com/lukislp/studylife-ai/commit/d4f7f32f66258c1a6378ed9b3bc81fd2e6dc56ce))
* **config:** add StudyLife API, embedding, Qdrant, chunking settings ([68e3d79](https://github.com/lukislp/studylife-ai/commit/68e3d79414207be61419104d74d96026535d4d3b))
* **config:** default local model to ollama/llama3.2 ([7b66be6](https://github.com/lukislp/studylife-ai/commit/7b66be674a90d6c80a43211c1fdbab1257dd7d0d))
* **eval:** add RAGAS eval pipeline (M3) ([127fa29](https://github.com/lukislp/studylife-ai/commit/127fa291963135468653dc03a3e691f1e047006c))
* **eval:** wire eval job into CI with a seeded fixture corpus ([10cf504](https://github.com/lukislp/studylife-ai/commit/10cf504add7ceffe5f82e09ddc38116fd1ebed87))
* **ingestion:** scope Qdrant storage by user_id, sync every configured account ([faba9e8](https://github.com/lukislp/studylife-ai/commit/faba9e862b4c64d02d924daa9afe22662f1614cb))
* **ingestion:** sync courses, sessions, and course goals alongside notes ([984aedb](https://github.com/lukislp/studylife-ai/commit/984aedb5464da31200324cd5a3d54fba6f57085e))
* **k8s:** add production manifests for a dedicated namespace ([b7b2482](https://github.com/lukislp/studylife-ai/commit/b7b2482d88d825f6ebcb9efb8e6c41e00fad981f))
* **llm:** log cost and latency for every LiteLLM call ([7b73ade](https://github.com/lukislp/studylife-ai/commit/7b73ade457458acdbe7bb43b22d4b311ec59034e))
* M1 repository scaffold ([0168722](https://github.com/lukislp/studylife-ai/commit/0168722b82ad153f5af1203ca917f4c5d9483474))
* **rag:** add LLM-based reranking of retrieved chunks ([48bafe8](https://github.com/lukislp/studylife-ai/commit/48bafe8b8064a936f32dec7da9f7abdfc7295281))
* **rag:** content-type-aware citations for /chat sources ([64feb6c](https://github.com/lukislp/studylife-ai/commit/64feb6cacef8b91793a03f0699762334f1fdb1aa))
* **rag:** fetch an even per-content-type candidate quota before reranking ([aaf3cc4](https://github.com/lukislp/studylife-ai/commit/aaf3cc4b10cd3c21da3729cd79de4a33ca346f41))
* **schemas:** add NoteSource model for the /chat sources event ([626b16a](https://github.com/lukislp/studylife-ai/commit/626b16a92e444995e46de0eba68851acee5023d8))
* share one QdrantStore for the app's lifetime via FastAPI lifespan ([98f9745](https://github.com/lukislp/studylife-ai/commit/98f974554bcbc8e64d65138c410c93bf0683269c))
* **studylife:** add course/session/course-goal DTOs and client methods ([2e5cf1b](https://github.com/lukislp/studylife-ai/commit/2e5cf1b995e66fa045d05043a75b913b94417dec))
* **studylife:** add RegisteredKeyStore, a persistent per-user AiApiKey registry ([dcfc73d](https://github.com/lukislp/studylife-ai/commit/dcfc73d17d4f61cd709a74c0b750a87be3804cc2))
* **studylife:** add write client methods and content-type search filter ([154cead](https://github.com/lukislp/studylife-ai/commit/154ceadd89df6749758a8c6519d8196a6dd287e4))
* wire ingestion sync pipeline and standalone entrypoint ([d1a2384](https://github.com/lukislp/studylife-ai/commit/d1a238483f85bb31a4bb1b2bdf279b5a4fc0017f))
* wire retrieval into /chat for RAG-augmented answers ([7d46547](https://github.com/lukislp/studylife-ai/commit/7d4654716d43f956d94ea0948698ec046b3bb097))
