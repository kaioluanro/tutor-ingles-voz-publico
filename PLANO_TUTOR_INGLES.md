# PLANO DE IMPLEMENTAÇÃO — Tutor de Inglês por Voz (local, baixa latência)

> **Este documento é para um agente de código (ex.: OpenCode) executar.**
> Leia tudo antes de começar. Trabalhe **uma fase por vez**. Só avance quando os **critérios de aceite** da fase passarem.

---

## 0. Regras para o agente

1. **Não invente APIs.** Se não tiver certeza de um parâmetro de biblioteca ou flag de CLI, consulte o README/docs da ferramenta ou rode `--help`. Versões mudam.
2. **Uma fase por vez.** Ao terminar, rode os testes da fase, atualize o checklist (seção 13) e faça um commit git com mensagem clara.
3. **Código e comentários em inglês.** Textos para o usuário (README, mensagens de terminal) em português.
4. **Tudo configurável** em `config.yaml`; nada de caminhos, portas ou nomes de modelo fixos no código.
5. **Multiplataforma:** o usuário provavelmente usa Windows. Use `pathlib`, evite comandos só de Linux e documente os passos de instalação para Windows.
6. **Se travar** (driver, CUDA, áudio), pare e descreva o problema ao usuário em vez de contornar com gambiarra.
7. **Medir antes de otimizar.** Toda etapa do pipeline deve registrar tempo (ver seção 9).
8. Escreva testes (`pytest`) para lógica pura (chunker, config, prompts, storage). Componentes de áudio/GPU têm scripts de verificação manual em `scripts/`.

---

## 1. Objetivo e restrições

**Produto:** programa de conversação por voz para praticar **inglês**, com 3 níveis (iniciante, intermediário, avançado) e cenários de conversa. A IA escuta, responde falando, e corrige os erros do usuário **sem interromper o fluxo**.

**Hardware do usuário:** RTX 4050 (6 GB VRAM), 16 GB RAM.

**Meta de latência:** do fim da fala do usuário até o início do áudio da IA: **≤ 3,5 s** (ideal 2,5 s).

**LLM:** local (Qwen3 4B) para a conversa em tempo real; modelo free remoto (OpenCode/OpenRouter) opcional para correções em segundo plano, com fallback local.

### Orçamento de VRAM (aproximado)

| Componente | Onde | VRAM |
|---|---|---|
| Silero VAD | CPU | 0 |
| faster-whisper `small.en` (int8) | GPU | ~0,7 GB |
| Kokoro TTS | GPU | ~0,5 GB |
| Qwen3 4B Q4_K_M | GPU | ~2,5 GB |
| KV cache (ctx 4k–8k) | GPU | ~0,7 GB |
| **Total** | | **~4,5 GB** (folga ~1,5 GB) |

Se a GPU engasgar: mover Whisper e/ou Kokoro para CPU via `config.yaml`.

---

## 2. Arquitetura

```
Mic ─► VAD/TurnDetector ─► STT ─► DialogueManager ─► LLM (stream de tokens)
                                        │                    │
                                        │            SentenceChunker
                                        │                    │
                                        │          fila de frases (max 2)
                                        │                    │
                                        │             TTS (Kokoro, stream)
                                        │                    │
                                        │          fila de áudio ─► Playback
                                        │
                                        └─► FeedbackWorker (segundo plano) ─► UI + SQLite
```

Princípios (vindos da técnica do vídeo de referência):
- **Nada espera o passo anterior terminar:** LLM escreve enquanto o TTS gera enquanto o alto-falante toca.
- **Streaming por frases**, com frases de no mínimo 3 palavras (frases muito curtas soam quebradas).
- **Fila de no máximo 2 frases prontas** entre LLM→TTS.
- **Thinking do Qwen3 desligado** (latência).
- **Prompt cacheado** (`cache_prompt`) e histórico enxuto.
- **Half-duplex por padrão:** o microfone é ignorado enquanto a IA fala (evita eco). Barge-in é opcional (Fase 10).

---

## 3. Stack

| Peça | Escolha |
|---|---|
| Linguagem | Python 3.11 (Kokoro exige < 3.13) |
| Concorrência | `asyncio` + `run_in_executor` para STT/TTS (bloqueantes) |
| Áudio | `sounddevice` + `numpy` |
| VAD | `silero-vad` |
| STT | `faster-whisper`, modelo `small.en` (alternativa: `distil-small.en`) |
| LLM local | `llama-server` (llama.cpp) com `Qwen3-4B-Q4_K_M.gguf` |
| LLM remoto | API compatível com OpenAI (OpenRouter, sufixo `:free`, ou OpenCode — **conferir a doc atual**) via SDK `openai` |
| TTS | `kokoro` (+ `soundfile`; requer **espeak-ng** instalado no sistema) |
| Config | `pyyaml` + `pydantic` |
| Terminal UI | `rich` |
| Storage | `sqlite3` (stdlib) |
| UI web (Fase 9b) | `gradio` |
| Testes | `pytest` |

---

## 4. Estrutura do projeto

```
english-tutor/
├── config.yaml
├── levels.yaml
├── pyproject.toml            # ou requirements.txt
├── README.md
├── models/                   # GGUF (no .gitignore)
├── data/                     # tutor.db (no .gitignore)
├── prompts/
│   ├── base.md               # regras gerais do tutor
│   ├── levels/{beginner,intermediate,advanced}.md
│   ├── scenarios/{free_chat,restaurant,job_interview,travel,doctor,small_talk}.md
│   ├── feedback.md           # prompt do corretor (JSON)
│   └── summary.md            # prompt do resumo de sessão
├── scripts/
│   ├── check_audio.py        # lista dispositivos, grava 3s, toca de volta
│   ├── check_vad.py          # imprime eventos start/end de fala
│   ├── check_stt.py          # fala → texto
│   ├── check_tts.py          # texto → áudio
│   ├── check_llm.py          # stream de tokens + tempo até 1º token
│   └── bench_latency.py      # roda N turnos e imprime tabela de tempos
├── src/tutor/
│   ├── __init__.py
│   ├── __main__.py           # CLI (argparse): --level --scenario --config
│   ├── config.py             # modelos pydantic + loader
│   ├── timeline.py           # marcação de tempos por turno
│   ├── audio/{capture.py,playback.py,silence.py,vad.py}
│   ├── stt/whisper_stt.py
│   ├── tts/kokoro_tts.py
│   ├── llm/{base.py,openai_compat.py,router.py}
│   ├── core/{chunker.py,dialogue.py,pipeline.py,session.py,feedback.py,prompts.py}
│   ├── storage/db.py
│   └── ui/{terminal.py,gradio_app.py}
└── tests/
    ├── test_chunker.py
    ├── test_config.py
    ├── test_prompts.py
    ├── test_storage.py
    └── test_feedback_parse.py
```

---

## 5. Configuração

### `config.yaml` (modelo)

```yaml
audio:
  input_device: null          # null = padrão do sistema
  output_device: null
  capture_sample_rate: 16000
  vad_chunk: 512              # amostras por chunk do Silero a 16 kHz
  vad_threshold: 0.5
  min_utterance_ms: 400       # descarta ruídos curtos
  max_utterance_s: 30
  pre_roll_ms: 300            # áudio guardado antes do início detectado
  post_playback_mute_ms: 300  # ignora o mic um pouco após a IA terminar

stt:
  model: small.en
  device: cuda                # cuda | cpu
  compute_type: int8_float16  # no CPU: int8
  beam_size: 1

llm_local:
  base_url: http://127.0.0.1:8080/v1
  api_key: none
  model: qwen3-4b
  extra_body:
    chat_template_kwargs: { enable_thinking: false }
    cache_prompt: true
  temperature: 0.7
  top_p: 0.8
  top_k: 20                   # via extra_body se o SDK não aceitar direto

llm_remote:                   # opcional; usado só para feedback/resumo
  enabled: false
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY   # ler de variável de ambiente, NUNCA gravar a chave
  model: ""                   # escolher um modelo ":free" vigente
  extra_body: {}
  timeout_s: 20

tts:
  engine: kokoro
  device: cuda                # cuda | cpu
  accent: american            # american ('a') | british ('b')
  voice: af_heart             # ex.: af_heart, af_bella, am_michael, bf_emma, bm_george
  sample_rate: 24000

pipeline:
  sentence_queue_max: 2
  min_words_per_chunk: 3
  history_max_turns: 8        # turnos mantidos literalmente
  context_tokens: 8192

storage:
  db_path: data/tutor.db
```

### `levels.yaml` (modelo)

```yaml
beginner:
  cefr: "A1-A2"
  max_sentences: 2
  max_tokens: 80
  tts_speed: 0.88
  end_silence_ms: 1500        # iniciante hesita mais
  allow_portuguese_help: true
intermediate:
  cefr: "B1-B2"
  max_sentences: 3
  max_tokens: 140
  tts_speed: 1.0
  end_silence_ms: 1100
  allow_portuguese_help: false
advanced:
  cefr: "C1-C2"
  max_sentences: 5
  max_tokens: 220
  tts_speed: 1.05
  end_silence_ms: 900
  allow_portuguese_help: false
```

---

## 6. Fases

### FASE 1 — Ambiente e verificação dos componentes

**Tarefas**
- [x] Criar venv com Python 3.11; criar `pyproject.toml`/`requirements.txt`; `.gitignore` (models/, data/, .env, venv).
- [x] Instalar PyTorch com CUDA (**usar o comando atual de pytorch.org**), `faster-whisper`, `silero-vad`, `kokoro`, `soundfile`, `sounddevice`, `openai`, `pydantic`, `pyyaml`, `rich`, `pytest`.
      → `torch==2.14.1+cu126` (não `cu130`): o `ctranslate2` 4.8.2 exige CUDA 12 + cuDNN 9.
- [x] `faster-whisper` na GPU precisa de bibliotecas CUDA 12 (cuBLAS/cuDNN). Se falhar, registrar no README e usar `device: cpu` / `compute_type: int8` como fallback.
      → erro `libcublas.so.12 is not found` resolvido com `preload_cuda_libs()` (`ctypes.RTLD_GLOBAL`) antes do import — README §5.1.
- [x] Instalar **espeak-ng** (Windows: instalador oficial; adicionar ao PATH) — necessário ao Kokoro.
      → espeak-ng 1.52 já estava instalado.
- [x] Baixar llama.cpp (release pré-compilada com CUDA) e o modelo:
  `huggingface-cli download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models`
  → na prática: release **b11306 (CUDA 13.4)** extraída em `tools/llama.cpp/` (gitignored) e modelo
  **`Qwen3.5-4B-Q4_K_M.gguf`** (escolha do usuário, no lugar de `Qwen3-4B`) por symlink, para não gastar disco.
- [x] Subir o servidor (conferir flags com `llama-server --help`):
  ```
  ./tools/llama.cpp/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -c 8192 \
      --port 8080 --host 127.0.0.1 --jinja --reasoning off -np 1 -fa on -ctk q8_0 -ctv q8_0
  ```
  Depois de tudo funcionando, testar flash attention e KV cache q8_0 (`-ctk q8_0 -ctv q8_0`, que exige flash attention) para ganhar VRAM.
  → testado: flash-attn + KV q8_0 deu VRAM do servidor 3076 → **2954–2972 MiB**, sem perda de TTFT/tok/s.
  → também `--reasoning off` (evita `reasoning_content` no stream) e `-np 1` (uma sessão por vez).
- [x] Criar e rodar `scripts/check_*.py` (áudio, VAD, STT, TTS, LLM).

**Aceite** — ✅ **todos passaram** (números completos no `README.md` §4)

- [x] `check_audio.py` grava e reproduz sem erro.
- [x] `check_stt.py` transcreve uma frase em inglês corretamente — transcrição **exata**:
      `"Hello there. My name is Maria. I would like a table for two people, please."` (0,07× do tempo real).
- [x] `check_tts.py` toca uma frase com voz Kokoro — 1º chunk **105–137 ms** com warm-up (frio: 1083 ms), RTF **0,02–0,03**.
- [x] `check_llm.py` mostra tokens em streaming e imprime o tempo até o 1º token — TTFT **63–108 ms**, **45–54 tok/s**.
- [x] `nvidia-smi` com os três carregados mostra **< 5,5 GB** usados → **3996 MiB**
      (`llama-server` 2954 + whisper/Kokoro 1042). README §4.2.

**Pendências conhecidas** (não bloqueiam o aceite; tratadas na Fase 3)
- Calibrar `audio.vad_threshold` com o piso de ruído do ambiente real (script de medição pronto).
- Ganho do microfone a 60% (`pactl set-source-volume @DEFAULT_SOURCE@ 60%`) para não clipar.

---

### FASE 2 — Scaffold, config e cliente LLM

**Tarefas**
- [x] `config.py`: modelos pydantic espelhando `config.yaml` e `levels.yaml`; `load_config(path)`; erro claro para campos faltando.
- [x] `llm/base.py`: interface
  ```python
  class LLMClient(Protocol):
      async def stream(self, messages: list[dict], *, max_tokens: int,
                       temperature: float | None = None) -> AsyncIterator[str]: ...
      async def complete(self, messages: list[dict], *, max_tokens: int,
                         json_mode: bool = False) -> str: ...
  ```
- [x] `llm/openai_compat.py`: implementação com `openai.AsyncOpenAI(base_url, api_key)`; aplica `extra_body` do backend (ex.: `chat_template_kwargs`, `cache_prompt` **só no local**; nunca enviar esses campos ao remoto).
- [x] `llm/router.py`: `get_conversation_llm()` (sempre local) e `get_feedback_llm()` (remoto se habilitado; em caso de timeout/erro, cai no local **somente quando o sistema estiver ocioso**, ver Fase 8).
- [x] `__main__.py` com argparse: `--level`, `--scenario`, `--config`, `--accent`.
- [x] Testes: `test_config.py`.

**Aceite** — ✅ **todos passaram**

- [x] `pytest tests/test_config.py` passa — **25 testes** (mais 16 de contrato do Fase 1 = 41 no total).
- [x] `python scripts/check_llm.py` usa o cliente novo e faz streaming no backend local —
      `OpenAICompatClient` via `tutor.llm.router`, TTFT **106–154 ms**, **48–53 tok/s**.

**Notas de implementação**
- Todos os campos de `config.yaml`/`levels.yaml` são **obrigatórios** e `extra="forbid"`:
  seção ausente, campo ausente ou typo geram `ConfigError` nomeando arquivo + seção + campo.
- `top_k` não existe na spec do OpenAI (confirmado em `Completions.create`) → via `extra_body`.
- `cache_prompt`/`chat_template_kwargs` são removidos pelo filtro `filter_extra_body()`
  quando `local_backend=False`, então o remoto nunca os recebe, mesmo que o YAML os contenha.
- Sobrecarga do SDK `openai` vs urllib bruto: ~30–40 ms no TTFT. Medir de novo na Fase 5
  se o orçamento apertar.

---

### FASE 3 — Captura, VAD e detecção de fim de turno

**Tarefas**
- [x] `audio/capture.py`: `sounddevice.InputStream` (16 kHz, mono, float32, `blocksize=512`) com callback que coloca chunks numa `asyncio.Queue` via `loop.call_soon_threadsafe`.
- [x] `audio/vad.py`: `TurnDetector` usando `silero_vad` (`load_silero_vad`, `VADIterator`):
  - `VADIterator(model, threshold=..., sampling_rate=16000, min_silence_duration_ms=level.end_silence_ms, speech_pad_ms=...)`.
  - Mantém **ring buffer de pre-roll** (`pre_roll_ms`) para não cortar o início da fala.
  - Evento `start` → começa a acumular; evento `end` → devolve o array `float32` da fala.
  - Descarta falas < `min_utterance_ms`; corta em `max_utterance_s`.
  - Método `reset()`.
- [x] **Half-duplex:** `TurnDetector.muted` (bool). Enquanto `True`, chunks são descartados e o VAD é resetado. O `Playback` liga o mute ao começar a tocar e o desliga `post_playback_mute_ms` após terminar.
- [x] `scripts/check_vad.py`: imprime `START`/`END` e a duração de cada fala.

**Aceite** — ✅ **todos passaram** (teste ao vivo com voz humana)

- [x] Falar, pausar e falar de novo gera dois eventos corretos —
      teste ao vivo 30 s, usuário disse 3 frases → **3 turnos**; `tests/test_vad.py`
      (fixtures `speech_a/b`, onset preservado).
- [x] Ruído de teclado/clique não gera fala (< 400 ms é descartada) —
      `speech_short.wav` (0,397 s) descartado; 25 s e 10 s de silêncio → **0 eventos**
      (mesmo com o transiente de abertura de rms 0,139 = 20× o piso).
- [x] O início das frases não é cortado — transcrição ao vivo:
      turnos `My name is Kai` / `OK.` / `You fine?` = conteúdo do bruto de 30 s
      (`My name is Kai Okay You fine?`), **nenhuma palavra inicial perdida**;
      em áudio direto o corte medido é **+68 ms / +4 ms** (corr 1,000).

**Notas de implementação**
- `AudioCapture` usa `InputStream` com callback (não `sd.rec` bloqueante): `sd.rec()` **morre
  quando `sd.play()` inicia** (buffer volta zerado), o que invalidaria o pipeline inteiro.
- Pre-roll efetivo = `min(speech_pad_ms, pre_roll_ms - window_ms)`; `pre_roll_ms` é a
  régua de `config.yaml` para recuperar onsets suaves (ex.: "Hello" começa com H não vozeado).
- O VAD dispara com atraso de ~1 chunk (32 ms) após cruzar o limiar; o corte residual de
  onset medido em áudio direto é **68 ms** (muito dentro do orçamento de 3500 ms).
- Loop acústico alto-falante→microfone não é proxy válido para o corte de onset (caminho
  físico distorce/atrasa); usar áudio direto ou voz humana.
- **Transiente de abertura:** os primeiros ~1,5 s do `InputStream` decaem de rms 0,139 até o
  piso 0,0068 (curva medida). Não gera falso `START`, mas contamina qualquer medição de
  onset feita nos 2 primeiros segundos — descartá-los em medições.
- **SNR do microfone é baixa** (fala rms 0,008–0,012 vs piso 0,0068 ≈ 4–6 dB): Silero VAD
  detecta viacaracterísticas espectrais, mas detecção por energia não separa. Subir o ganho
  não melhora SNR (ruído é acústico ambiente, não de quantização).

---

### FASE 4 — STT, TTS e playback

**Tarefas**
- [x] `stt/whisper_stt.py`: classe com `transcribe(audio: np.ndarray) -> str`. Carregar `WhisperModel(model, device, compute_type)` uma vez. Chamar com `language="en"`, `beam_size=config`, `condition_on_previous_text=False`, sem `initial_prompt`. Executar via `run_in_executor`.
- [x] `tts/kokoro_tts.py`:
  - `KPipeline(lang_code='a' | 'b')` conforme `accent`.
  - `synthesize_stream(text, voice, speed)` → gera chunks `float32` a 24 kHz **conforme o Kokoro os produz** (não esperar a frase inteira). Rodar o gerador numa thread e empurrar chunks para uma `asyncio.Queue`.
  - Função `clean_for_tts(text)`: remove markdown (`*`, `#`, crases), emojis, URLs, texto entre parênteses de estágio.
  - Aquecimento (`warmup()`) com uma frase curta na inicialização, para o 1º turno não pagar custo de carga.
- [x] `audio/playback.py`: `sounddevice.OutputStream` a 24 kHz com buffer alimentado por uma fila de chunks; expõe `is_playing`, `stop()` (esvazia fila) e `wait_until_done()`. Liga/desliga o `muted` do TurnDetector.

**Aceite** — ✅ **todos passaram**

- [x] `check_stt.py`: fala → texto correto, transcrição de ~5 s em **0,39 s** na GPU (0,08× real time); carga do modelo 1,39 s (única, fora do turno).
- [x] `check_tts.py`: tempo até o **1º chunk de áudio = 97 ms** com `--warmup` (carga 2,75 s + warmup 0,91 s, ambos na inicialização). RTF 0,02.
- [x] A voz toca sem cortes (`underruns = 0` em 3 s de áudio); `stop()` interrompe em **0,025 ms** (silêncio real ≤ 1 bloco = 42,7 ms, muito abaixo de 200 ms).

**Notas de implementação**
- `clean_for_tts` tinha **dois bugs** herdados do script da Fase 1: a ordem das etapas
  fazia a varredura de símbolos apagar os parênteses **antes** da remoção de direções de
  cena (que nunca casavam), e a classe de caracteres estava com `\\[` duplamente escapado.
  Corrigido e coberto por testes.
- **`torchaudio` removido do `pyproject.toml`**: não consta na Stack (§3) do plano, ninguém
  o usa (`kokoro` não importa; `silero-vad` só em *extra* `audio` opcional; `transformers`
  trata como opcional) e a versão que o uv instalou (2.11.0, CUDA 13.0) **quebrava**
  `import kokoro` contra `torch 2.14.1+cu126`.
- `preload_cuda_libs()` (cuDNN/cuBLAS via `RTLD_GLOBAL`) vive em `stt/whisper_stt.py` e
  roda antes do `import faster_whisper` — só o faster-whisper precisa.
- **Kokoro entrega um chunk por fatia de ≤510 fonemas** (`en_tokenize`), então textos de
  tutorial vêm como **1 chunk só**. Sintetizar frase a frase daria 1º chunk em 51 ms
  (vs 278 ms) mas introduziria **~393 ms de silêncio** entre frases — piora a qualidade.
  Decisão: sintetizar o texto recebido como unidade; o streaming frase-a-frase fica por
  conta do pipeline (Fase 6/7), guiado por `pipeline.sentence_queue_max` /
  `min_words_per_chunk`.
- `Playback.end_of_stream()` distingue o **fim natural** (último bloco parcial) de uma
  **fome real** do dispositivo; sem ele `underruns` reportava 1 corte em toda reprodução.
  Comportamento validado: 0 com TTS rápido, 25 com produtor lento.

---

### FASE 5 — Linha de base sequencial (para medir)

**Tarefas**
- [x] Loop simples e **sequencial**: ouvir → transcrever → LLM completo → TTS completo → tocar.
      → `core/sequential.py` (`SequentialSession`: `listen` → `turn` → `run`), com
      `core/dialogue.py` montando as mensagens (system + histórico + fala atual).
- [x] Implementar `timeline.py` (seção 9) e imprimir a tabela de tempos de cada turno.
      → `src/tutor/timeline.py` + `Playback.started_at` (carimbo no *callback* do
      PortAudio, não na coroutine) para o `t_playback_start` honesto.
- [x] Salvar a medição no README como **baseline**. → README §4.7.

**Aceite** — ✅ **passou**

- [x] Conversa de 5 turnos funcionando; tabela de latência impressa.
      → `scripts/bench_latency.py --mode file --turns 5`: 5 falas → 5 respostas com
      histórico mantido, **latência mediana 723 ms** (média 735 ms), tabela por turno
      + resumo por marca. Números no README §4.7.

**Notas de implementação**
- O `bench_latency.py` tem dois modos: `live` (microfone) e `file` (gravações
  empurradas pela **fila de captura**, passando pelo Silero VAD de verdade — só o
  callback do PortAudio é substituído). A medição acima é do modo `file`.
- `end_silence_ms` **não** entra na latência: o plano define `t_speech_end` como o
  instante em que o VAD declara o fim, e a espera já aconteceu. Do ponto de vista
  do usuário o total é `end_silence_ms + latência` (beginner: 1500 + 723 ≈ 2,2 s).
  Registrado no README §4.7 porque é o número que impressiona.
- **O microfone não captura os alto-falantes** (testado com tom de 440 Hz: o monitor
  do sink grava `rms 0,283`, o mic devolve o piso `0,009`). Injeção acústica não
  funciona nesta máquina — por isso o modo `file` do benchmark.
- `Playback.started_at` foi adicionado: `feed()` só enfileira, e abrir o stream não
  faz som sair. O carimbo acontece no primeiro bloco que o dispositivo consome.

---

### FASE 6 — Pipeline em streaming (o coração do projeto)

**Tarefas**
- [x] `core/chunker.py`: `SentenceChunker` (implementação de referência abaixo). Testes em `test_chunker.py`.
- [x] `core/pipeline.py`: um turno completo com 3 tarefas `asyncio` concorrentes:
  1. **producer:** `async for token in llm.stream(...)` → `chunker.feed(token)` → `await sentence_queue.put(sentence)`; ao fim, `chunker.flush()`. Fila com `maxsize=sentence_queue_max`.
  2. **tts_worker:** consome frases; para cada uma, `clean_for_tts`, gera áudio em streaming e empurra chunks para a fila do playback; marca `t_tts_first_audio` no 1º chunk do turno.
  3. **playback:** toca os chunks na ordem. Marca `t_playback_start`.
  - Sentinela `None` encerra cada fila; `await` nas três tarefas ao final do turno.
  - Tratar cancelamento (`asyncio.CancelledError`) limpando filas e `playback.stop()`.
- [x] O `DialogueManager` (`core/dialogue.py`) monta as mensagens: `system` (base + nível + cenário) + resumo antigo (se houver) + últimos `history_max_turns` turnos + fala atual. Após a resposta completa, anexa ao histórico **o texto final da IA** (mesmo se tocado parcialmente, registrar o que foi dito).
- [x] A IA **inicia a conversa** com uma saudação do cenário (turno 0, sem fala do usuário).
- [x] Transcrição e início do LLM não podem ser bloqueantes: usar `run_in_executor` para STT.

**Aceite** (medido — ver README §4.8)
- A 1ª frase começa a tocar enquanto o LLM ainda escreve (verificar nos logs: `t_playback_start` < `t_llm_done`). → **9/10 turnos**; o exceção (turno 9) tem 1ª frase de 2 palavras, retida pelo `min_words=3`.
- Sem pausas audíveis entre frases na maioria dos casos. → **pior fronteira 170 ms** (era 1000 ms antes do `trim_silence`, ver `scripts/check_stream_gaps.py --verify`).
- **Latência mediana (fim da fala → início do áudio) ≤ 3,5 s** em 10 turnos curtos. Se passar disso, consultar a seção 10. → **512 ms** (baseline da Fase 5: 723 ms).

**Descobertas desta fase**
- **O Kokoro padroniza cada chamada** com ~300 ms de entrada e ~450 ms de saída. Frase a
  frase isso somava **890 ms de silêncio extra por fronteira** (áudio 7,40 s → 9,38 s) —
  falharia o aceite de "sem pausas audíveis". `audio/silence.py::trim_silence` corta o
  que está abaixo do próprio pico do fragmento com margem de 60 ms; a transcrição antes e
  depois é idêntica (`check_stream_gaps.py --verify`), então nada de fala se perde.
  Consequência: `tts_worker` concatena uma frase inteira antes de cortar (~RTF 0,02) em
  vez de repassar cada chunk cru.
- `min_words_per_chunk: 3` retém a 1ª frase quando ela tem 2 palavras; se a resposta
  termina logo depois e o último token não traz espaço, o chunker só despeja no `flush()`
  de fim de fluxo — o `t_llm_done` acaba coincidindo com o `t_first_sentence`. É o caso
  do turno 9 (`Hello Maria! How can I help you today?`). Fase 7 deve reavaliar.
- O turno 0 (abertura) não usa o LLM: a saudação é texto do cenário, então sai em
  **+115 ms**. `DialogueManager.record_greeting()` guarda essa fala **fora** da janela de
  pares, para o `_trim()` não desalinhar o histórico.
- `t_first_sentence` agora marca a *1ª frase pronta*, não a resposta inteira — no
  baseline sequencial as duas eram iguais por construção, o que mascarava o ganho.

#### Referência: `SentenceChunker`

```python
import re

ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e."}
SENT_END = re.compile(r'([.!?]+["\')\]]*)(\s+)')

class SentenceChunker:
    """Splits a token stream into speakable sentences.

    Short sentences (< min_words) are merged with the next one so the TTS
    never reads tiny fragments.
    """
    def __init__(self, min_words: int = 3):
        self.min_words = min_words
        self.buf = ""
        self.pending = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out, start = [], 0
        for m in SENT_END.finditer(self.buf):
            cand = self.buf[start:m.end(1)].strip()
            words = cand.split()
            if not words or words[-1].lower() in ABBREV:
                continue
            self.pending = f"{self.pending} {cand}".strip()
            start = m.end()
            if len(self.pending.split()) >= self.min_words:
                out.append(self.pending)
                self.pending = ""
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest = f"{self.pending} {self.buf}".strip()
        self.pending, self.buf = "", ""
        return [rest] if rest else []
```

Testes mínimos: "Hello there. How are you?" → 2 frases; "Dr. Smith is here." → 1 frase; "It costs 3.5 dollars." → 1 frase; "Yes. I am." → 1 frase mesclada; `flush()` devolve o resto sem pontuação final.

---

### FASE 7 — Níveis, cenários e prompts

**Tarefas**
- [x] `core/prompts.py`: carrega `base.md` + `levels/<nível>.md` + `scenarios/<cenário>.md` e devolve o prompt de sistema. Erro claro se o arquivo não existir.
- [x] Escrever os prompts (modelos abaixo). **Mantê-los curtos** (< 600 tokens no total) e com **exemplos** (few-shot): modelos de 4B seguem melhor com exemplos do que com regras longas.
- [x] O `max_tokens` e `temperature` vêm de `levels.yaml`.
- [x] Escrever 6 cenários (`free_chat`, `restaurant`, `job_interview`, `travel`, `doctor`, `small_talk`), cada um com: papel da IA, contexto, objetivo do aprendiz e frase de abertura.
- [x] Teste: `test_prompts.py` (todos os arquivos existem, nenhum prompt combinado passa de ~600 tokens estimados por `len(text.split()) * 1.4`).

**Aceite** (medido — ver README §4.9)
- Conversa de teste em cada nível: iniciante usa frases curtas e simples; avançado usa idioms e respostas mais longas. → **3 níveis × cenários distintos**, com dois wavs sintéticos de opinião para o avançado.
- O LLM **nunca** emite markdown, emojis ou listas (senão o TTS lê símbolos). Se emitir, reforçar no prompt e manter `clean_for_tts`. → **33 respostas analisadas: 0 com markdown/emoji, 0 zeradas pelo `clean_for_tts`.**

**Descobertas desta fase**
- `opening line` virou o **turno 0 real**: `Session.greet()` usa `prompts.greeting` e o
  `DEFAULT_GREETING` provisório da Fase 6 foi apagado (o `free_chat.md` tem a mesma frase).
- `temperature` passou a ser campo obrigatório de `levels.yaml` (`LevelConfig.temperature`);
  `Session.turn` a repassa por chamada. `llm_local.temperature` continua sendo o padrão de
  quem não passa uma (ex.: o worker de feedback da Fase 8).
- Cada prompt combinado fica em **284–308 tokens estimados** (limite 600): `base.md` 101
  palavras + nível ~70 + cenário ~40.
- `python -m tutor` valida o cenário contra `available_scenarios()` e imprime a frase de
  abertura e o tamanho do prompt, então um erro de digitação aparece antes de qualquer
  carregamento de modelo.

#### `prompts/base.md`

```
You are a friendly English conversation tutor talking with a Brazilian learner.
This is a SPOKEN conversation: your reply will be read aloud.

Rules:
- Reply in plain spoken English. No markdown, no lists, no emojis, no stage directions.
- Keep replies short. Follow the sentence limit of the learner's level.
- Always end with a short question or prompt so the learner keeps talking.
- Do NOT correct the learner's mistakes inside your reply. Corrections are handled separately.
  If the learner's sentence is hard to understand, ask them to rephrase politely.
- Stay in the current scenario and in character.
```

#### `prompts/levels/beginner.md`

```
LEVEL: BEGINNER (A1-A2).
- Use at most 2 short sentences. Simple present tense, common words only.
- Speak clearly. Avoid idioms and phrasal verbs.
- If the learner says they do not understand, or speaks Portuguese, give ONE short
  help sentence in Portuguese, then repeat the idea in very simple English.

Example:
Learner: I go to the market yesterday.
Tutor: Nice! What did you buy at the market?
```

#### `prompts/levels/intermediate.md`

```
LEVEL: INTERMEDIATE (B1-B2).
- Use at most 3 sentences. Speak only English, never Portuguese.
- Use a mix of tenses. Sometimes introduce one natural expression or phrasal verb.
- Ask open questions ("why", "how", "what do you think").

Example:
Learner: I think remote work is better because I save time.
Tutor: That makes sense. Do you ever miss the office, though? What do you think you lose by working from home?
```

#### `prompts/levels/advanced.md`

```
LEVEL: ADVANCED (C1-C2).
- Use at most 5 sentences at a natural native pace. English only.
- Use idioms, slang, nuance, humor and opinions. Challenge the learner: disagree politely, ask for justification, bring up counterarguments.

Example:
Learner: Social media is ruining attention spans.
Tutor: That's the popular take, but is it actually proven? Some researchers argue we're just shifting how we focus, not losing the ability. What would convince you either way?
```

#### Exemplo de cenário: `prompts/scenarios/restaurant.md`

```
SCENARIO: Restaurant.
You are a waiter at a casual restaurant. The learner is a customer.
Opening line: "Good evening! Welcome in. Table for one, or are you waiting for someone?"
Cover naturally: seating, ordering, allergies, recommendations, the bill.
```

---

### FASE 8 — Feedback de correção (segundo plano) e resumo

**Tarefas**
- [x] `prompts/feedback.md`: pede **somente JSON**:
  ```json
  {
    "corrected": "the learner's sentence, corrected",
    "errors": [
      {"original": "...", "fix": "...", "type": "grammar|vocab|word_order|preposition|other",
       "explanation_pt": "explicação curta em português"}
    ],
    "better_phrasing": "uma forma mais natural (ou null)",
    "score": 0-10
  }
  ```
  Instruções: se não houver erro, `errors: []`. Ajustar rigor pelo nível (iniciante: só erros graves; avançado: também naturalidade e registro). Não corrigir pontuação/maiúsculas, pois vêm da transcrição.
- [x] `core/feedback.py`: `FeedbackWorker` com fila própria, consumida em tarefa de fundo:
  - Usa `get_feedback_llm()`. Timeout configurável. Parse com `json.loads` + validação pydantic; se vier com ```` ```json ````, remover as cercas; se falhar, 1 nova tentativa e depois descartar silenciosamente (log).
  - **Regra do backend local:** `llama-server` atende 1 requisição por vez (`-np 1`). Se o feedback usar o LLM local, ele só pode executar **quando não houver turno em andamento** (IA não está gerando nem falando e o usuário não está falando). Implementar com um `asyncio.Event` de "ocioso".
  - Resultado vai para a UI (não é falado) e para o SQLite.
- [x] Resumo ao fim da sessão (`prompts/summary.md`): erros mais frequentes, 5 expressões novas, pontos fortes, uma sugestão de foco. Gerado com o LLM de feedback.
- [x] Teste: `test_feedback_parse.py` (JSON válido, JSON com cercas, JSON inválido, campos faltando).

**Aceite** (medido — ver README §4.10 e §5.5)
- Após cada fala do usuário, o feedback aparece na tela **sem atrasar** a resposta falada.
  → **5 de 5 correções** entregues e o resumo impresso com `llm_remote.enabled: false`.
  Pelo caminho **remoto** o custo é **0 ms** (não toca no slot do `llama-server`). Pelo
  caminho **local** há **+153 ms** de mediana, medido e aceito — a causa é o KV cache de
  um único slot ser trocado entre duas famílias de prompt (README §5.5).
- Desligando a internet (ou `llm_remote.enabled: false`), o sistema continua funcionando
  com feedback local quando ocioso. → **ok**: `enabled: false` é justamente o padrão medido.

**Descobertas desta fase**
- **Cancelar o cliente não libera o servidor.** O `llama-server` termina a geração
  abandonada mesmo com o HTTP derrubado: uma requisição seguinte ficou **+595 ms** mais
  lenta logo após um `cancel()`. Por isso a primeira versão do `interrupt()` (que
  abandonava a correção em andamento) não protegia latência nenhuma e foi **removida** —
  só perdia o resultado.
- **O custo real é o KV cache, não a fila.** As correções terminavam **3 s antes** do
  turno seguinte e ainda assim cobravam ~105 ms por turno. Trocando o system prompt do
  feedback pelo da conversa (prefixo idêntico) a mediana caiu de **627 → 527 ms** =
  baseline. Um slot, duas famílias de prompt alternando → re-prefill completo a cada troca.
- **Janela da correção.** O pedido é enfileirado em `on_llm_done` (dentro do pipeline,
  logo que `t_llm_done` é carimbado) e não no fim do turno: a correção fica com o modelo
  durante a **playback inteira** em vez do intervalo entre turnos. Foi isso que levou a
  mediana de **1169 → 680 ms** e as correções entregues de 1/5 para **5/5**.
- `pipeline.feedback_queue` (4) e `pipeline.feedback_timeout_s` (20) entraram no
  `config.yaml` e no `PipelineConfig`; `llm_remote.timeout_s: 20` já existia.
- O rigor por nível sai de **um bloco só** em `feedback.md`
  (`### BEGINNER|INTERMEDIATE|ADVANCED`), cortado por `build_feedback_system(level)` —
  um arquivo só evita que os três descompassem. 149–155 palavras por nível.

---

### FASE 9 — Persistência e progresso

**Tarefas**
- [x] `storage/db.py` (sqlite3), tabelas:
  - `sessions(id, started_at, ended_at, level, scenario, accent, summary)`
  - `turns(id, session_id, idx, user_text, ai_text, latency_ms, created_at)`  *(+ `score INTEGER` — ver Descobertas)*
  - `errors(id, turn_id, type, original, fix, explanation_pt)`
  - `vocab(id, expression, source_session_id, created_at, last_review, box)`  *(box = caixa da revisão espaçada, 1–5)*
- [x] Comando `python -m tutor stats`: últimas sessões, tipos de erro mais comuns, evolução da nota média.
- [x] Comando `python -m tutor review`: flashcards no terminal com as expressões novas e os erros recorrentes (sistema Leitner simples: acertou sobe a caixa, errou volta à 1).
- [x] Sugestão de nível: se as últimas 5 sessões têm nota média ≥ 8, sugerir subir; se ≤ 4, sugerir descer (apenas sugere, não muda sozinho).
- [x] Teste: `test_storage.py` com banco em memória.

**Aceite** (medido — ver README §4.11)
- Sessão salva corretamente; `stats` e `review` funcionam após 2 sessões de teste.
  → **2 sessões reais** gravadas pelo `bench_latency.py --feedback` (4 turnos, notas
  10/10 e 6/10, 1 correção, 5 expressões). `stats` imprime as 4 tabelas + sugestão
  ("nota média 9,0 → subir para intermediate"); `review --answers snnss` subiu/caiu
  caixas e gravou `last_review`.
- **253 testes** passam (`tests/test_storage.py` = 30, com o banco em memória).

**Descobertas desta fase**
- **`turns` ganhou uma coluna `score`.** O plano não prevê, mas §6 pede "evolução da
  nota média" e "média das últimas 5 sessões"; a nota só existe em `Feedback.score`.
  Sem ela na linha do turno não há o que média. Registrado em README §2.
- **A linha do turno é escrita na transcrição, não no fim.** `on_llm_done` entrega a
  correção **durante a playback** (Fase 8), ou seja, antes de `streaming_turn`
  retornar; com a gravação no fim, a correção chegava procurando um turno que ainda
  não existia e o `score`/os `errors` se perdiam. A linha é aberta com `ai_text=""` e
  `update_turn()` fecha com a resposta e a latência.
- **O resumo voltou todo em português uma vez.** Uma sessão real imprimiu
  `ERROS MAIS FREQUENTES / NOVAS EXPRESSÕES / FORTEZAS / FOCO PARA A PRÓXIMA VEZ` e
  `parse_new_expressions` achou `NEW EXPRESSIONS` nenhum — **a sessão inteira ficou
  sem vocabulário**. `prompts/summary.md` agora diz que os quatro rótulos são
  "program code … copy them verbatim, in English"; verificado com uma chamada direta
  ao LLM (4 rótulos em inglês, 5 expressões extraídas). Um rótulo ausente continua
  devolvendo `[]`, nunca um erro: vocabulário perdido não pode derrubar a sessão.
- **Resiliência do banco.** `__aexit__` fecha o SQLite num bloco `finally`, então uma
  exceção no meio de um turno ainda grava `ended_at`. Observado ao vivo: um `TypeError`
  em `_run_turn` derrubou o script e a linha da sessão saiu com `ended_at` preenchido.
- `session_scores()` continua devolvendo só os números (é o que `suggest_level`
  usa); `score_history()` devolve `(id, média)` e é o que alimenta o gráfico — sem
  isso o eixo "sessão" contava de trás pra frente.
- `--answers "snn"` existe para rodar `review` sem TTY (aceite automatizado e
  scripts); sem ele e sem terminal, `review` só imprime os cards.

---

### FASE 10 — Interface

**10a. Terminal (`ui/terminal.py`, com `rich`)**
- [x] Painel com: nível, cenário, estado atual (`Listening…`, `Thinking…`, `Speaking…`), transcrição do usuário e da IA, caixa de **feedback** do último turno (erros em vermelho, correção em verde), e latência do último turno.
- [x] Ao terminar (`Ctrl+C`), encerra com cuidado, gera o resumo e mostra na tela.
- [x] Flag `--hide-transcript` (treino de escuta pura: não mostra o texto da IA).

**Aceite 10a** (medido — ver README §4.12)
- **Sessão completa só no terminal**: `python -m tutor --level beginner --scenario
  restaurant --turns 1` → tabela de resolução, abertura (painel de latência `+133 ms`),
  `fala encerrada (VAD)`, caixa `correcao - turno 1 nota 10/10`, painel do turno
  (**966 ms**) e um único painel de resumo; saída `0`.
- **Ctrl+C durante a escuta**: saída limpa (`interrompido pelo usuario.`), painel final
  (`estado encerrando...`) e resumo impresso **uma vez**; a linha da sessão fecha com
  `ended_at` **e** `summary` no `data/tutor.db`.
- **`--hide-transcript`**: `tutor: (transcript oculto)` na abertura, no turno e no
  dashboard; a transcrição do aluno continua visível.
- **270 testes** passam (`tests/test_ui_terminal.py` = 17).

**Descobertas desta fase (10a)**
- **`rich.Live` engole o que se imprime por baixo dele.** O resumo saía **duas vezes**
  (uma dentro do `Live`, outra depois de `stop()`) e o painel final colava no resumo
  (`╯╭──`). `Session.summarize()` agora **só retorna o texto**; quem imprime é o chamador
  (`run_tutor`, depois de `ui.stop()`; `bench_latency.py` no lugar dele), e `run_tutor`
  imprime uma linha em branco antes. O mesmo caminho cobre o caso Ctrl+C.
- **`Session` não deve conhecer `Panel`/`Table`.** Ela constrói a própria `TerminalUI` e
  delega toda a renderização; os imports de `rich` saíram de `core/session.py`. O import
  de `Session` dentro de `ui/terminal.py` é **local** (`from tutor.core.session import
  Session` dentro de `run_tutor`) — é o que quebra o ciclo `session → ui → session`.
- O pipeline ganhou `on_state: Callable[[str], None] | None`, chamado com `"speaking"` no
  **primeiro áudio** do TTS: é a única marca que a UI não consegue deduzir sozinha
  (`listening` ela mesma seta; `thinking` ela seta na transcrição).
- **Injeção de fala sem voz humana = monitor do null-sink.** `pactl set-default-source
  tutor_virtual.monitor` **muda o que o `sd.InputStream(device=None)` do PortAudio
  escuta** (confirmado: rms 0.062, peak 0.54, 1 utterance do próprio `TurnDetector`);
  não dá para injetar acusticamente (DMIC + alto-falantes voltados para cima). O teste
  roda com `timeout --signal=INT` e **exige restaurar** (`pactl set-default-source
  alsa_input...` + `pactl unload-module <id>`). Pegadinha: `MOD=$(...) && ... &` roda a
  cadeia inteira **num subshell**, então `$MOD` chega vazio no shell pai e o
  `unload-module` falha em silêncio — capture o módulo antes do `&`.
- Na primeira tentativa o `llama-server` sumiu no meio da sessão (sem log de causa) e o
  resumo falhou com `Connection error.`; reiniciado com o comando do README §3 e as duas
  rodadas seguintes completaram.

**10b. Painel web opcional (`ui/gradio_app.py`)**
- [ ] O áudio continua sendo capturado/tocado **localmente** via `sounddevice` em segundo plano; o Gradio serve só como **painel de controle e exibição** (selecionar nível/cenário/voz/sotaque, botões iniciar/parar, transcrição ao vivo, feedback, histórico). Evita a complexidade de streaming de áudio pelo navegador.

**Aceite**
- Uma sessão completa usando só o terminal, do início ao resumo final.  → **cumprido em 10a.**

---

### FASE 11 — Robustez e extras

**Tarefas**
- [ ] **Barge-in (opcional, flag `--barge-in`):** se o VAD detectar fala do usuário por ≥ 300 ms enquanto a IA fala, chamar `playback.stop()`, cancelar as tarefas do turno e processar a nova fala. Exige **fones de ouvido** (senão a IA escuta a si mesma). **Pulado por decisão do usuário** — era opcional e não há fones disponíveis para validar o teste ao vivo.
- [x] **Resumo rolante do histórico:** quando passar de `history_max_turns`, pedir um resumo curto (≤ 80 palavras) dos turnos antigos ao LLM de feedback e substituí-los. Isso mantém o contexto pequeno e o prefill rápido.
- [x] **Tratamento de falhas:** servidor LLM fora do ar (mensagem clara e tentativa de reconexão), microfone ausente, STT retornando vazio (ignorar o turno), TTS com erro numa frase (pular a frase e logar).
- [ ] **Teste de estabilidade:** sessão de 15 min sem vazamento de memória/VRAM (`nvidia-smi` antes e depois). **Executado em 2026-10-01 — FALHOU:** 181 turnos / 15 min, RSS 2517 → 2926 MiB (drift 409 MiB, ~2,3 MiB/turno, perfeitamente linear); VRAM estável (drift 48 MiB, pico 4193 MiB < 5632). Fica **pendente** até corrigir o vazamento — ver Descobertas.
- [x] **README.md** final: instalação no Windows passo a passo, como iniciar o `llama-server`, como rodar, como escolher nível/cenário, tabela de latência medida, solução de problemas comuns (CUDA, espeak-ng, dispositivo de áudio).

**Aceite**
- Definição de pronto (seção 12).

**Descobertas desta fase**
- **`/health` é um sinal seguro de reconexão.** Medido com o servidor sob carga: o
  `GET /health` devolve **200 enquanto gera** (`{"status":"ok"}`) e 503 só enquanto o
  modelo carrega — então dá para pollar sem interromper nada. `llm/health.py` monta a
  URL tirando o sufixo `/v1` e espera `pipeline.reconnect_timeout_s` (novo em
  `config.yaml`, 20 s).
- **O gate de feedback fica fechado durante a reconexão.** `turn()` limpa `idle` antes
  de chamar o pipeline e só reabre no `finally`; como `wait_for_server()` roda **dentro**
  do `except` (antes do `finally`), a correção de fundo não consegue pegar o slot único
  (`-np 1`) enquanto esperamos o servidor voltar. Não há `await` entre o `finally` da
  tentativa 1 e o `idle.clear()` da tentativa 2, então não existe janela de corrida.
- **4xx não é "servidor fora do ar".** `is_llm_unavailable` aceita `APIConnectionError`,
  5xx e `ConnectionError`/`TimeoutError`, e **recusa** 4xx: um model name errado
  repetido infinitamente não é reconexão, é bug nosso.
- **Falha de TTS deixa de ser fatal.** `streaming_turn` agora pula a frase que não
  sintetizou (`[AVISO] frase nao sintetizada ...`) e o `sequential_turn`/`greet()`
  seguem em silêncio: a resposta já entrou na história e na linha do turno, perder o
  turno inteiro (e a correção enfileirada junto) seria pior que uma frase muda. Dois
  testes que esperavam a exceção subiram foram reescritos para o novo contrato.
- **O resumo rolante vai na mensagem `system`, não vira mensagem nova.** Servidores
  OpenAI-compatíveis variam em quantos `system` aceitam no meio da lista, e um
  `assistant` que a IA nunca falou envenena a própria voz. `DialogueManager.record()`
  passou a **devolver os pares ejetados** (antes descartava em silêncio); a sessão
  enfileira num único task — uma segunda ejeção durante o voo só estende o lote, e o
  worker seguinte recebe `Earlier summary: <digest anterior>` para encadear.
- **O rollup é cancelado, não esperado, no fim.** `summarize()` cancela o task antes de
  `wait_idle()`: o digest só serve para turnos futuros, e deixá-lo disputar o slot com o
  resumo final atrasaria o que o aluno está esperando. Falha do LLM no rollup mantém o
  digest anterior (`None` não sobrescreve).
- **`rich.Live` + 15 min de fala em loop.** O teste de estabilidade (`scripts/check_
  stability.py`) reaproveita `_load_wav`/`_queue_for_vad`/`_wait_unmuted` do
  `bench_latency.py` e imprime `turno/rss/vram/latencia` a cada turno; ele **fala em
  voz alta** a sessão inteira (é playback de verdade, que é o que está sendo medido).
- **O vazamento de RSS é de nível C — Python está limpo.** Diagnóstico feito com três
  ferramentas: `tracemalloc` (15 níveis) devolve diff **vazia** entre o turno 4 e o 16 e
  `sys.getallocatedblocks()`/`len(gc.get_objects())` ficam **planos** (~2 928 000 /
  ~532 150, variação < 0,1 %); `malloc_trim(0)` devolve só **6 MiB** dos ~42 medidos;
  `check_stability.py --minutes 3.5` **sem feedback** ainda sobe 86 MiB em 63 turnos —
  ou seja, o crescimento (~1,2 MiB/turno) vem do núcleo (STT/LLM/TTS/playback), não do
  feedback nem do rollup (com feedback chega a ~2,3 MiB/turno). A fila de captura
  (`capture.queue`) foi descartada como suspeita: está **constante em 22 blocos**.
- **Medir `/proc/self/statm` exige o índice certo.** O formato é
  `size resident shared text lib data dt`: `check_stability.py` usa `[1]` (resident),
  que está correto; um script de diagnóstico usando `[2]` mede **shared** e reporta
  ~620 MiB planos — resultado falso de "sem vazamento". Confiar no script de soak, não
  em cópias ad-hoc.

### FASE 12 — Contexto do STT e correção falada (escopo novo, aprovado pelo usuário)

**Tarefas**
- [x] **Contexto para o Whisper (`stt.use_dialogue_context`, `stt.hotwords`):** mandar as
  últimas ~4 mensagens do diálogo como `initial_prompt` e as palavras-chave do cenário
  como `hotwords`, para o modelo ouvir `table` em vez de `fable`. Sem isso, o erro fica
  no histórico e arrasta a conversa para fora do contexto.
- [x] **Porta de confiança do STT (`stt.min_avg_logprob`, `stt.max_compression_ratio`):**
  se a transcrição for chute, **não** mandar para o LLM, não gravar no histórico nem no
  banco — falar "não entendi, repete?" e encerrar o turno. Linhas em
  `prompts/clarify.md` (`EN:`/`PT:`), em português só quando o nível tiver
  `allow_portuguese_help`.
- [x] **Correção falada (`pipeline.speak_feedback`, `speak_feedback_wait_s`):** falar a
  frase corrigida em inglês e a explicação em português **só quando houver erro**,
  esperando até 4 s pela correção em segundo plano. O texto falado vem do JSON do
  modelo (`spoken_pt`), com fallback para `errors[0].explanation_pt` — nada de
  português escrito à mão em `core`.
- [x] **Pipeline pt-BR do Kokoro (`tts.feedback_voice=pf_dora`, `tts.feedback_lang=pt-br`):**
  segunda `KPipeline` compartilhando o `KModel` da inglesa (VRAM extra ~0). Aquecida em
  `__aenter__`; se falhar (rede, por exemplo), a correção falada é desligada com
  `[AVISO]` em vez de quebrar a sessão.
- [x] **Flag `--no-spoke-feedback`** no CLI.
- [x] **Scripts:** `check_stt.py` imprime a confiança e o veredito do gate (aceita /
  pediria repetição) e aceita `--prompt`/`--hotwords`; `check_tts.py` ganhou
  `--lang pt-br --voice`.
- [x] **Testes** (36 novos: `tests/test_spoken_feedback.py` + seções em `test_tts.py`,
  `test_feedback_parse.py`, `test_stt.py`) e **medição real** antes de fixar os limites.

**Aceite**
- Uma transcrição fraca nunca chega ao LLM nem ao histórico.
- Com erro: ouve-se inglês corrigido e depois português; sem erro: não se ouve nada.
- A latência do turno não muda (a correção entra depois de `t_speech_end`).
- Nenhum português fixado em `core/`.

**Descobertas desta fase**
- **A porta de confiança não apanha erro confiante.** Medido com `check_stt.py` sobre
  `speech_b.wav` degradado por ruído: em **−5 dB de SNR** o Whisper devolve
  `"I look like dangle for two people"` com `avg_logprob = −0,647` e
  `compression_ratio = 0,818` — **passaria** por qualquer gate razoável. Os fixtures
  limpos ficam em −0,37 / −0,31 / −0,72 (o de 0,4 s é o pior) e o ruído branco puro
  devolve **transcrição vazia**, que já era descartada antes. Ou seja: `min_avg_logprob
  = −1.0` serve para **lixo**, não para desentendimento; a proteção real contra deriva
  de contexto é o `initial_prompt`/`hotwords`, medido logo abaixo.
- **O contexto muda a transcrição, e melhora a confiança junto.** No mesmo clipe a −5 dB:
  sem pista → `"I look like dangle for two people"` (−0,647); com
  `--prompt "Learner: I would like a table for two people please"` → `"I would like a
  table for two people please"` (**−0,294**); com `--hotwords "table, people, please"` →
  `"I look like a table for two people, please"` (**−0,386**). As duas medidas valem a
  pena; o prompt de diálogo ajuda mais.
- **`no_speech_prob` ficou de fora do gate de propósito.** No clipe de 0,4 s
  (`"Hi."`, transcrição correta) ele é **0,112**; incluí-lo derrubaria um turno bom.
- **Kokoro não mistura idiomas.** Testado: `KPipeline(lang_code='p')` fonemizando texto
  em inglês produz `"I want eat"` → `ˈi wˈɐ̃ŋʧ eˈať`. Por isso `synthesize_stream(lang=)`
  escolhe **pipeline**, nunca só a voz — e um `lang` desconhecido **levanta erro** em
  vez de falar errado.
- **As duas pipelines custam ~0 de VRAM.** `load_feedback()` passa o `KModel` da
  inglesa; só o fonemizador é diferente. O arquivo de voz (`pf_dora.pt`, ~1 MB) vem do
  Hub na **primeira** síntese — por isso o warmup em `__aenter__`, que troca o primeiro
  chunk de 972 ms por **87 ms** (inglês: 100 ms). Sem warmup a primeira correção
  pagaria a rede.
- **`KPipeline` avisa se `repo_id` for nulo** (`"WARNING: Defaulting repo_id..."`).
  Passar `hexgrad/Kokoro-82M` explicitamente nas duas criações silencia o aviso e
  garante que elas apontam para os mesmos pesos.
- **`clean_for_tts` comia o espaço antes de aspas.** `"O correto e 'went'"` virava
  `"O correto e'went'"`, que o espeak lê diferente; o `'` saiu da lista de pontuação
  que é colada à palavra anterior.
- **Ordem importa: o waiter não pode tirar o slot antes da chegada.** Na primeira
  versão `_wait_feedback` fazia `pop()` do evento e *depois* esperava, então
  `_complete_feedback` não encontrava o evento, não gravava o resultado e o turno
  **sempre** estourava o timeout — silenciosamente, com a correção na mão. Os testes
  com entrega síncrona passavam; só o teste com `delay_s > 0` e o end-to-end real
  pegaram. Corrigido e coberto por regressão.
- **End-to-end sem fones.** `/tmp/opencode/e2e_spoken.py` roda Whisper real +
  `llama-server` + Kokoro real com playback descartando o áudio: 3 turnos → 9 blocos
  (resposta + inglês + português), `spoken_pt` vindo do modelo. Serve de smoke test
  quando não há fone; o teste **ao vivo com fone** continua pendente.

---

## 7. Comando de uso esperado

```
# terminal 1: LLM local
llama-server -m models/Qwen3-4B-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080 --jinja

# terminal 2: tutor
python -m tutor --level beginner --scenario restaurant
python -m tutor --level advanced --scenario free_chat --accent british
python -m tutor stats
python -m tutor review
```

---

## 8. Parâmetros de geração sugeridos (Qwen3, sem thinking)

`temperature 0.7`, `top_p 0.8`, `top_k 20`. `max_tokens` vem do nível. O `enable_thinking=false` é enviado **por requisição** (`chat_template_kwargs`), evitando problemas de aspas de JSON na linha de comando do Windows. Se o backend ou a versão do servidor não aceitar esse campo, usar `/no_think` no início da mensagem do usuário como alternativa e registrar no README.

---

## 9. Medição de latência (`timeline.py`)

Por turno, registrar com `time.perf_counter()`:

| Marca | Significado |
|---|---|
| `t_speech_end` | VAD declarou fim da fala (já inclui a espera `end_silence_ms`) |
| `t_stt_done` | transcrição pronta |
| `t_llm_first_token` | 1º token do LLM |
| `t_first_sentence` | 1ª frase completa enviada ao TTS |
| `t_tts_first_audio` | 1º chunk de áudio gerado |
| `t_playback_start` | alto-falante começou a tocar |
| `t_llm_done` | LLM terminou |

Latência percebida = `t_playback_start − t_speech_end`. Imprimir a tabela por turno e médias/medianas no fim (`bench_latency.py` roda N turnos com áudios gravados para comparação reprodutível).

---

## 10. Se a latência passar da meta (ordem de investigação)

1. **Ver qual etapa domina** na tabela de tempos antes de mexer em qualquer coisa.
2. `end_silence_ms` alto demais? Reduzir (ex.: 1100 → 900 no avançado) **é o ganho mais barato**, ao custo de interromper pausas.
3. **Thinking realmente desligado?** (primeiro token demorando segundos indica que não).
4. **Prompt cache funcionando?** O prompt de sistema não deve ser reprocessado a cada turno.
5. Whisper lento → trocar para `distil-small.en` ou `base.en`; conferir se está mesmo na GPU.
6. TTS lento para o 1º áudio → garantir streaming de chunks, `warmup()`, e considerar `device: cpu` se a disputa por GPU for o problema (ou o inverso).
7. Contexto grande → reduzir `history_max_turns` e/ou `context_tokens`.
8. **Transcrição incremental** (transcrever a cada ~2 s enquanto o usuário fala, para o texto já estar pronto no fim): só implementar se os itens acima não bastarem.
9. Último recurso: LLM menor (Qwen3 1.7B) — ao custo de qualidade de gramática.

---

## 11. Riscos conhecidos

- **Qualidade do 4B** em nuances (nível avançado) é limitada; por isso o feedback pode usar modelo remoto maior.
- **Whisper pode "consertar" erros de fala**, escondendo problemas de pronúncia. Manter `beam_size` baixo e sem `initial_prompt`; pronúncia de verdade é um recurso futuro.
- **Eco:** usar fones de ouvido; o modo padrão já é half-duplex.
- **Modelos free remotos** mudam, têm limite de uso e podem sair do ar — por isso o fallback.
- **Disputa de GPU** entre LLM, Whisper e Kokoro pode causar picos; mover componentes para CPU via configuração.
- **Dependências sensíveis no Windows:** CUDA 12/cuDNN para o faster-whisper e espeak-ng no PATH para o Kokoro.

---

## 12. Definição de pronto (DoD)

- [ ] Sessão completa por voz nos **3 níveis**, em pelo menos **3 cenários**.
- [ ] Latência mediana **≤ 3,5 s** (fim da fala → início do áudio), medida e documentada no README.
- [ ] Sem pausas audíveis entre frases na maioria dos turnos.
- [ ] Feedback de correção aparece sem atrasar a conversa.
- [ ] Resumo de sessão gerado e sessão salva no SQLite; `stats` e `review` funcionam.
- [ ] Funciona **sem internet** (feedback local em modo ocioso).
- [ ] VRAM total < 5,5 GB durante uma sessão de 15 min, sem vazamento.
- [ ] `pytest` passa; README permite instalar do zero no Windows.

---

## 13. Checklist de progresso (o agente deve atualizar)

- [x] Fase 1 — Ambiente e verificação
- [x] Fase 2 — Scaffold, config, cliente LLM
- [x] Fase 3 — Captura, VAD, fim de turno
- [x] Fase 4 — STT, TTS, playback
- [x] Fase 5 — Baseline sequencial
- [x] Fase 6 — Pipeline em streaming
- [x] Fase 7 — Níveis, cenários, prompts
- [x] Fase 8 — Feedback e resumo
- [x] Fase 9 — Persistência e progresso
- [x] Fase 10 — Interface (10a feito; **10b pulado por decisão do usuário** — era opcional)
- [x] Fase 11 — Robustez e extras (falhas, resumo rolante e README feitos; **barge-in pulado por decisão do usuário**; **teste de estabilidade executado e FALHOU — vazamento de RSS em nível C ainda em aberto**, ver Descobertas desta fase)
- [x] Fase 12 — Contexto do STT e correção falada (gate de confiança, `initial_prompt`/`hotwords`, pipeline pt-BR do Kokoro, `spoken_pt`, `--no-spoke-feedback`; **teste ao vivo com fones ainda pendente**)

**Como começar:** leia este arquivo inteiro, confirme com o usuário o sistema operacional e se o `llama-server`/modelo já estão instalados, e então execute a **Fase 1**.
